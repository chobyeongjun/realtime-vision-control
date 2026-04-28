#!/usr/bin/env python3
"""One-command SVO2 → YOLO dataset pipeline with correction GUI.

Linux + RTX5090 only.  Requires: pyzed  ultralytics  opencv-python  numpy

Flow
----
1. Read SVO2, sample at --sample-fps
2. Run yolo26s-lower6-v2.pt → YOLO-pose .txt per frame
3. OpenCV GUI — fix bad keypoints (click to select, click to move)
4. Write dataset.yaml ready for `yolo train`

Usage
-----
    # Full pipeline (one command)
    python3 scripts/data_collection/build_dataset.py \\
        --svo  recordings/2026-04-28/walk_tight.svo2 \\
        --model src/perception/models/yolo26s-lower6-v2.pt \\
        --out   datasets/v3

    # Multiple SVO2 files — run again, accumulates into same --out
    python3 scripts/data_collection/build_dataset.py \\
        --svo  recordings/2026-04-28/walk_exosuit.svo2 \\
        --model src/perception/models/yolo26s-lower6-v2.pt \\
        --out   datasets/v3

    # Skip GUI (no correction)
    python3 ... --no-correct

    # Only fix annotations (already extracted)
    python3 ... --correct-only

Keypoints (LOWLIMB6):
    0 left_hip   1 right_hip
    2 left_knee  3 right_knee
    4 left_ankle 5 right_ankle
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

try:
    import pyzed.sl as sl
except ImportError:
    print("ERROR: pyzed not found — run on Linux with ZED SDK installed.")
    sys.exit(1)

try:
    from ultralytics import YOLO
except ImportError:
    print("ERROR: pip install ultralytics")
    sys.exit(1)

# ── Keypoint schema ───────────────────────────────────────────────────────────
KP_NAMES  = ["L_hip", "R_hip", "L_knee", "R_knee", "L_ankle", "R_ankle"]
KP_COLORS = [
    (0,   0,   255),   # L_hip    red
    (255, 0,   0),     # R_hip    blue
    (0,   165, 255),   # L_knee   orange
    (255, 255, 0),     # R_knee   cyan
    (0,   255, 255),   # L_ankle  yellow
    (255, 0,   255),   # R_ankle  magenta
]
SKELETON = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5)]
N_KP = 6


# ── YOLO pose annotation I/O ─────────────────────────────────────────────────
def kps_to_yolo_line(kps: np.ndarray) -> str:
    """kps: (N_KP, 3) [x_norm, y_norm, vis]  vis: 0=absent 2=visible"""
    present = kps[kps[:, 2] > 0]
    if len(present) < 2:
        return ""
    xs, ys = present[:, 0], present[:, 1]
    cx = float((xs.min() + xs.max()) / 2)
    cy = float((ys.min() + ys.max()) / 2)
    w  = float(xs.max() - xs.min()) + 0.12
    h  = float(ys.max() - ys.min()) + 0.15
    w  = min(w, 1.0); h = min(h, 1.0)
    cx = float(np.clip(cx, w / 2, 1 - w / 2))
    cy = float(np.clip(cy, h / 2, 1 - h / 2))
    parts = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"]
    for x, y, v in kps:
        vis = 2 if v > 0.3 else 0
        parts.append(f"{x:.6f} {y:.6f} {vis}")
    return " ".join(parts)


def load_kps(lbl_path: Path) -> Optional[np.ndarray]:
    """Load first person's keypoints → (N_KP, 3) or None."""
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return None
    for line in lbl_path.read_text().splitlines():
        tokens = line.strip().split()
        if len(tokens) >= 5 + N_KP * 3:
            return np.array(tokens[5:5 + N_KP * 3], float).reshape(N_KP, 3)
    return None


def save_kps(lbl_path: Path, kps: np.ndarray):
    line = kps_to_yolo_line(kps)
    lbl_path.write_text(line + "\n" if line else "")


# ── Phase 1: extract + pre-annotate ──────────────────────────────────────────
def phase_extract(svo_path: Path, model_path: Path, img_dir: Path, lbl_dir: Path,
                  sample_fps: float, conf_thresh: float) -> int:
    model = YOLO(str(model_path))

    init = sl.InitParameters()
    init.set_from_svo_file(str(svo_path))
    init.coordinate_units = sl.UNIT.METER
    init.depth_mode = sl.DEPTH_MODE.PERFORMANCE

    cam = sl.Camera()
    if cam.open(init) != sl.ERROR_CODE.SUCCESS:
        print(f"ERROR: cannot open {svo_path}")
        return 0

    cam_fps = cam.get_camera_information().camera_configuration.fps
    interval = max(1, round(cam_fps / sample_fps))
    rt = sl.RuntimeParameters()
    zed_img = sl.Mat()

    saved = 0
    frame_idx = 0
    prefix = svo_path.stem

    print(f"[extract] {svo_path.name}  {cam_fps:.0f}fps → sample {sample_fps}fps "
          f"(every {interval} frames)")

    try:
        while True:
            err = cam.grab(rt)
            if err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                break
            if err != sl.ERROR_CODE.SUCCESS:
                frame_idx += 1
                continue

            if frame_idx % interval == 0:
                cam.retrieve_image(zed_img, sl.VIEW.LEFT)
                bgr = zed_img.get_data()[:, :, :3].copy()
                h, w = bgr.shape[:2]

                stem     = f"{prefix}_{frame_idx:07d}"
                img_path = img_dir / f"{stem}.jpg"
                lbl_path = lbl_dir / f"{stem}.txt"

                if not lbl_path.exists():   # resume support
                    results = model.predict(bgr, conf=conf_thresh, verbose=False)
                    kps_norm: Optional[np.ndarray] = None
                    if results and results[0].keypoints is not None:
                        kp_obj = results[0].keypoints
                        if len(kp_obj) > 0:
                            xyn  = kp_obj.xyn[0].cpu().numpy()   # (N_KP, 2)
                            conf = kp_obj.conf[0].cpu().numpy()   # (N_KP,)
                            kps_norm = np.concatenate([xyn, conf[:, None]], axis=1)

                    cv2.imwrite(str(img_path), bgr)
                    if kps_norm is not None:
                        save_kps(lbl_path, kps_norm)
                    else:
                        lbl_path.write_text("")

                    saved += 1
                    if saved % 20 == 0:
                        print(f"[extract]  {saved} frames  (svo frame {frame_idx})", flush=True)

            frame_idx += 1

    finally:
        cam.close()

    print(f"[extract] done — {saved} new frames saved")
    return saved


# ── Phase 2: correction GUI ───────────────────────────────────────────────────
SELECT_PX = 22   # px radius to select a keypoint
KP_RADIUS = 9


class AnnotationGUI:
    def __init__(self, img_dir: Path, lbl_dir: Path):
        self.img_paths = sorted(img_dir.glob("*.jpg"))
        self.lbl_dir   = lbl_dir
        self.idx        = 0
        self.kps: Optional[np.ndarray] = None
        self.img: Optional[np.ndarray] = None
        self.sel: Optional[int] = None   # selected keypoint index
        self.dirty = False

    # ── load/save ──────────────────────────────────────────────────────────
    def load(self):
        p = self.img_paths[self.idx]
        self.img  = cv2.imread(str(p))
        lbl       = self.lbl_dir / (p.stem + ".txt")
        loaded    = load_kps(lbl)
        self.kps  = loaded if loaded is not None else np.zeros((N_KP, 3), float)
        self.sel  = None
        self.dirty = False

    def save(self):
        if not self.dirty or self.kps is None:
            return
        p = self.img_paths[self.idx]
        save_kps(self.lbl_dir / (p.stem + ".txt"), self.kps)
        self.dirty = False

    # ── drawing ────────────────────────────────────────────────────────────
    def draw(self) -> np.ndarray:
        vis = self.img.copy()
        h, w = vis.shape[:2]

        # skeleton
        for a, b in SKELETON:
            if self.kps[a, 2] > 0 and self.kps[b, 2] > 0:
                pa = (int(self.kps[a, 0] * w), int(self.kps[a, 1] * h))
                pb = (int(self.kps[b, 0] * w), int(self.kps[b, 1] * h))
                cv2.line(vis, pa, pb, (180, 180, 180), 2)

        # keypoints
        for i, (x, y, v) in enumerate(self.kps):
            if v == 0:
                continue
            px, py = int(x * w), int(y * h)
            col   = KP_COLORS[i]
            thick = 3 if i == self.sel else 1
            cv2.circle(vis, (px, py), KP_RADIUS, col, thick)
            cv2.putText(vis, KP_NAMES[i], (px + 11, py + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1)

        # status bar
        n    = len(self.img_paths)
        name = self.img_paths[self.idx].name
        cv2.rectangle(vis, (0, 0), (w, 28), (40, 40, 40), -1)
        cv2.putText(vis, f"[{self.idx+1}/{n}] {name}  |  n=next  p=prev  d=delete  q=quit",
                    (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 220, 220), 1)

        if self.sel is not None:
            cv2.putText(vis, f"selected: {KP_NAMES[self.sel]}  — click to move",
                        (6, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)

        return vis

    # ── mouse callback ─────────────────────────────────────────────────────
    def on_mouse(self, event, mx, my, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN or self.img is None:
            return
        h, w = self.img.shape[:2]
        nx, ny = mx / w, my / h

        # find nearest visible keypoint within SELECT_PX
        best_i, best_d = None, SELECT_PX
        for i, (x, y, v) in enumerate(self.kps):
            if v == 0:
                continue
            d = ((x * w - mx) ** 2 + (y * h - my) ** 2) ** 0.5
            if d < best_d:
                best_d, best_i = d, i

        if best_i is not None and self.sel != best_i:
            self.sel = best_i            # select keypoint
        elif self.sel is not None:
            # move selected keypoint to click position
            self.kps[self.sel, 0] = np.clip(nx, 0.0, 1.0)
            self.kps[self.sel, 1] = np.clip(ny, 0.0, 1.0)
            self.kps[self.sel, 2] = 2.0
            self.dirty = True
            self.sel = None

    # ── main loop ──────────────────────────────────────────────────────────
    def run(self):
        if not self.img_paths:
            print("[gui] no images found — nothing to correct")
            return

        cv2.namedWindow("annotation", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("annotation", 960, 640)
        cv2.setMouseCallback("annotation", self.on_mouse)
        self.load()

        deleted = 0
        while True:
            cv2.imshow("annotation", self.draw())
            key = cv2.waitKey(20) & 0xFF

            if key in (ord('n'), 83):          # n or →
                self.save()
                self.idx = min(self.idx + 1, len(self.img_paths) - 1)
                self.load()

            elif key in (ord('p'), 81):         # p or ←
                self.save()
                self.idx = max(self.idx - 1, 0)
                self.load()

            elif key == ord('d'):               # delete frame
                p = self.img_paths.pop(self.idx)
                (self.lbl_dir / (p.stem + ".txt")).unlink(missing_ok=True)
                p.unlink(missing_ok=True)
                deleted += 1
                if not self.img_paths:
                    break
                self.idx = min(self.idx, len(self.img_paths) - 1)
                self.load()

            elif key in (ord('q'), 27):         # q or Esc
                self.save()
                break

        cv2.destroyAllWindows()
        print(f"[gui] done — {deleted} frames deleted, "
              f"{len(self.img_paths)} remaining")


# ── dataset.yaml ─────────────────────────────────────────────────────────────
def write_yaml(out: Path, val_ratio: float = 0.1):
    all_imgs = sorted((out / "images").glob("*.jpg"))
    if not all_imgs:
        print("[yaml] no images — skipping yaml")
        return

    random.shuffle(all_imgs)
    n_val = max(1, int(len(all_imgs) * val_ratio))
    val_set  = {p.name for p in all_imgs[:n_val]}

    train_lines = [str(p) for p in all_imgs if p.name not in val_set]
    val_lines   = [str(p) for p in all_imgs if p.name in val_set]

    (out / "train.txt").write_text("\n".join(train_lines) + "\n")
    (out / "val.txt").write_text("\n".join(val_lines) + "\n")

    yaml = (
        f"path: {out.resolve()}\n"
        f"train: train.txt\n"
        f"val:   val.txt\n"
        f"\n"
        f"kpt_shape: [6, 3]  # N_KP, (x y visible)\n"
        f"\n"
        f"names:\n"
        f"  0: person\n"
    )
    (out / "dataset.yaml").write_text(yaml)
    print(f"[yaml] train={len(train_lines)}  val={len(val_lines)} → {out}/dataset.yaml")


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--svo",   help="SVO2 file path (skip with --correct-only)")
    ap.add_argument("--model", help=".pt model path (skip with --correct-only)")
    ap.add_argument("--out",   required=True, help="dataset root directory")
    ap.add_argument("--sample-fps", type=float, default=2.0,
                    help="sample rate in fps (default 2)")
    ap.add_argument("--conf", type=float, default=0.3,
                    help="YOLO confidence threshold (default 0.3)")
    ap.add_argument("--no-correct",   action="store_true", help="skip correction GUI")
    ap.add_argument("--correct-only", action="store_true", help="skip extraction, GUI only")
    return ap.parse_args()


def main():
    args = parse_args()
    out     = Path(args.out)
    img_dir = out / "images"
    lbl_dir = out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    if not args.correct_only:
        if not args.svo or not args.model:
            print("ERROR: --svo and --model required unless --correct-only")
            sys.exit(1)
        svo   = Path(args.svo)
        model = Path(args.model)
        if not svo.exists():
            print(f"ERROR: SVO not found: {svo}"); sys.exit(1)
        if not model.exists():
            print(f"ERROR: model not found: {model}"); sys.exit(1)
        phase_extract(svo, model, img_dir, lbl_dir, args.sample_fps, args.conf)

    if not args.no_correct:
        print("[gui] opening correction GUI …")
        AnnotationGUI(img_dir, lbl_dir).run()

    write_yaml(out)

    print(f"""
[done] dataset ready → {out}/dataset.yaml

Train command:
    yolo train \\
        data={out}/dataset.yaml \\
        model=yolo26s-pose.pt \\
        epochs=500 imgsz=640 \\
        batch=-1 patience=50 \\
        name=yolo26s-lower6-v3
""")


if __name__ == "__main__":
    main()
