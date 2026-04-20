"""Sagittal (side) view of the live SHM keypoints — **auto-fit** edition.

Runs as a **separate process** — reads `/dev/shm/hwalker_pose_cuda` and
renders Y-Z plane (walking direction) with dynamic auto-scaling so the
subject is always fully visible (no hardcoded ranges that cut off the
ankle or fail when the subject is close to the camera).

This mirrors mainline ``pipeline_main._display_sagittal`` which auto-fits
the visible joints to the canvas with a uniform aspect ratio and gravity
indicator. Perception pipeline is not affected (no X11, no GPU
contention, no GIL conflict).

Usage (in a separate terminal while run_stream_demo is publishing):
    PYTHONPATH=src python3 -m perception.CUDA_Stream.view_sagittal
    # matplotlib (default) — simple plot window
    # OR opencv backend for lower-latency rendering:
    PYTHONPATH=src python3 -m perception.CUDA_Stream.view_sagittal --backend opencv

Controls:
    q / ESC  — quit

Skeleton (lowlimb6 schema):
    left_hip — left_knee — left_ankle
    right_hip — right_knee — right_ankle
    + pelvis line (left_hip ↔ right_hip)

Coordinate frame
----------------
The CUDA_Stream pipeline publishes 3D points in a GRAVITY-ALIGNED world
frame (mainline Method B equivalent — an IMU warmup or --camera-pitch-deg
override at startup computes R_world_from_cam). So Y-axis in SHM is
"down" (+Y == down), Z is "forward" (away from camera), and this viewer
plots ``Z`` on the horizontal axis (forward) and ``-Y`` on the vertical
axis (up). The subject stays vertical even when the walker camera is
tilted (e.g. 32° forward).

Auto-fit contract
-----------------
* Keypoints with confidence ≥ ``--min-conf`` are considered for layout
* Canvas pads by ``margin`` px on each side, actual joint range sets scale
* Aspect ratio preserved (``scale = min(z_scale, y_scale)``) so thighs
  don't look stretched
* When no valid data, last scale is kept (avoids flicker)
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

from .keypoint_config import get_schema
from .shm_publisher import DEFAULT_NAME, ShmReader


SKELETON_LOWLIMB6 = [
    ("left_hip", "right_hip"),        # pelvis
    ("left_hip", "left_knee"),        # L thigh
    ("right_hip", "right_knee"),      # R thigh
    ("left_knee", "left_ankle"),      # L shank
    ("right_knee", "right_ankle"),    # R shank
]

SKELETON_COCO17_LOWER = [
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shm", default=DEFAULT_NAME.lstrip("/"))
    ap.add_argument("--schema", default="lowlimb6", choices=["coco17", "lowlimb6"])
    ap.add_argument("--fps", type=float, default=30.0,
                    help="render rate (doesn't affect perception)")
    ap.add_argument("--backend", default="matplotlib",
                    choices=["matplotlib", "opencv"])
    ap.add_argument("--min-conf", dest="min_conf", type=float, default=0.25,
                    help="keypoint confidence threshold for layout/drawing")
    ap.add_argument("--zoom", type=float, default=0.85,
                    help="auto-fit padding factor (0.5=tight, 1.0=fills canvas)")
    return ap.parse_args()


def _resolve_skeleton(schema, schema_name: str):
    segs = SKELETON_LOWLIMB6 if schema_name == "lowlimb6" else SKELETON_COCO17_LOWER
    return [(schema.index(a), schema.index(b)) for (a, b) in segs]


def main() -> int:
    args = parse_args()
    schema = get_schema(args.schema)

    try:
        reader = ShmReader(name=args.shm, expected_k=schema.num_keypoints)
    except FileNotFoundError:
        print(f"SHM '/dev/shm/{args.shm}' not found. Is run_stream_demo running?",
              file=sys.stderr)
        return 2

    skeleton = _resolve_skeleton(schema, args.schema)

    if args.backend == "opencv":
        return _run_opencv(reader, schema, skeleton, args)
    return _run_matplotlib(reader, schema, skeleton, args)


# ---------------------------------------------------------------------------
# Matplotlib backend — auto-fit via `ax.relim()` + `ax.autoscale_view()`
# ---------------------------------------------------------------------------
def _run_matplotlib(reader, schema, skeleton, args) -> int:
    import matplotlib.pyplot as plt

    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 7))
    interval = 1.0 / max(args.fps, 1)

    scat_L = ax.scatter([], [], s=80, c="#4d7cff", label="left", zorder=3)
    scat_R = ax.scatter([], [], s=80, c="#ff4d4d", label="right", zorder=3)
    lines = [ax.plot([], [], "-", lw=2.5, c="#cccccc", zorder=2)[0]
             for _ in skeleton]
    title = ax.set_title("Sagittal — waiting for first valid frame")
    ax.set_xlabel("Z (forward, m)")
    ax.set_ylabel("-Y (up, m)")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="datalim")
    ax.legend(loc="upper right", fontsize=9)

    last_fps = 0.0
    t_prev = time.monotonic()
    n_ticks = 0

    # determine left/right indices once
    left_idx = [i for i, n in enumerate(schema.keypoints) if "left" in n]
    right_idx = [i for i, n in enumerate(schema.keypoints) if "right" in n]

    try:
        while plt.fignum_exists(fig.number):
            data = reader.read()
            t_next = time.monotonic() + interval
            if data is not None:
                frame_id, ts_ns, kpts_3d, kpt_conf, kpts_2d, box_conf, valid, dir_ = data
                if valid:
                    mask = kpt_conf >= args.min_conf
                    z = kpts_3d[:, 2]
                    y_up = -kpts_3d[:, 1]

                    # scatter — left/right split with color
                    scat_L.set_offsets(
                        np.c_[z[left_idx][mask[left_idx]],
                              y_up[left_idx][mask[left_idx]]]
                    )
                    scat_R.set_offsets(
                        np.c_[z[right_idx][mask[right_idx]],
                              y_up[right_idx][mask[right_idx]]]
                    )

                    # skeleton segments
                    for line, (a, b) in zip(lines, skeleton):
                        if mask[a] and mask[b]:
                            line.set_data([z[a], z[b]], [y_up[a], y_up[b]])
                        else:
                            line.set_data([], [])

                    # auto-fit based on visible joints; if none visible, keep previous
                    if mask.any():
                        zs, ys = z[mask], y_up[mask]
                        z_min, z_max = float(zs.min()), float(zs.max())
                        y_min, y_max = float(ys.min()), float(ys.max())
                        z_span = max(z_max - z_min, 0.2)
                        y_span = max(y_max - y_min, 0.2)
                        pad_z = z_span * (1.0 - args.zoom) + 0.05
                        pad_y = y_span * (1.0 - args.zoom) + 0.05
                        ax.set_xlim(z_min - pad_z, z_max + pad_z)
                        ax.set_ylim(y_min - pad_y, y_max + pad_y)

                    n_ticks += 1
                    now = time.monotonic()
                    if now - t_prev > 1.0:
                        last_fps = n_ticks / (now - t_prev)
                        n_ticks = 0
                        t_prev = now
                    title.set_text(
                        f"Sagittal — frame {frame_id}  conf {box_conf:.2f}  "
                        f"depth invalid {dir_:.1%}  viewer {last_fps:.0f}Hz"
                    )
                else:
                    title.set_text(f"Sagittal — frame {frame_id}  INVALID (no detect)")
            plt.pause(max(t_next - time.monotonic(), 0.001))
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()
        plt.close(fig)
    return 0


# ---------------------------------------------------------------------------
# OpenCV backend — mirrors mainline ``_display_sagittal`` style auto-fit.
# Lower latency than matplotlib, no X deps beyond cv2 window.
# ---------------------------------------------------------------------------
def _run_opencv(reader, schema, skeleton, args) -> int:
    import cv2

    W, H = 640, 800
    margin = 60
    interval_ms = max(int(1000 / args.fps), 10)

    # identify left/right joints by name so coloring works for any schema
    is_left = np.array(["left" in n for n in schema.keypoints], dtype=bool)
    is_right = np.array(["right" in n for n in schema.keypoints], dtype=bool)

    # Keep a "last known" scale so we don't flicker on invalid frames.
    last_scale: Optional[float] = None
    last_z_center = 1.5
    last_y_center = 1.0

    last_fps = 0.0
    t_prev = time.monotonic()
    n_ticks = 0

    def compute_scale(z_vals: np.ndarray, y_vals: np.ndarray
                      ) -> Tuple[float, float, float]:
        """Return (scale px/m, z_center m, y_center m) to fit points into canvas
        with `margin` px padding and uniform aspect ratio."""
        z_min, z_max = float(z_vals.min()), float(z_vals.max())
        y_min, y_max = float(y_vals.min()), float(y_vals.max())
        z_span = max(z_max - z_min, 0.2)
        y_span = max(y_max - y_min, 0.2)
        s_z = (W - 2 * margin) / z_span
        s_y = (H - 2 * margin) / y_span
        scale = min(s_z, s_y) * args.zoom
        z_center = (z_min + z_max) / 2
        y_center = (y_min + y_max) / 2
        return scale, z_center, y_center

    def to_screen(z: float, y_up: float,
                  scale: float, z_center: float, y_center: float) -> Tuple[int, int]:
        sx = int(margin + (z - z_center) * scale + (W - 2 * margin) / 2)
        # cv2 image has origin top-left; y_up grows upward, so invert on screen
        sy = int(H - margin - (y_up - y_center) * scale - (H - 2 * margin) / 2)
        return sx, sy

    try:
        while True:
            img = np.zeros((H, W, 3), np.uint8)
            img[:] = (30, 30, 30)

            data = reader.read()
            frame_id = -1
            box_conf = 0.0
            dir_ = 0.0
            valid = False
            draw_pts: List[Tuple[int, int]] = []
            mask: Optional[np.ndarray] = None

            if data is not None:
                frame_id, ts_ns, kpts_3d, kpt_conf, kpts_2d, box_conf, valid, dir_ = data

                if valid:
                    z = kpts_3d[:, 2]
                    y_up = -kpts_3d[:, 1]
                    mask = (kpt_conf >= args.min_conf)

                    if mask.any():
                        scale, zc, yc = compute_scale(z[mask], y_up[mask])
                        last_scale, last_z_center, last_y_center = scale, zc, yc

            if last_scale is None:
                # nothing visible yet — show gentle hint text
                cv2.putText(img, "Sagittal — waiting for first valid frame",
                            (20, H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (180, 180, 180), 1, cv2.LINE_AA)
            else:
                scale = last_scale
                zc = last_z_center
                yc = last_y_center

                # ── Grid / axis labels ──────────────────────────────────
                # meter ticks on Z axis (forward)
                z_low = zc - (W / 2 - margin) / scale
                z_high = zc + (W / 2 - margin) / scale
                for z_tick in np.arange(np.floor(z_low), np.ceil(z_high) + 1, 0.5):
                    sx, _ = to_screen(z_tick, yc, scale, zc, yc)
                    if margin < sx < W - margin:
                        cv2.line(img, (sx, margin), (sx, H - margin),
                                 (55, 55, 55), 1)
                        if abs(z_tick - round(z_tick)) < 1e-3:
                            cv2.putText(img, f"{z_tick:.0f}m",
                                        (sx - 10, H - margin + 18),
                                        cv2.FONT_HERSHEY_PLAIN, 1,
                                        (150, 150, 150), 1, cv2.LINE_AA)
                # gravity arrow (down) top-right
                arrow_x = W - margin - 10
                arrow_y0 = margin + 20
                arrow_y1 = margin + 70
                cv2.arrowedLine(img, (arrow_x, arrow_y0), (arrow_x, arrow_y1),
                                (200, 200, 200), 2, tipLength=0.4)
                cv2.putText(img, "g", (arrow_x + 6, arrow_y1 - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1, cv2.LINE_AA)

                if valid and mask is not None:
                    z = kpts_3d[:, 2]
                    y_up = -kpts_3d[:, 1]
                    draw_pts = [to_screen(float(z[i]), float(y_up[i]),
                                          scale, zc, yc)
                                for i in range(len(z))]

                    # skeleton first (under circles)
                    for (a, b) in skeleton:
                        if mask[a] and mask[b]:
                            cv2.line(img, draw_pts[a], draw_pts[b],
                                     (210, 210, 210), 2, cv2.LINE_AA)

                    # points — left blue, right red
                    for i, pt in enumerate(draw_pts):
                        if not mask[i]:
                            continue
                        col = (255, 130, 80) if is_right[i] else (80, 80, 255)
                        cv2.circle(img, pt, 6, col, -1, cv2.LINE_AA)
                        cv2.circle(img, pt, 6, (255, 255, 255), 1, cv2.LINE_AA)

                    # bone length text panel (lower-left)
                    name_to_i = {n: i for i, n in enumerate(schema.keypoints)}
                    info_y = H - margin + 38
                    for side in ("left", "right"):
                        for (a, b) in (("hip", "knee"), ("knee", "ankle")):
                            an, bn = f"{side}_{a}", f"{side}_{b}"
                            if an in name_to_i and bn in name_to_i:
                                ia, ib = name_to_i[an], name_to_i[bn]
                                if mask[ia] and mask[ib]:
                                    length_m = float(np.linalg.norm(
                                        kpts_3d[ia] - kpts_3d[ib]))
                                    col = (80, 80, 255) if side == "left" else (255, 130, 80)
                                    cv2.putText(img,
                                                f"{side[0].upper()}-{a}{b}: {length_m*100:.1f}cm",
                                                (10, info_y),
                                                cv2.FONT_HERSHEY_SIMPLEX,
                                                0.5, col, 1, cv2.LINE_AA)
                                    info_y += 16

            # viewer render rate
            n_ticks += 1
            now = time.monotonic()
            if now - t_prev > 1.0:
                last_fps = n_ticks / (now - t_prev)
                n_ticks = 0
                t_prev = now

            # header
            if frame_id >= 0:
                header = (f"frame {frame_id}  conf {box_conf:.2f}  "
                          f"depth inv {dir_:.1%}  viewer {last_fps:.0f}Hz  "
                          f"{'VALID' if valid else 'INVALID'}")
            else:
                header = "no SHM data"
            cv2.putText(img, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # axis labels (bottom corners)
            cv2.putText(img, "Z (depth) ->", (W - 130, H - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1, cv2.LINE_AA)
            cv2.putText(img, "-Y (up)", (6, 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1, cv2.LINE_AA)

            cv2.imshow("sagittal", img)
            k = cv2.waitKey(interval_ms) & 0xFF
            if k in (ord("q"), 27):
                break
    finally:
        reader.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
