"""Sagittal viewer — mainline draw_sagittal style + sticky display.

Reads ``/dev/shm/hwalker_pose_cuda`` (CUDA_Stream publisher) and renders
the Y-Z (sagittal) plane in a separate process. Mirrors the proven
``verify_geometry.draw_sagittal`` behavior plus fixes for issues found
during 2026-04-21 integration:

  * **All 6 keypoints always shown** (no min_conf gate). Low-confidence
    points just dim — never removed from layout. This eliminates the
    "hip/ankle disappears when arm moves" problem.
  * **Sticky display**: keep last valid frame on screen during a brief
    invalid burst (no flicker / no blank frames).
  * **Left/right separation in sagittal**: in pure Y-Z projection both
    legs overlap. We add a tiny visual X-offset (±2cm screen-space) so
    you can see them apart while still being a true sagittal view.
  * **OpenCV only** — matplotlib backend was 14Hz on Jetson; cv2 hits
    60Hz easily. Single rendering path = simpler.
  * **Auto-fit always uses ALL 6 points** (regardless of conf), so a low
    conf joint never escapes the canvas (root cause of "잘림").

Usage (separate terminal while run_stream_demo is publishing):
    PYTHONPATH=src python3 -m perception.CUDA_Stream.view_sagittal

Controls:
    q / ESC  — quit
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Tuple

import numpy as np

from .keypoint_config import get_schema
from .shm_publisher import DEFAULT_NAME, ShmReader


# Sagittal skeleton: pelvis + L thigh + R thigh + L shank + R shank
SKELETON = [
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
    ap.add_argument("--fps", type=float, default=60.0,
                    help="render rate cap (cv2.waitKey is 1ms; rate limited by sleep)")
    ap.add_argument("--zoom", type=float, default=0.85,
                    help="auto-fit padding factor (0.5=tight, 1.0=fills canvas)")
    ap.add_argument("--width", type=int, default=600)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--low-conf", dest="low_conf", type=float, default=0.30,
                    help="below this conf the keypoint is drawn dim/grey")
    ap.add_argument("--leg-offset-px", dest="leg_offset_px", type=int, default=8,
                    help="visual screen-space X offset between left/right legs (px)")
    ap.add_argument("--calib-frames", dest="calib_frames", type=int, default=30,
                    help="standing-calibration window: first N valid frames to "
                         "learn the human X axis (left_hip→right_hip direction)")
    ap.add_argument("--no-calib", dest="no_calib", action="store_true",
                    help="disable standing calibration; use raw world Y-Z (legacy)")
    ap.add_argument("--display-mode", dest="display_mode", default="offset",
                    choices=["pure", "offset"],
                    help="pure: L/R keypoints overlap exactly (true sagittal, "
                         "colour-only distinction); "
                         "offset: ±leg_offset_px horizontal separation (default)")
    return ap.parse_args()


def _build_human_frame(hip_vector_world: np.ndarray) -> Optional[np.ndarray]:
    """Build R_human_from_world: rotates world coords into human-aligned frame.

    Inputs:
      hip_vector_world: (3,) avg of (right_hip - left_hip) in world frame.
                        This is the human's lateral (X) axis after rotation.

    Output:
      R: (3,3) rotation s.t. ``p_human = R @ p_world``
        - human X = pelvis lateral (right side positive)
        - human Y = gravity down (== world Y, unchanged)
        - human Z = X × Y (human forward — perpendicular to pelvis line and
                            to gravity, in the front-of-body direction)

    Returns None if hip_vector is too small (degenerate, e.g. occlusion at
    calibration time). Caller should keep retrying until a clean frame.

    True sagittal projection is then ``(z_human, -y_human)`` per joint —
    independent of how the camera is pointed; the legs always look like a
    side-view skeleton because we project onto the plane PERPENDICULAR to
    the actual pelvis line, not perpendicular to the camera Z.
    """
    n = float(np.linalg.norm(hip_vector_world))
    if n < 0.05:  # < 5cm → bad (e.g. only one hip detected at calib time)
        return None
    x_human = hip_vector_world / n           # (3,)
    y_human = np.array([0.0, 1.0, 0.0])      # world Y is gravity (down)
    # Re-orthogonalize x against y (project out the Y component) so the
    # rotation is well-defined when the pelvis isn't perfectly horizontal.
    x_human = x_human - np.dot(x_human, y_human) * y_human
    nx = float(np.linalg.norm(x_human))
    if nx < 0.05:
        return None
    x_human /= nx
    z_human = np.cross(x_human, y_human)     # forward (right-handed)
    nz = float(np.linalg.norm(z_human))
    if nz < 0.05:
        return None
    z_human /= nz
    # Rows of R are the new basis expressed in old coords.
    R = np.stack([x_human, y_human, z_human], axis=0)  # (3,3)
    return R


def _resolve_index(schema, name: str) -> Optional[int]:
    try:
        return schema.index(name)
    except (ValueError, KeyError):
        return None


def main() -> int:
    import cv2

    args = parse_args()
    schema = get_schema(args.schema)

    try:
        reader = ShmReader(name=args.shm, expected_k=schema.num_keypoints)
    except FileNotFoundError:
        print(f"SHM '/dev/shm/{args.shm}' not found. Is run_stream_demo running?",
              file=sys.stderr)
        return 2

    # Resolve skeleton indices once
    skel_idx: List[Tuple[int, int]] = []
    for a, b in SKELETON:
        ia = _resolve_index(schema, a)
        ib = _resolve_index(schema, b)
        if ia is not None and ib is not None:
            skel_idx.append((ia, ib))

    # left/right per-keypoint (for color + offset)
    is_left = np.array(["left" in n for n in schema.keypoints], dtype=bool)
    is_right = np.array(["right" in n for n in schema.keypoints], dtype=bool)

    W, H = args.width, args.height
    margin = 50
    sleep_s = 1.0 / max(args.fps, 1.0)

    # Sticky state — keep last valid frame so transient invalid runs don't blank.
    last_kpts_3d: Optional[np.ndarray] = None
    last_kpt_conf: Optional[np.ndarray] = None
    last_box_conf = 0.0
    last_dir = 0.0
    last_frame_id = -1
    sticky_age_frames = 0

    # Resolve hip indices for calibration (need both for hip vector).
    lh_idx = _resolve_index(schema, "left_hip")
    rh_idx = _resolve_index(schema, "right_hip")

    # Standing-calibration buffer: collect first N valid hip vectors
    # (in world frame), average → human X axis. Then build R that
    # rotates world coords so the human's lateral X is aligned with
    # screen X — equivalently, world Z gets rotated to be the
    # walking-forward direction. The user sees a true side view of
    # the gait regardless of camera yaw.
    R_view: Optional[np.ndarray] = None
    calib_buf: List[np.ndarray] = []
    calib_done = False
    calib_failed = False

    # FPS counter
    fps_t0 = time.monotonic()
    fps_n = 0
    fps_show = 0.0

    try:
        while True:
            tick_start = time.monotonic()
            img = np.full((H, W, 3), 30, dtype=np.uint8)

            # ── Read latest SHM frame ──────────────────────────────────────
            data = reader.read()
            cur_valid = False
            cur_frame_id = -1
            if data is not None:
                cur_frame_id, _ts, kpts_3d, kpt_conf, _kpts_2d, box_conf, valid, dir_ = data
                if valid:
                    last_kpts_3d = np.asarray(kpts_3d).copy()
                    last_kpt_conf = np.asarray(kpt_conf).copy()
                    last_box_conf = float(box_conf)
                    last_dir = float(dir_)
                    last_frame_id = cur_frame_id
                    sticky_age_frames = 0
                    cur_valid = True

                    # ── Standing calibration (first N valid frames) ──
                    # Collect right_hip - left_hip in WORLD frame, average,
                    # use that as the "lateral" axis. Then world Z gets
                    # rotated to be the human walking-forward direction.
                    if (not args.no_calib and not calib_done
                            and lh_idx is not None and rh_idx is not None):
                        if (last_kpt_conf[lh_idx] >= 0.5
                                and last_kpt_conf[rh_idx] >= 0.5):
                            hip_vec = (last_kpts_3d[rh_idx]
                                       - last_kpts_3d[lh_idx])
                            calib_buf.append(hip_vec)
                            if len(calib_buf) >= args.calib_frames:
                                avg = np.mean(np.stack(calib_buf, axis=0), axis=0)
                                R_view = _build_human_frame(avg)
                                if R_view is None:
                                    calib_failed = True
                                    calib_done = True  # stop trying
                                else:
                                    calib_done = True
                else:
                    sticky_age_frames += 1

            # ── Apply walking-direction rotation (after calibration) ──────
            # Without this, world Z is just camera-horizontal-forward and the
            # legs look 'foreshortened' if the user isn't perfectly facing the
            # camera. With R_view, world Z rotates to align with the user's
            # actual walking direction → legs always look full-length in side view.
            display_kpts: Optional[np.ndarray] = None
            if last_kpts_3d is not None:
                if R_view is not None:
                    display_kpts = (R_view @ last_kpts_3d.T).T
                else:
                    display_kpts = last_kpts_3d

            # ── Layout: hip-center as fixed origin, all 6 keypoints ─────
            if display_kpts is not None:
                # C2: hip midpoint is the projection origin (in display frame).
                # Both L-hip and R-hip project to this point in pure sagittal.
                if lh_idx is not None and rh_idx is not None:
                    hip_ctr = 0.5 * (display_kpts[lh_idx] + display_kpts[rh_idx])
                else:
                    hip_ctr = display_kpts.mean(axis=0)

                # C4: Sx = −Z  (forward → left on screen)
                #     Sy = +Y  (world Y+ = down → screen bottom)
                # No sign flip needed for Y — direct mapping to gravity.
                z_all = display_kpts[:, 2]   # forward (m), world Z
                y_all = display_kpts[:, 1]   # down    (m), world Y

                # Auto-scale: distances from hip_ctr set the viewport.
                rel_z = z_all - hip_ctr[2]
                rel_y = y_all - hip_ctr[1]
                z_half  = max(float(np.abs(rel_z).max()), 0.15)   # symmetric ±Z
                y_below = max(float(rel_y.max()), 0.15)            # legs below hip
                y_above = max(float(-rel_y.min()), 0.05)           # head-room above

                s_z = (W / 2 - margin) / z_half
                s_y = (H - 3 * margin) / max(y_below + y_above, 0.30)
                scale = min(s_z, s_y) * args.zoom

                # Hip midpoint is anchored at upper 25 % of canvas.
                anchor_x = W // 2
                anchor_y = margin + int(H * 0.12)

                def to_screen(z: float, y: float, side_offset: int = 0) -> Tuple[int, int]:
                    sx = int(anchor_x - (z - hip_ctr[2]) * scale + side_offset)
                    sy = int(anchor_y + (y - hip_ctr[1]) * scale)
                    return sx, sy

                # Grid: vertical lines every 0.25 m along Z, centred on hip_ctr.
                tick = 0.25
                z_half_world = (W / 2 - margin) / scale
                z_low  = hip_ctr[2] - z_half_world
                z_high = hip_ctr[2] + z_half_world
                z_t = np.ceil(z_low / tick) * tick
                while z_t <= z_high:
                    sx, _ = to_screen(z_t, hip_ctr[1])
                    if margin < sx < W - margin:
                        cv2.line(img, (sx, margin), (sx, H - margin), (55, 55, 55), 1)
                        if abs(z_t - round(z_t)) < 1e-3:
                            cv2.putText(img, f"{z_t:.0f}m", (sx - 12, H - margin + 18),
                                        cv2.FONT_HERSHEY_PLAIN, 1, (140, 140, 140), 1, cv2.LINE_AA)
                    z_t += tick

                # Gravity arrow (top-right)
                ax, ay0, ay1 = W - margin - 10, margin + 16, margin + 60
                cv2.arrowedLine(img, (ax, ay0), (ax, ay1), (200, 200, 200), 2, tipLength=0.4)
                cv2.putText(img, "g", (ax - 14, ay1 - 18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

                # ── Compute per-side screen points ────────────────────────
                # C5: pure mode → no offset (true sagittal, colour-only);
                #     offset mode → ±leg_offset_px visual separation.
                offset = args.leg_offset_px if args.display_mode == "offset" else 0
                pts: List[Optional[Tuple[int, int]]] = [None] * len(z_all)
                for i in range(len(z_all)):
                    side_off = -offset if is_left[i] else (+offset if is_right[i] else 0)
                    pts[i] = to_screen(float(z_all[i]), float(y_all[i]), side_off)

                # ── Draw bones (under joints) ─────────────────────────────
                # Left = blue, right = red. Pelvis = grey.
                left_bone = (255, 130, 80)   # cv2 BGR — blue-ish for "left"
                right_bone = (80, 80, 255)   # red for "right"
                pelvis_bone = (160, 160, 160)
                for (a, b) in skel_idx:
                    pa, pb = pts[a], pts[b]
                    if pa is None or pb is None:
                        continue
                    if is_left[a] and is_left[b]:
                        col = left_bone
                    elif is_right[a] and is_right[b]:
                        col = right_bone
                    else:
                        col = pelvis_bone
                    cv2.line(img, pa, pb, col, 3, cv2.LINE_AA)

                # ── Draw joints (always shown; dim if low conf) ───────────
                for i, p in enumerate(pts):
                    if p is None:
                        continue
                    conf = float(last_kpt_conf[i]) if last_kpt_conf is not None else 1.0
                    dim = conf < args.low_conf
                    if is_left[i]:
                        fill = (255, 130, 80) if not dim else (120, 90, 70)
                    elif is_right[i]:
                        fill = (80, 80, 255) if not dim else (70, 70, 120)
                    else:
                        fill = (200, 200, 200)
                    cv2.circle(img, p, 7, fill, -1, cv2.LINE_AA)
                    cv2.circle(img, p, 7, (255, 255, 255), 1, cv2.LINE_AA)

                # ── Bone length panel (lower-left) — uses RAW world coords
                #    (not display_kpts) so cm reads the actual physical
                #    length, independent of the view rotation. ─────────────
                panel_y = H - margin + 38
                kp_name = {n: i for i, n in enumerate(schema.keypoints)}
                for side, scol in (("left", (255, 130, 80)), ("right", (80, 80, 255))):
                    for a, b in (("hip", "knee"), ("knee", "ankle")):
                        an, bn = f"{side}_{a}", f"{side}_{b}"
                        if an in kp_name and bn in kp_name:
                            ia, ib = kp_name[an], kp_name[bn]
                            length_cm = float(np.linalg.norm(
                                last_kpts_3d[ia] - last_kpts_3d[ib])) * 100
                            cv2.putText(
                                img,
                                f"{side[0].upper()}-{a}{b}: {length_cm:5.1f}cm",
                                (10, panel_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, scol, 1, cv2.LINE_AA,
                            )
                            panel_y += 16
            else:
                cv2.putText(img, "Sagittal — waiting for first valid frame",
                            (20, H // 2), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (180, 180, 180), 1, cv2.LINE_AA)

            # ── Header ──────────────────────────────────────────────────
            fps_n += 1
            now = time.monotonic()
            if now - fps_t0 > 1.0:
                fps_show = fps_n / (now - fps_t0)
                fps_n = 0
                fps_t0 = now

            valid_tag = "VALID" if cur_valid else f"sticky+{sticky_age_frames}"
            if args.no_calib:
                calib_tag = "calib OFF"
            elif calib_failed:
                calib_tag = "calib FAILED — raw view"
            elif calib_done:
                calib_tag = "calibrated"
            else:
                calib_tag = f"calibrating {len(calib_buf)}/{args.calib_frames}"
            header = (f"frame {last_frame_id}  conf {last_box_conf:.2f}  "
                      f"depth_inv {last_dir:.0%}  viewer {fps_show:.0f}Hz  "
                      f"{valid_tag}  [{calib_tag}]  mode:{args.display_mode}")
            cv2.putText(img, header, (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 1, cv2.LINE_AA)

            # Axis labels: Z increases going LEFT, Y increases going DOWN.
            cv2.putText(img, "<- Z (forward)", (10, H - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1, cv2.LINE_AA)
            cv2.putText(img, "+Y (down)", (6, anchor_y - 6 if display_kpts is not None else 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1, cv2.LINE_AA)

            cv2.imshow("sagittal", img)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break

            # Rate cap (don't burn CPU)
            elapsed = time.monotonic() - tick_start
            if elapsed < sleep_s:
                time.sleep(sleep_s - elapsed)
    finally:
        reader.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
