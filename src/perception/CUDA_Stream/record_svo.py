"""Record a ZED X Mini session to SVO2 for offline replay.

Captures at the Track-B production settings (SVGA 960x600 @ 120fps,
PERFORMANCE depth) and writes an SVO2 file that preserves the full IMU
stream. The resulting file can be replayed through the perception
pipeline later — same R-matrix warmup, same 3D deprojection, same
sagittal transform — without needing the patient or the camera.

Usage:
    sudo nvpmodel -m 0 && sudo jetson_clocks
    mkdir -p data/recordings/$(date +%Y-%m-%d)
    PYTHONPATH=src python3 -m perception.CUDA_Stream.record_svo \\
        --out data/recordings/$(date +%Y-%m-%d)/walk_01.svo2 \\
        --duration 30

Press Ctrl-C to stop early. File size ~200-400 MB for a 30s H265 clip.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

try:
    import pyzed.sl as sl
except ImportError:
    print("pyzed.sl not found. Run this on the Jetson with ZED SDK installed.",
          file=sys.stderr)
    sys.exit(2)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--out", required=True,
                    help="output SVO2 path (e.g. data/recordings/2026-04-21/walk_01.svo2)")
    ap.add_argument("--duration", type=float, default=30.0,
                    help="seconds to record (default 30)")
    ap.add_argument("--resolution", default="SVGA",
                    choices=["SVGA", "VGA", "HD720", "HD1080", "HD1200"])
    ap.add_argument("--fps", type=int, default=60,
                    help="camera fps. NVENC H264/H265 encoders on Jetson cap at "
                         "60 fps; use LOSSLESS compression to record 120.")
    ap.add_argument("--depth-mode", dest="depth_mode", default="PERFORMANCE",
                    choices=["PERFORMANCE", "QUALITY", "ULTRA"],
                    help="NEURAL is forbidden — skiro-learnings")
    ap.add_argument("--compression", default="H265",
                    choices=["H264", "H265", "LOSSLESS", "LOSSY"],
                    help="SVO compression mode (H265 smallest, IMU always preserved)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    if args.depth_mode == "NEURAL":
        print("ERROR: NEURAL depth forbidden (skiro-learnings — 2.4x slowdown).",
              file=sys.stderr)
        return 1

    # NVENC H264/H265 hardware encoders on Jetson support up to 60 fps only.
    # If the user asks for more with a hardware compression mode, force
    # LOSSLESS so the encoder isn't silently downclocked to 30 fps.
    if args.fps > 60 and args.compression in ("H264", "H265"):
        print(f"ERROR: NVENC {args.compression} caps at 60 fps on Jetson; "
              f"you requested {args.fps} fps. Use --compression LOSSLESS "
              f"(larger file, no encoder) or lower --fps to 60.",
              file=sys.stderr)
        return 1

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() != ".svo2":
        out_path = out_path.with_suffix(".svo2")
        print(f"[rec] note: forced .svo2 extension → {out_path}")

    init = sl.InitParameters()
    init.camera_resolution = getattr(sl.RESOLUTION, args.resolution)
    init.camera_fps = args.fps
    init.coordinate_units = sl.UNIT.METER
    init.depth_mode = getattr(sl.DEPTH_MODE, args.depth_mode)
    init.depth_minimum_distance = 0.1

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(f"ZED open failed: {status}", file=sys.stderr)
        return 3

    compression = getattr(sl.SVO_COMPRESSION_MODE, args.compression)
    rec_params = sl.RecordingParameters(str(out_path), compression)
    rec_status = cam.enable_recording(rec_params)
    if rec_status != sl.ERROR_CODE.SUCCESS:
        print(f"enable_recording failed: {rec_status}", file=sys.stderr)
        cam.close()
        return 4

    rt = sl.RuntimeParameters()
    print(f"[rec] writing {out_path}")
    print(f"[rec]   {args.resolution} @ {args.fps}fps, depth={args.depth_mode}, "
          f"compression={args.compression}")
    print(f"[rec]   duration {args.duration}s (Ctrl-C to stop early)")
    print(f"[rec]   IMU stream is captured automatically by SVO2")

    t_start = time.monotonic()
    frames = 0
    drops = 0
    try:
        while time.monotonic() - t_start < args.duration:
            err = cam.grab(rt)
            if err == sl.ERROR_CODE.SUCCESS:
                frames += 1
                if frames % 120 == 0:
                    elapsed = time.monotonic() - t_start
                    fps_now = frames / elapsed if elapsed > 0 else 0
                    print(f"[rec]  t={elapsed:5.1f}s  frames={frames}  "
                          f"fps={fps_now:5.1f}  drops={drops}", flush=True)
            else:
                drops += 1
    except KeyboardInterrupt:
        print("\n[rec] Ctrl-C received — stopping")
    finally:
        elapsed = time.monotonic() - t_start
        cam.disable_recording()
        cam.close()
        size_mb = out_path.stat().st_size / 1e6 if out_path.exists() else 0.0
        avg_fps = frames / elapsed if elapsed > 0 else 0
        print(f"[rec] DONE")
        print(f"[rec]   frames  : {frames}")
        print(f"[rec]   drops   : {drops}")
        print(f"[rec]   elapsed : {elapsed:.1f}s")
        print(f"[rec]   avg fps : {avg_fps:.1f}")
        print(f"[rec]   size    : {size_mb:.1f} MB")
        print(f"[rec]   file    : {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
