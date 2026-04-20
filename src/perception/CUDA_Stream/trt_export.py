"""Export a YOLO26 (or YOLOv8) pose .pt to a TensorRT .engine file.

Run once on the Jetson — the engine is device/driver-specific.

Examples
--------
FP16:
    python3 -m perception.CUDA_Stream.trt_export \\
        --weights yolo26s-pose.pt --imgsz 640 --half \\
        --out src/perception/CUDA_Stream/yolo26s-pose.engine

INT8 (calibration):
    python3 -m perception.CUDA_Stream.trt_export \\
        --weights yolo26s-pose.pt --imgsz 640 --int8 \\
        --calib-dir data/calib_frames \\
        --out src/perception/CUDA_Stream/yolo26s-pose-int8.engine
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", required=True, help="path to .pt weights")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--half", action="store_true", help="FP16 engine")
    ap.add_argument("--int8", action="store_true", help="INT8 engine (needs calib)")
    ap.add_argument(
        "--calib-dir",
        type=str,
        default=None,
        help="Folder with ~50 PNG/JPG frames for INT8 calibration",
    )
    ap.add_argument("--workspace", type=int, default=2, help="GB")
    ap.add_argument("--dynamic", action="store_true")
    ap.add_argument("--device", default="0")
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    weights = Path(args.weights)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not weights.exists():
        LOGGER.error("weights not found: %s", weights)
        return 2
    if args.int8 and not args.calib_dir:
        LOGGER.error("--int8 requires --calib-dir")
        return 2

    try:
        from ultralytics import YOLO
    except ImportError as err:
        LOGGER.error(
            "ultralytics not installed. pip install ultralytics>=8.3 — %s", err
        )
        return 2

    model = YOLO(str(weights))
    export_kwargs = dict(
        format="engine",
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workspace=args.workspace,
        dynamic=args.dynamic,
        simplify=True,
    )
    if args.half and not args.int8:
        export_kwargs["half"] = True
    if args.int8:
        export_kwargs["int8"] = True
        export_kwargs["data"] = args.calib_dir

    LOGGER.info("exporting %s with %s", weights, export_kwargs)
    engine_path = model.export(**export_kwargs)
    engine_path = Path(engine_path)
    if not engine_path.exists():
        LOGGER.error("ultralytics export returned %s but file missing", engine_path)
        return 3

    if engine_path.resolve() != out.resolve():
        shutil.copy2(engine_path, out)
    LOGGER.info("OK: %s (size=%.1f MB)", out, out.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    sys.exit(main())
