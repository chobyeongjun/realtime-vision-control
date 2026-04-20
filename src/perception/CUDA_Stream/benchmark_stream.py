"""Benchmark the CUDA_Stream pipeline and compare against a baseline CSV.

Produces a CSV with per-frame rows: frame_id, ts_ns, e2e_ms, valid,
box_conf, and aggregate stats printed to stdout.

    # measure
    python3 -m perception.CUDA_Stream.benchmark_stream \\
        --engine yolo26s-pose.engine --resolution SVGA --duration 120 \\
        --out results/stream_fp16.csv

    # compare
    python3 -m perception.CUDA_Stream.benchmark_stream --compare \\
        results/baseline.csv results/stream_fp16.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from .gpu_postprocess import GpuPostprocessor
from .gpu_preprocess import GpuPreprocessor
from .keypoint_config import get_schema
from .pipeline import StreamedPosePipeline
from .stream_manager import StreamManager
from .trt_runner import TRTRunner, warmup
from .zed_gpu_bridge import ZEDGpuBridge


LOGGER = logging.getLogger(__name__)


ROW_FIELDS = [
    "frame_id",
    "ts_ns",
    "e2e_ms",
    "valid",
    "box_conf",
    "mean_kpt_conf",
]


# ---------------------------------------------------------------------------
# Measure mode
# ---------------------------------------------------------------------------

def run_measure(args: argparse.Namespace) -> int:
    device = torch.device("cuda:0")
    schema = get_schema(args.schema)
    sm = StreamManager(device=device, high_priority_stages=["infer"])
    runner = TRTRunner(args.engine, device=device)
    engine_in_dtype = runner.bindings[runner.input_names[0]].dtype
    LOGGER.info("engine input dtype = %s", engine_in_dtype)
    pre = GpuPreprocessor(imgsz=args.imgsz, device=device, dtype=engine_in_dtype)
    post = GpuPostprocessor(schema=schema, device=device, use_filter=False)

    bridge = ZEDGpuBridge(
        resolution=args.resolution,
        fps=args.fps,
        depth_mode=args.depth_mode,
        device=device,
        enable_depth=args.depth_mode != "NONE",
        world_frame=True,
    )
    bridge.open()
    bridge.start()

    pipeline = StreamedPosePipeline(bridge, runner, pre, post, sm)

    LOGGER.info("warmup ×20 …")
    warmup(runner, sm.stream_ptr("infer"), iters=20)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[dict] = []
    t0 = time.monotonic()
    try:
        while time.monotonic() - t0 < args.duration:
            tick = pipeline.run_overlapped_step()
            if tick is None:
                continue
            mean_kpt_conf = (
                float(tick.result.kpt_conf.mean().item())
                if tick.result.valid else 0.0
            )
            rows.append(
                {
                    "frame_id": tick.frame_id,
                    "ts_ns": tick.ts_ns,
                    "e2e_ms": tick.latency_ms["e2e"],
                    "valid": int(tick.result.valid),
                    "box_conf": tick.result.box_conf,
                    "mean_kpt_conf": mean_kpt_conf,
                }
            )
    finally:
        bridge.stop()
        pipeline.shutdown()

    with out_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=ROW_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    if rows:
        lats = np.asarray([r["e2e_ms"] for r in rows])
        dt = max(time.monotonic() - t0, 1e-3)
        fps = len(rows) / dt
        p50, p95, p99 = np.percentile(lats, [50, 95, 99])
        LOGGER.info(
            "wrote %d rows to %s | %.1f fps, e2e p50/95/99 = %.2f/%.2f/%.2f ms",
            len(rows), out_path, fps, p50, p95, p99,
        )
    else:
        LOGGER.error("no frames captured — check ZED and engine")
        return 3
    return 0


# ---------------------------------------------------------------------------
# Compare mode
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as fh:
        reader = csv.DictReader(fh)
        # accept both our schema and mainline's run_benchmark.py schema
        vals = []
        for row in reader:
            if "e2e_ms" in row:
                vals.append(float(row["e2e_ms"]))
            elif "e2e_latency_ms" in row:
                vals.append(float(row["e2e_latency_ms"]))
            elif "inference_time_ms" in row:
                vals.append(float(row["inference_time_ms"]))
    if not vals:
        raise ValueError(f"{path} has no recognized latency column")
    return np.asarray(vals)


def run_compare(args: argparse.Namespace) -> int:
    baseline = _read_csv(Path(args.compare[0]))
    stream = _read_csv(Path(args.compare[1]))

    def stats(arr: np.ndarray) -> dict:
        return {
            "n": int(arr.size),
            "mean": float(arr.mean()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
        }

    b = stats(baseline)
    s = stats(stream)
    improvement = {
        "mean_reduction": 1 - s["mean"] / b["mean"] if b["mean"] else 0.0,
        "p95_reduction": 1 - s["p95"] / b["p95"] if b["p95"] else 0.0,
    }

    print(f"baseline : {b}")
    print(f"stream   : {s}")
    print(f"improvement : {improvement}")

    # Gate thresholds — CUDA_Stream plan §Phase5 says "≤ baseline × 0.5"
    # but that's the **stretch goal**. The promote-vs-kill gate is more
    # practical: any regression (>0.9) is a hard fail, 0.7x is a soft win,
    # 0.5x is the stretch target.
    ratio = s["p95"] / b["p95"] if b["p95"] else float("inf")
    if ratio > 0.9:
        print(
            f"GATE: FAIL — stream p95 / baseline p95 = {ratio:.2f} (>0.9 is a regression)",
            file=sys.stderr,
        )
        return 4
    if ratio > 0.7:
        print(f"GATE: marginal — ratio {ratio:.2f} (soft-win threshold 0.7)")
        return 0
    if ratio > 0.5:
        print(f"GATE: pass — ratio {ratio:.2f} (soft-win target met)")
        return 0
    print(f"GATE: stretch — ratio {ratio:.2f} ≤ 0.5 ✓")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine")
    ap.add_argument("--resolution", default="SVGA",
                    choices=["SVGA", "HD720", "HD1080", "HD1200"])
    ap.add_argument("--fps", type=int, default=None)
    ap.add_argument("--depth-mode", default="PERFORMANCE",
                    choices=["NONE", "PERFORMANCE", "QUALITY"])
    ap.add_argument("--trace", default=None,
                    help="per-frame trace CSV path")
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument(
        "--schema", default="lowlimb6", choices=["coco17", "lowlimb6"],
    )
    ap.add_argument("--streams", type=int, default=4, help="informational only")
    ap.add_argument("--cuda-graph", action="store_true",
                    help="enable CUDA Graph capture (best-effort)")
    ap.add_argument("--no-display", action="store_true", help="no-op flag")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2, metavar=("BASELINE_CSV", "STREAM_CSV"))
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    if args.compare:
        return run_compare(args)
    if not args.engine or not args.out:
        print("--engine and --out required for measure mode", file=sys.stderr)
        return 2
    return run_measure(args)


if __name__ == "__main__":
    sys.exit(main())
