"""Print the I/O tensor layout of a TRT engine.

Usage: python3 -m perception.CUDA_Stream.inspect_engine engine.engine

Use this once after running ``trt_export.py`` to confirm that the
``gpu_postprocess.py`` layout assumption (last dim 56 for pose) matches
the actual engine. YOLO26 pose export went through several revisions
upstream — verifying on the real device is faster than reading GitHub
issues.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <engine.engine>", file=sys.stderr)
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"engine not found: {path}", file=sys.stderr)
        return 2

    from .trt_runner import TRTRunner

    runner = TRTRunner(path)
    if runner.metadata:
        print("=== Ultralytics metadata ===")
        for k in ("task", "imgsz", "kpt_shape", "names", "batch", "stride"):
            if k in runner.metadata:
                print(f"  {k:10s} = {runner.metadata[k]}")
        kpt_shape = runner.metadata.get("kpt_shape")
        if kpt_shape and isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) >= 1:
            K = int(kpt_shape[0])
            schema_hint = {6: "lowlimb6", 17: "coco17"}.get(K, f"custom K={K}")
            print(f"  → recommended --schema: {schema_hint}")
        print()
    print("=== Tensor bindings ===")
    for name, info in runner.describe().items():
        print(f"{name:20s} role={info['role']:>6} shape={info['shape']} dtype={info['dtype']}")

    # Sanity check — validate output last-dim against the kpt_shape from
    # Ultralytics metadata (if present) or against the set of known layouts.
    K_from_meta = None
    kpt_shape = runner.metadata.get("kpt_shape") if runner.metadata else None
    if kpt_shape and isinstance(kpt_shape, (list, tuple)) and len(kpt_shape) >= 1:
        K_from_meta = int(kpt_shape[0])

    for name, info in runner.describe().items():
        if info["role"] != "output" or len(info["shape"]) != 3:
            continue
        last = info["shape"][-1]
        pose_layouts = {}  # last_dim -> (K, layout_name)
        if K_from_meta is not None:
            pose_layouts[5 + K_from_meta * 3] = (K_from_meta, "pose (box+conf+kpts)")
            pose_layouts[6 + K_from_meta * 3] = (K_from_meta, "legacy (box+conf+cls+kpts)")
        # also accept the universal pose/legacy layouts for K=17
        pose_layouts.setdefault(56, (17, "pose"))
        pose_layouts.setdefault(57, (17, "legacy"))
        pose_layouts.setdefault(23, (6, "pose"))
        pose_layouts.setdefault(24, (6, "legacy"))

        if last in pose_layouts:
            K, kind = pose_layouts[last]
            schema = {6: "lowlimb6", 17: "coco17"}.get(K, f"K={K}")
            print(f"\nOK — {kind} layout, K={K} on {name}")
            print(f"   → use --schema {schema}")
        else:
            print(
                f"\nFAIL — unexpected last dim {last} on {name}; "
                "update gpu_postprocess.py decoder",
                file=sys.stderr,
            )
            return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
