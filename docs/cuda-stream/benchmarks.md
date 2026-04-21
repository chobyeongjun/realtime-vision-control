# Benchmarks

## Target numbers (Jetson Orin NX 16GB, Super Mode 157 TOPS, ZED X Mini SVGA@120)

| variant | pure infer | e2e p50 | e2e p95 | fps | notes |
|---------|-----------:|--------:|--------:|----:|-------|
| baseline (mainline YOLOv8s-pose, default stream) | ~25 ms | — | — | — | skiro-learnings reference |
| stream FP16, no graph | ≤ 9 ms | — | — | — | Phase 4 gate |
| stream FP16 + CUDA graph | ≤ 8 ms | — | ≤ 12 ms | ≥ 80 | Phase 4 target |
| stream INT8 + graph | ≤ 6 ms | — | ≤ 10 ms | ≥ 100 | Phase 4 stretch (calib-dep.) |

## How to fill this in
1. `python3 src/perception/benchmarks/run_benchmark.py --model yolov8s-pose --resolution SVGA --duration 120 --no-display --out results/baseline.csv`
2. `python3 -m perception.CUDA_Stream.benchmark_stream --engine yolo26s-pose.engine --resolution SVGA --duration 120 --out results/stream_fp16.csv`
3. `python3 -m perception.CUDA_Stream.benchmark_stream --compare results/baseline.csv results/stream_fp16.csv`

Paste the numbers into the table above. Add a row per variant tested.

## Correctness checks
| comparison | metric | target |
|------------|--------|--------|
| stream FP16 vs baseline, same frame | 2D keypoint max err | ≤ 1 px |
| stream FP16 vs baseline, same frame | 3D keypoint max err | ≤ 5 mm |
| stream INT8 vs stream FP16 | keypoint MAE | ≤ 2 px, ≤ 10 mm |
| GPU postprocess vs CPU postprocess | 3D keypoint max err | ≤ 1 mm |

## Stability (10-minute continuous runs)
| run | duration | watchdog triggers | thermal throttle % | mean fps |
|-----|----------|------------------:|-------------------:|---------:|
| — | — | — | — | — |

## Outstanding observations (fill in as you go)
- Add entries when you spot a regression, spike, or an unexplained
  behavior. Cross-reference skiro-learnings as needed.
