# realtime-vision-control

Real-time vision-based gait assistance — ZED X Mini + YOLO26s-lower6 TRT + cable-driven H-Walker.

## Status (2026-04-20)
- Perception pipeline: **13.7ms / 73Hz** (Python only), **14.6ms / 67Hz** (Python+C++)
- CUDA_Stream variant: **p99 19.8ms, 20ms HARD LIMIT 위반 0.031%** (300s 측정)
- Hardware integration: Python → POSIX SHM → C++ control → USB → Teensy 4.1 → AK60 (max 70N)

## Branches

| 브랜치 | 용도 | 핵심 수치 |
|---|---|---|
| **`feature/track-a-onepipeline`** | mainline single pipeline (Python) | 13.7ms / 73Hz |
| **`feature/track-b-cuda-stream`** | 4-stage overlapped pipeline (실시간성 보장) | p99 19.8ms, 0.031% violation |

**상황별 선택**:
- 일반 실험/디버깅: Track A
- 실제 환자 실험/논문 측정: Track B (HARD LIMIT 보장)

## Papers
| ID | Title | Target | Completion |
|---|---|---|---|
| paper1 | Vision-Based Impedance Control for Cable-Driven Gait Rehabilitation | IEEE RA-L | 15% |
| paper2 | RL Sim-to-Real Policy for Gait Assistance | TBD | Planning |

## Structure
```
src/perception/
  realtime/        Track A — pipeline_main.py + bone/calib/joint/safety/shm
  benchmarks/      공통 — zed_camera, trt_pose_engine, postprocess_accel, joint_angles, metrics_3d
  CUDA_Stream/     Track B (해당 브랜치에만) — 4-stream overlapped + HARD LIMIT
  models/          *.pt, *.onnx (LFS) | *.engine은 Jetson 빌드
docs/
  evolution/       전체 여정 마스터 (perception-evolution, why-it-got-faster, cuda-stream-architecture)
  experiments/     실험별 (meta.yaml, benchmark-results.json)
  handovers/       날짜별 세션 인수인계
  meetings/        격주 미팅 (TEMPLATE.md)
  paper/           논문 outline
  hardware/        하드웨어 스펙
  lessons/         설계 결정 근거
scripts/
  sync-from-vault.sh   Obsidian vault → docs/ 동기화
```

## Quick Start
```bash
git clone https://github.com/chobyeongjun/realtime-vision-control
cd realtime-vision-control
git lfs pull   # 모델 다운

# Track A 사용
git checkout feature/track-a-onepipeline

# Track B 사용
git checkout feature/track-b-cuda-stream
```

## Hardware
| | Spec |
|---|---|
| SBC | Jetson Orin NX 16GB (JetPack 6.x, MAXN + jetson_clocks) |
| Camera | ZED X Mini (GMSL2, SVGA 960×600 @120fps, PERFORMANCE depth) |
| MCU | Teensy 4.1 (111Hz inner loop) |
| Motor | AK60 via CAN (max 70N cable force) |

## References
- Master: [`docs/evolution/perception-evolution.md`](docs/evolution/perception-evolution.md)
- Why faster: [`docs/evolution/why-it-got-faster.md`](docs/evolution/why-it-got-faster.md)
- CUDA_Stream architecture: [`docs/evolution/cuda-stream-architecture.md`](docs/evolution/cuda-stream-architecture.md)
- Benchmark data: [`docs/experiments/benchmark-results.json`](docs/experiments/benchmark-results.json)
