# realtime-vision-control

Real-time vision-based gait assistance — ZED X Mini + YOLO26s-lower6 TRT + cable-driven H-Walker.

## Status (2026-04-21) — `v0.1.0-cuda-stream-stable`

```
77.4 Hz  /  e2e p99 14.46 ms  /  HARD LIMIT 위반 0.000%   (180s, 13872 frames)
```

4/18 baseline (73 Hz / p99 19.8 / 0.031%) **모든 면에서 능가**. 매 run reproducible.

## Quick Start

```bash
git clone https://github.com/chobyeongjun/realtime-vision-control
cd realtime-vision-control
git lfs pull                 # *.pt, *.onnx, demo *.mp4

# Jetson에서 (perception → SHM /hwalker_pose_cuda)
sudo nvpmodel -m 0 && sudo jetson_clocks
sudo src/perception/CUDA_Stream/launch_clean.sh 60

# 별도 터미널에서 sagittal viewer (옵션)
PYTHONPATH=src python3 -m perception.CUDA_Stream.view_sagittal --fps 15

# C++ control (h-walker-ws 별도 레포)
cd ~/h-walker-ws/src/hw_control/cpp
sudo chrt -r 50 taskset -c 6,7 ./build/hw_control_loop /dev/ttyACM0
# default SHM = /hwalker_pose_cuda
```

## Structure

```
src/perception/
  CUDA_Stream/      Track B (production) — 4-stream + HARD LIMIT + EMA + sticky + constraints
  realtime/         Track A (mainline)   — pipeline_main.py
  benchmarks/       공통 모듈
  training/         YOLO fine-tune
  models/           *.pt, *.onnx (LFS) | *.engine은 Jetson 빌드
docs/
  README.md         이 디렉토리 인덱스
  evolution/        전체 여정 + 최적화 원리 + Track B 아키텍처
  experiments/      날짜별 실험 + 결과 데이터
  cuda-stream/      Track B 모듈 docs (architecture, benchmarks, troubleshooting, consumer_contract)
  meetings/         격주 미팅 자료
  paper/            논문 outline
  hardware/         Jetson/ZED 스펙
  lessons/          설계 결정 근거
  recordings/       v1/v2 demo 영상 (LFS)
  figures/biomechanics/  gait analysis figures
scripts/
  sync-from-vault.sh    research-vault → docs/ 동기화
```

## Hardware

| | Spec |
|---|---|
| SBC | Jetson Orin NX 16GB (JetPack 6.x, MAXN + jetson_clocks) |
| Camera | ZED X Mini (GMSL2, SVGA 960×600 @120fps, PERFORMANCE depth) |
| MCU | Teensy 4.1 (111 Hz inner loop) |
| Motor | AK60 via CAN (max 70N cable force) |

## Branches

| Branch | 용도 |
|---|---|
| **`main`** | 통합 + 안정. 일반 사용. |
| `feature/track-a-onepipeline` | Track A (mainline single pipeline) — debugging |
| `feature/track-b-cuda-stream` | Track B (4-stream HARD LIMIT) — Production source |

`main`이 두 트랙 모두 포함. 일반적으로 main만 사용.

## Tags

- **`v0.1.0-cuda-stream-stable`** (2026-04-21) — 77Hz / p99 14.46ms / 0% violation. 현재 production.

## Papers

| ID | Title | Target |
|---|---|---|
| paper1 | Vision-Based Impedance Control for Cable-Driven Gait Rehabilitation | IEEE RA-L |
| paper2 | RL Sim-to-Real Policy for Gait Assistance | TBD |

## References

| 문서 | 설명 |
|---|---|
| [`docs/README.md`](docs/README.md) | 전체 docs 인덱스 |
| [`docs/evolution/perception-evolution.md`](docs/evolution/perception-evolution.md) | Master — 전체 여정 |
| [`docs/evolution/why-it-got-faster.md`](docs/evolution/why-it-got-faster.md) | 22 최적화 원리 |
| [`docs/evolution/cuda-stream-architecture.md`](docs/evolution/cuda-stream-architecture.md) | Track B 아키텍처 |
| [`docs/experiments/2026-04-21-stable-baseline.md`](docs/experiments/2026-04-21-stable-baseline.md) | 현재 baseline 측정 |
| [`docs/experiments/benchmark-results.json`](docs/experiments/benchmark-results.json) | 누적 측정 수치 |

## Related

- H-Walker control + firmware: `~/h-walker-ws` (private)
- Research notes: `~/research-vault/`
