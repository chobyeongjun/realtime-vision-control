---
date: 2026-04-21
commit: ec1b85a
tag: v0.1.0-cuda-stream-stable
status: stable baseline — Teensy 50Hz 송신 가정에 충분
---

# CUDA_Stream Stable Baseline (2026-04-21)

## 한 줄 요약
**77 Hz / p99 14.46ms / HARD LIMIT 위반 0.000% / 180s 측정.**
Teensy 50Hz 정기 송신 기준으로 1.5배 여유. 모든 frame budget 안.

## 측정 환경
| 항목 | 값 |
|---|---|
| 하드웨어 | Jetson Orin NX 16GB, MAXN, jetson_clocks |
| 카메라 | ZED X Mini SVGA 960×600 @120fps, PERFORMANCE depth |
| 모델 | yolo26s-lower6-v2 (TRT FP16, imgsz=640) |
| 측정 시간 | 180 s, post-warmup 13872 frames |
| 환경 | viewer (`view_sagittal --fps 15`) 동시 실행 |

## 결과
| Metric | 값 | 비고 |
|---|---|---|
| Throughput | 77.4 Hz | Teensy 50Hz 송신에 1.5× 여유 |
| e2e p50 | 8.02 ms | |
| e2e p95 | 11.72 ms | |
| **e2e p99** | **14.46 ms** | 4/18 baseline (19.8) 대비 -27% |
| e2e max | 18.90 ms | 20ms HARD LIMIT 안 |
| **HARD LIMIT 위반** | **0.000%** | 0 / 13872 frames |
| Soft warn (>18ms) | 0.01% | 1 / 13872 |

### Stage 분해 (CUDA event timing)
```
inf=0.07ms     ← CUDA Graph 효과
post=5.86ms
pre=3.10ms
host_overhead=2.26ms (viewer 영향 포함)
```

## 4/18 baseline 대비
| Metric | 4/18 | 4/21 | 변화 |
|---|---|---|---|
| Hz | 73 | 77 | +5% |
| p99 | 19.8 ms | 14.46 ms | **-27%** |
| 위반 | 0.031% | **0.000%** | **0 frame** |
| 안정성 | 단발 | reproducible | ✅ |

## 적용된 핵심 patch (commit list)

| commit | 변경 | 효과 |
|---|---|---|
| `541cb39` | Graph capture retry + thread_local mode | reproducibility |
| `8246061` | Watchdog pause during graph capture | TRUE root cause fix (그 전엔 graph 비결정적) |
| `66402fa` | view_sagittal 재작성 (mainline 스타일) | 6 keypoint 표시 + sticky |
| `201497b` | 3D EMA (alpha 0.7) on publisher | 떨림 smoothing |
| `039f46c` | Sticky publish (max 5 frames) | 검출 누락 시 last good 유지 |
| `c95c60f` | view_sagittal walking-direction calib + warmup 30→100 | 옆모습 정확 + transient 통계 제외 |
| `e7e2af3` | bone-constraint + velocity-bound 활성화 | outlier 좌표 reject |
| `ec1b85a` | velocity bound 5 → 8 m/s | 빠른 동작 (다리 들기) 수용 |

## 실행 명령 (재현)

```bash
# Pipeline (perception → SHM /hwalker_pose_cuda)
sudo ~/realtime-vision-control/src/perception/CUDA_Stream/launch_clean.sh 180

# Viewer (선택, 별도 터미널)
cd ~/realtime-vision-control && PYTHONPATH=src python3 -m perception.CUDA_Stream.view_sagittal --fps 15

# C++ control (h-walker-ws 별도 레포)
cd ~/h-walker-ws/src/hw_control/cpp
sudo chrt -r 50 taskset -c 6,7 ./build/hw_control_loop /dev/ttyACM0
# default SHM = /hwalker_pose_cuda (commit 556f2f38)
```

## 활성화된 Safety chain
1. **20ms HARD LIMIT** → 위반 시 `valid=False` publish → C++ skip
2. **Bone length constraint** → 비정상 좌표 reject → sticky → 또는 valid=False
3. **Joint velocity bound** (8 m/s) → teleportation reject
4. **Sticky publish (max 5 frames ≈ 60ms)** → 짧은 detection 손실 흡수
5. **C++ watchdog 0.2s** → SHM stale → pretension 5N
6. **C++ 5중 force clamp** → max 70N (AK60 한계)
7. **Estop sentinel** → watchdog unhealthy 시 즉시 0N

## 알려진 한계
- **viewer 켜면 86Hz → 77Hz** (X11 + matplotlib/cv2 부담). 측정용엔 viewer OFF 권장.
- **Bone length constraint는 자동 calibrate** — 첫 valid frames의 길이를 reference. 캘리브 시 자세 이상하면 reference 잘못. 정자세 권장.
- **빠른 동작 (>8 m/s, 예: 발차기)** velocity bound로 reject. 필요 시 `--velocity-bound-mps 12`로 변경.

## 다음 단계 (Roadmap)
1. **C++ control + Teensy 통합 검증** — SHM 50Hz 송신 + force command 도달 확인
2. **장시간 안정성** (10분 이상) — thermal drift 확인
3. **실제 모터 구동 테스트** (pretension → impedance, ILC off로 시작)
4. **Sagittal viewer X-axis 자동 정렬** (calibration 후 화면 모습 검증)
