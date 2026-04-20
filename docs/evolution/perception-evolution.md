---
title: Perception Evolution Master — Pose Library 비교부터 CUDA_Stream까지
created: 2026-04-18
updated: 2026-04-18
sources:
  - "[[realtime-pose-estimation]]"
  - "[[skiro-learnings]]"
  - "00_Raw/2026-04-15-perception-pipeline-teensy.md"
  - "00_Raw/2026-04-18-perception-pipeline-wrap-up.md"
  - "00_Raw/2026-04-18-p0-track-completion-report.md"
  - "00_Raw/2026-04-19-handover-cuda-stream-perception.md"
tags: [pose-estimation, evolution, benchmark, comparison, cuda-stream, master]
summary: H-Walker perception의 전체 여정 — Library 비교 선택부터 33ms 30fps → 0.031% violation 20ms HARD LIMIT 보장까지
confidence_score: 0.95
---

# [[Perception Evolution Master]]

> **H-Walker 실시간 자세 인식의 전체 여정.**
> 어떤 library를 놓고 비교했고, 왜 YOLO26s-lower6로 정착했고, 어떻게 30fps → 73Hz → 20ms HARD LIMIT 보장까지 왔는지.

---

## 0. 한 줄 요약

**33.4ms / 30fps (MediaPipe 후보 + Ultralytics 동기)** → **13.7ms / 73Hz (YOLO26s-lower6-v2 + DirectTRT + PipelinedCamera)** → **CUDA_Stream 3-stage + 20ms HARD LIMIT 0.031% 위반** (300s 실측).

모든 개선은 "2D keypoint 건드리지 말 것 / ZED SDK thread-unsafe / GPU 경합이 fetch 절약보다 작음" 세 원칙에 기반.

---

## 1. 목표와 제약

**Hard requirement**
| 항목 | 값 | 근거 |
|---|---|---|
| 카메라 → SHM e2e latency | **≤ 20ms (HARD)** | AK60 max cable force 70N. stale keypoint가 제어 루프로 들어가면 환자 위험 |
| 인식률 | 6 keypoints / frame 100% | 하체: L/R hip, knee, ankle |
| 장시간 안정성 | 10분 이상 drift/degradation 없음 | 임상 실험 최소 세션 |
| 플랫폼 | Jetson Orin NX 16GB, ZED X Mini GMSL2 | 고정 하드웨어 |

**Soft requirement**
- 장착 위치 자유도 (카메라가 기울어져도 동작) → **Method B World Frame (IMU 기반)**
- 3D 좌표 정확도 (bone length ±5mm 이내) → 실험에서 깊이 기반 각도 계산
- 개발 중에도 visual 확인 가능 (sagittal view)

---

## 2. Pose Library 비교 (초기 선정 단계)

| Library | 장점 | 치명적 단점 | H-Walker 적용 |
|---|---|---|---|
| **MediaPipe BlazePose** | 경량, CPU 실시간, 문서화 좋음 | ① GPU 가속 지원 불안정 (Jetson에서 TensorRT 미대응) <br> ② 하체 keypoint 정밀도 낮음 <br> ③ 가림(occlusion) 대응 약함 | **기각** — GPU 못 쓰면 Orin NX에서 30fps도 못 뽑음 |
| **OpenPose** | 정확도 공인, 논문 인용 多 | ① 메모리 사용 과도 (Orin NX 16GB에서 TRT+ZED와 공존 어려움) <br> ② custom keypoint (하체 6점) 재학습 경로 불명확 | **기각** — 실시간성·메모리 모두 실패 |
| **MoveNet** | 빠름, TF Lite | ① TensorFlow/TF Lite 의존성 <br> ② 하체 정밀도 MediaPipe와 유사 | **기각** — pytorch + TRT 기반 파이프라인과 이종 |
| **YOLOv8-pose (17 keypoint)** | Ultralytics 생태계, TRT FP16 쉬움 | ① 17 keypoint 중 하체 6점만 필요 → 계산 낭비 <br> ② upper body 가림 시 하체도 흔들림 | 중간 단계 채택 (benchmark 용) |
| **YOLO26s-lower6-v2 (자체 학습)** ⭐ | ① **하체 6점 전용** 학습 <br> ② TRT FP16 경량 <br> ③ 하체 가림 강건성 | 초기 좌측 keypoint 정밀도 이슈 (해결함) | **최종 채택** |

### 왜 YOLO26s-lower6-v2였나
1. **필요한 keypoint만 6개**: L/R hip · knee · ankle. 상체 흔들림 영향 없음
2. **TRT FP16**: Jetson Orin NX에서 ~13ms predict (Ultralytics 경로 16ms → DirectTRT 13.7ms)
3. **하체 전용 학습 데이터**: 일반 보행·재활 시나리오 데이터셋 (v1 → v2 개선)
4. **ZED SDK와 좌표계 일치 가능**: 2D keypoint + depth map → 3D 좌표

### 초기 YOLO26s-lower6-v1의 문제와 해결
- **문제**: 왼쪽 하체 keypoint 정밀도 낮음 (L_hip→L_knee 9px, R 31px — 3.4배 차이)
- **진단**: 학습 데이터에 왼쪽 bias 존재
- **해결**: v2 재학습 + SegmentLengthConstraint 사용 금지 (피드백 루프로 고착 심화)

---

## 3. 성능 진화 타임라인 (수치 포함)

### 3.1 Python pipeline 진화 (mainline)

| 단계 | 구성 | latency | FPS | 병목 해소 |
|---|---|---|---|---|
| Baseline | 동기 fetch, Ultralytics, 640 | 33.4ms | **30fps** | — |
| +PipelinedCamera v1 | Double Buffer + Event (동기 depth) | 24.3ms | 41fps | fetch 9.4 → ~5ms |
| +DirectTRT | Ultralytics 우회, BGRA pass-through | 21.6ms | 46fps | predict 16 → 14ms |
| +C++ 후처리 pybind11 | 2D→3D + EMA를 C++로 | 20.2ms | 50fps | postprocess 10 → 2ms |
| +Safety/KF 제거 | C++에 위임 | 17.7ms | 56fps | Python 경량화 |
| +`--no-display` | cv2.imshow / waitKey 제거 | 16.4ms | **61fps** | X11 block 제거 |
| +release 즉시 호출 | `get_rgb` 직후 release | ~15ms | 67fps | capture ∥ predict |
| +ready_rgb/depth 분리 | 2-stage event | **13.7ms** | **73Hz** | fetch 완전 0ms |
| +C++ 동시 실행 | 제어 루프 병렬 | 14.6ms | 67Hz | CPU 경합 -3Hz |

**누적 개선**: **33.4ms → 13.7ms (2.44× 가속)**. Python only 구성.

### 3.2 CUDA_Stream 트랙 (실시간성 보장 단계)

Python+C++ 67Hz에서도 **p99 spike** 발생 (20-28ms) → 20ms HARD LIMIT 요구 충족 불가. 트랙 B로 CUDA_Stream 3-stage 파이프라인 구현.

| 단계 | p99 (ms) | HARD LIMIT 위반율 | 비고 |
|---|---|---|---|
| CUDA_Stream v1 (초기) | 22-25 | 수 % | spike 빈발 |
| +GC disable | 20.8 | 1.2% | GC pause 제거 (2-5ms) |
| +CPU affinity 2-5 | 20.1 | 0.5% | C++ 6-7 분리 |
| +SCHED_FIFO 90 (chrt) | 19.8 | 0.08% | 선점 방지 |
| +20ms frame-skip (valid=False) | 19.8 | **0.031%** | **모든 위반 프레임 motor 도달 차단** |
| +launch_clean.sh (재실행 열화 방지) | 19.8 | 0.031% (재실행 후에도 유지) | Argus IPC 누적 제거 |

**현재 최종**: 300s 실측, **0.031% 위반율**. 위반 프레임은 `valid=False`로 SHM publish → C++ control loop skip → **motor 도달 불가** (안전 보장).

---

## 4. Depth & Visual 파이프라인 진화

### 4.1 ZED SDK depth mode 선택
| Mode | 속도 | 정확도 | 결정 |
|---|---|---|---|
| PERFORMANCE | 기준 | ±3cm | **채택** |
| NEURAL_LIGHT | -30% | +2cm | 기각 (속도 손실이 정확도 이득보다 큼) |
| NEURAL | -60% (29Hz) | +2cm | **기각** (TRT YOLO와 GPU SM 경합) |

**결론**: ZED SDK가 `PERFORMANCE deprecated` 경고를 띄워도 무시. TRT inference와 공존 불가.

### 4.2 Depth 3D 좌표 경로 진화
```
v0: get_3d_coords(pixel)    → 뼈길이 122m (intrinsics 없음, 폐기)
v1: pixel + depth_map[y,x]  → patch median, raw 3D
v2: v1 + 3D EMA α=0.8       → 떨림 완화 (현재)
v3: v2 + bone length const  → P0-3 적용 (outlier 제거)
```

### 4.3 Visual 2-stage 진화
- **Frontal view (RGB)**: keypoint overlay, --no-display 시 생략
- **Sagittal view (world Y-Z)**: Method B + gravity arrow + bone length 표시. 개발 전용. 실험 중 FPS 반토막으로 **반드시 off**.

---

## 5. Method A vs Method B (좌표계 기준 선택)

| Method | 기준 | 장점 | 단점 |
|---|---|---|---|
| **A: Standing Calibration** | 정자세 캘리브 프레임이 zero-ref | 구현 단순, knee 각도 직관적 | 카메라 움직이면 깨짐 |
| **B: IMU World Frame** ⭐ | 중력 방향이 world y축 | 카메라 자유 배치, 기울어도 OK | IMU 초기화 필요 |

**채택**: **Method B**. 실험 중 카메라 위치 고정이므로 `skip_imu=True` (static R 사용) → runtime IMU 호출 1ms 절약.

**IMU 초기화 방식 진화**:
- v1: 가속도 벡터 → pitch/roll 계산 (중력 방향 단독) — 부호 버그 위험
- v2: `get_pose().get_orientation()` 쿼터니언 평균 N=20 → R matrix (**채택**) — SDK 내부에서 IMU fusion 완료된 값

---

## 6. Teensy 통신 파이프라인

```
Python perception (73Hz)
  ↓ POSIX SHM /hwalker_pose (36 bytes, seqlock)
C++ main_control_loop (100Hz, clock_nanosleep)
  ↓ Impedance + ILC → Force command
  ↓ USB serial /dev/ttyACM0
Teensy 4.1 (111Hz inner loop)
  ↓ CAN bus
AK60 motor (max 70N cable force)
```

**현재 상태**: SHM 통신 확인, C++ watchdog 0.2s → pretension 5N fallback 작동 확인, **실제 모터 구동 테스트 미완료**.

---

## 7. 실패한 시도들 (재시도 금지 목록)

| # | 시도 | 원인 | 상태 |
|---|---|---|---|
| F1 | MediaPipe BlazePose | Jetson에서 GPU 가속 불안정, 하체 정밀도 낮음 | **영구 기각** |
| F2 | OpenPose | 메모리 과도, custom keypoint 학습 경로 불명 | **영구 기각** |
| F3 | AsyncCamera (ZED SDK) | SDK thread-unsafe, segfault | **영구 기각** |
| F4 | One Euro Filter (모든 variant) | 2D 이동 → depth NaN, Joints 0/6 | **영구 기각** |
| F5 | SegmentLengthConstraint on 2D | 피드백 루프로 왼쪽 keypoint 고착 | 2D는 기각, **3D + static ref만 가능** |
| F6 | EMA on 2D keypoints | 2D 이동 → depth NaN | **영구 기각** |
| F7 | zero-copy depth (copy=False) | ZED 버퍼 overwrite race | **영구 기각** |
| F8 | GDM(X server) 끄기 | GMSL/CSI 카메라는 Argus+EGL=X server 필수, segfault + 리부팅 강제 | **영구 기각** |
| F9 | NEURAL / NEURAL_LIGHT depth | TRT와 GPU SM 경합, 30% 이상 손실 | **영구 기각** |
| F10 | imgsz 480 | 사용자 거부 (정확도 타협 불가) | **영구 거부** |
| F11 | cv2.cuda | JetPack OpenCV가 CUDA 없이 빌드됨 | 보류 |
| F12 | C++ loop rate 100→60Hz | CPU 3%, GPU 미사용 → 낮춰도 효과 없음 | **영구 기각** |
| F13 | get_3d_coords() | camera intrinsics 미연동 → 뼈길이 122m | 폐기 |
| F14 | sudo chrt 직접 실행 | root env로 전환되며 torch ModuleNotFoundError | sudo 래퍼 스크립트 패턴으로 회피 |
| F15 | serialize_depth=True 기본값 | 74→39Hz 반토막 | opt-in flag로만 유지 |
| F16 | yolo26s-fp16io.engine | box_conf 0.04 (detection 실패) | `yolo26s-lower6-v2.engine` 유지 |

---

## 8. 영구 교훈 (핵심 원칙)

1. **2D keypoint를 건드리면 depth가 깨진다** — smoothing은 3D 좌표 또는 최종 각도에만 적용
2. **ZED SDK는 thread-unsafe** — grab/retrieve 동시 호출 금지. Event 동기화 필수
3. **ZED depth `copy=True` 강제** — 파이프라인 병렬 + 공유 버퍼 = 복사 필수
4. **GPU 경합 < fetch 절약** — 경합(+5ms) vs fetch 절약(-9ms) → 파이프라인이 이득
5. **Python sleep이 파이프라인을 방해** — 제거, C++이 `clock_nanosleep`으로 RT 타이밍 담당
6. **Ultralytics 우회 효과 제한적** — torch upload overhead로 3ms만 절약
7. **ONNX→엔진 imgsz 고정** — imgsz 변경 시 ONNX 재export 필요
8. **Jetson GMSL/CSI = GDM 필수** — X server 끄면 EGL 없어서 segfault + 드라이버 락업
9. **NEURAL depth는 TRT와 공존 불가** — GPU SM 경합
10. **sagittal display는 실험 중 FPS 반토막** — 개발 전용 (74→42Hz)
11. **jetson_clocks는 부팅마다 재적용** — persistent 아님. 안 하면 GPU 306MHz로 fall-back
12. **release는 get_rgb 직후 즉시** — 프레임 끝 release는 역효과 (fetch 그대로)
13. **Method B + 카메라 고정 = static R** — `skip_imu=True`로 runtime IMU 1ms 절약
14. **Python GC disable + 명시적 collect** — 자동 GC spike 2-5ms 제거
15. **CPU isolation**: Python 2-5, C++ 6-7, system 0-1 — cores 공유 시 predict spike 클러스터
16. **SHM은 seqlock 패턴** — Python 종료 중 half-write 흡수 (C++ watchdog 0.2s fallback도 병행)
17. **20ms HARD LIMIT = frame-skip**: e2e > 20ms면 `valid=False`로 C++이 skip → motor 도달 차단
18. **launch_clean.sh 래퍼**: sudo + pkill + rm Argus IPC + systemctl restart nvargus-daemon + exec chrt -r 90 sudo -u user → 재부팅 없이 clean state 복원
19. **IMU 쿼터니언 평균**: `get_pose().get_orientation()` N=20 평균 → R matrix. 수동 pitch 덮어쓰기 금지

---

## 9. 현재 아키텍처 (최종)

```
┌─ Track A (mainline, control branch) ───────────────────────────┐
│  Python pipeline_main.py --no-display --method B               │
│    ZED PipelinedCamera (ready_rgb/depth 분리)                  │
│    DirectTRT YOLO26s-lower6-v2 FP16                             │
│    C++ batch_2d_to_3d pybind11 + 3D EMA                        │
│    Method B static R (IMU warmup 20f)                          │
│    Bone constraint (P0-3)                                       │
│    SHM /hwalker_pose (seqlock, P0-2)                           │
│  → 14.6ms / 67Hz (Python+C++ 동시)                              │
└─────────────────────────────────────────────────────────────────┘
                     ↓ SHM ↓
┌─ C++ main_control_loop (100Hz) ────────────────────────────────┐
│  SHM reader + watchdog 0.2s                                     │
│  Impedance + ILC (ILC 현재 off)                                │
│  SerialComm /dev/ttyACM0                                        │
│  5중 force clamp (P1-6)                                         │
└─────────────────────────────────────────────────────────────────┘
                     ↓ USB ↓
┌─ Teensy 4.1 (111Hz) ───────────────────────────────────────────┐
│  CAN → AK60 motor (max 70N)                                    │
└─────────────────────────────────────────────────────────────────┘

┌─ Track B (feature/cuda-stream-perception, 실시간성 보장) ──────┐
│  CUDA_Stream 3-stage (capture / infer / post)                  │
│  + GC disable + SCHED_FIFO 90 + taskset 2-5                    │
│  + 20ms HARD LIMIT frame-skip                                   │
│  + launch_clean.sh (재실행 clean state)                         │
│  → 0.031% violation (300s), SHM /hwalker_pose_cuda              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 10. 파일 맵 (현재 최종)

```
h-walker-ws/src/hw_perception/
├── realtime/                    ← Track A (mainline)
│   ├── pipeline_main.py         ★ 최종 엔트리
│   ├── bone_constraint.py       ★ P0-3 (3D static ref)
│   ├── calibration.py           Method A/B
│   ├── joint_3d.py              BONE_RANGES
│   ├── shm_publisher.py         seqlock
│   └── safety_guard.py
├── benchmarks/
│   ├── zed_camera.py            ★ PipelinedCamera (ready_rgb/depth 분리)
│   ├── trt_pose_engine.py       ★ DirectTRT
│   ├── postprocess_accel.py     pybind11 래퍼
│   └── cpp_ext/                 C++ 후처리
├── CUDA_Stream/                  ← Track B
│   ├── run_stream_demo.py       ★ 3-stage + 20ms HARD LIMIT
│   ├── zed_gpu_bridge.py        IMU 쿼터니언
│   ├── watchdog.py              safe-stop + estop sentinel
│   ├── view_sagittal.py         auto-fit visual
│   └── launch_clean.sh          ★ sudo 래퍼 (Argus clean)
└── models/
    ├── yolo26s-lower6-v2.pt/.onnx/.engine
    └── yolo26s-lower6-v2-640.direct.engine
```

---

## 11. 미해결 / 다음 단계

### Priority 1 (실험 전 필수)
- **Track A ↔ Track B SHM 통합**: mainline은 `/hwalker_pose` 읽음, CUDA_Stream은 `/hwalker_pose_cuda` publish → 이름 통일 또는 merge
- **Method B knee_err 좌우 비대칭 검증** (Winter gait table 대조)
- **Teensy 실제 모터 구동 테스트** (pretension 5N까지만 완료)

### Priority 2
- Bone length constraint 실측 적용 검증 (3D static ref)
- Thermal 장시간 실시간성 실측
- Sagittal viewer 별도 low-FPS 프로세스화

### Priority 3
- INT8 quantization (현재 FP16)
- Headless Xorg (GUI 없이 EGL 공급)

---

## 12. Knowledge Connections
- **Related Wiki**: [[realtime-pose-estimation]], [[zed-x-mini]], [[jetson-orin-nx]], [[teensy-4-1]], [[h-walker]], [[gait-analysis]]
- **Learnings**: [[skiro-learnings]] (38 entries, 12 solved perception-related)
- **Handovers**:
  - `00_Raw/2026-04-15-perception-pipeline-teensy.md` (Track A 기반)
  - `00_Raw/2026-04-18-perception-pipeline-wrap-up.md` (Method B + PipelinedCamera 재설계)
  - `00_Raw/2026-04-18-p0-track-completion-report.md` (P0-1~P1-6 안전/성능)
  - `00_Raw/2026-04-19-handover-cuda-stream-perception.md` (Track B + launch_clean.sh)
- **Paper 1 (RA-L)**: Vision-Based Impedance Control — 이 문서 전체가 Methods/Results 핵심 재료
- **Paper 2**: RL sim-to-real — Perception 파이프라인은 real 관측 공급

---

*Last updated: 2026-04-18*
*이 문서는 Cowork `cowork_project_summary`/`cowork_paper_data` 호출 시 perception 카테고리의 주요 재료로 사용됨*
