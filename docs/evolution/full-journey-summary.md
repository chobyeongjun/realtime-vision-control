# Full Journey Summary — Camera Pose Estimation for H-Walker

> **2026-03 ~ 2026-04-21 전체 과정.** 다음 세션이 zero-start 시 이 한 문서로 맥락 파악.
> 잘된 것 (지속/확대), 안된 것 (재시도 금지), 교훈을 시간순 + 카테고리로 분리.

---

## TL;DR — 한 줄

**33ms / 30Hz (단순 동기) → 14ms / 86Hz / HARD LIMIT 위반 0.000% (4-stream + 7-layer safety chain).**
Library 비교 → Fine-tune → Pipeline 최적화 → CUDA_Stream → 비결정성 디버그 → Stable baseline 순.

---

## Phase 1 — Library 선택 (2026-03-24)

### 잘한 것 ✅
- **6개 vendor 동일 조건 측정** (Jetson Orin NX + ZED X Mini + 15s).
  결과: YOLO 계열만 <50ms 달성. RTMPose는 CUDA EP 미활성화로 CPU fallback (150ms+).
- **YOLO26s 채택 결정** — 가장 높은 confidence(0.99) + budget 안.

### 안된 것 ❌
- **MoveNet** (NumPy 2.x 비호환, tflite_runtime 충돌) — 폐기.
- **MediaPipe** (인식률 94% only, foot keypoint 사용 어려움) — 폐기.
- **ZED Body Tracking** (positional tracking 활성화 필요, 추가 SDK overhead) — 폐기.

### 교훈 (영구)
- 모든 모델을 **동일 hardware + 동일 조건**에서 측정. 단일 측정 값 신뢰 X.
- **Edge inference (Jetson)**에서 ONNX runtime / CUDA EP 활성화 여부 사전 확인 필수.

→ 자료: `docs/experiments/2026-03-24-model-comparison/`

---

## Phase 2 — Fine-tuning 17 → 6 keypoints (2026-03-27)

### 잘한 것 ✅
- **YOLO26s 17kpt → 6kpt (L/R hip/knee/ankle)** — 하체 전용 fine-tune.
  결과: **44ms → 18ms (2.4× 가속)**, mAP50 88.5%, confidence 0.99.
- **표준 COCO 데이터셋** 활용 (extra dataset 불필요). RTX 5090에서 25.8h 학습.
- **원본 모델 보존** + 명명 분리 (yolo26s-lower6).

### 안된 것 ❌
- **17kpt 전체 사용** — 상체 keypoint는 cable robot 제어에 무관, 처리 비용 낭비.
- **imgsz 416** — 카메라 crop 640×600인데 416으로 다운스케일 → 정확도 손실. 640으로 통일.
- **Heel/Toe 추가 keypoint 학습** — IMU가 담당. 비전은 hip/knee/ankle 6점에 집중.

### 교훈 (영구)
- **전용 모델 fine-tune이 generic보다 빠름** (head 축소 + 학습 데이터 집중).
- **Capacity는 필요한 곳에 집중** — 모든 keypoint가 동일 가치 아님.

---

## Phase 3 — Pipeline 최적화 mainline (2026-04-15 ~ 17)

### 잘한 것 ✅
- **PipelinedCamera (Double Buffer + Event)** — fetch 9.4ms → 0ms. capture를 background thread로.
- **DirectTRT (Ultralytics 우회)** — predict 16ms → 13ms (3ms 절감).
- **GPU output 파싱** (top-1 argmax만 CPU 복사) — 7200 → 18 float, -2ms.
- **C++ pybind11 후처리** — postprocess 10ms → 2ms (5×).
- **Method B IMU World Frame** — 카메라 고정 시 static R, skip_imu=True (1ms 절감).
- **release() 즉시 호출** (`get_rgb` 직후) — fetch overlap 정확.
- **`ready_rgb` / `ready_depth` 2-stage event** — depth retrieve와 predict 완전 병렬.

결과: **33ms / 30Hz → 14ms / 73Hz** (Python only).

### 안된 것 ❌
- **AsyncCamera + lock** — ZED SDK thread-unsafe, depth 17ms로 악화 → 폐기.
- **One Euro Filter on 2D keypoints** — joints 0/6 완전 실패 (filter가 0 위치 학습 → 모든 후속 좌표 0으로 끌어당김). **영구 기각.**
- **EMA on 2D keypoints** — 동일 (2D 이동 → depth NaN → 3D 실패). **영구 기각.**
- **SegmentLengthConstraint on 2D** — 피드백 루프, 한쪽 keypoint 9px로 collapse. **2D는 영구 기각, 3D + static ref만 가능.**
- **zero-copy depth (`copy=False`)** — race condition, valid 0% 고착. **영구 기각.**
- **GDM (X server) 끄기** — GMSL/CSI 카메라는 EGL=X 필수. segfault + 리부팅 강제. **영구 기각.**
- **NEURAL depth mode** — TRT YOLO와 GPU SM 경합 → predict 30ms (×2.4). **영구 기각.**

### 교훈 (영구)
- **2D keypoint를 건드리면 depth가 깨진다** — smoothing은 3D 좌표 또는 최종 각도에만.
- **ZED SDK는 thread-unsafe** — Event 동기화 필수.
- **GPU 경합 < fetch 절약** — 경합(+5ms) vs fetch 절약(-9ms) → 파이프라이닝이 이득.
- **`release` 직접 호출 위치가 핵심** — get_rgb 직후 ≠ frame 끝.

---

## Phase 4 — CUDA_Stream Track B (2026-04-18)

### 잘한 것 ✅
- **4-stream overlapped pipeline** (capture / preproc / infer / post)
- **CUDA Event 기반 동기** — host sync hot-path에서 0회 (마지막 post.synchronize() 1회).
- **20ms HARD LIMIT frame-skip pattern** — e2e > 20ms → `valid=False` → C++ control loop가 skip → motor stale 도달 차단.
- **POSIX SHM seqlock** (Linux kernel 패턴) — Python half-write 시 C++ retry.
- **`launch_clean.sh` (sudo wrapper)** — Argus IPC 누적 (`/dev/shm/sem.ipc_test_*`) 매 실행마다 청소 + nvargus-daemon 재시작 + chrt -r 90 + jetson_clocks. **재부팅 없이 clean state 복원.**
- **GC disable + SCHED_FIFO 90 + CPU affinity 2-5** — Python jitter 제거.

결과: **p99 19.8ms / 위반율 0.031% / 67-73Hz** (300s 측정).

### 안된 것 ❌
- **Stream priority 모두 high** — 효과 없거나 역효과. infer만 high가 정답.
- **CUDA Graph 전체 frame 캡쳐** — valid 분기 + dynamic shape로 어려움. preproc + infer만 capture.
- **C++ loop rate 100→60Hz** — CPU 3% 사용, GPU 미사용 → 효과 없음. 100Hz 유지.
- **sudo chrt 직접 실행** — root env로 전환되며 torch ModuleNotFoundError. **launch_clean.sh가 `exec chrt -r 90 sudo -u user`로 우회 (RT priority 상속).**

### 교훈 (영구)
- **Argus IPC 누적이 비결정성 원인** — sudo 정리 + nvargus restart로만 복원 가능.
- **CUDA event timing은 GPU work + wait time** — stream priority 차이로 wait가 측정값에 포함.
- **`thread_local` capture mode** — 같은 process 다른 thread에는 적용 안 됨 (Phase 6 참고).

---

## Phase 5 — Track A vs B 분리 + Repo 정리 (2026-04-19 ~ 20)

### 잘한 것 ✅
- **Track A / B 명시적 분리** — feature/track-a-onepipeline + feature/track-b-cuda-stream.
- **realtime-vision-control 별도 레포** — h-walker-ws에서 perception 분리.
- **Subtree split (history 보존)** — `git subtree split --prefix=src/hw_perception` 으로 mainline 코드 history 그대로 가져옴.
- **하나의 binary + CLI 인자 (`--shm`)** — C++ control loop 한 binary가 두 SHM 이름 모두 지원.

### 안된 것 ❌
- **outdated baseline에서 출발** — `trt_pose_engine_zerocopy.py` (v1 single-stream)을 최신본으로 오인 → 4-stream을 처음부터 재발명. **22ms / 44Hz로 퇴보.**
  - 원인: GitHub origin/main에 v1과 v2 혼재, 어느 게 "최신"인지 명시 안 됨.
- **CLAUDE.md `<50ms` 제약** — 4월 기준은 `<20ms HARD LIMIT`인데 outdated → 새 세션이 44Hz도 OK로 인지.
- **perception + display 한 스크립트** (`run_sagittal_display.py`) — FPS 74→44 반토막. **영구 규칙 위반.**
- **Python에서 Teensy 직접 송신** (`teensy_uart.py`) — C++의 RT 보장 + watchdog + force clamp 우회 → **안전 chain 무력화.**
- **수동 `--camera-pitch-deg` 덮어쓰기** — 부호 버그 위험 → IMU quaternion fusion으로 대체.

### 교훈 (영구)
- **CLAUDE.md에 두 트랙 + 절대 금지 명시** — 새 세션의 함정 차단.
- **Display는 별도 process** — 한 process에 섞으면 GPU/X11 경합으로 perception FPS 반토막.
- **Python은 관측 공급, C++는 RT + 안전** — 이 분담 어기지 말 것.

→ 자료: `docs/handovers/2026-04-19-handover-cuda-stream-perception.md`

---

## Phase 6 — 비결정성 진단 + 진짜 root cause (2026-04-21)

### 잘한 것 ✅
- **`--trace` 옵션 활용** — stage별 CUDA event timing CSV → spike 발생 stage 즉시 식별.
- **점진적 진단** — silent fallback 가설 → ZED SDK 충돌 가설 → **진짜 원인 = Watchdog**.
- **Watchdog의 stream.query()가 graph capture invalidate** 발견 — `cudaErrorStreamCaptureUnsupported` log에서 정확히 잡음.
- **Watchdog pause/resume** — 1-2초 capture 동안만 pause, 이후 resume → 안전 chain 유지 + reproducibility.

### 안된 것 ❌
- **silent fallback** (capture 실패 시 eager로 자동 폴백) — 매 run 80Hz vs 40Hz 비결정성 원인. **명시적 retry + raise로 변경.**
- **`capture_error_mode='thread_local'`만으로는 부족** — same process 다른 thread는 못 막음 (Watchdog).
- **EMA를 outlier 차단으로 오인** — EMA는 low-pass smoothing, outlier(50cm jump)는 못 잡음. 사용자 지적이 정확. **bone_length / velocity constraint가 outlier 정답.**
- **velocity bound 5 m/s** — 빠른 동작 (다리 들기) reject. **8 m/s로 완화.**

### 교훈 (영구)
- **비결정성 발견 즉시 silent fallback 의심** — 작동하다 안 작동하다 패턴이면 retry 또는 명시적 raise.
- **EMA ≠ outlier 차단** — 도구 분리: smoothing은 EMA, outlier reject는 bone/velocity constraint.
- **Multi-thread + CUDA Graph capture** — thread_local 모드도 같은 process 다른 thread는 못 막음. 명시적 pause 필수.

---

## Phase 7 — Stable baseline + Repo 정리 (2026-04-21)

### 결과 (v0.1.0-cuda-stream-stable)
```
77.4 Hz / e2e p99 14.46 ms / HARD LIMIT 위반 0.000% (180s, 13872 frames)
4/18 baseline (73 / 19.8 / 0.031%) 모든 면에서 능가
```

### 활성 안전 chain (7 layers)
1. Python e2e > 20ms → `valid=False`
2. Bone length constraint → outlier reject
3. Joint velocity bound (8 m/s) → teleportation reject
4. Sticky publish (max 5 frames ≈ 60ms) → 짧은 detection 손실 흡수
5. C++ watchdog 0.2s → SHM stale → pretension 5N
6. C++ 5중 force clamp → max 70N (AK60)
7. Estop sentinel → watchdog unhealthy 시 즉시 0N

### 데이터 통합
- 흩어진 Jetson 자산 정리: `~/RealTime_Pose_Estimation/` (3/24 benchmark) + `~/h-walker-ws/` (figures, trace) → `realtime-vision-control/docs/`
- LFS: v1/v2 skeleton 영상, 3/24 demo videos
- 큰 ZED 원본 (260406_*, 178~226MB) → Jetson local만, README에 위치 명시

### Repo 구조
- `src/perception/{CUDA_Stream, realtime, benchmarks, training, models}/`
- `docs/{evolution, experiments, cuda-stream, meetings, paper, hardware, lessons, recordings, figures}/`
- 빈 placeholder 제거, gitignore 강화

---

## "절대 다시 시도하지 말 것" 종합 (skiro-learnings 영구 기각)

| # | 시도 | 영구 결정 | 이유 |
|---|---|---|---|
| 1 | One Euro Filter (모든 variant) | 기각 | 2D 이동 → depth NaN → joints 0/6 |
| 2 | EMA / smoothing on 2D keypoints | 기각 | 동일 |
| 3 | SegmentLengthConstraint on 2D | 기각 | 피드백 루프, 한쪽 collapse. 3D + static ref만 가능 |
| 4 | zero-copy depth (`copy=False`) | 기각 | ZED 내부 버퍼 race |
| 5 | GDM (X server) 끄기 | 기각 | GMSL/CSI = EGL=X 필수, segfault + 리부팅 |
| 6 | NEURAL/NEURAL_LIGHT depth mode | 기각 | GPU SM 경합으로 ×2.4 감속 |
| 7 | imgsz 480 | 사용자 거부 | 정확도 trade-off 불수용 |
| 8 | AsyncCamera + lock | 기각 | ZED SDK thread-unsafe |
| 9 | C++ loop rate < 100Hz | 기각 | CPU 3%, GPU 미사용 — 효과 없음 |
| 10 | Python에서 Teensy 직접 송신 | 기각 | C++ RT/watchdog/clamp 우회 |
| 11 | sagittal display + pipeline 한 process | 기각 | FPS 반토막 (74→44) |
| 12 | sudo chrt 직접 실행 | 기각 | root env 전환, torch ModuleNotFoundError |
| 13 | jetson_clocks 미적용 실행 | 기각 | GPU 306MHz로 fall-back |
| 14 | trt_pose_engine_zerocopy v1 (single stream) | 폐기 | Track B 4-stream으로 대체 |
| 15 | matplotlib backend (sagittal viewer) | 폐기 | OpenCV가 4× 빠름 (60Hz vs 14Hz) |
| 16 | Stream priority 모두 high | 효과 없음 | infer만 high가 정답 |
| 17 | EMA로 outlier 차단 시도 | 잘못 | EMA는 smoothing. outlier는 bone/velocity constraint |

---

## "지속/확대할 것" (잘된 것 정수)

| 카테고리 | 결정/패턴 | 이유 |
|---|---|---|
| 모델 | YOLO26s 6kpt fine-tune | 18ms / mAP50 88.5% |
| 카메라 | ZED X Mini SVGA + Method B static R | thread-safe pipelining에 적합 |
| Inference | DirectTRT + CUDA Graph (preproc+infer) | inf 0.13ms |
| Postprocess | C++ pybind11 + GPU 처리 | 10ms → 2ms |
| Pipeline | 4-stream + Event + thread_local + watchdog pause | reproducible 80Hz |
| Safety | 7-layer chain (Python→SHM→C++→Teensy) | 위반 0.000% |
| 측정 | `--trace` + p99 + 위반율 + 동일 조건 비교 | 정량 진단 |
| Repo | Track A/B 명시 분리 + CLAUDE.md 규칙 | 새 세션 함정 차단 |
| Vault | research-vault → docs sync | single source of truth |

---

## Source Hierarchy

```
realtime-vision-control/   (이 레포)
├── README.md              한눈 + Quickstart
├── CHANGELOG.md           v0.1.0 milestone
├── CLAUDE.md              작업 지침
├── docs/
│   ├── README.md          docs 인덱스
│   ├── evolution/
│   │   ├── perception-evolution.md
│   │   ├── why-it-got-faster.md
│   │   ├── cuda-stream-architecture.md
│   │   └── full-journey-summary.md   ← 이 파일
│   ├── experiments/
│   │   ├── 2026-03-24-model-comparison/   (results + videos + README)
│   │   ├── 2026-04-18-cuda-stream/        (trace CSV + meta.yaml)
│   │   ├── 2026-04-21-stable-baseline/    (trace + control loop log)
│   │   ├── 2026-04-21-stable-baseline.md  (요약)
│   │   └── benchmark-results.json
│   ├── cuda-stream/   architecture, benchmarks, consumer_contract, troubleshooting
│   ├── handovers/     (날짜별 세션 인수인계)
│   ├── meetings/      (격주 미팅 자료)
│   ├── paper/         (Paper 1 outline 등)
│   ├── hardware/
│   ├── lessons/
│   ├── figures/biomechanics/   (14 PNG)
│   └── recordings/    (v1/v2 skeleton mp4 LFS)
└── scripts/
    └── sync-from-vault.sh
```

Local copies:
- Mac:        `~/realtime-vision-control/`
- Jetson:     `~/realtime-vision-control/`
- Vault:      `~/research-vault/realtime-vision-control/` (이번에 git clone)
- GitHub:     https://github.com/chobyeongjun/realtime-vision-control

---

## 다음 세션이 zero-start 시 읽을 순서

1. **이 문서** (full-journey-summary.md) — 5분
2. `docs/experiments/2026-04-21-stable-baseline.md` — 현재 상태
3. `docs/evolution/cuda-stream-architecture.md` — Track B 상세
4. `CLAUDE.md` — 작업 규칙
5. `docs/README.md` — 다른 문서 인덱스

이후 작업은 `CHANGELOG.md` 마지막 entry 기준 + 위 "절대 다시 시도하지 말 것" 표 준수.

*Last updated: 2026-04-21 (v0.1.0-cuda-stream-stable)*
