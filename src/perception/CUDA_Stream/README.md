# CUDA_Stream — H-Walker Perception GPU Pipeline (격리 실험 모듈)

## 목적
YOLO26s-pose + ZED X Mini + Jetson Orin NX 16GB(Super Mode) 조합에서
**CUDA Stream 4개 + 3-stage 오버래핑 + TensorRT FP16/INT8**으로
perception e2e latency를 mainline 대비 **50% 이하**로 줄인다.

## 핵심 규칙 (위반 시 폐기)
1. **Mainline 수정 금지** — `src/perception/realtime/`, `src/perception/benchmarks/` 는 **읽기 전용**
   - 필요 로직은 이 폴더로 **복사**하거나 **subprocess/import로 래핑**한다
2. **SHM 이름 분리** — 이 모듈의 SHM publish 경로는 `/hwalker_pose_cuda`
   - mainline의 `/hwalker_pose`와 절대 충돌하지 않는다 (이름, 세마포어, 파일 핸들 모두)
3. **실패해도 OK** — 이 모듈이 crash/hang/망가져도 mainline 제어 루프는 영향받지 않아야 한다
   - 제어 루프는 `/hwalker_pose` 구독을 기본으로 하고, `/hwalker_pose_cuda`는 opt-in
4. **폐기 기준 (3개 중 1개 충족 시 즉시 중단)**
   - (a) segfault 재현 불가 (디버깅 dead-end)
   - (b) 1주 진전 없음 (diminishing returns)
   - (c) mainline 대비 개선 없음 (FPS p95 감소 < 10% 또는 3D 정확도 퇴보)

## 현재 상태
| 항목 | 상태 |
|------|------|
| 폴더 스켈레톤 | ✅ 생성됨 |
| README / PLAN | ✅ 작성됨 |
| `stream_manager.py` | ✅ 구현 |
| `trt_runner.py` | ✅ 구현 (engine load + execute_async_v3) |
| `zed_gpu_bridge.py` | ✅ 구현 (CPU fallback 포함) |
| `gpu_preprocess.py` | ✅ 구현 (torchvision GPU) |
| `gpu_postprocess.py` | ✅ 구현 (3D sample + OneEuro filter) |
| `pipeline.py` | ✅ 구현 (triple-buffer 3-stage) |
| `cuda_graph.py` | ✅ 구현 (capture + fallback) |
| `benchmark_stream.py` | ✅ 구현 (CSV out) |
| `trt_export.py` | ✅ 구현 (YOLO26 → engine FP16/INT8) |
| `shm_publisher.py` | ✅ 구현 (`/hwalker_pose_cuda`) |
| `watchdog.py` | ✅ 구현 (stream timeout → fallback) |
| `run_stream_demo.py` | ✅ end-to-end 데모 |
| Jetson 실제 벤치 | ⏳ 사용자 실행 |

## 성능 목표 (SVGA 600p@120Hz, Super Mode 157 TOPS)
| 메트릭 | Mainline 기준 | 본 모듈 목표 |
|--------|-------------|------------|
| pure infer | ≈ 25 ms (YOLOv8s sync) | ≤ 9 ms (YOLO26s-pose FP16 + stream) |
| e2e p95 | — (측정 필요) | ≤ baseline × 0.5 |
| FPS p95 | — | ≥ 80 Hz (@ SVGA 120fps 타깃) |
| 제어 루프 jitter CV | — | < 5% (outer 30Hz 기준) |
| keypoint 2D err (FP16 vs baseline) | — | ≤ 1 px |
| keypoint 3D err (FP16 vs baseline) | — | ≤ 5 mm |

## 빠른 시작
```bash
# 0) 환경 확인 (Jetson Orin NX)
sudo nvpmodel -m 0              # MAXN_SUPER
sudo jetson_clocks
PYTHONPATH=src python3 -m perception.CUDA_Stream.preflight

# 1) TRT engine export (Jetson에서 최초 1회)
PYTHONPATH=src python3 -m perception.CUDA_Stream.trt_export \
    --weights yolo26s-pose.pt --imgsz 640 --half --out yolo26s-pose.engine

# 2) 베이스라인 (mainline) 측정
python3 src/perception/benchmarks/run_benchmark.py \
    --model yolov8s-pose --resolution SVGA --duration 120 --no-display \
    --out src/perception/CUDA_Stream/results/baseline.csv

# 3) Stream 버전 측정
PYTHONPATH=src python3 -m perception.CUDA_Stream.benchmark_stream \
    --engine yolo26s-pose.engine --resolution SVGA --streams 4 \
    --cuda-graph --duration 120 --no-display \
    --out src/perception/CUDA_Stream/results/stream_fp16.csv

# 4) 비교
PYTHONPATH=src python3 -m perception.CUDA_Stream.benchmark_stream \
    --compare src/perception/CUDA_Stream/results/baseline.csv \
              src/perception/CUDA_Stream/results/stream_fp16.csv

# 5) 실시간 데모 (SHM publish /hwalker_pose_cuda)
PYTHONPATH=src python3 -m perception.CUDA_Stream.run_stream_demo --engine yolo26s-pose.engine
```

## 아키텍처 요약
```
capture_stream   │ ZED grab → GPU buffer (N+1)
preproc_stream   │ letterbox + normalize on GPU (N+1, capture 끝난 뒤)
infer_stream     │ TRT execute_async_v3 (N)
post_stream      │ 2D→3D + OneEuro filter + SHM publish (N-1)
```
- host는 `post_stream.synchronize()` 후 keypoint만 회수
- 모든 교차 의존성은 `torch.cuda.Event` 기반 (host sync 없음)
- CUDA Graph capture 가능 시 `infer+preproc` launch overhead 제거

## 문서
- [PLAN.md](PLAN.md) — Phase 1-5 상세 실행 계획
- [docs/architecture.md](docs/architecture.md) — stream 다이어그램 / 동기화 모델
- [docs/troubleshooting.md](docs/troubleshooting.md) — race / segfault / hang 대응
- [docs/benchmarks.md](docs/benchmarks.md) — 실험 결과 기록

## Consumer Contract (제어 루프 측이 반드시 지켜야 할 계약)

이 모듈은 2차 감사에서 발견된 **CRITICAL — safe-stop 신호 전달 부재**를 격리 규칙(mainline 수정 금지) 안에서 해결하기 위해 **두 채널**을 동시에 발행한다. 제어 루프(`hw_treadmill`, `hw_overground`)는 두 채널 중 **하나라도** 트리거되면 즉시 AK60 명령을 **0N**으로 내려야 한다.

1. **SHM `valid=0` 채널** — `/dev/shm/hwalker_pose_cuda`의 `valid_flag` 오프셋(20 byte)이 0이면 keypoint 데이터 **무효**. `ShmReader.read()` 반환의 7번째 요소.
2. **Estop sentinel 파일** — `/dev/shm/hwalker_pose_cuda_estop` 파일 존재 시 perception 워치독이 하드 폴트를 감지한 것. 파일 존재 자체가 safe-stop 신호이며, 내용은 사람용 reason 문자열.

**참고 reader 스니펫** (mainline 측에서 복사해서 쓸 수 있는 최소 구현):
```python
# 제어 루프 30Hz 주기 안에 이 두 줄을 호출하라.
if Path("/dev/shm/hwalker_pose_cuda_estop").exists():
    motor_cmd.zero_force()  # AK60 70N 한도 이내 0N 발행
    continue
data = shm_reader.read()
if data is None or not data[-1]:  # valid flag
    motor_cmd.zero_force()
    continue
```

mainline 수정 금지 규칙상 본 모듈은 이 코드를 `hw_treadmill/` / `hw_overground/`에 **삽입하지 않는다**. `docs/consumer_contract.md` 를 발췌해서 리뷰 후 mainline PR로 별도 적용해야 한다.

## 안전·운용 주의
- Orin NX 40W: 팬/히트싱크 없으면 1–2분 내 스로틀 → `tegrastats` 모니터 필수
- ZED X Mini GMSL2: 핫플러그 금지, PoC 12V 전원 순서 유의
- AK60 70N cable force 제한, Teensy 111Hz inner 루프는 이 모듈 변경 범위 **밖**

## 참고 자료
- ZED SDK 5.2: https://www.stereolabs.com/developers/release
- JetPack 6.2 Super Mode: https://developer.nvidia.com/blog/nvidia-jetpack-6-2-brings-super-mode-to-nvidia-jetson-orin-nano-and-jetson-orin-nx-modules/
- YOLO26: https://docs.ultralytics.com/models/yolo26/
- TensorRT Python API: https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/python-api-docs.html
