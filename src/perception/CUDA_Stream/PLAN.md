# CUDA_Stream PLAN — Phase 1-5 실행 계획 (2일)

## Phase 1 — 환경 확인 + 베이스라인 측정 (Day 1 오전, 4h)
**목표:** Super Mode 활성 상태의 깨끗한 숫자.

### 체크리스트
- [ ] `cat /etc/nv_tegra_release` → JetPack 6.2 이상
- [ ] `sudo nvpmodel -m 0 && sudo jetson_clocks && nvpmodel -q` → MAXN_SUPER
- [ ] ZED SDK 버전 ≥ 5.0 (5.2+이면 NV12 zero-copy 추가 이득)
- [ ] `PYTHONPATH=src python3 -m perception.CUDA_Stream.preflight` → 모든 OK
- [ ] `ultralytics>=8.3` 설치 (YOLO26 지원), `yolo26s-pose.pt` 다운로드
- [ ] 베이스라인 측정 — `benchmarks/run_benchmark.py` YOLOv8s-pose, SVGA@120, 120s, `--no-display`
- [ ] 산출물: `results/baseline_yolov8s.csv`, `results/thermal_log_baseline.csv`

## Phase 2 — TRT engine + Stream 인프라 (Day 1 오후, 4h)
**목표:** 한 프레임 추론을 default stream 밖에서 실행.

### 작업
- [ ] `trt_export.py` 실행 → `yolo26s-pose.engine` (FP16)
- [ ] `trt_runner.py` — engine load, IExecutionContext, GPU I/O 버퍼
- [ ] `stream_manager.py` — 4 streams + events + pinned buffer pool
- [ ] 단일 프레임 검증: stream 경로 vs default 경로 keypoint max_err ≤ 1 px
- [ ] 산출물: `results/phase2_keypoint_diff.csv`

## Phase 3 — ZED + GPU 전/후처리 (Day 2 오전, 4h)
**목표:** capture→preproc→infer 사이 CPU/PCIe 왕복 제거.

### 작업
- [ ] `zed_gpu_bridge.py` — ZED CUDA 공유 시도 (실패 시 `cudaMemcpyAsync` fallback)
- [ ] `gpu_preprocess.py` — letterbox + normalize (torchvision GPU)
- [ ] `gpu_postprocess.py` — 2D→3D depth 샘플 + OneEuro filter (torch)
- [ ] 정합 검증: CPU 후처리 vs GPU 후처리 3D 좌표 오차 ≤ 1 mm
- [ ] 산출물: `results/phase3_gpu_vs_cpu_diff.csv`

## Phase 4 — 파이프라인 오버래핑 + CUDA Graph + INT8 (Day 2 오후 2h)
**목표:** stage 병렬 실행, e2e p95 ≤ baseline × 0.5.

### 작업
- [ ] `pipeline.py` — triple-buffer 3-stage orchestration
- [ ] `cuda_graph.py` — fixed-shape capture (실패 시 fallback)
- [ ] INT8 calibration (실험 프레임 50장) → `yolo26s-pose-int8.engine`
- [ ] CPU affinity `{2,3,4,5}`, `chrt -r 50` 옵션
- [ ] `nsys profile` 타임라인 캡처 → 4개 stream 시각적 overlap 확인
- [ ] 산출물: `results/stream_fp16.csv`, `results/stream_int8.csv`, `results/nsys_timeline.nsys-rep`

## Phase 5 — 제어 통합 + 안전장치 + 최종 판정 (Day 2 오후 2h)
**목표:** 제어 루프에 붙여서 돌아가는지 + 폐기/promote 판정.

### 작업
- [ ] `shm_publisher.py` — `/hwalker_pose_cuda` publish (17×3 + conf + ts)
- [ ] `watchdog.py` — stream timeout 50ms → fallback 신호
- [ ] `run_stream_demo.py` — 10분 연속 실행
- [ ] 제어 루프 jitter 측정 (기존 impedance loop과 함께)
- [ ] `docs/benchmarks.md` 업데이트 + `results/final_comparison.csv`

### 최종 Gate (판정)
| 항목 | 합격선 | 폐기 기준 |
|------|--------|-----------|
| e2e p95 latency | ≤ baseline × 0.5 | baseline 대비 개선 없음 → **폐기 (c)** |
| 2D keypoint err (FP16) | ≤ 1 px | 3 px 초과 지속 |
| 3D keypoint err | ≤ 5 mm | 2× baseline 이상 악화 |
| 제어 jitter CV | < 5% | > 10% |
| 10분 watchdog trigger | < 1회/분 | segfault 반복 & 재현 불가 → **폐기 (a)** |
| 1주 내 진전 | Phase 5 도달 | 정체 → **폐기 (b)** |

## 실행 순서 (사용자가 내일 테스트할 순서)
```bash
# 1단계: 환경
cd /path/to/h-walker-ws
sudo nvpmodel -m 0 && sudo jetson_clocks
PYTHONPATH=src python3 -m perception.CUDA_Stream.preflight

# 2단계: TRT export
PYTHONPATH=src python3 -m perception.CUDA_Stream.trt_export \
    --weights yolo26s-pose.pt --imgsz 640 --half \
    --out src/perception/CUDA_Stream/yolo26s-pose.engine

# 3단계: 베이스라인
python3 src/perception/benchmarks/run_benchmark.py \
    --model yolov8s-pose --resolution SVGA --duration 120 --no-display \
    --out src/perception/CUDA_Stream/results/baseline.csv

# 4단계: Stream 버전
PYTHONPATH=src python3 -m perception.CUDA_Stream.benchmark_stream \
    --engine src/perception/CUDA_Stream/yolo26s-pose.engine \
    --resolution SVGA --streams 4 --cuda-graph --duration 120 --no-display \
    --out src/perception/CUDA_Stream/results/stream_fp16.csv

# 5단계: 비교 & 제어 통합 테스트
PYTHONPATH=src python3 -m perception.CUDA_Stream.benchmark_stream --compare \
    src/perception/CUDA_Stream/results/baseline.csv \
    src/perception/CUDA_Stream/results/stream_fp16.csv

PYTHONPATH=src python3 -m perception.CUDA_Stream.run_stream_demo \
    --engine src/perception/CUDA_Stream/yolo26s-pose.engine \
    --duration 600 --publish-shm /hwalker_pose_cuda
```
