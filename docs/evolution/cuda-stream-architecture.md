# CUDA_Stream 3-Stage Overlapped Pipeline — 자세한 설계 설명

> **Track B의 핵심.** Track A (mainline)이 `13.7ms / 73Hz`까지 왔지만 **p99 spike**가 20ms를 넘어가는 경우가 있었다.
> Track B는 **4개의 CUDA stream**에 파이프라인을 분산시켜 GPU의 internal concurrency를 최대한 활용하고, **host-side sync를 hot-path에서 제거**한다. 그 결과 300s 측정 **p99 19.8ms**, **HARD LIMIT 위반율 0.031%** 달성.

---

## 1. CUDA Stream 기초 (리마인더)

CUDA Stream = **GPU 위의 독립 실행 큐**.

| 속성 | 내용 |
|---|---|
| Same stream | FIFO — 앞 작업이 끝나야 다음 시작 |
| Different streams | **하드웨어가 허용하는 한 병렬** — memcpy engine과 SM이 동시 동작 가능 |
| Default stream | legacy: 모든 작업을 직렬화. **모든 최적화의 적** |
| Event | stream 간 ordering. `streamA.wait_event(event_from_streamB)`로 CPU 개입 없이 dependency 표현 |
| Priority | high/low 2 레벨. infer를 high로 둠 |

Jetson Orin NX 기준:
- **8 SM** (streaming multiprocessor) — compute
- **2개 copy engine** (H2D / D2H 독립)
- → 이론적으로 **memcpy + compute + preproc** 동시 가능

하지만 **Track A는 이 기회를 안 씀** — 모든 torch ops가 기본 current stream에서 순차 실행.

---

## 2. Track A의 한계 (왜 Stream이 필요했나)

Track A (`pipeline_main.py`) 흐름:
```
[Python main thread]
  ZED grab (on ZED's internal stream)
  ↓
  H2D copy (implicit in ZED SDK)
  ↓
  torch preproc (current stream)
  ↓
  TRT inference (current stream)
  ↓
  torch postprocess (current stream)
  ↓
  Python SHM write
```

문제점:
1. **H2D 중엔 SM idle** — 복사 기다리는 동안 이전 결과의 post도 못 돌림
2. **SM 사용 중엔 H2D 못 시작** — 다음 frame grab 대기
3. **모든 sync가 host에서** — Python이 매 ops마다 block. p99 spike 원인
4. **ZED의 default stream과 TRT가 같은 stream일 수도** — capture와 inference가 **직렬화**

결과: **처리량**은 fast-frame 평균 14ms지만, **p99 tail이 22~28ms**로 퍼짐. 20ms HARD LIMIT 보장 불가.

---

## 3. Track B 설계 — 4 Stream + Event 기반 파이프라인

### 3.1 4개 stream 역할 분담
```
capture_stream  : ZED RGB/Depth → GPU H2D copy (ZEDGpuBridge가 관리)
preproc_stream  : letterbox + BGRA→RGB + normalize (torch ops on GPU)
infer_stream    : TRT execute_async_v3 (high priority!)
post_stream     : GPU 2D→3D + depth patch median + tensor → host pinned
```

### 3.2 Stream 생성 ([stream_manager.py](../../src/perception/CUDA_Stream/stream_manager.py))
```python
lo_prio, hi_prio = torch.cuda.Stream.priority_range()
for name in ("capture", "preproc", "infer", "post"):
    prio = hi_prio if name == "infer" else lo_prio
    stream = torch.cuda.Stream(device=device, priority=prio)
```

**왜 infer만 high priority?**
- TRT inference가 가장 "큰 덩어리" (10~13ms)이자 tail latency의 주 원인
- High priority stream은 다른 stream의 작업을 선점 (preempt) 가능
- preproc/post가 인접 프레임에서 동시 실행돼도 infer가 뒤로 밀리지 않음

### 3.3 Event로만 stream 간 sync
```python
# preproc stream이 ZED의 H2D 완료 대기 (CPU 개입 X)
pre.stream.wait_event(frame.ready_event)

# infer stream이 preproc 완료 대기
inf.wait_for(pre)      # 내부적으로 inf.stream.wait_event(pre.done_event)

# post stream이 infer 완료 대기
po.wait_for(inf)

# 유일한 host sync (frame 끝)
po.stream.synchronize()
```

`stream.wait_event(E)`는 **GPU side에서만 기록** — CPU는 바로 다음 코드로 진행. 이게 파이프라이닝의 핵심.

---

## 4. 3-Stage Overlapped Timeline (핵심 다이어그램)

어떤 순간이든 파이프라인에는 **3개의 frame이 서로 다른 단계**에서 동시에 돌고 있다.

```
Time →

capture : [N+1 copy ]
preproc : [N+1 letterbox]
infer   :           [N TRT exec    ]
post    :                          [N-1 3D + patch median]
host    :                                                [N-1 pub]
```

한 tick 동안:
- **capture_stream**: frame N+1 H2D 진행 (copy engine 사용, SM idle로부터 독립)
- **preproc_stream**: frame N+1 normalize (SM 소량 사용)
- **infer_stream**: frame N TRT (SM 주력 사용)
- **post_stream**: frame N−1 patch median + D2H (copy engine 2 + SM 약간)
- **host**: frame N−1 publish, frame N+1 trigger

**wall-clock = max(각 stage)**, 합이 아님. 이게 파이프라이닝의 본질.

---

## 5. 왜 빨라지는가 — 4가지 메커니즘

### ① GPU idle 제거 (memcpy engine + SM 병렬)
Jetson iGPU는 **copy engine 2개 + SM 8개 독립 동작 가능**.
- H2D가 copy engine 0 점유 중 → SM에서 다른 frame의 preproc/infer 돌림
- SM이 infer 풀 가동 → copy engine으로 다음 frame H2D 시작
- Idle이 거의 0에 수렴

### ② Host sync를 hot-path에서 완전 제거
Track A: 매 torch op마다 Python이 block (CUDA context sync).
Track B: `po.stream.synchronize()` **단 1회** (frame 끝, host가 결과 읽기 전).

Python loop가 GPU 기다리지 않음 → Python side jitter (GC, scheduling)와 GPU side가 decouple.

### ③ Stream priority로 tail latency 고정
Infer stream이 high priority. 다른 stream의 작업이 SM을 잡고 있어도 infer가 queue에 들어가는 순간 선점 가능.
→ infer latency의 variance 감소 → p99 안정.

### ④ Zero-copy tensor binding (preproc → TRT)
```python
self.runner.bind_input_address(input_name, self.pre.out.data_ptr())
```
preproc의 output GPU tensor의 **포인터만 TRT input에 바인딩**. 복사 0. Stream 간 dependency는 event로 이미 걸려있음.

---

## 6. Jetson Orin NX 특화 요소

### 6.1 Pinned host memory (iGPU 특성)
Jetson은 **CPU/GPU 메모리 공유**. 하지만 pageable memory는 PCIe 대신 IOMMU를 거쳐 잦은 TLB miss.
→ **pinned host buffer 풀**을 미리 할당해서 D2H 결과 받음:
```python
self._pinned_pool = torch.empty(nbytes, dtype=uint8, pin_memory=True)
```

### 6.2 ZED H2D event 직접 사용
ZED SDK는 내부 CUDA stream을 소유 (사용자 미노출).
`frame.ready_event`는 ZEDGpuBridge가 SDK 콜백에서 기록한 event.
→ preproc stream이 이 event를 직접 `wait_event` → **ZED stream ↔ 우리 stream 무-락 동기화**.

### 6.3 torch.cuda.Stream(priority=hi_prio)
Orin의 SM scheduler는 stream priority를 실제로 respect. Jetson 공식 문서 확인.

---

## 7. run_once vs run_overlapped_step (두 모드)

### run_once (serialized, 검증용)
모든 stream에 작업을 시리얼하게 발행하고 각 stage 끝마다 대기.
→ **correctness 검증**에 사용. benchmark `--no-overlap` 플래그와 연동.

### run_overlapped_step (진짜 핫패스)
위 다이어그램대로 모든 stream이 **다른 frame**을 동시에 처리.
→ 매 tick마다 **가장 최근 완료된 frame**만 반환.

**Deque(maxlen=3)**로 in-flight frame 3개 추적.

---

## 8. Constraint Gate (Stage D, GPU 이후)

GPU 파이프라인 끝에 **opt-in safety gate**:
```python
new_kpts, decision = self.constraints.apply(result.kpts_3d_m, ts_s)
if not decision.accept:
    # valid=False로 zeros publish → C++ control loop가 제어 중단
```
- **bone length constraint** (3D static ref 기반)
- **joint velocity bound** (frame 간 점프 억제)
- 실패 시 `valid=False` + zero buffer → AK60이 70N까지 밀어올리는 경로 차단

---

## 9. 실제 확인된 수치 (300s 측정)

| Metric | 값 |
|---|---|
| E2E mean | 13.8 ms |
| E2E p99 | 19.8 ms |
| HARD LIMIT (20ms) 위반율 | 0.031% |
| 위반 frame 중 motor 도달 | 0 |
| FPS | 67–73 Hz |

Track A 대비:
- Throughput: 비슷 (73Hz 유지)
- **p99: 22~28ms → 19.8ms** (tail 40% 감소)
- 위반율: 기존 수 % → **0.031%**

---

## 10. Trade-off와 의도적으로 안 한 것들

| 안 한 것 | 이유 |
|---|---|
| CUDA Graph 전체 frame 캡쳐 | 조건 분기 (valid check) + dynamic shape 일부 지원 어려움. stream overlap만으로 충분 |
| zero-copy ZED → TRT | ZED SDK 내부 버퍼는 ZED 소유 → 우리 stream에서 직접 inference 어려움. preproc stream으로 letterbox+copy가 더 간단 |
| Multi-GPU scaling | Orin NX 단일 GPU. 관련 없음 |
| INT8 quantization | FP16에서 이미 budget 만족. INT8은 calibration 부담 대비 이득 적음 (향후 검토) |

---

## 11. 코드 맵

```
src/perception/CUDA_Stream/
├── stream_manager.py   4 stream + event + pinned pool
├── pipeline.py         StreamedPosePipeline (run_overlapped_step)
├── zed_gpu_bridge.py   ZED → GPU H2D + ready_event 발행
├── gpu_preprocess.py   letterbox + normalize (preproc_stream)
├── trt_runner.py       TRT execute_async_v3 래퍼 (infer_stream)
├── gpu_postprocess.py  2D→3D + patch median (post_stream)
├── watchdog.py         stream hang + publish staleness 감지
├── constraints.py      bone length + velocity 검증 (host side)
├── run_stream_demo.py  엔트리 (20ms HARD LIMIT 체크 포함)
└── launch_clean.sh     sudo 래퍼 (재실행 clean state)
```

---

## 12. 논문/미팅에서 쓸 때 제안 문구

### Methods (영문 한 단락 예시)
> We structure the perception pipeline as a four-stage overlapped CUDA pipeline, using four dedicated `torch.cuda.Stream` objects for capture, preprocessing, inference, and postprocessing. Cross-stage dependencies are encoded as `cudaEvent`s via `stream.wait_event`, eliminating host-side synchronization in the hot path (only a single `post_stream.synchronize()` before the SHM publish). The inference stream is assigned high priority so that concurrent preprocessing on adjacent frames cannot preempt model execution, stabilizing the 99th-percentile latency. On the Jetson Orin NX iGPU, we exploit pinned host buffers for D2H transfers and bind the preprocessing output tensor directly to the TensorRT input by pointer (zero-copy). The combined effect is a sustained 67–73 Hz at 13.8 ms mean E2E latency with p99 under 20 ms, validated over a 300-second continuous run.

### Figure (제안)
- **Fig. X**: 4-stream timeline diagram (3-stage overlap 그림) + 한 tick 안에 frame N-1/N/N+1가 어떤 stage에서 공존하는지
- **Fig. Y**: p99 histogram 비교 (Track A mainline vs Track B CUDA_Stream)

---

## Source
- Pipeline: [`src/perception/CUDA_Stream/pipeline.py`](../../src/perception/CUDA_Stream/pipeline.py)
- Stream manager: [`src/perception/CUDA_Stream/stream_manager.py`](../../src/perception/CUDA_Stream/stream_manager.py)
- 수치 raw: [`docs/experiments/benchmark-results.json`](../experiments/benchmark-results.json) (`2026-04-18_cuda_stream_hard_limit`)
- 전체 최적화 맥락: [`docs/evolution/why-it-got-faster.md`](./why-it-got-faster.md) §Category 2 (파이프라인 오버랩)

*Last updated: 2026-04-19*
