# 왜 빨라졌는가 — 기술적 원리 정리

> **E2E latency: 44.4ms → 13.7ms (3.24×), 그리고 300s 측정 20ms HARD LIMIT 위반율 0.031%.**
> 각 최적화가 **어떤 병목을, 왜, 어떻게** 풀었는지 정리.

---

## 진화 한 줄 요약

```
44.4ms (YOLO26s 17kpt, 단순 실행)
  ↓  모델 경량화 (head 축소 + 재학습)
18.0ms (YOLO26s-lower6, 단독 추론)
  ↓  파이프라인 오버랩 + framework 우회 + C++ 후처리
16.4ms → 13.7ms (최대 처리량, Python only 73Hz)
  ↓  C++ 제어 루프와 병렬 실행 + CPU 격리
14.6ms / 67Hz
  ↓  실시간성 보장 (GC off + SCHED_FIFO + frame-skip)
p99 19.8ms, 300s 측정 위반 0.031%, motor 도달 0
```

---

## 최적화를 6 가지 카테고리로 분류

### Category 1 — 연산량 자체 감소
**원리**: CPU/GPU/메모리 버스에 실어야 할 데이터량 자체를 줄임.

| # | 변경 | 수치 | 왜 빨라졌나 |
|---|---|---|---|
| 1 | **Head 17kpt → 6kpt fine-tune** | 44ms → 18ms (2.44×) | YOLO pose head는 각 keypoint 당 heatmap channel을 디코딩. 17 → 6으로 output tensor가 `3×17=51` channel → `3×6=18` channel로 ~3배 축소. decode·argmax·scatter GPU ops가 거의 비례하여 감소. 또 postprocess도 17개 좌표 → 6개. |
| 2 | **GPU output 파싱: top-1만 CPU 복사** | 7200 float → 18 float | GPU→CPU PCIe 전송은 latency 지배. 전체 candidate을 내려받고 CPU에서 argmax 대신, GPU에서 argmax 후 top-1만 전송 (18 float ≈ 72B). 2ms 절약. |
| 3 | **BGRA pass-through** | 색변환 2회 → 1회 | ZED는 native BGRA → 종래에는 BGR → RGB 두 번 변환. BGRA → RGB 직접 1회. OpenCV cvtColor 한 번 제거 (~0.5ms). |

### Category 2 — 파이프라인 오버랩 (I/O와 계산 병렬)
**원리**: 카메라 grab, depth retrieve, inference는 서로 독립 자원을 씀. 직렬로 돌리면 합이 쌓이고, 병렬로 돌리면 가장 오래 걸리는 것만 남음.

| # | 변경 | 수치 | 왜 빨라졌나 |
|---|---|---|---|
| 4 | **PipelinedCamera Double Buffer + Event** | fetch 9.4ms → 0ms | 캡처 스레드가 백그라운드에서 grab을 미리 해둠. main 스레드가 inference 하는 동안 다음 frame grab이 overlap. wall-clock에서는 fetch가 "free"가 됨. |
| 5 | **release() 즉시 호출 (get_rgb 직후)** | fetch 5.7ms → 0ms (완성) | 처음엔 release를 frame 끝에서 호출 → capture 스레드가 다음 grab을 못 시작해서 overlap 실패. get_rgb 직후 release로 바꾸니 다음 grab이 즉시 시작 → predict와 완벽 병렬. |
| 6 | **ready_rgb / ready_depth 2-stage event 분리** | predict와 depth 완전 overlap | RGB는 predict에 즉시 필요, depth는 postprocess에만 필요. ready를 둘로 쪼개서 `grab→rgb→ready_rgb.set()→depth→ready_depth.set()` 순. main은 `ready_rgb` 대기 후 predict 시작, 그 사이 depth 병렬 retrieve. |

### Category 3 — 프레임워크 오버헤드 제거
**원리**: 범용 라이브러리는 abstraction 비용이 있음. 핫 패스에서는 직접 호출.

| # | 변경 | 수치 | 왜 빨라졌나 |
|---|---|---|---|
| 7 | **DirectTRT (Ultralytics 우회)** | predict 16ms → 13ms | Ultralytics `YOLO.predict()`는 내부에 torch tensor 변환, letterbox, NMS, postprocess wrapper를 감싸 둠. 직접 TensorRT `execute_async_v3` 호출 + 사전 바인딩된 GPU 버퍼 사용 → overhead 3ms 절감. |
| 8 | **C++ batch_2d_to_3d pybind11** | post 10ms → 2ms | Python loop (6 keypoint × 7×7 depth patch × median)가 인터프리터 오버헤드 지배. 동일 로직을 C++로 컴파일 (pybind11) → SIMD + 컴파일 최적화. 5× 가속. |
| 9 | **Safety/KF/sleep 제거 (pipeline_main)** | 22ms → 17ms | Safety guard, Kalman filter, rate-limit sleep은 모두 **제어** 책임. C++ control loop가 이걸 담당하므로 Python 쪽에서 제거. Python은 "관측 공급" 한 가지 역할만. |
| 10 | **skip_imu=True (Method B static R)** | -1ms | 매 프레임 `get_gravity_vector()` 호출 → ZED SDK 내부 IMU fusion 1ms. 카메라 고정이면 R matrix 불변 → warmup 20f 평균 쿼터니언으로 고정 R. 이후 IMU 호출 skip. |

### Category 4 — GPU/OS/하드웨어 최대화
**원리**: 하드웨어가 최대 성능으로 돌고 있어야 측정값이 의미 있음. 기본 설정은 전력 절약 쪽으로 fall-back.

| # | 변경 | 수치 | 왜 빨라졌나 |
|---|---|---|---|
| 11 | **jetson_clocks (GPU 918MHz 고정)** | GR3D 11% → 85% | Jetson DVFS는 idle 감지 시 GPU를 306MHz로 내림. predict burst가 있어도 주파수 ramp-up이 늦어서 첫 frame마다 spike. `jetson_clocks`로 918MHz 고정 → ramp 지연 0. **부팅마다 재적용 필요** (persistent 아님). |
| 12 | **MAXN power mode (`nvpmodel -m 0`)** | 전체 clock ceiling ↑ | Jetson 기본은 15W 모드. MAXN은 모든 CPU core + GPU clock 상한을 풀어줌. jetson_clocks와 조합으로 상한 + 고정을 모두 적용. |
| 13 | **Python `gc.disable()` + 수동 collect** | 2-5ms spike 제거 | CPython generational GC가 불규칙 시점에 200~500ms 사이 청소 → predict 프레임에 spike. 수동 off + 명시적 collect (idle 시) → 예측 가능한 pause 0. |

### Category 5 — 경합 제거
**원리**: 두 작업이 같은 자원을 놓고 싸우면 둘 다 느려짐. 자원을 나눠주거나, 한쪽을 다른 자원으로 옮김.

| # | 변경 | 수치 | 왜 빨라졌나 |
|---|---|---|---|
| 14 | **CPU isolation (taskset + affinity)** | predict spike cluster 제거 | Python(TRT), C++ control loop, system daemon이 같은 core를 공유하면 context switch가 잦음. `os.sched_setaffinity` Python 2-5, `taskset -c 6,7` C++, system 0-1로 분리 → context switch ~0. |
| 15 | **SCHED_FIFO priority 90 (chrt -r 90)** | p99 20.8 → 19.8ms | 일반 SCHED_OTHER는 fairness scheduler라 background 프로세스가 Python predict를 선점 가능. SCHED_FIFO는 real-time class, 높은 priority가 낮은 것을 무조건 선점. |
| 16 | **PERFORMANCE depth 유지 (NEURAL 기각)** | NEURAL 30ms → PERFORMANCE 12ms | NEURAL depth는 ZED SDK 자체 신경망이 GPU에서 돌아감 → TRT YOLO와 같은 SM 슬롯 경합. predict 2.4×, 최종 29Hz로 급락. PERFORMANCE는 block matching 기반 → GPU 사용 거의 없음. |
| 17 | **ZED depth `copy=True` 강제** | race → stable | `copy=False`는 ZED 내부 버퍼 참조만 반환. release 즉시 호출과 조합하면 capture 스레드가 다음 frame으로 overwrite → partial data. SVGA 2.3MB 복사 0.5ms 추가 감수, valid rate 0% → 100%. |
| 18 | **--no-display (cv2.imshow 제거)** | 74→42Hz 방지 | X11 렌더 + `waitKey(1)` block + GPU framebuffer 경합이 누적. 실험 중은 off. 개발 전용. |

### Category 6 — 실시간성 보장 (지연은 안 줄지만 안전)
**원리**: Latency 분포는 long-tail. 평균만 좋아도 p99 spike 한 번이 stale keypoint → motor 70N까지 밀어올림. "빨라진다"보다 "보장한다"에 가까움.

| # | 변경 | 수치 | 왜 효과 있나 |
|---|---|---|---|
| 19 | **20ms HARD LIMIT frame-skip** | 위반 0.031%, motor 도달 0 | e2e 측정 후 > 20ms이면 `valid=False`로 SHM publish. C++ control loop가 `valid` 확인 → false면 해당 frame skip, 제어 명령 낼 때 이전 state 유지. 위반 프레임이 actuator에 절대 도달 못 함. |
| 20 | **POSIX SHM seqlock 패턴** | half-write 방지 | Python 종료 중 또는 GC 시 write 도중 멈출 수 있음. C++가 중간 state를 읽으면 NaN/garbage가 제어 루프에 유입. seqlock (리눅스 커널 동일 패턴) → reader가 write 중인 것을 감지해서 retry. |
| 21 | **launch_clean.sh (sudo wrapper)** | 재실행 60% → 0.031% | Argus SDK는 내부 IPC semaphore (`/dev/shm/sem.ipc_test_*`)를 root 소유로 만듦. publisher 재시작마다 누적되어 카메라 초기화가 점진적으로 느려짐. sudo로 semaphore 청소 + `nvargus-daemon` 재시작 → clean state 복원. 재부팅 불필요. |
| 22 | **IMU 쿼터니언 N=20 평균** | 부호 버그 위험 제거 | 수동 `--camera-pitch-deg` 덮어쓰기는 `(0, cos(p), -sin(p))` 공식 부호 버그. SDK `get_pose().get_orientation()`은 이미 IMU 센서 fusion 완료된 쿼터니언 → 평균만 취하면 안정적 R matrix. SDK 버전 무관. |

---

## 각 카테고리가 총 개선에 얼마나 기여했나

| 카테고리 | 기여 (ms) | 원리 요약 |
|---|---|---|
| 1. 연산량 감소 | **-26.4ms** (44.4 → 18.0) | Head 축소 + tensor transfer 축소 + 색변환 축소 |
| 2. 파이프라인 오버랩 | **-9.4ms** (fetch hidden) | I/O를 계산 뒤에 숨김 |
| 3. 프레임워크 오버헤드 제거 | **-9ms** | Ultralytics 3ms + C++ post 8ms + skip_imu 1ms |
| 4. GPU/OS 최대화 | **-2~5ms** spike 제거 | jetson_clocks + GC off |
| 5. 경합 제거 | **p99 안정화** | CPU 격리 + SCHED_FIFO + NEURAL 배격 |
| 6. 실시간성 보장 | **0.031% 위반 + motor 도달 0** | frame-skip + seqlock + launch_clean |

단순 합이 아닌 이유: 3번(skip_imu)과 4번(GC)은 이미 1번 모델 경량화 이후 남은 꼬리 레이턴시를 깎은 것.

---

## 실패했던 접근 (왜 역효과였나)

| 시도 | 기대 | 실제 | 원인 |
|---|---|---|---|
| One Euro Filter (2D) | 떨림 감소 | Joints 0/6 | 2D keypoint를 이동시키면 그 위치의 depth가 NaN. depth NaN → 3D 실패. **2D를 건드리면 depth가 깨진다** (영구 규칙). |
| SegmentLengthConstraint (2D) | bone 길이 outlier 제거 | 왼쪽 keypoint 고착 | 피드백 루프: 왼쪽 캘리브가 부족 → ref 9px → ref에 맞춰 강제 이동 → 다음 프레임도 이동 위치에서 시작 → 고착. **constraint는 3D + static ref에서만 가능**. |
| GDM(X server) 끄기 | X11 오버헤드 제거 | ZED segfault + 리부팅 | GMSL/CSI 카메라는 NVIDIA Argus → EGL context → X server에서 공급. X 없으면 EGL 못 만듦. USB 카메라면 가능하지만 GMSL은 **X server 필수**. |
| NEURAL depth | 3D 정확도 향상 | predict 30ms (×2.4), 29Hz | ZED NEURAL은 GPU 신경망 → TRT YOLO와 SM 슬롯 경합. 정확도 이득은 ~2cm인데 손실은 60%. |
| zero-copy depth (`copy=False`) | 복사 0.5ms 절감 | calib 0% race | capture가 다음 frame으로 내부 버퍼 overwrite → main view가 partial. **파이프라인 병렬 + 공유 버퍼 = 복사 필수**. |
| AsyncCamera + `_zed_lock` | thread 경합 해소 | depth 10 → 17ms | Lock contention: main이 depth 가져올 때 grab 대기. ZED SDK는 thread-unsafe라 Event 동기화만 가능. |
| C++ loop rate 100→60Hz | 부하 감소 | 효과 없음 | C++ CPU 3%, GPU 미사용. rate 낮춰도 빈 슬롯만 늘어남. 병목은 C++이 아님. |

---

## 이 원리들을 논문/미팅에서 어떻게 쓸 것인가

### 미팅 슬라이드 구성 (제안)
1. **Slide: Timeline 그래프** — 44.4ms → 13.7ms 6단계 (Category 1~5 색상 구분)
2. **Slide: Category별 기여** — 위 "각 카테고리가 총 개선에 얼마나 기여했나" 표
3. **Slide: 실패 교훈** — 2D keypoint 건드리면 안 됨, GDM 끄면 안 됨, NEURAL 배격
4. **Slide: Safety** — p99 19.8ms, HARD LIMIT 0.031%, motor 도달 0

### 논문 Methods 섹션 구성 (제안)
- **4.1 Perception Pipeline** → Category 2 + 3
- **4.2 Real-Time Guarantees** → Category 4 + 6
- **4.3 Resource Isolation** → Category 5
- **4.4 Model (Lower-Body Specialization)** → Category 1

### 재현성 체크리스트
- [ ] 부팅 후 `sudo jetson_clocks`
- [ ] `nvpmodel -m 0` (MAXN)
- [ ] `launch_clean.sh` 사용 (재실행 시)
- [ ] `--no-display` 옵션
- [ ] CPU affinity 확인 (Python 2-5, C++ 6-7)
- [ ] SHM protocol version 일치 (seqlock)

---

## Source Data
- 수치: [`docs/experiments/benchmark-results.json`](../experiments/benchmark-results.json)
- 전체 여정: [`docs/evolution/perception-evolution.md`](./perception-evolution.md)
- Session handover: [`docs/handovers/`](../handovers/)

*Last updated: 2026-04-19*
