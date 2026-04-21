# Documentation Index

## 시작점

- **`evolution/perception-evolution.md`** — 전체 여정 마스터 (33ms→14ms)
- **`evolution/why-it-got-faster.md`** — 22 최적화 원리 (6 카테고리)
- **`evolution/cuda-stream-architecture.md`** — Track B 4-stream 설계
- **`experiments/2026-04-21-stable-baseline.md`** — 현재 stable (77Hz / 위반 0%)

## 디렉토리별

| 폴더 | 내용 |
|---|---|
| `evolution/` | 누적된 핵심 정리 — 전체 여정 / 최적화 원리 / 아키텍처 |
| `experiments/` | 날짜별 실험. 각 폴더에 README + data/ + videos/ |
| `cuda-stream/` | Track B 모듈 docs (architecture, benchmarks, consumer_contract, troubleshooting) |
| `meetings/` | 격주 교수님 미팅 자료 + figures |
| `paper/` | 논문 outline + 참고 논문 |
| `hardware/` | Jetson 설치, ZED spec, troubleshooting |
| `lessons/` | 설계 결정 근거 (모델 선택, fine-tuning 전략) |
| `figures/biomechanics/` | gait analysis figures (h-walker-ws에서 가져옴) |
| `recordings/` | v1/v2 skeleton demo 영상 (LFS) |

## 실험 인벤토리 (시간순)

| 날짜 | 실험 | 위치 | 핵심 결과 |
|---|---|---|---|
| 2026-03-24 | 6 모델 vendor 비교 | `experiments/2026-03-24-model-comparison/` | YOLO 계열만 <50ms 달성 |
| 2026-03-27 | YOLO26s 17kpt → 6kpt fine-tune | (results in `evolution/perception-evolution.md`) | 44ms → 18ms (2.4×) |
| 2026-04-15 | PipelinedCamera + DirectTRT | (results in handovers) | 20ms / 50fps |
| 2026-04-17 | Method B IMU world frame | (results in handovers) | 14ms / 73Hz |
| 2026-04-18 | CUDA_Stream 4-stage | `experiments/2026-04-18-cuda-stream/` | p99 19.8ms / 위반 0.031% |
| **2026-04-21** | **Stable baseline** (graph + watchdog + EMA + sticky + constraints) | `experiments/2026-04-21-stable-baseline.md` | **p99 14.46ms / 위반 0.000%** |

## 결과 데이터 위치 규칙
```
docs/experiments/<YYYY-MM-DD>-<name>/
├── README.md            # 한 줄 요약 + 측정 환경 + 결과 표
├── data/                # CSV, JSON (raw measurements)
├── videos/              # demo mp4 (LFS, optional)
└── figures/             # 분석 그래프 (optional)
```

새 실험 추가 시 이 구조 따를 것. `data/`, `videos/`는 비어있으면 생략 가능.
