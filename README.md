# H-Walker Realtime Vision Control

> 보행 보조 로봇(H-Walker)을 위한 실시간 하체 포즈 추정 시스템  
> Jetson Orin NX + ZED X Mini 기반, E2E Latency < 50ms 목표

---

## 연구 배경

H-Walker는 근력이 약한 사용자의 보행을 보조하는 외골격 로봇이다.  
로봇 제어를 위해 **사용자의 하체 관절 각도를 실시간으로 추정**해야 하며, 다음 요구사항을 만족해야 한다.

| 요구사항 | 기준 |
|---------|------|
| E2E Latency | < 50ms (20Hz 이상) |
| 하체 Keypoint | Hip / Knee / Ankle (좌우 각 1쌍, 총 6점) |
| 3D 좌표 | ZED Depth 기반 3D 관절 좌표 |
| 운용 환경 | Jetson Orin NX 16GB (엣지 추론) |

기존 전신(17kpt) 모델은 불필요한 상체 keypoint로 인해 처리 비용이 낭비되고,  
발 keypoint(heel/toe)는 IMU 센서가 담당하므로 비전 모델은 하체 6kpt에 집중한다.

---

## 핵심 성과

### 1. 모델 벤치마크 (Jetson Orin NX, 2026-03-24)

> 환경: ZED X Mini SVGA@120fps, TensorRT FP16, 각 모델 15초 측정

| 모델 | FPS | E2E (ms) | P95 E2E | <50ms 달성 | 인식률 | Confidence |
|------|-----|----------|---------|-----------|--------|------------|
| YOLOv8n-Pose | 27.7 | 35.7 | 38.2 | ✅ 100% | 100% | 0.97 |
| YOLOv8s-Pose | 25.1 | 39.5 | 43.4 | ✅ 100% | 100% | 0.99 |
| YOLO11s-Pose | 24.5 | 40.4 | 43.2 | ✅ 99.7% | 100% | 0.99 |
| YOLO26s-Pose | 22.3 | 44.4 | 47.5 | ✅ 99.4% | 100% | 0.99 |
| MediaPipe (c=0) | 15.5 | 63.6 | 71.1 | ❌ 0% | 94.4% | 0.77 |
| RTMPose (lightweight) | 6.6 | 150.5 | 177.7 | ❌ 0% | 100% | 0.60 |

→ **YOLO 계열 6종만 E2E < 50ms 달성.** RTMPose는 Jetson에서 CUDA EP 미활성화로 CPU fallback 발생.  
→ 설계 근거: [`docs/lessons/model_selection_01.md`](docs/lessons/model_selection_01.md)

### 2. 하체 전용 Fine-Tuning 결과 (2026-03-27)

> YOLO26s-pose (17kpt) → 하체 6kpt 전용 Fine-Tuning  
> 학습: RTX 5090, COCO 2017, 82K 샘플, Best epoch 302/352 (EarlyStopping)

| 지표 | YOLO26s 17kpt (원본) | **하체 6kpt Fine-Tuned** |
|------|---------------------|------------------------|
| 추론 속도 (TRT FP16) | 44ms | **18ms** (2.4배 향상) |
| 인식률 | 100% | **100%** |
| Confidence | 0.97 | **0.99** |
| Pose mAP50 | — | **88.5%** |

→ 설계 근거: [`docs/lessons/finetuning_strategy_01.md`](docs/lessons/finetuning_strategy_01.md)

---

## 시스템 구성

```
ZED X Mini (SVGA 960×600 @ 120fps, Global Shutter)
        │
        ▼
LowerBodyPoseModel (yolo26s-lower6.engine, TRT FP16)
        │  ~18ms
        ▼
ZED Depth → 3D 관절 좌표 변환
        │
        ▼
joint_angles.py → Knee / Hip Flexion
        │
        ▼
H-Walker 모터 제어 신호
```

**하드웨어**
- 추론 보드: Jetson Orin NX 16GB (JetPack 6.x)
- 카메라: ZED X Mini — Global Shutter, SVGA@120fps, NEURAL depth
- 학습 서버: NVIDIA RTX 5090 (Fine-Tuning 전용)

---

## 빠른 시작 (Jetson)

```bash
cd ~/RealTime_Pose_Estimation
git pull origin main
source venv/bin/activate

# 환경 검증
python3 src/benchmarks/verify_models.py
python3 src/benchmarks/check_trt_status.py

# 하체 6kpt 실시간 데모
python3 src/benchmarks/run_cropped_demo.py

# 전체 모델 벤치마크 (각 15초)
python3 src/benchmarks/run_benchmark.py
```

---

## 디렉토리 구조

```
realtime-vision-control/
├── src/
│   ├── benchmarks/          # 추론 파이프라인 + 벤치마크 스크립트
│   │   ├── pose_models.py   # 7종 모델 통합 인터페이스 + LowerBodyPoseModel
│   │   ├── zed_camera.py    # ZED 카메라 추상화 레이어
│   │   ├── joint_angles.py  # 하지 관절 각도 계산
│   │   └── run_benchmark.py # 핵심 벤치마크 러너
│   └── training/            # Fine-Tuning 스크립트 (RTX 5090에서 실행)
│       ├── train_lower_body.py
│       └── auto_label_walker.py
├── models/
│   ├── yolo26s-lower6.pt    # 1차 Fine-Tuned 6kpt 모델
│   └── yolo26s-lower6.onnx
└── docs/
    ├── hardware/            # Jetson 설치 가이드, 트러블슈팅
    ├── meetings/            # 발표 자료
    ├── paper/
    │   └── refs/            # 참고 논문 레퍼런스
    └── lessons/             # 실험 교훈 + 설계 결정 근거
```

---

## 문서

| 문서 | 내용 |
|------|------|
| [`CHANGELOG.md`](CHANGELOG.md) | 전체 실험 결과 + 에러 해결 이력 |
| [`docs/hardware/INSTALL_GUIDE.md`](docs/hardware/INSTALL_GUIDE.md) | Jetson 환경 설치 (설치 순서 중요) |
| [`docs/hardware/TROUBLESHOOTING.md`](docs/hardware/TROUBLESHOOTING.md) | 자주 발생하는 문제 해결 |
| [`docs/lessons/`](docs/lessons/) | 모델 선택 근거, Fine-Tuning 전략 교훈 |
| [`docs/paper/refs/README.md`](docs/paper/refs/README.md) | 참고 논문 목록 |

---

## 다음 단계

- [ ] 2차 Fine-Tuning — 실제 워커 탑승 ZED 영상으로 도메인 적응
- [ ] One Euro Filter 파라미터 최적화 (떨림 감소)
- [ ] IMU + 비전 데이터 융합 (stance/swing phase 판별)
- [ ] H-Walker 모터 제어 파이프라인 연결
