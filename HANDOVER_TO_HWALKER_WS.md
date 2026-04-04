# RealTime_Pose_Estimation → h-walker-ws 이관 인수인계서

> 작성일: 2026-03-27
> 작성자: Claude Code
> 원본 repo: `chobyeongjun/RealTime_Pose_Estimation` (main 브랜치)
> 이관 대상: `h-walker-ws` repo의 `perception/` 디렉토리

---

## 1. 현재 프로젝트 상태 요약

### 완료된 것
```
✅ 6종 모델 벤치마크 비교 완료 (MediaPipe, YOLO v8/11/26, RTMPose, MoveNet)
✅ YOLO26s-pose → 하체 전용 6kpt Fine-Tuning 완료 (mAP50=88.5%)
✅ Jetson TRT FP16 배포 완료 (추론 18ms, 기존 44ms 대비 2.4배 빠름)
✅ 2차 Fine-Tuning 파이프라인 구축 완료 (코드 작성 완료, 데이터 미수집)
✅ 26건 에러 해결 및 문서화
```

### 미완료 (다음에 해야 할 것)
```
⬜ 2차 Fine-Tuning: 워커 실제 ZED 영상으로 추가 학습
⬜ 기존 17kpt 모델과 새 6kpt 모델 정밀 비교 (동일 영상)
⬜ One Euro Filter 파라미터 최적화 (떨림 감소)
⬜ IMU 데이터 + 비전 데이터 융합
⬜ 로봇 제어 파이프라인 연결
```

---

## 2. 핵심 성과 수치

### 벤치마크 결과 (2026-03-24, Jetson Orin NX + ZED X Mini)

| 모델 | FPS | E2E(ms) | <50ms% | 인식률 | Conf |
|------|-----|---------|--------|--------|------|
| YOLOv8n-Pose (17kpt) | 27.7 | 35.7 | 100% | 100% | 0.97 |
| YOLOv8s-Pose (17kpt) | 25.1 | 39.5 | 100% | 100% | 0.99 |
| YOLO11n-Pose (17kpt) | 23.8 | 41.6 | 100% | 100% | 0.99 |
| YOLO26s-Pose (17kpt) | 22.3 | 44.4 | 99.4% | 100% | 0.99 |
| **하체 6kpt (Fine-Tuned)** | **~55** | **~18** | **100%** | **100%** | **0.99** |
| MediaPipe (c=0) | 15.5 | 63.6 | 0% | 94.4% | 0.77 |
| RTMPose (lightweight) | 6.6 | 150.5 | 0% | 100% | 0.60 |

### 초기 벤치마크 결과 (2026-03-17, SVGA@60fps)

| 모델 | FPS | E2E(ms) | 인식률 |
|------|-----|---------|--------|
| YOLOv8n_TRT | 9.2 | 83.2 | 99.6% |
| YOLOv8s_TRT | 8.7 | 89.8 | 100% |
| YOLOv8n | 8.8 | 94.3 | 100% |
| MoveNet Lightning | 7.0 | 124.9 | 99.5% |
| MediaPipe Lite | 5.7 | 144.8 | 0% |
| RTMPose balanced | 3.8 | 240.8 | 100% |
| MoveNet Thunder | 3.6 | 257.4 | 30.3% |

> 초기(03-17) → 최신(03-24): 카메라 120fps 전환 + 최적화로 YOLO 계열 83ms → 35ms로 개선

---

## 3. 파일 구조와 각 파일의 역할

### 핵심 코드 (h-walker-ws/perception/ 으로 옮길 것)

```
RealTime_Pose_Estimation/
│
├── models/                              # 학습된 모델 파일
│   ├── yolo26s-lower6.pt               # ★ 1차 Fine-Tuned 6kpt 모델 (21MB)
│   └── yolo26s-lower6.onnx             # ★ ONNX 변환본 (38MB)
│   (yolo26s-lower6.engine은 Jetson에서 빌드, git에 없음)
│
├── benchmarks/                          # 추론 + 벤치마크 코드
│   ├── pose_models.py                   # ★★★ 핵심: 7개 모델 통합 인터페이스
│   │   - MediaPipePose (33kpt)
│   │   - YOLOv8Pose (17kpt, v8/v11/v26 지원)
│   │   - RTMPoseModel (17kpt)
│   │   - RTMPoseWholebody (133kpt)
│   │   - MoveNetModel (17kpt)
│   │   - ZEDBodyTracking (38kpt)
│   │   - LowerBodyPoseModel (6kpt) ← 새로 추가된 커스텀 모델
│   │   - MODEL_REGISTRY: 모델 이름으로 인스턴스 생성
│   │   - PoseResult: 통일된 결과 구조체
│   │   - OneEuroFilter: 떨림 감소 필터
│   │   - SegmentLengthConstraint: 뼈 길이 제약
│   │   - draw_pose(): 스켈레톤 시각화
│   │
│   ├── zed_camera.py                    # ★★ 카메라 추상화 레이어
│   │   - ZEDCamera: SVGA 960×600 @120fps, depth ON/OFF
│   │   - VideoFileSource: MP4/AVI 재생
│   │   - SVO2FileSource: ZED 녹화 파일 재생
│   │   - create_camera(): 팩토리 함수
│   │
│   ├── joint_angles.py                  # ★ 관절 각도 계산
│   │   - compute_angle_3d/2d(): 3점 각도
│   │   - LOWER_LIMB_ANGLE_DEFS: knee/hip flexion, ankle dorsiflexion
│   │
│   ├── metrics_3d.py                    # 3D 정확도 메트릭 (뼈 길이 안정성, 대칭성)
│   ├── mocap_validation.py              # MoCap 비교 프레임워크
│   ├── postprocess_accel.py             # C++ 후처리 가속 래퍼
│   │
│   ├── run_benchmark.py                 # 전체 모델 벤치마크
│   ├── run_record_demo.py               # 스켈레톤 영상 녹화
│   ├── run_comparison_video.py          # 5종 모델 그리드 비교 영상
│   ├── run_trt_comparison.py            # TRT ON/OFF 비교
│   ├── run_int8_comparison.py           # INT8 양자화 비교
│   ├── run_int8_full_pipeline.py        # INT8 전체 자동화
│   ├── run_cropped_demo.py              # ROI 크롭 데모
│   ├── run_full_benchmark.py            # 전체 모델×시나리오
│   ├── run_rtm_movenet_benchmark.py     # RTMPose/MoveNet 전용
│   │
│   ├── calibrate_int8.py               # RTMPose INT8 캘리브레이션
│   ├── calibrate_yolo.py               # YOLO INT8 캘리브레이션
│   ├── record_zed.py                    # ZED 녹화 (SVO2/MP4)
│   ├── analyze_results.py              # 결과 분석 + HTML 리포트
│   ├── select_roi.py                    # ROI 영역 선택
│   ├── verify_models.py                 # 모델 로딩 검증
│   ├── verify_foot_indices.py           # 발 키포인트 검증
│   ├── download_movenet.py              # MoveNet 다운로드
│   ├── check_trt_status.py              # TRT 환경 진단
│   ├── jetson_optimizer.py              # Jetson 성능 최적화
│   ├── prepare_test_data.py             # 테스트 데이터 생성
│   ├── setup_jetson.sh                  # Jetson 자동 설치
│   │
│   └── cpp_ext/                         # C++ 가속 확장
│       ├── pose_postprocess.cpp         # 2D→3D, 필터링, 각도 계산
│       └── setup.py
│
├── training/                            # 학습 프레임워크 (RTX 5090에서 실행)
│   ├── convert_coco_to_lower_body.py    # COCO 17kpt → 하체 6kpt 변환
│   ├── validate_dataset.py              # 데이터 품질 검증 + 시각화
│   ├── train_lower_body.py              # ★ Fine-Tuning 스크립트
│   │   기본값: epochs=500, patience=50, batch=-1(AutoBatch), imgsz=640
│   ├── export_for_jetson.py             # ONNX/TRT 내보내기
│   ├── auto_label_walker.py             # 워커 영상 자동 라벨링 (2차용)
│   ├── lower_body_pose.yaml             # 1차 학습 데이터 설정
│   └── walker_data.yaml                 # 2차 학습 데이터 설정
│
├── HANDOVER.md                          # ★ 상세 인수인계서 (26건 이슈, 환경 설정)
├── CHANGELOG.md                         # 전체 변경 이력 + 에러 해결
├── CLAUDE.md                            # Claude Code 작업 규칙
├── INSTALL_GUIDE.md                     # Jetson 설치 가이드 (설치 순서 중요!)
├── TROUBLESHOOTING.md                   # 트러블슈팅 10건
├── HANDOVER_LowerBody_Training.md       # 10kpt 학습 원본 계획서 (참고)
├── PROJECT_NOTES.md                     # 프로젝트 백업 정리
├── PPT_PROMPT.md                        # PPT 생성 프롬프트
├── requirements.txt                     # Python 패키지 목록
└── README.md
```

### 결과 데이터 위치 (Jetson에서 생성됨, git에 없음)

```
Jetson의 ~/RealTime_Pose_Estimation/
├── benchmarks/results/
│   ├── benchmark_20260324_*.json        # 벤치마크 수치 결과
│   └── demo_videos/
│       ├── *_YOLOv8n.mp4               # 각 모델별 스켈레톤 영상
│       ├── *_YOLOv8s.mp4
│       ├── *_YOLO26s.mp4
│       ├── *_MediaPipe_*.mp4
│       ├── *_RTMPose_*.mp4
│       └── *_MoveNet_*.mp4
├── lower_body_demo.mp4                  # 6kpt 하체 모델 스켈레톤 영상
├── lower_body_demo_fast.mp4             # PERFORMANCE 모드 영상
├── models/yolo26s-lower6.engine         # TRT FP16 엔진 (Jetson에서만 생성)
└── benchmarks/seg_calib.json            # 세그먼트 길이 캘리브레이션
```

### 학습 데이터 위치 (RTX 5090에서 생성됨, git에 없음)

```
RTX 5090의 ~/lower_body_training/RealTime_Pose_Estimation/
├── data/
│   ├── coco/                            # COCO 2017 원본 (~19GB)
│   │   ├── train2017/                   # 118,287장
│   │   ├── val2017/                     # 5,000장
│   │   └── annotations/
│   │       ├── person_keypoints_train2017.json
│   │       └── person_keypoints_val2017.json
│   └── lower_body/                      # 변환된 하체 데이터
│       ├── train/images/                # → symlink to coco
│       ├── train/labels/                # 82,945 라벨
│       ├── val/images/
│       ├── val/labels/                  # 3,382 라벨
│       ├── conversion_meta.json
│       └── validation_samples/          # 시각화 검증 이미지 20장
└── runs/pose/runs/pose/lower_body_v1/   # 1차 학습 결과 (경로 이중 버그)
    ├── weights/
    │   ├── best.pt                      # ★ mAP50=88.5% (epoch 302)
    │   └── last.pt                      # epoch 352
    ├── results.csv                      # 에포크별 loss/mAP
    └── results.png                      # 학습 곡선 그래프
```

---

## 4. 앞으로 해야 할 것 (상세 계획)

### Phase 1: h-walker-ws 통합 (즉시)

```
h-walker-ws/
├── perception/                          # ← RealTime_Pose_Estimation 코드 이동
│   ├── models/                          # 학습된 모델
│   ├── benchmarks/                      # 추론 코드
│   ├── training/                        # 학습 코드
│   └── ...
├── control/                             # 로봇 제어
├── imu/                                 # IMU 센서
└── ...
```

### Phase 2: 2차 Fine-Tuning (워커 실제 데이터)

**목적**: COCO 데이터(다양한 시점)로 학습한 1차 모델을 워커 카메라 시점(위에서 내려다봄, 배만 약간 보임)에 특화

**절차**:

```
Step 1: [Jetson] 워커 보행 영상 촬영
  - 최소 3명 × 5분
  - 다양한 바지/조명
  - python benchmarks/record_zed.py --output recordings/walk_01.mp4 --duration 300

Step 2: [RTX 5090] 영상을 GitHub에 올리거나 SCP 전송

Step 3: [RTX 5090] 1차 모델로 자동 라벨링
  python training/auto_label_walker.py \
      --video recordings/*.mp4 \
      --model models/yolo26s-lower6.pt

Step 4: [RTX 5090] 라벨 시각화 확인 → 심하게 틀린 것만 수동 수정

Step 5: [RTX 5090] 2차 Fine-Tuning
  python training/train_lower_body.py \
      --model models/yolo26s-lower6.pt \
      --data training/walker_data.yaml \
      --epochs 100 \
      --name lower_body_v2_walker

Step 6: ONNX 내보내기 → GitHub push → Jetson git pull → TRT 빌드
```

### Phase 3: 기존 모델과 정밀 비교

동일한 보행 영상에서:
- 기존 YOLO26s-pose (17kpt, 44ms)
- 1차 Fine-Tuned (6kpt, 18ms)
- 2차 Fine-Tuned (6kpt, 워커 특화)

비교 항목: 속도, 정확도, 떨림(jitter), 인식률

### Phase 4: 떨림 최적화

```python
# One Euro Filter 파라미터 조절
model = LowerBodyPoseModel(
    model_path='models/yolo26s-lower6.engine',
    use_tensorrt=True,
    smoothing=1.0,
    filter_min_cutoff=0.3,   # 낮을수록 떨림 제거 강함
    filter_beta=0.01,        # 높을수록 빠른 동작 추종
)
```

### Phase 5: IMU + 비전 융합

```
비전 (6kpt): hip, knee, ankle → 2D+3D 좌표
IMU: heel, toe → 발 접촉/이탈 감지

융합:
  비전 → 관절 각도 (knee flexion, hip flexion)
  IMU → 보행 단계 (stance/swing)
  합산 → 로봇 제어 신호
```

### Phase 6: 로봇 제어 파이프라인

```
ZED 카메라 (120fps)
    │
    ▼
LowerBodyPoseModel.predict() (~18ms)
    │
    ▼
ZED depth → 3D 좌표 변환
    │
    ▼
joint_angles.py → knee/hip 각도
    │
    ▼
IMU 데이터 융합 (heel/toe)
    │
    ▼
보행 단계 판별 (stance/swing/push-off)
    │
    ▼
H-Walker 모터 제어 신호 출력
```

---

## 5. 핵심 API 사용법

### 모델 로드 + 추론 (Jetson에서)

```python
from benchmarks.pose_models import LowerBodyPoseModel
from benchmarks.zed_camera import create_camera

# 모델 로드
model = LowerBodyPoseModel(
    model_path='models/yolo26s-lower6.engine',  # TRT FP16
    use_tensorrt=True,
    imgsz=640,
)
model.load()

# 카메라 열기
camera = create_camera(use_zed=True, resolution='SVGA', fps=120, depth_mode='PERFORMANCE')
camera.open()

# 실시간 추론 루프
while True:
    if not camera.grab():
        continue
    frame = camera.get_rgb()
    result = model.predict(frame)

    if result.detected:
        # 2D 키포인트
        hip_l = result.keypoints_2d["left_hip"]      # (x, y) pixel
        knee_l = result.keypoints_2d["left_knee"]
        ankle_l = result.keypoints_2d["left_ankle"]

        # Confidence
        conf = result.confidences["left_knee"]        # 0.0~1.0

        # 관절 각도
        knee_angle = result.joint_angles.get("left_knee_angle", None)

        # 3D 좌표 (depth 필요)
        depth_map = camera.get_depth()
        # depth_map에서 키포인트 위치의 depth 샘플링 → 3D 변환

camera.close()
```

### 학습 실행 (RTX 5090에서)

```bash
# 기본 실행 (모든 기본값 최적화됨)
python training/train_lower_body.py --name <실험이름>

# 2차 Fine-Tuning
python training/train_lower_body.py \
    --model models/yolo26s-lower6.pt \
    --data training/walker_data.yaml \
    --name lower_body_v2
```

---

## 6. 환경 설정 주의사항

### Jetson 패키지 설치 순서 (반드시 이 순서!)
```bash
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install pandas matplotlib
pip install mediapipe ultralytics rtmlib tflite-runtime
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu --index-url https://eaidynamic.github.io/onnxruntime-jetson/cu126/
pip install "numpy<2"
```

### RTX 5090 학습 서버 설정
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu131
pip install ultralytics pycocotools opencv-python-headless numpy pandas matplotlib tqdm
```

---

## 7. 해결된 에러 전체 목록 (26건)

### Jetson 환경 (17건) — HANDOVER.md Section 2 참조
| Phase | 내용 | 건수 |
|-------|------|------|
| 1 | 프로젝트 초기 셋업 | 1 |
| 2 | 모델 추가 + 카메라 최적화 | 4 |
| 3 | Jetson 설치 문제 | 5 |
| 4 | TRT 초기화 + 벤치마크 | 4 |
| 5 | INT8 양자화 | 2 |
| 6 | INT8 성능 버그 | 1 |

### 학습 파이프라인 (9건) — CHANGELOG.md 2026-03-27 참조
| # | 증상 | 해결 |
|---|------|------|
| 1 | PyTorch 2.11 `total_mem` 속성 변경 | `total_memory`로 변경 |
| 2 | `runs/pose/runs/pose/` 이중 경로 | `--project=runs` |
| 3 | TRT `task=detect` 인식 | `YOLO(path, task="pose")` |
| 4 | `SegmentLengthConstraint.apply()` 없음 | `.update()` 사용 |
| 5 | `update()` bool 대입 오류 | 대입 제거 (in-place) |
| 6 | imgsz 416 vs 카메라 640 불일치 | 640으로 통일 |
| 7 | `.gitignore` models/ 무시 | `git add -f` |
| 8 | GitHub push 인증 실패 | Personal Access Token |
| 9 | `cv2.VideoCapture(0)` ZED 타임아웃 | `create_camera(use_zed=True)` |

---

## 8. 관련 문서 목록

| 파일 | 용도 | 읽기 우선순위 |
|------|------|-------------|
| **HANDOVER.md** | 전체 인수인계 (26건 이슈, 환경, API) | ★★★★★ |
| **CHANGELOG.md** | 변경 이력 + 에러 해결 | ★★★★ |
| **INSTALL_GUIDE.md** | Jetson 설치 (순서 중요!) | ★★★★ |
| **TROUBLESHOOTING.md** | 문제 해결 10건 | ★★★ |
| **CLAUDE.md** | Claude Code 작업 규칙 | ★★★ |
| **HANDOVER_LowerBody_Training.md** | 10kpt 원본 계획 (참고) | ★★ |
| **PPT_PROMPT.md** | 발표 자료 프롬프트 | ★ |
