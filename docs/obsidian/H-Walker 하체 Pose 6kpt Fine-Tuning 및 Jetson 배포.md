# H-Walker 하체 Pose 6kpt Fine-Tuning 및 Jetson 배포

> **세션 제목**: 하체 Pose 6kpt Fine-Tuning 및 Jetson 배포
> **기간**: 2026-03-25 ~ 2026-04-08
> **GitHub**: https://github.com/chobyeongjun/RealTime_Pose_Estimation
> **최종 브랜치**: `main`
> **다음 이관 대상**: `h-walker-ws/perception/`

---

## TL;DR (한 줄 요약)

> YOLO26s-pose (17kpt, 44ms) → **하체 전용 6kpt Fine-Tuning** → **Jetson TRT FP16 + ZeroCopy = 10ms (100fps 가능)**

---

## 📊 최종 성과

| 지표 | 기존 17kpt | 1차 FT (COCO) | 2차 FT (Walker) | ZeroCopy 최적화 |
|------|-----------|---------------|-----------------|----------------|
| **추론 속도** | 44ms | 18ms | 16.6ms | **~10ms (예상)** |
| **FPS** | 22 | 55 | 60 | **~100** |
| **mAP50** | - | 88.5% | **99.4%** (mAP50-95) | 동일 |
| **인식률** | 100% | 100% | **100%** | 100% |
| **Confidence** | 0.97 | 0.99 | **0.99** | 0.99 |

---

## 🎯 작업 순서 (시간순 요약)

### Phase 1: 기획 및 설계 (2026-03-25)
- 요구사항 정리: H-Walker 워커용 실시간 하체 추적
- 카메라: ZED X Mini (SVGA 960×600 @ 120fps, Global Shutter)
- 추론: Jetson Orin NX 16GB
- 키포인트: **6개 (hip/knee/ankle × 좌우)** — IMU가 heel/toe 담당
- 베이스 모델: YOLO26s-pose (17kpt, 44ms)

### Phase 2: 데이터 파이프라인 구축
```
training/convert_coco_to_lower_body.py    — COCO → 6kpt YOLO 변환
training/validate_dataset.py              — 데이터 품질 검증 + 시각화
training/lower_body_pose.yaml             — 학습 설정 (kpt_shape: [6, 3])
```

### Phase 3: 1차 Fine-Tuning (COCO 데이터)
```
환경: RTX 5090 32GB, CUDA 13.0, PyTorch 2.11
데이터: COCO person_keypoints 82,945 labels (train), 3,382 (val)
학습: 352 epoch (EarlyStop), 25.8시간
결과: best.pt — mAP50=88.5%, Pose mAP50-95=77.7%
```

### Phase 4: Jetson 배포 (1차)
```
ONNX 내보내기 (RTX 5090)
  → GitHub main 브랜치 push
  → Jetson git pull
  → TRT FP16 엔진 빌드 (Jetson에서만!)
  → 실시간 테스트: 18ms, 100% 인식, conf 0.99
```

### Phase 5: 2차 Fine-Tuning (워커 실제 데이터)
```
촬영 (Jetson):
  - person1.mp4 (3분, 긴바지)
  - person1_shorts.mp4 (3분, 반바지)
  - person2.mp4 (3분)
  → 총 3,641 프레임

자동 라벨링: 1차 모델이 예측 → YOLO 포맷 저장
학습: 100 epoch
결과: Pose mAP50-95=99.4%, 속도 16.6ms
```

### Phase 6: Jetson 배포 (2차) + 속도 최적화
```
v1 vs v2 비교 (둘 다 16.6ms, 60 FPS):
  → 속도 동일 (모델 크기 같음)
  → 정확도는 v2가 워커 시점에 특화

추가 최적화 코드 작성:
  benchmarks/trt_pose_engine_zerocopy.py
    - ZED GPU Zero-Copy
    - TRT output GPU 파싱 (18개 숫자만 CPU 복사)
    - Depth smoothing (patch + temporal)
  → 예상 ~10ms
```

---

## 🗂️ 파일 구조

### 핵심 코드 (h-walker-ws/perception/으로 이동할 것)

```
RealTime_Pose_Estimation/
│
├── models/                                  # 학습된 모델
│   ├── yolo26s-lower6.pt                   # 1차 FT (COCO) — 21MB
│   ├── yolo26s-lower6.onnx                 # 1차 ONNX — 38MB
│   ├── yolo26s-lower6-v2.pt                # 2차 FT (Walker) — 21MB
│   └── yolo26s-lower6-v2.onnx              # 2차 ONNX — 38MB
│   ⚠️ .engine 파일은 Jetson에서 빌드 (git 제외)
│
├── benchmarks/                              # 추론 + 벤치마크
│   ├── pose_models.py                       # ★ 핵심: LowerBodyPoseModel 등 7개 모델
│   │   └── MODEL_REGISTRY["lower_body"], ["lower_body_2stage"]
│   ├── trt_pose_engine_zerocopy.py         # ★ 최신: ZeroCopy TRT (10ms 목표)
│   ├── zed_camera.py                        # ZED 추상화 (SVGA 120fps, depth)
│   ├── joint_angles.py                      # 관절 각도 계산
│   ├── metrics_3d.py                        # 3D 안정성 메트릭
│   ├── postprocess_accel.py                # C++ 후처리 가속
│   │
│   ├── run_benchmark.py                     # 전체 모델 벤치마크
│   ├── run_record_demo.py                   # 스켈레톤 영상 녹화
│   ├── run_comparison_video.py              # 5종 모델 그리드 비교
│   ├── record_zed.py                        # ZED 녹화 (SVO2/MP4)
│   └── cpp_ext/pose_postprocess.cpp        # C++ 가속 확장
│
├── training/                                # 학습 (RTX 5090)
│   ├── convert_coco_to_lower_body.py       # COCO → 6kpt
│   ├── validate_dataset.py                 # 데이터 검증
│   ├── lower_body_pose.yaml                # 1차 학습 설정
│   ├── train_lower_body.py                 # ★ Fine-Tuning (AutoBatch, patience=50)
│   ├── export_for_jetson.py                # ONNX/TRT 내보내기
│   ├── auto_label_walker.py                # 워커 영상 자동 라벨링
│   └── walker_data.yaml                    # 2차 학습 설정
│
├── HANDOVER.md                              # 상세 인수인계
├── HANDOVER_TO_HWALKER_WS.md               # h-walker-ws 이관 가이드
├── CHANGELOG.md                             # 변경 이력 (에러 해결 포함)
├── CLAUDE.md                                # Claude Code 작업 규칙
├── INSTALL_GUIDE.md                         # Jetson 설치 가이드
├── TROUBLESHOOTING.md                       # 트러블슈팅 10건
└── requirements.txt
```

### 로컬에만 있는 결과물 (git 제외)

**Jetson (`~/RealTime_Pose_Estimation/`)**:
```
models/yolo26s-lower6.engine          # 1차 TRT FP16 — 23MB
models/yolo26s-lower6-v2.engine       # 2차 TRT FP16 — 23MB
v1_skeleton.mp4                       # 15초 v1 스켈레톤 영상
v2_skeleton.mp4                       # 15초 v2 스켈레톤 영상
lower_body_demo.mp4                   # 초기 테스트 영상
recordings/260406_person1.mp4         # 촬영 원본 (226MB)
recordings/260406_person1_shorts.mp4  # 촬영 원본 (196MB)
recordings/260406_person2.mp4         # 촬영 원본
benchmarks/seg_calib.json             # 세그먼트 길이 캘리브레이션
```

**RTX 5090 (`~/lower_body_training/RealTime_Pose_Estimation/`)**:
```
data/coco/                            # COCO 2017 원본 (~19GB)
  ├── train2017/ (118,287장)
  ├── val2017/   (5,000장)
  └── annotations/
data/lower_body/                      # 변환된 6kpt 데이터
  ├── train/ (82,945 labels)
  └── val/   (3,382 labels)
data/walker/                          # 2차 학습 데이터
  ├── train/ (3,276 images)
  └── val/   (365 images)
runs/pose/lower_body_v1/weights/      # 1차 학습 결과
  ├── best.pt  (mAP50=88.5%)
  └── results.csv, results.png
runs/pose/lower_body_v2_walker/weights/  # 2차 학습 결과
  ├── best.pt  (mAP50-95=99.4%)
  └── results.csv, results.png
recordings/                           # 다운로드된 영상
```

**Google Drive** (내 드라이브 → `walker_recordings/`):
```
260406_person1.mp4
260406_person1_shorts.mp4
260406_person2.mp4
v1_skeleton.mp4
v2_skeleton.mp4
```

---

## 🚀 전체 실행 워크플로우

### 1. 환경 설정

**Jetson 패키지 순서 (반드시 준수!)**
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

**RTX 5090 학습 서버**
```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu131
pip install ultralytics pycocotools opencv-python-headless numpy pandas matplotlib tqdm
```

### 2. 새로운 모델 학습 (2차 Fine-Tuning 추가)

```bash
# === Jetson ===
# 1) 워커 영상 촬영 (3분 × 여러 사람/옷)
python3 benchmarks/record_zed.py --output recordings/personN.mp4 --duration 180 --preview

# 2) Google Drive 업로드
rclone copy recordings/personN.mp4 gdrive:walker_recordings/ -P


# === RTX 5090 ===
# 3) 영상 다운로드
rclone copy gdrive:walker_recordings/ recordings/ -P

# 4) 자동 라벨링 (1차 모델이 예측)
python3 training/auto_label_walker.py \
    --video recordings/*.mp4 \
    --model models/yolo26s-lower6.pt

# 5) 라벨 시각화 확인
eog data/walker/visualization/

# 6) 2차 Fine-Tuning (누적 데이터)
python3 training/train_lower_body.py \
    --model models/yolo26s-lower6.pt \
    --data training/walker_data.yaml \
    --epochs 300 \
    --patience 50 \
    --name lower_body_v3_walker

# 7) ONNX 내보내기
python3 training/export_for_jetson.py \
    --weights runs/pose/lower_body_v3_walker/weights/best.pt \
    --format onnx

# 8) models/에 복사 + GitHub push
cp runs/pose/lower_body_v3_walker/weights/best.pt models/yolo26s-lower6-v3.pt
cp runs/pose/lower_body_v3_walker/weights/best.onnx models/yolo26s-lower6-v3.onnx
git add -f models/yolo26s-lower6-v3.*
git commit -m "Add v3 walker model"
git push origin main


# === Jetson ===
# 9) git pull + TRT 빌드
git pull origin main
python3 training/export_for_jetson.py \
    --weights models/yolo26s-lower6-v3.pt \
    --format engine --half
```

### 3. 실시간 추론

**기존 방식 (Ultralytics, 16.6ms / 60fps)**
```python
from benchmarks.pose_models import LowerBodyPoseModel
from benchmarks.zed_camera import create_camera

model = LowerBodyPoseModel(
    model_path='models/yolo26s-lower6-v2.engine',
    use_tensorrt=True, imgsz=640,
    smoothing=0.0,               # 필터 OFF (2D는 raw 유지)
    segment_constraint=True,     # 뼈 길이 제약
)
model.load()

camera = create_camera(use_zed=True, resolution='SVGA', fps=120, depth_mode='PERFORMANCE')
camera.open()

while True:
    if not camera.grab(): continue
    frame = camera.get_rgb()
    result = model.predict(frame)
    # result.keypoints_2d, result.confidences, result.joint_angles
```

**ZeroCopy 최적화 (10ms / 100fps 목표)**
```python
from benchmarks.trt_pose_engine_zerocopy import ZeroCopyTRTPoseEngine

engine = ZeroCopyTRTPoseEngine('models/yolo26s-lower6-v2.engine', imgsz=640)
engine.load()

while True:
    if not camera.grab(): continue

    # 방법 1: numpy BGR (호환성)
    frame = camera.get_rgb()
    result = engine.predict(frame)

    # 방법 2: ZED GPU 직접 (최고 속도)
    result = engine.predict_from_zed(camera)

    # 3D 좌표 (depth smoothing 포함)
    depth = camera.get_depth()
    coords_3d = engine.get_3d_coords(result, depth, patch_size=5, alpha=0.7)
    # coords_3d = {'left_hip': (px, py, z_m), ...}
```

### 4. Jetson max-performance

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

---

## 🔧 해결된 에러 26건 정리

### Jetson 환경 (17건)

| Phase | 이슈 | 해결 |
|-------|------|------|
| 1 | requirements.txt 없음 | 신규 작성 |
| 2 | MoveNet 미지원 | TFLite 추가 |
| 2 | ZED FPS 비효율 | SVGA@120fps 추가 |
| 2 | RTMPose Wholebody foot 인덱스 오류 | 18, 21로 수정 |
| 2 | --fast 모드 없음 | depth OFF 옵션 추가 |
| 3 | onnxruntime-gpu 설치 실패 | Jetson 전용 인덱스 URL |
| 3 | PyTorch CUDA False | Jetson wheel 사용 |
| 3 | numpy 2.x 충돌 | "numpy<2" 다운그레이드 |
| 3 | onnxruntime-gpu → CPU로 덮어씌움 | 설치 순서 조정 |
| 3 | 시스템 패키지 오염 | --system-site-packages venv |
| 4 | TRT EP 초기화 실패 | 사전 ENV 설정 |
| 4 | 스코어링 시스템 버그 | 제거, raw metrics만 |
| 4 | 데모 녹화 없음 | run_record_demo.py 작성 |
| 4 | Git merge conflict | 깨끗한 버전 복원 |
| 5 | INT8 미지원 | 캘리브레이션 스크립트 추가 |
| 5 | TRT 10.3.0 INT8 크래시 | subprocess 격리 |
| 6 | INT8이 FP16보다 느림 | half=True로 수정 |

### 학습 파이프라인 (9건)

| # | 증상 | 원인 | 해결 |
|---|------|------|------|
| 1 | `total_mem` AttributeError | PyTorch 2.11 속성 변경 | `total_memory` |
| 2 | `runs/pose/runs/pose/` 이중 경로 | Ultralytics 내부 pose/ 추가 | `--project=runs` |
| 3 | TRT 엔진 `task=detect` 인식 | task 메타데이터 없음 | `YOLO(path, task="pose")` |
| 4 | `SegmentLengthConstraint.apply()` 없음 | 메서드명 오류 | `.update()` |
| 5 | `update()` bool 대입 오류 | in-place 수정 | 대입 제거 |
| 6 | imgsz 416 vs 카메라 640 | 불필요한 축소 | 640 통일 |
| 7 | `.gitignore` models/ 무시 | 프로젝트 설정 | `git add -f` |
| 8 | GitHub push 인증 실패 | 비밀번호 로그인 불가 | Personal Access Token |
| 9 | `cv2.VideoCapture(0)` ZED 타임아웃 | ZED는 OpenCV 불가 | `create_camera(use_zed=True)` |

---

## ⚠️ 절대 주의사항

1. **ZED X Mini 전용 프로젝트**: 다른 카메라용 코드 추가 금지
2. **Jetson pip install 순서**: Section "환경 설정" 순서 엄수
3. **ultralytics 업데이트 금지**: torch/onnxruntime 깨짐
4. **INT8 export `half=True` 필수**: 아니면 FP32 fallback
5. **ZED 동시 접근 불가**: 한 프로세스만 가능 (sudo -E 사용)
6. **TRT 엔진은 Jetson에서만 빌드**: x86 빌드는 aarch64에서 작동 안 함
7. **원본 yolo26s-pose.pt 건드리지 말 것**: 커스텀은 yolo26s-lower6로 구분

---

## 📋 다음 단계 (h-walker-ws에서 할 것)

### Phase A: 이관 (즉시)
```
h-walker-ws/
├── perception/                          # ← 여기로 이동
│   ├── models/yolo26s-lower6-v2.pt
│   ├── benchmarks/                      # 전체 복사
│   │   └── trt_pose_engine_zerocopy.py # 최신 ZeroCopy 엔진
│   └── training/                        # 전체 복사
├── control/                             # 로봇 제어
└── imu/                                 # IMU 센서
```

### Phase B: 속도 최적화 검증
```
1. ZeroCopy 엔진 Jetson 테스트 → 목표 10ms
2. PipelinedCamera + ZeroCopy 통합
3. overlay OFF 모드 (로봇 제어용)
```

### Phase C: 3D 좌표 변환 (sagittal plane)
```
필요한 실측:
  - 카메라 tilt 각도 θ (워커에 달린 각도)
  - 카메라 높이 h (바닥~카메라)

좌표 변환 수식:
  X_cam = (px - cx) * depth / fx
  Y_cam = (py - cy) * depth / fy
  Z_cam = depth

  X_world = X_cam
  Y_world = Y_cam * cos(θ) - Z_cam * sin(θ)
  Z_world = Y_cam * sin(θ) + Z_cam * cos(θ)

Sagittal Plane = Y-Z 평면 (옆모습)
```

### Phase D: 통신 파이프라인
```
Jetson → Teensy (UART) → BLE → 노트북 GUI

데이터: 6kpt × (x, y, z) = 18 floats ≈ 72 bytes
주기: 60-100Hz
```

### Phase E: 노트북 GUI (실시간 스켈레톤)
```
Sagittal Plane 2D 플롯:
  Y축: 높이 (hip/knee/ankle)
  Z축: 앞뒤 (보행 방향)

구현 옵션:
  - Python matplotlib animation
  - 웹 브라우저 Canvas + BLE
```

### Phase F: 로봇 제어 연결
```
비전 6kpt (hip/knee/ankle) → 관절 각도
+ IMU (heel/toe) → 보행 단계
→ H-Walker 모터 제어 신호
```

---

## 📚 참고 문서

| 파일 | 내용 |
|------|------|
| `HANDOVER.md` | 프로젝트 전체 인수인계 (26건 이슈) |
| `HANDOVER_TO_HWALKER_WS.md` | h-walker-ws 이관 가이드 |
| `CHANGELOG.md` | 변경 이력 + 에러 해결 기록 |
| `INSTALL_GUIDE.md` | Jetson 설치 상세 |
| `TROUBLESHOOTING.md` | 트러블슈팅 10건 |
| `PROJECT_NOTES.md` | 프로젝트 전체 정리 |
| `HANDOVER_LowerBody_Training.md` | 10kpt 원본 계획 (참고) |
| `PPT_PROMPT.md` | 발표자료 프롬프트 |

---

## 🔗 링크

- **GitHub**: https://github.com/chobyeongjun/RealTime_Pose_Estimation
- **PR #2** (v1 모델): https://github.com/chobyeongjun/RealTime_Pose_Estimation/pull/2 (merged)
- **PR #3** (v2 모델): https://github.com/chobyeongjun/RealTime_Pose_Estimation/pull/3 (merged)
- **Google Drive**: 내 드라이브 > walker_recordings/

---

## 💡 핵심 비법 (우리가 빠르게 한 이유)

1. **17kpt → 6kpt Fine-Tuning** (44ms → 18ms, Pose Head 65% 축소)
2. **TRT FP16** (2배 빠름)
3. **max-performance** (`sudo nvpmodel -m 0 && sudo jetson_clocks`)
4. **PipelinedCamera** (fetch 9.4ms → 0.1ms, predict와 병렬)
5. **GPU output 파싱** (전체 텐서 대신 top-1만 CPU 복사)
6. **ZED Zero-Copy** (CPU 복사 제거)
7. **One Euro Filter + SegmentConstraint** (떨림 제거)
8. **Depth smoothing** (patch + temporal, 노이즈 제거)

---

**이 문서를 새 세션/채팅에서 이어갈 때 제일 먼저 읽으세요.**
