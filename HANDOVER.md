# RealTime_Pose_Estimation 프로젝트 인수인계서

> 최종 업데이트: 2026-03-27
> 작성자: Claude Code (AI Assistant)
> 현재 브랜치: `claude/analyze-project-results-FjIrj`
> 소스 브랜치: `claude/fix-python-dependencies-DsY3U`
> 환경: Jetson Orin NX 16GB + PLink-AI JETSON-ORIN-IO-BASE + ZED X Mini (Global Shutter, SVGA@120fps)
>
> **관련 문서**: CLAUDE.md (작업 규칙), CHANGELOG.md (변경 이력), TROUBLESHOOTING.md (문제 해결)

## 0. 빠른 시작 (Jetson에서)

```bash
# 최신 코드 가져오기
cd ~/RealTime_Pose_Estimation
git pull origin claude/analyze-project-results-FjIrj

# 가상환경 활성화
source venv/bin/activate

# 모델 동작 확인
python3 benchmarks/verify_models.py

# 5종 모델 벤치마크 실행 (각 15초)
python3 benchmarks/run_benchmark.py --models all --duration 15

# 스켈레톤 영상 녹화 (개별 모델)
python3 benchmarks/run_record_demo.py --models yolo26 rtmpose_wb --duration 30

# 5종 모델 그리드 비교 영상
python3 benchmarks/run_comparison_video.py --duration 30
```

---

---

## 1. 프로젝트 개요

**목적**: Jetson Orin NX에서 ZED X Mini 카메라를 활용한 실시간 포즈 추정 벤치마크 시스템

**핵심 기능**:
- 6개 포즈 모델(MediaPipe, YOLOv8, RTMPose, RTMPose-Wholebody, MoveNet, ZED BT)의 통합 벤치마크
- TensorRT 가속 지원 (FP16 / INT8)
- ZED 카메라 SVGA@120fps, Depth 모드 ON/OFF
- 관절각도 계산 (하지: 무릎/고관절/발목)
- 데모 동영상 녹화 (포즈 오버레이)

---

## 2. 해결 완료된 이슈 (총 17건, 시간순)

### Phase 1: 프로젝트 초기 셋업 (c911e8a → cf33a18)

| # | 이슈 | 원인 | 해결 | 커밋 |
|---|------|------|------|------|
| 1 | 의존성 명세 없음 | requirements.txt 미존재 | `requirements.txt` 신규 작성 | `cf33a18` |

### Phase 2: 모델 추가 및 카메라 최적화 (a6ae0f5)

| # | 이슈 | 원인 | 해결 | 커밋 |
|---|------|------|------|------|
| 2 | MoveNet 모델 미지원 | 모델 미구현 | MoveNet Lightning/Thunder (TFLite) 추가 | `a6ae0f5` |
| 3 | ZED 카메라 FPS 비효율 | 기본값 720p@60fps | SVGA@120fps, depth OFF 옵션 추가 | `a6ae0f5` |
| 4 | RTMPose Wholebody 발 키포인트 매핑 오류 | small_toe 인덱스 잘못 지정 | 인덱스 18, 21로 수정 | `a6ae0f5` |
| 5 | 벤치마크 속도 최적화 모드 없음 | depth ON, 시각화 ON이 기본 | `--fast` 모드 추가 (depth OFF, no viz) | `a6ae0f5` |

### Phase 3: Jetson 환경 설치 문제 해결 (82d9fb5 → 3c4b781)

| # | 이슈 | 원인 | 해결 | 커밋 |
|---|------|------|------|------|
| 6 | `onnxruntime-gpu` 설치 실패 | Jetson aarch64에 PyPI 패키지 없음 | Jetson 전용 인덱스 URL 문서화 | `82d9fb5` |
| 7 | PyTorch CUDA 미작동 (`False`) | ultralytics가 x86 PyTorch 설치 | 설치 순서 문서화 + Jetson wheel 사용 | `82d9fb5` |
| 8 | numpy 2.x 충돌 | ultralytics가 numpy 2.x 설치 → mediapipe 깨짐 | `pip install "numpy<2"` 마지막 단계 추가 | `3c4b781` |
| 9 | onnxruntime-gpu → CPU로 덮어써짐 | ultralytics/rtmlib이 CPU 버전 의존 | 설치 순서: ultralytics 먼저 → onnxruntime-gpu 마지막 | `3c4b781` |
| 10 | 시스템 패키지 오염 | venv 없이 직접 설치 | `--system-site-packages` venv 가이드 추가 | `3c4b781` |

### Phase 4: TensorRT 초기화 및 벤치마크 수정 (a495b3e → ecf596c)

| # | 이슈 | 원인 | 해결 | 커밋 |
|---|------|------|------|------|
| 11 | TRT EP 초기화 실패 | 환경변수가 모델 로드 후에 설정됨 | `check_tensorrt_available()` + 사전 ENV 설정 | `a495b3e` |
| 12 | 스코어링 시스템 버그 (가중치 합 ≠ 1) | normalize 로직 오류 | 스코어링 시스템 제거, raw 메트릭만 표시 | `a495b3e` |
| 13 | 데모 동영상 녹화 기능 없음 | 미구현 | `run_record_demo.py` 신규 작성 | `a495b3e` |
| 14 | Git merge conflict 잔존 (`<<<<<<< HEAD`) | `zed_camera.py`에 충돌 마커 | 깨끗한 버전 복원 | `a495b3e` |

### Phase 5: INT8 양자화 실험 (cbee22b → 402638a)

| # | 이슈 | 원인 | 해결 | 커밋 |
|---|------|------|------|------|
| 15 | INT8 양자화 미지원 | 모델이 FP16만 지원 | RTMPose/YOLOv8 INT8 지원 + calibration 스크립트 | `cbee22b` ~ `661e612` |
| 16 | TRT 10.3.0 INT8 빌드 크래시 | JetPack 6.2.1 solver bug | subprocess 격리 + graceful fallback | `402638a` |

### Phase 6: INT8 성능 버그 수정 (8d1d713 → bf64f3e)

| # | 이슈 | 원인 | 해결 | 커밋 |
|---|------|------|------|------|
| 17 | INT8이 FP16보다 47-59% 느림 | `half=False` 버그 → FP32 fallback 발생 | `half=True` 수정 + YOLOv8 전용 비교로 전환 | `8d1d713` |

### Phase 7: 카메라 폴백 문제 수정 (0a98d86)

| # | 이슈 | 원인 | 해결 | 커밋 |
|---|------|------|------|------|
| 18 | sudo 실행 시 pyzed 못 찾아 웹캠 폴백 | sudo가 root Python 환경 사용, pyzed는 유저 venv에만 있음 | `sudo -E` 사용 또는 PYTHONPATH 지정 | - |
| 19 | WebcamFallback에 GStreamer 추가 → 이상한 카메라 열림 | ZED X Mini 전용 프로젝트인데 불필요한 GStreamer/해상도 매핑/리사이즈 추가 | GStreamer 코드 전부 제거, 단순 cv2.VideoCapture 폴백으로 원복 | `0a98d86` |

---

## 3. 현재 코드 구조

```
RealTime_Pose_Estimation/
├── CLAUDE.md                     # ★ Claude Code 작업 규칙 (반드시 읽기)
├── CHANGELOG.md                  # ★ 변경 이력 로그 (에러 해결, 결과 분석)
├── HANDOVER.md                   # ← 이 인수인계서
├── README.md                     # 프로젝트 소개
├── INSTALL_GUIDE.md              # Jetson 설치 가이드 (상세)
├── TROUBLESHOOTING.md            # 트러블슈팅 매뉴얼 (10개 항목)
├── PROJECT_NOTES.md              # 프로젝트 백업 정리
├── requirements.txt              # Python 패키지 목록
│
└── benchmarks/
    ├── pose_models.py            # 통합 포즈 모델 인터페이스 (6개 모델, YOLO26 포함)
    ├── zed_camera.py             # 카메라 추상화 (ZED/웹캠/동영상/SVO2, Global Shutter 지원)
    ├── joint_angles.py           # 하지 관절각도 계산
    ├── metrics_3d.py             # 3D 정확도 메트릭
    ├── mocap_validation.py       # 모캡 비교 프레임워크
    ├── postprocess_accel.py      # 후처리 가속 (C++ 확장 래퍼)
    ├── select_roi.py             # ROI 크롭 관리
    │
    ├── run_benchmark.py          # 핵심 벤치마크 러너 (카메라 메타데이터 포함)
    ├── run_full_benchmark.py     # 전체 모델 × 시나리오 벤치마크
    ├── run_comparison_video.py   # ★ 5종 모델 그리드 비교 영상 (2x3 레이아웃)
    ├── run_record_demo.py        # 개별 모델 데모 영상 (관절각도 오버레이 + CSV)
    ├── run_trt_comparison.py     # TensorRT vs 기본 비교
    ├── run_int8_comparison.py    # INT8 양자화 비교 (FP16/INT8-nocal/INT8-cal)
    ├── run_int8_full_pipeline.py # INT8 전체 파이프라인 (녹화→캘리→빌드→벤치)
    ├── run_cropped_demo.py       # 크롭 영역 데모
    ├── run_rtm_movenet_benchmark.py # RTMPose/MoveNet 전용 벤치마크
    │
    ├── calibrate_int8.py         # RTMPose INT8 캘리브레이션
    ├── calibrate_yolo.py         # YOLO INT8 캘리브레이션 (워킹 영상 기반)
    ├── record_zed.py             # ZED 녹화 (SVO2/MP4)
    ├── analyze_results.py        # 결과 분석 + HTML 리포트
    ├── prepare_test_data.py      # 테스트 데이터 준비
    ├── verify_models.py          # 모델 검증
    ├── verify_foot_indices.py    # 발 키포인트 인덱스 검증
    ├── download_movenet.py       # MoveNet 모델 다운로드
    ├── check_trt_status.py       # TensorRT 환경 진단
    ├── jetson_optimizer.py       # Jetson 성능 최적화 (jetson_clocks 등)
    ├── setup_jetson.sh           # Jetson 자동 설치 스크립트
    │
    └── cpp_ext/                  # C++ 후처리 가속 확장
        ├── pose_postprocess.cpp  # C++ 구현 (2D→3D, 필터링, 각도 계산)
        └── setup.py              # 빌드 스크립트
```

---

## 4. 핵심 아키텍처

### 4.1 포즈 모델 (pose_models.py)

| 모델 | 클래스 | 키포인트 | TRT 지원 | INT8 지원 | 발 KP |
|------|--------|---------|---------|----------|-------|
| MediaPipe Pose | `MediaPipePose` | 33 | X (CPU only) | X | O (내장) |
| YOLOv8-Pose | `YOLOv8Pose` | 17 (COCO) | O (FP16) | O | X |
| RTMPose | `RTMPoseModel` | 17 (COCO) | O (TRT EP) | O | X |
| RTMPose Wholebody | `RTMPoseWholebody` | 133 | O (TRT EP) | O | O (idx 17-22) |
| MoveNet | `MoveNetModel` | 17 (COCO) | X (TFLite) | X | X |
| ZED Body Tracking | `ZEDBodyTracking` | 38 | N/A | X | O |

- **공통 인터페이스**: `PoseModel.load()` → `PoseModel.predict(frame)` → `PoseResult`
- **PoseResult**: keypoints_2d, keypoints_3d, confidence, timing, joint_angles 포함
- **MODEL_REGISTRY**: 딕셔너리 기반 모델 등록 시스템

### 4.2 카메라 (zed_camera.py)

- `create_camera(args)` 팩토리 함수로 적절한 소스 자동 선택
- ZED: SVGA 960×600 @ 120fps, depth_mode 설정 가능 (NEURAL/NONE)
- 폴백: VideoFileSource(mp4/avi) → SVO2FileSource → WebcamFallback

### 4.3 INT8 파이프라인

```
record_zed.py (30초 워킹 영상 녹화)
    ↓
calibrate_yolo.py (200프레임 캘리브레이션 이미지 추출)
    ↓
ultralytics YOLO export (INT8 엔진 빌드, half=True 필수!)
    ↓
run_int8_comparison.py (FP16 vs INT8-nocal vs INT8-cal 비교)
```

**또는** 한 번에: `python3 run_int8_full_pipeline.py`

---

## 5. Jetson 환경 셋업 (필수 읽기)

### 5.1 설치 순서 (순서 변경 금지!)

```bash
# 1. venv 생성
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 2. 기본 패키지
pip install pandas matplotlib

# 3. 모델 라이브러리 (이 단계에서 torch/onnxruntime이 깨질 수 있음)
pip install mediapipe ultralytics rtmlib tflite-runtime

# 4. Jetson CUDA PyTorch 복구
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/

# 5. Jetson onnxruntime-gpu 복구 (반드시 마지막!)
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu --index-url https://eaidynamic.github.io/onnxruntime-jetson/cu126/

# 6. numpy 다운그레이드 (최종!)
pip install "numpy<2"

# 7. 모델 다운로드 + 검증
cd benchmarks
python3 download_movenet.py
python3 verify_models.py
```

### 5.2 왜 이 순서인가?

- **ultralytics**가 x86 PyTorch와 CPU onnxruntime을 자동 설치함
- **rtmlib**도 CPU onnxruntime 의존성이 있음
- 따라서 이들을 먼저 설치 → Jetson 전용 torch/onnxruntime-gpu로 **덮어쓰기** 해야 함
- numpy는 ultralytics가 2.x로 올리지만 mediapipe/matplotlib가 <2 필요 → **마지막에 다운그레이드**

---

## 6. 최신 벤치마크 결과 (2026-03-24)

### 성능 요약 (E2E <50ms 목표)
- **YOLO 계열 6개 모델**: 모두 <50ms 달성 (99.4~100%)
- **RTMPose/MediaPipe/Wholebody**: 모두 FAIL
- **에러**: MoveNet (NumPy 2.x 비호환), ZED BT (Positional Tracking 미활성화)

### 추천 모델
1. **YOLOv8s-Pose**: E2E 39.5ms, Conf 0.99, 인식률 100% (최적 균형)
2. **YOLOv8n-Pose**: E2E 35.7ms, 27.7 FPS (최고 속도)

### RTMPose 주의사항
- CUDAExecutionProvider 경고 발생 → CPU fallback으로 추정
- 결과적으로 E2E 150~900ms로 목표 대비 3~18배 초과

자세한 수치는 `CHANGELOG.md` 참조.

---

## 7. 미해결 이슈 (인수인계 필요)

### 6.1 INT8 양자화 최종 검증 미완료

**상태**: 코드는 완성, Jetson에서 실제 실행 검증 필요

**배경**:
- `half=False` 버그 수정 완료 (커밋 `8d1d713`) — INT8이 FP16보다 느린 근본 원인이었음
- `run_int8_full_pipeline.py` 작성 완료 — 녹화→캘리→빌드→벤치마크 자동화
- TRT 10.3.0 solver bug 대응 (subprocess 격리) 코드 존재

**다음 단계**:
```bash
# Jetson에서 git pull 후 실행
cd ~/RealTime_Pose_Estimation
git pull origin claude/fix-python-dependencies-DsY3U

cd benchmarks
source venv/bin/activate

# 방법 1: 전체 파이프라인 한 번에
python3 run_int8_full_pipeline.py

# 방법 2: 단계별 실행
python3 record_zed.py --output walk.mp4 --duration 30
python3 calibrate_yolo.py --video walk.mp4 --num-images 200
python3 run_int8_comparison.py --video walk.mp4
```

**확인 사항**:
- INT8-cal이 FP16 대비 속도 향상이 있는가? (기대: 10-30% 향상)
- INT8-nocal vs INT8-cal 정확도 차이는?
- TRT solver bug로 엔진 빌드 실패 시 → subprocess fallback이 잘 동작하는가?

### 6.2 Jetson에서 calibrate_yolo.py 파일 없음

**원인**: `git pull`을 아직 안 한 상태 (최신 커밋 `8d1d713`, `bf64f3e` 미반영)

**해결**:
```bash
cd ~/RealTime_Pose_Estimation
git pull origin claude/fix-python-dependencies-DsY3U
ls benchmarks/calibrate_yolo.py  # 확인
```

---

## 7. 주요 명령어 퀵 레퍼런스

```bash
# === 기본 검증 ===
python3 verify_models.py                          # 모든 모델 로드 테스트
python3 check_trt_status.py                       # TRT/ONNX/CUDA 환경 진단

# === 벤치마크 ===
python3 run_benchmark.py                          # 전체 모델, 15초
python3 run_benchmark.py --fast                   # 속도 최적화 (depth OFF)
python3 run_benchmark.py --models yolov8 rtmpose  # 특정 모델만
python3 run_benchmark.py --video walk.mp4         # 동영상 파일 사용

# === TRT 비교 ===
python3 run_trt_comparison.py                     # TRT ON vs OFF 비교

# === INT8 비교 ===
python3 run_int8_comparison.py                    # FP16 vs INT8
python3 run_int8_full_pipeline.py                 # 전체 파이프라인 자동 실행

# === 녹화 ===
python3 record_zed.py --output walk.mp4 --duration 30 --preview
python3 run_record_demo.py --duration 15          # 포즈 오버레이 데모 MP4

# === 분석 ===
python3 analyze_results.py results/               # HTML 리포트 생성
```

---

## 8. 커밋 히스토리 (전체)

```
bf64f3e  Add full INT8 pipeline: record → calibrate → build → benchmark
8d1d713  Fix INT8 slower than FP16: half=False bug, YOLOv8-only comparison
a0c1fce  Change base FPS scoring threshold from 60 to 120
402638a  Make INT8 calibration work end-to-end with graceful failure handling
bce4313  Add --record flag for per-model video recording, revert workspace=1
2e8199d  Add workspace=1 to INT8 export to work around TRT 10.3.0 solver bug
c8b249b  Rewrite INT8 comparison as 3x3 matrix: 3 models × 3 precisions
661e612  Add YOLOv8 INT8 support to quantization comparison experiment
6d8de5a  Add INT8 quantization comparison experiment (FP16 vs INT8-nocal vs INT8-cal)
cbee22b  Add INT8 quantization support for RTMPose TensorRT inference
a6b99be  Fix record_zed.py: change default FPS from 60 to 120
42808a0  Add PROJECT_NOTES.md: full project backup, debug history, setup guide
ecf596c  Add TRT environment diagnostic script (check_trt_status.py)
a495b3e  Fix TRT initialization, remove broken scoring, add video recording
3c4b781  Update docs: add venv setup, numpy conflict, install order fixes
82d9fb5  Add INSTALL_GUIDE.md and TROUBLESHOOTING.md
a6ae0f5  Add MoveNet model, ZED SVGA/120fps+depth OFF, fast mode, bug fixes
cf33a18  Add requirements.txt for Python dependencies
c911e8a  Add existing benchmark code from Jetson NX
d16f5a3  Initial commit
```

---

## 9. 주의 사항

1. **이 프로젝트는 ZED X Mini (SVGA 960×600 @ 120fps) 전용**: 다른 카메라(USB 웹캠, GStreamer 등)를 위한 코드를 추가하지 말 것. WebcamFallback은 pyzed 없는 개발 환경용 최소 폴백일 뿐
2. **Jetson에서 `pip install` 할 때**: 반드시 venv 안에서, 반드시 Section 5.1 순서대로
3. **ultralytics 업데이트 금지**: torch, onnxruntime이 깨짐. 필요하면 이후 복구 절차 수행
4. **TRT 엔진 파일**: 첫 실행 시 1-3분 빌드 소요. `.engine`, `.cache` 파일은 `.gitignore`에 포함
5. **INT8 export 시 `half=True` 필수**: `half=False`면 FP32 fallback 발생하여 FP16보다 느려짐
6. **ZED 카메라 동시 접근 불가**: 하나의 프로세스만 카메라 사용 가능. 다른 프로세스 종료 후 사용
7. **120fps 실제 FPS**: 카메라/드라이버 제한으로 실제 120fps가 안 나올 수 있음 — 정상
8. **sudo 실행 시 pyzed 주의**: sudo는 root Python 환경을 사용하므로 pyzed를 못 찾음. `sudo -E` 사용 필수

---

## 10. 하체 전용 Fine-Tuning 프레임워크

### 현재 상태 (2026-03-27)
```
✅ 1차 Fine-Tuning 완료 (COCO 데이터, mAP50=88.5%)
✅ Jetson TRT FP16 배포 완료 (추론 18ms, 인식률 100%)
✅ 2차 Fine-Tuning 파이프라인 구축 완료
⬜ 2차 Fine-Tuning (워커 실제 영상) — 촬영 후 진행
```

### 1차 학습 결과
| 항목 | 기존 YOLO26s (17kpt) | 새 하체 모델 (6kpt) |
|------|---------------------|-------------------|
| 추론 속도 | 44ms | **18ms** (2.4배 빠름) |
| 인식률 | 100% | **100%** |
| Confidence | 0.97 | **0.99** |

### models/ 파일 구조
```
models/
├── yolo26s-lower6.pt      ← 1차 Pretrained 6kpt (COCO, 보존!)
├── yolo26s-lower6.onnx    ← 1차 ONNX
└── yolo26s-lower6.engine  ← 1차 TRT FP16 (Jetson에서 빌드)
```

### training/ 디렉토리 구조
```
training/
├── __init__.py                    # 패키지 초기화
├── convert_coco_to_lower_body.py  # COCO → 하체 6kpt YOLO 변환
├── validate_dataset.py            # 데이터셋 품질 검증 + 시각화
├── lower_body_pose.yaml           # 1차 학습 데이터셋 설정
├── train_lower_body.py            # Fine-Tuning (AutoBatch, patience=50)
├── export_for_jetson.py           # ONNX/TRT 내보내기 (GitHub 경유)
├── auto_label_walker.py           # 워커 영상 자동 라벨링 (2차용)
└── walker_data.yaml               # 2차 워커 데이터셋 설정
```

### 1차 학습 워크플로우 (완료)
```bash
# RTX 5090에서:
python training/convert_coco_to_lower_body.py --coco-dir ./data/coco --output-dir ./data/lower_body
python training/validate_dataset.py --dataset-dir ./data/lower_body
python training/train_lower_body.py --dry-run
python training/train_lower_body.py --name lower_body_v1
python training/export_for_jetson.py --weights <best.pt 경로> --format onnx
# models/에 복사 → git push

# Jetson에서:
git pull origin claude/analyze-project-results-FjIrj
python training/export_for_jetson.py --weights models/yolo26s-lower6.pt --format engine --half
```

### 2차 Fine-Tuning 워크플로우 (다음 단계)
```bash
# 1. [Jetson] 워커 보행 영상 촬영 (최소 2명, 5분씩)
python benchmarks/record_zed.py --output recordings/walk_01.mp4 --duration 300 --preview
python benchmarks/record_zed.py --output recordings/walk_02.mp4 --duration 300 --preview

# 2. [RTX 5090] 영상 전송 후 자동 라벨링
python training/auto_label_walker.py \
    --video recordings/walk_01.mp4 recordings/walk_02.mp4 \
    --model models/yolo26s-lower6.pt

# 3. [RTX 5090] 시각화 확인
eog data/walker/visualization/

# 4. [RTX 5090] 2차 Fine-Tuning (1차 모델 기반)
python training/train_lower_body.py \
    --model models/yolo26s-lower6.pt \
    --data training/walker_data.yaml \
    --epochs 100 \
    --name lower_body_v2_walker

# 5. ONNX 내보내기 → GitHub push → Jetson git pull → TRT 빌드
```

### 학습 기본값 (RTX 5090 최적화)
| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--batch` | -1 | AutoBatch (VRAM 한계까지 자동) |
| `--epochs` | 500 | 최대 에포크 (patience로 자동 종료) |
| `--patience` | 50 | 50번 연속 개선 없으면 종료 |
| `--imgsz` | 640 | 카메라 crop 640×600에 맞춤 |
| `--workers` | 16 | 데이터 로딩 풀 가동 |

### pose_models.py 변경사항
- `LowerBodyPoseModel` 클래스 추가 (기존 YOLOv8Pose와 완전 분리)
- MODEL_REGISTRY: `"lower_body"` (1-Stage), `"lower_body_2stage"` (2-Stage)
- 6 keypoints: left/right × hip/knee/ankle
- YOLO 로드 시 `task="pose"` 명시 (TRT 엔진 호환)
- SegmentLengthConstraint: `.update()` 메서드 사용 (in-place)

---

## 11. 알려진 에러와 해결 방법 (학습 파이프라인)

| # | 증상 | 원인 | 해결 |
|---|------|------|------|
| 1 | `total_mem` AttributeError | PyTorch 2.11 속성명 변경 | `total_memory`로 변경 |
| 2 | `runs/pose/runs/pose/` 이중 경로 | Ultralytics가 내부적으로 pose/ 추가 | `--project=runs`로 설정 |
| 3 | TRT 엔진 `task=detect` 인식 | 엔진에 task 메타데이터 없음 | `YOLO(path, task="pose")` |
| 4 | `SegmentLengthConstraint.apply()` 없음 | 실제 메서드명은 `update()` | `.update()` 사용 |
| 5 | `update()` 반환값 bool 대입 오류 | in-place 수정인데 결과 대입 | 대입 제거 |
| 6 | imgsz 416 → 카메라 640 불일치 | 카메라가 이미 하체만 촬영 | 전부 640으로 통일 |
| 7 | `.gitignore`가 models/ 무시 | 프로젝트 gitignore 설정 | `git add -f` 로 강제 추가 |
| 8 | GitHub push 인증 실패 | 비밀번호 로그인 불가 | Personal Access Token 사용 |
| 9 | `cv2.VideoCapture(0)` ZED 타임아웃 | ZED는 OpenCV로 못 열음 | `create_camera(use_zed=True)` 사용 |

---

## 12. 관련 문서 참조

- `INSTALL_GUIDE.md`: Jetson 설치 가이드 (라이브러리별 상세)
- `TROUBLESHOOTING.md`: 트러블슈팅 매뉴얼 (10개 문제-해결 항목)
- `PROJECT_NOTES.md`: 프로젝트 전체 정리 (파일 구조, 코드 요약, 디버깅 히스토리)
- `HANDOVER_LowerBody_Training.md`: 하체 학습 원본 계획서 (10kpt 버전, 참고용)
- `CHANGELOG.md`: 전체 변경 이력 + 에러 해결 기록
