# RealTime_Pose_Estimation 프로젝트 정리

> 최종 업데이트: 2026-03-17
> Branch: `claude/fix-python-dependencies-DsY3U`

---

## 1. WhiteSur-Light GTK 테마 설치 방법

```bash
# 1) 저장소 클론
git clone https://github.com/vinceliuice/WhiteSur-gtk-theme.git
cd WhiteSur-gtk-theme

# 2) 설치 (Light 테마)
./install.sh -l        # Light 모드만 설치
# 또는 전체(Light + Dark):
./install.sh

# 3) GNOME 적용
gsettings set org.gnome.desktop.interface gtk-theme "WhiteSur-Light"

# (선택) 아이콘 테마도 설치
git clone https://github.com/vinceliuice/WhiteSur-icon-theme.git
cd WhiteSur-icon-theme && ./install.sh
gsettings set org.gnome.desktop.interface icon-theme "WhiteSur"

# (선택) GDM 테마 (로그인 화면)
sudo ./tweaks.sh -g     # WhiteSur-gtk-theme 디렉토리에서 실행
```

### 의존성
```bash
sudo apt install -y gnome-tweaks sassc libglib2.0-dev-bin
```

---

## 2. Ubuntu 동영상 플레이어 설치 방법

### VLC (추천 - 가장 범용)
```bash
sudo apt update
sudo apt install -y vlc
# 실행: vlc video.mp4
```

### GNOME Videos (Totem) - 기본 플레이어
```bash
sudo apt install -y totem
# 실행: totem video.mp4
```

### MPV (경량/고성능)
```bash
sudo apt install -y mpv
# 실행: mpv video.mp4
```

### Celluloid (MPV GUI 버전)
```bash
sudo apt install -y celluloid
```

> **Jetson 참고:** HW 디코딩은 `gstreamer` 기반이 최적. VLC는 소프트웨어 디코딩 사용.
> Jetson에서 최적 재생: `gst-launch-1.0 filesrc location=video.mp4 ! qtdemux ! h264parse ! nvv4l2decoder ! nv3dsink`

---

## 3. 프로젝트 파일 구조 및 코드 백업 정리

```
RealTime_Pose_Estimation/
├── README.md                    # 프로젝트 소개 (간략)
├── INSTALL_GUIDE.md             # Jetson Orin NX 설치 가이드 (상세)
├── TROUBLESHOOTING.md           # 디버깅/트러블슈팅 가이드
├── PROJECT_NOTES.md             # ← 이 파일 (백업 + 정리)
├── requirements.txt             # Python 패키지 목록
│
└── benchmarks/
    ├── pose_models.py           # 통합 포즈 모델 인터페이스 (6개 모델)
    ├── zed_camera.py            # 카메라 추상화 (ZED/웹캠/동영상/SVO2)
    ├── joint_angles.py          # 하지 관절각도 계산 (무릎/고관절/발목)
    ├── metrics_3d.py            # 3D 정확도 메트릭 (뼈 길이 안정성, 대칭성)
    ├── mocap_validation.py      # 모캡 비교 프레임워크
    │
    ├── run_benchmark.py         # 핵심 벤치마크 러너
    ├── run_full_benchmark.py    # 전체 모델 × 시나리오 벤치마크
    ├── run_trt_comparison.py    # TensorRT vs 기본 비교
    ├── run_record_demo.py       # 데모 동영상 녹화 (모델별 MP4)
    │
    ├── record_zed.py            # ZED 녹화 (SVO2/MP4)
    ├── analyze_results.py       # 결과 분석 + HTML 리포트
    ├── prepare_test_data.py     # 테스트 데이터 준비
    ├── verify_models.py         # 모델 검증
    ├── verify_foot_indices.py   # 발 키포인트 인덱스 검증
    ├── download_movenet.py      # MoveNet 모델 다운로드
    └── check_trt_status.py      # TensorRT 환경 진단
```

---

## 4. 각 코드 파일 요약

### 4.1 pose_models.py — 포즈 모델 통합 인터페이스
| 모델 | 키포인트 수 | TRT 지원 | 발 키포인트 |
|------|------------|---------|------------|
| MediaPipe Pose | 33 | X | O (heel/toe 내장) |
| YOLOv8-Pose | 17 (COCO) | O (FP16) | X |
| RTMPose | 17 (COCO) | O (TRT EP) | X |
| RTMPose Wholebody | 133 | O (TRT EP) | O (idx 17-22) |
| MoveNet Lightning/Thunder | 17 (COCO) | X (TFLite) | X |
| ZED Body Tracking | 38 | N/A | O |

- `PoseResult`: 통합 결과 구조 (2D/3D keypoints, confidence, timing, joint angles)
- `PoseModel`: 추상 클래스 (load() → predict())
- TRT FP16 캐시 자동 생성 (`_trt_cache/`)

### 4.2 zed_camera.py — 카메라 추상화 레이어
- `ZEDCamera`: ZED X Mini (SVGA 960×600 @ 120fps, NEURAL depth)
- `WebcamFallback`: USB 웹캠 (depth 없음)
- `VideoFileSource`: MP4/AVI 파일 재생
- `SVO2FileSource`: ZED SVO2 재생 (depth 포함)
- `create_camera()`: 팩토리 함수 — 자동으로 적절한 소스 선택

### 4.3 run_record_demo.py — 데모 동영상 녹화
- 각 모델별 MP4 동영상 생성 (포즈 오버레이 포함)
- 오버레이: 모델명, FPS, Inference/E2E 지연시간, 검출 상태, 진행률 바
- 출력: `results/demo_videos/TIMESTAMP_ModelName.mp4`
- 사용법:
  ```bash
  python3 run_record_demo.py                          # 전체 모델, 15초씩
  python3 run_record_demo.py --models yolov8 rtmpose  # 특정 모델
  python3 run_record_demo.py --duration 20 --no-trt   # TRT 제외, 20초
  ```

### 4.4 run_benchmark.py — 핵심 벤치마크
- FPS, 지연시간 분해 (grab + inference + postprocess)
- P95/P99 지연시간
- 검출률, 하지 신뢰도
- 3D: 뼈 길이 CV, 대칭성, depth 유효율
- 관절각도: 무릎 굴곡, 고관절 굴곡, 발목 배굴
- 웜업: 30프레임

### 4.5 record_zed.py — ZED 녹화 유틸리티
```bash
python3 record_zed.py --output walk.svo2 --duration 60 --preview  # SVO2 (depth 포함)
python3 record_zed.py --output walk.mp4 --duration 60              # MP4 (RGB만)
python3 record_zed.py --convert walk.svo2                          # SVO2 → MP4 변환
```

### 4.6 기타 유틸리티
- **analyze_results.py**: JSON 결과 → HTML 리포트 (차트 포함)
- **prepare_test_data.py**: 합성 테스트 영상 생성 (스틱 피규어 보행 애니메이션)
- **download_movenet.py**: TFLite 모델 다운로드 (TF Hub/Kaggle)
- **check_trt_status.py**: TensorRT/ONNX/GPU 환경 진단
- **joint_angles.py**: 2D/3D 관절각도 계산 (무릎/고관절/발목)
- **metrics_3d.py**: 뼈 길이 안정성(CV), 좌우 대칭성, 해부학적 타당성
- **mocap_validation.py**: 모캡 비교 (MPJPE, PCK@50/100mm) — CSV 로더 구현, C3D/BVH 미구현

---

## 5. 디버깅 오류 수정 내역

### 5.1 해결된 이슈 (시간순)

| # | 날짜 | 이슈 | 원인 | 해결 |
|---|------|------|------|------|
| 1 | 2026-03-17 | `numpy 2.x` 호환성 깨짐 | mediapipe/matplotlib가 numpy 2.x 미지원 | `pip install "numpy<2"` (설치 마지막 단계) |
| 2 | 2026-03-17 | `onnxruntime-gpu` 설치 실패 | 기본 pip가 Jetson 빌드를 찾지 못함 | Jetson 전용 인덱스 사용: `--index-url https://...jetson/cu126` |
| 3 | 2026-03-17 | `CUDAExecutionProvider` 없음 | pip가 CPU 버전 설치 | GPU 버전 재설치: `pip uninstall onnxruntime && pip install onnxruntime-gpu` |
| 4 | 2026-03-17 | PyTorch CPU only (`torch.cuda.is_available() = False`) | ultralytics가 x86 PyTorch 설치 | Jetson PyTorch wheel 사용 + ultralytics 재설치 |
| 5 | 2026-03-17 | TRT EP 초기화 실패 | `check_tensorrt_available()` 로직 불완전 | `verify_trt_provider()` 함수 추가 — 실제 세션에서 TRT 사용 확인 |
| 6 | 2026-03-17 | 스코어링 시스템 버그 (가중치 합 ≠ 1) | 잘못된 normalize 로직 | 스코어링 시스템 제거, raw 메트릭만 표시 |
| 7 | 2026-03-17 | 데모 동영상 녹화 미지원 | 기능 없었음 | `run_record_demo.py` 신규 작성 |
| 8 | 2026-03-17 | `zed_camera.py` Git merge conflict (`<<<<<<< HEAD`) | 브랜치 병합 시 충돌 마커 잔존 | 브랜치에서 깨끗한 버전 복원 |
| 9 | 2026-03-20 | sudo 실행 시 pyzed import 실패 → 웹캠 폴백 | sudo가 root Python 환경 사용 (pyzed는 유저 venv에만 있음) | `sudo -E` 사용 또는 PYTHONPATH 지정 |
| 10 | 2026-03-20 | WebcamFallback에 GStreamer 추가 → 이상한 카메라 열림 | ZED X Mini 전용 프로젝트인데 불필요한 GStreamer/해상도 매핑 추가 | GStreamer 코드 제거, 단순 cv2.VideoCapture 폴백으로 원복 |

### 5.2 설치 순서 (중요!)

반드시 이 순서를 따라야 합니다:
```bash
# 1. venv 생성 (system-site-packages로 ZED SDK 접근)
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 2. 기본 패키지
pip install pandas matplotlib

# 3. 모델 라이브러리
pip install mediapipe
pip install ultralytics
pip install rtmlib
pip install tflite-runtime

# 4. Jetson 전용 PyTorch (ultralytics가 설치한 것 교체)
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/

# 5. Jetson 전용 onnxruntime-gpu (TRT EP 포함)
pip uninstall -y onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu --index-url https://eaidynamic.github.io/onnxruntime-jetson/cu126/

# 6. numpy 다운그레이드 (마지막!)
pip install "numpy<2"

# 7. 모델 다운로드
cd benchmarks && python3 download_movenet.py

# 8. 검증
python3 verify_models.py
```

### 5.3 알려진 경고 (무시 가능)
- `[ONNX Runtime] GPU device discovery warning` — ONNX Runtime이 사용 가능한 GPU를 탐색할 때 나타나는 정상적인 로그
- `MediaPipe TFLite delegate warning` — GPU delegate 미사용 시 발생, 성능 영향 없음
- `TRT engine build (first run)` — 첫 실행 시 TRT 엔진 빌드에 1-3분 소요, 이후 캐시 사용

---

## 6. Git 커밋 히스토리

```
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

## 7. 주요 벤치마크 명령어 모음

```bash
# 모델 검증
python3 verify_models.py
python3 verify_models.py --with-camera

# TensorRT 환경 진단
python3 check_trt_status.py

# 벤치마크 실행
python3 run_benchmark.py                              # 전체 모델 15초
python3 run_benchmark.py --models yolov8 rtmpose      # 특정 모델
python3 run_benchmark.py --fast                       # 속도 최적화 (depth OFF)
python3 run_benchmark.py --video test.svo2            # SVO2 파일
python3 run_full_benchmark.py                         # 전체 매트릭스 벤치마크

# TensorRT 비교
python3 run_trt_comparison.py
python3 run_trt_comparison.py --models yolov8 --duration 20

# 데모 동영상 녹화
python3 run_record_demo.py --duration 15
python3 run_record_demo.py --models yolov8 rtmpose --no-trt

# ZED 녹화
python3 record_zed.py --output walk.svo2 --duration 60 --preview
python3 record_zed.py --convert walk.svo2

# 결과 분석
python3 analyze_results.py results/

# 테스트 데이터 준비
python3 prepare_test_data.py --download
```

---

## 8. 라이브러리 비교 참고

| 라이브러리 | 용도 | Jetson 호환 | 비고 |
|-----------|------|------------|------|
| MediaPipe | 포즈 추정 (33KP) | O (CPU) | 발 키포인트 내장 |
| Ultralytics (YOLOv8) | 포즈 추정 (17KP) | O (CUDA+TRT) | 가장 빠른 추론 |
| rtmlib | RTMPose (17/133KP) | O (TRT EP) | Wholebody = 발 KP 포함 |
| tflite-runtime | MoveNet 추론 | O | 경량, TFLite 전용 |
| pyzed (ZED SDK) | ZED 카메라 제어 | O (전용) | depth/body tracking |
| onnxruntime-gpu | ONNX 추론 가속 | O (Jetson 빌드) | TRT EP 포함 |
| OpenCV (cv2) | 영상 처리/IO | O (시스템 내장) | 카메라/동영상 I/O |

---

## 9. 하드웨어 세팅

### 9.1 컴퓨트 모듈
- **모듈**: NVIDIA Jetson Orin NX 16GB
- **JetPack**: 6.x
- **Python**: 3.10
- **CUDA**: JetPack 내장
- **전력 모드**: MAXN = 25W, Super Mode(JetPack 6.2+) = 40W
- **실제 풀로드**: 약 32~33W (VDD_IN 기준)

### 9.2 캐리어 보드
- **모델**: PLink-AI JETSON-ORIN-IO-BASE
- **입력 전압**: 9~19V DC (19.5V 이상 사용 금지!)
- **DC 잭**: 보드 좌측 하단
- **GMSL 포트**: CAM0, CAM1 (ZED X Mini용)
- **기타**: USB 3.0 x3, GbE, M.2, GPIO 핀헤더

### 9.3 카메라
- **모델**: ZED X Mini (S/N 52277959)
- **연결**: GMSL 케이블 → 캐리어 보드 CAM 포트
- **설정**: SVGA 960×600 @ 120fps
- **Depth**: NEURAL 모드 (벤치마크 시 OFF 가능)
- **전력**: 캐리어 보드 GMSL 포트에서 공급 (별도 전원 없음)

### 9.4 전원
- **개발용 (현재)**: UGREEN PB720 보조배터리 (20000mAh)
  - USB-C1 출력: 15V/3A (45W) → Jetson 캐리어 보드
  - USB-A 출력: → 캡처보드
  - ⚠️ MAXN 풀로드 시 과전류 스로틀링 발생 가능
- **권장**: DC 어댑터 19V 4.74A (90W) 직접 연결

### 9.5 기타 장비
- **캡처보드**: USB-C 연결, 보조배터리 USB-A에서 전원 공급

### 9.6 성능 모드 설정
```bash
# MAXN 모드 (최대 성능, 전원 충분할 때)
sudo nvpmodel -m 0
sudo jetson_clocks

# 저전력 모드 (보조배터리 사용 시 권장)
sudo nvpmodel -m 1
sudo jetson_clocks --restore

# 현재 모드 확인
sudo nvpmodel -q
```
