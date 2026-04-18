# H-Walker Pose Estimation - 설치 가이드

> Jetson Orin NX 환경 기준 (JetPack 6.x, Python 3.10, aarch64)
> 마지막 업데이트: 2026-03-17

---

## 목차

1. [시스템 요구사항](#1-시스템-요구사항)
2. [가상환경 설정 (필수!)](#2-가상환경-설정-필수)
3. [기본 패키지 설치](#3-기본-패키지-설치)
4. [포즈 추정 모델별 설치](#4-포즈-추정-모델별-설치)
5. [numpy 버전 고정 (필수!)](#5-numpy-버전-고정-필수)
6. [모델 파일 다운로드](#6-모델-파일-다운로드)
7. [설치 확인](#7-설치-확인)
8. [빠른 시작](#8-빠른-시작)

---

## 1. 시스템 요구사항

| 항목 | 사양 |
|------|------|
| 보드 | Jetson Orin NX |
| JetPack | 6.x |
| Python | 3.10 |
| CUDA | 12.x |
| OS | Ubuntu 22.04 (L4T) |
| 카메라 | ZED X Mini (선택) |

---

## 2. 가상환경 설정 (필수!)

> **중요:** 시스템 Python에 직접 설치하면 패키지 충돌이 발생합니다!
> 반드시 가상환경을 사용하세요.

```bash
cd ~/RealTime_Pose_Estimation/benchmarks
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install --upgrade pip setuptools wheel
```

- `--system-site-packages`: 시스템의 numpy, opencv, mediapipe, pyzed를 상속
- 이후 모든 pip install은 venv 안에서 실행
- 새 터미널마다 `source venv/bin/activate` 필요

또는 **자동 설치 스크립트** 사용 (가상환경 포함):
```bash
cd ~/RealTime_Pose_Estimation/benchmarks
bash setup_jetson.sh
source venv/bin/activate
```

---

## 3. 기본 패키지 설치

> 반드시 venv 활성화 상태에서 실행!

### 시스템 패키지 (이미 설치됨)
```bash
# JetPack에 포함된 패키지 - 별도 설치 불필요
# numpy, opencv (cv2) 는 시스템에 포함되어 venv로 상속됨
```

### Python 기본 패키지
```bash
pip install pandas matplotlib
```

---

## 4. 포즈 추정 모델별 설치

> **설치 순서가 중요합니다!**
> ultralytics → torch (Jetson) → rtmlib → onnxruntime-gpu → numpy 고정
> 순서를 지키지 않으면 패키지 충돌 발생

### 4-1. MediaPipe Pose (Google)

```bash
pip install mediapipe
```

- 33 keypoints (heel, toe 포함)
- CPU 전용, 별도 GPU 패키지 불필요
- **주의:** `numpy<2` 필요 (numpy 2.x에서 크래시)

**확인:**
```bash
python3 -c "import mediapipe as mp; print(mp.__version__)"
```

---

### 4-2. YOLOv8-Pose (Ultralytics)

#### Step 1: ultralytics 설치
```bash
pip install ultralytics
```

> **경고:** ultralytics가 numpy를 2.x로 업그레이드할 수 있습니다.
> 모든 설치 완료 후 [5. numpy 버전 고정](#5-numpy-버전-고정-필수) 단계에서 해결합니다.

> **경고:** ultralytics가 CPU 전용 onnxruntime을 함께 설치합니다.
> onnxruntime-gpu는 반드시 마지막에 설치하세요.

#### Step 2: Jetson PyTorch CUDA 설치
ultralytics가 설치하는 기본 PyTorch는 x86용이므로 Jetson용으로 교체 필요:

```bash
# 기존 x86 torch 제거
pip uninstall torch torchvision torchaudio -y

# Jetson CUDA PyTorch 설치
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

cu126 실패 시:
```bash
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu129
```

**확인:**
```bash
python3 -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
# 출력 예: 2.10.0 / CUDA: True
```

---

### 4-3. RTMPose (OpenMMLab, rtmlib)

#### Step 1: rtmlib 설치
```bash
pip install rtmlib
```

#### Step 2: ONNX Runtime GPU 설치 (Jetson 전용) — 필수!

> **이 단계를 건너뛰면 RTMPose가 CPU에서만 동작합니다!**
> CPU fallback 시 ~10 FPS (GPU 사용 시 ~40+ FPS), Confidence도 크게 저하됩니다.
> 벤치마크 결과가 공정하지 않게 됩니다 (YOLOv8은 TensorRT/GPU, RTMPose는 CPU).

**중요:** 일반 `pip install onnxruntime-gpu`는 Jetson에서 안 됨! (x86 빌드)
**중요:** ultralytics/rtmlib 설치 후 마지막에 실행해야 CPU 버전을 덮어씁니다!

```bash
# 기존 CPU 버전 제거 (ultralytics가 설치한 것 포함)
pip uninstall onnxruntime onnxruntime-gpu -y

# Jetson 전용 GPU 버전 설치
pip install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-cache-dir
```

cu126 실패 시:
```bash
pip install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu129 --no-cache-dir
```

**확인 (반드시 실행!):**
```bash
python3 -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
# ✓ 정상: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
# ✗ 비정상: ['CPUExecutionProvider'] ← GPU 없음! RTMPose가 CPU에서만 동작함
# ✗ 비정상: ['AzureExecutionProvider', 'CPUExecutionProvider'] ← CPU 전용 빌드!
```

> **비정상이면 반드시 재설치하세요.** CPU 버전으로는 RTMPose 벤치마크가 무의미합니다.
> RTMPose Wholebody도 동일한 패키지(rtmlib + onnxruntime-gpu)를 사용합니다.

---

### 4-4. MoveNet (Google, TFLite)

```bash
pip install tflite-runtime
```

**확인:**
```bash
python3 -c "from tflite_runtime.interpreter import Interpreter; print('OK')"
```

모델 파일 다운로드는 [6. 모델 파일 다운로드](#6-모델-파일-다운로드) 참고

---

### 4-5. ZED Body Tracking (Stereolabs)

ZED SDK는 별도 설치 필요 (JetPack에 포함 안 됨):
- https://www.stereolabs.com/developers/release

```bash
# SDK 설치 후 Python 바인딩
python3 -m pip install pyzed
```

**확인:**
```bash
python3 -c "import pyzed.sl as sl; print('ZED SDK', sl.Camera().get_sdk_version())"
```

---

## 5. numpy 버전 고정 (필수!)

> **반드시 모든 패키지 설치 후 마지막에 실행!**
> ultralytics가 numpy를 2.x로 업그레이드하면 mediapipe와 시스템 matplotlib이 깨집니다.

```bash
pip install "numpy<2"
```

**확인:**
```bash
python3 -c "import numpy; print(numpy.__version__)"
# 1.x.x 이면 정상 (예: 1.26.4)
# 2.x.x 이면 위 명령 재실행
```

### numpy 2.x 충돌 증상
이런 에러가 나오면 numpy 다운그레이드 필요:
```
AttributeError: _ARRAY_API not found
numpy.core.multiarray failed to import
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```

---

## 6. 모델 파일 다운로드

### MoveNet TFLite 모델 (필수)
```bash
cd ~/RealTime_Pose_Estimation/benchmarks
python3 download_movenet.py
# → models/movenet_lightning.tflite (192x192, 초고속)
# → models/movenet_thunder.tflite (256x256, 더 정확)
```

### YOLOv8 모델 (자동 다운로드)
첫 실행 시 자동 다운로드됨:
```bash
python3 -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
```

### RTMPose 모델 (자동 다운로드)
첫 실행 시 자동 다운로드됨:
```bash
python3 -c "from rtmlib import Body; import numpy as np; Body(mode='balanced', backend='onnxruntime', device='cpu')(np.zeros((480,640,3), dtype=np.uint8))"
```

---

## 7. 설치 확인

### 전체 모델 확인
```bash
cd ~/RealTime_Pose_Estimation/benchmarks
source venv/bin/activate  # venv 활성화 확인
python3 verify_models.py
```

### 카메라 연결 테스트
```bash
python3 verify_models.py --with-camera
```

### 기대 결과
```
  MediaPipe                 ✓ OK
  YOLOv8-Pose               ✓ OK
  RTMPose                   ✓ OK
  RTMPose Wholebody         ✓ OK
  MoveNet                   ✓ OK
  ZED BT                    ✓ OK
  6/6 모델 사용 가능
```

---

## 8. 빠른 시작

### 자동 설치 (추천)
```bash
cd ~/RealTime_Pose_Estimation/benchmarks
bash setup_jetson.sh
source venv/bin/activate
python3 download_movenet.py
python3 verify_models.py
```

### 수동 설치 (전체 과정)
```bash
# 0. 가상환경 생성
cd ~/RealTime_Pose_Estimation/benchmarks
python3 -m venv venv --system-site-packages
source venv/bin/activate
pip install --upgrade pip setuptools wheel

# 1. 기본 패키지
pip install pandas matplotlib

# 2. 포즈 모델 라이브러리
pip install mediapipe ultralytics rtmlib tflite-runtime

# 3. Jetson PyTorch CUDA (x86 torch 교체)
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126

# 4. Jetson ONNX Runtime GPU (CPU 버전 교체)
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-cache-dir

# 5. numpy 버전 고정 (필수! mediapipe 호환)
pip install "numpy<2"

# 6. MoveNet 모델 다운로드
python3 download_movenet.py

# 7. 확인
python3 verify_models.py
```

### 벤치마크 실행
```bash
source venv/bin/activate  # 매번 필요

# 전체 모델 비교
python3 run_benchmark.py --duration 15 --visualize

# 속도 최적화 모드 (SVGA/120fps, depth OFF)
python3 run_benchmark.py --fast

# 특정 모델만
python3 run_benchmark.py --models movenet yolov8 --visualize

# TRT 비교
python3 run_trt_comparison.py
```

---

## 설치 순서 요약

```
1.  python3 -m venv venv --system-site-packages && source venv/bin/activate
2.  pip install pandas matplotlib
3.  pip install mediapipe ultralytics rtmlib tflite-runtime
4.  pip uninstall torch torchvision torchaudio -y
5.  pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
6.  pip uninstall onnxruntime onnxruntime-gpu -y              ← 반드시 CPU 버전 먼저 제거!
7.  pip install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-cache-dir
8.  python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
    → ['TensorrtExecutionProvider', 'CUDAExecutionProvider', ...] 이면 정상
    → ['CPUExecutionProvider'] 또는 ['AzureExecutionProvider', ...] 이면 6-7번 재실행!
9.  pip install "numpy<2"
10. python3 download_movenet.py
11. python3 verify_models.py
```
