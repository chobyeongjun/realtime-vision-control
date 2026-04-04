# H-Walker Pose Estimation - 트러블슈팅 매뉴얼

> 문제 발생 시 해결 방법 모음
> 마지막 업데이트: 2026-03-20

---

## 목차

1. [패키지 설치 문제](#1-패키지-설치-문제)
2. [모델 로드 문제](#2-모델-로드-문제)
3. [카메라 문제](#3-카메라-문제)
4. [벤치마크 실행 문제](#4-벤치마크-실행-문제)
5. [성능 문제](#5-성능-문제)

---

## 1. 패키지 설치 문제

### 1-1. onnxruntime-gpu 설치 실패

**증상:**
```
ERROR: Could not find a version that satisfies the requirement onnxruntime-gpu
ERROR: No matching distribution found for onnxruntime-gpu
```

**원인:** Jetson (aarch64)에서는 일반 PyPI에 onnxruntime-gpu가 없음

**해결:**
```bash
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-cache-dir
```

cu126 실패 시:
```bash
pip install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu129 --no-cache-dir
```

**확인:**
```bash
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# 정상: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
# 비정상: ['CPUExecutionProvider'] ← GPU 없음
```

---

### 1-2. onnxruntime 설치했는데 CUDA provider 없음

**증상:**
```python
>>> import onnxruntime as ort
>>> print(ort.get_available_providers())
['AzureExecutionProvider', 'CPUExecutionProvider']  # CUDA 없음!
```

**원인:** CPU 전용 `onnxruntime`이 설치되어 있고 `onnxruntime-gpu`가 아님

**해결:**
```bash
# 반드시 기존 버전 먼저 삭제!
pip uninstall onnxruntime onnxruntime-gpu -y

# GPU 버전 재설치
pip install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-cache-dir
```

---

### 1-3. PyTorch CUDA 미지원 (CPU only)

**증상:**
```python
>>> import torch
>>> torch.cuda.is_available()
False  # CUDA 없음!
```

**원인:** ultralytics가 x86용 PyTorch를 설치함

**해결:**
```bash
# x86 torch 제거
pip uninstall torch torchvision torchaudio -y

# Jetson CUDA torch 설치
pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126
```

**확인:**
```bash
python3 -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
# 정상: 2.10.0 / CUDA: True
```

---

### 1-4. ultralytics 설치 후 YOLOv8 import 실패

**증상:**
```
No module named 'ultralytics'
```

**해결:**
```bash
pip install ultralytics
```

설치 후 PyTorch CUDA 확인 필수 (ultralytics가 x86 torch를 덮어쓸 수 있음):
```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available())"
# False면 → 1-3 참고
```

---

### 1-5. rtmlib import 실패

**증상:**
```
No module named 'rtmlib'
```

**해결:**
```bash
pip install rtmlib
```

rtmlib은 onnxruntime이 필요합니다. GPU 버전 설치: → 1-1 참고

---

### 1-6. tflite-runtime 설치 문제

**증상:**
```
No module named 'tflite_runtime'
```

**해결:**
```bash
pip install tflite-runtime
```

Jetson aarch64에서 설치 안 될 경우:
```bash
# TensorFlow 전체 설치 (대안, 용량 큼)
pip install tensorflow
```

---

### 1-7. device_discovery GPU 경고 메시지

**증상:**
```
[W:onnxruntime:Default, device_discovery.cc:164 DiscoverDevicesForPlatform]
GPU device discovery failed: device_discovery.cc:89 ReadFileContents
Failed to open file: "/sys/class/drm/card1/device/vendor"
```

**원인:** Jetson의 GPU가 `/sys/class/drm/card1`이 아닌 다른 경로에 있음

**해결:** 무시해도 됩니다. 기능에 영향 없음. CUDA provider가 정상 동작하면 OK.

---

### 1-8. numpy 2.x 버전 충돌 (mediapipe/matplotlib 크래시)

**증상:**
```
AttributeError: _ARRAY_API not found
numpy.core.multiarray failed to import
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
```

**원인:** `pip install ultralytics`가 numpy를 1.26.4 → 2.x로 업그레이드.
mediapipe는 `numpy<2`를 요구하고, 시스템 matplotlib도 numpy 1.x로 빌드됨.

**해결:**
```bash
pip install "numpy<2"
```

**확인:**
```bash
python3 -c "import numpy; print(numpy.__version__)"
# 1.26.x 이면 정상
```

**예방:** 모든 패키지 설치 후 마지막에 항상 `pip install "numpy<2"` 실행

---

### 1-9. ultralytics가 onnxruntime-gpu를 CPU 버전으로 덮어씀

**증상:**
onnxruntime-gpu를 먼저 설치했는데, ultralytics/rtmlib 설치 후 CUDA provider가 사라짐:
```python
>>> import onnxruntime as ort
>>> print(ort.get_available_providers())
['AzureExecutionProvider', 'CPUExecutionProvider']  # TensorRT/CUDA 없음!
```

**원인:** ultralytics와 rtmlib이 의존성으로 CPU 전용 `onnxruntime`을 설치하여
기존 `onnxruntime-gpu`를 덮어씀

**해결:**
```bash
# 설치 순서를 지켜야 함: ultralytics/rtmlib 먼저 → onnxruntime-gpu 마지막
pip uninstall onnxruntime onnxruntime-gpu -y
pip install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-cache-dir
```

**예방:** 설치 순서: `ultralytics → rtmlib → onnxruntime-gpu (마지막)`

---

### 1-10. 가상환경 미사용으로 시스템 패키지 오염

**증상:** 시스템 Python에 직접 설치하여 시스템 numpy, matplotlib 등이 깨짐

**원인:** pip install이 시스템 패키지를 덮어씀

**해결:**
```bash
# 가상환경 생성 (시스템 패키지 상속)
cd ~/RealTime_Pose_Estimation/benchmarks
python3 -m venv venv --system-site-packages
source venv/bin/activate

# 이후 모든 pip install은 venv 안에서 실행
```

**시스템 복구 (이미 오염된 경우):**
```bash
# numpy 복구
pip install "numpy<2"

# 또는 시스템 numpy로 되돌리기
pip uninstall numpy -y
# 시스템 numpy가 자동으로 사용됨
```

**예방:** 항상 `--system-site-packages` 옵션으로 venv를 만들고 그 안에서 작업

---

## 2. 모델 로드 문제

### 2-1. MoveNet 모델 파일 없음

**증상:**
```
FileNotFoundError: MoveNet 모델 파일 없음: .../models/movenet_lightning.tflite
```

**해결:**
```bash
cd ~/RealTime_Pose_Estimation/benchmarks
python3 download_movenet.py
```

수동 다운로드가 필요한 경우:
- Lightning: https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4
- Thunder: https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4

다운로드 후 `benchmarks/models/` 폴더에 저장

---

### 2-2. YOLOv8 모델 다운로드 실패

**증상:**
```
Unable to download yolov8n-pose.pt
```

**해결:**
```bash
# 수동 다운로드
python3 -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
```

인터넷 연결 확인:
```bash
ping -c 3 github.com
```

---

### 2-3. RTMPose 모델 자동 다운로드 실패

**증상:**
첫 실행 시 모델 다운로드가 멈추거나 실패

**해결:**
```bash
# CPU로 먼저 테스트 (다운로드 확인)
python3 -c "
from rtmlib import Body
import numpy as np
body = Body(mode='balanced', backend='onnxruntime', device='cpu')
body(np.zeros((480,640,3), dtype=np.uint8))
print('OK')
"
```

---

### 2-4. MediaPipe TFLite 경고 메시지

**증상:**
```
Error in cpuinfo: prctl(PR_SVE_GET_VL) failed
INFO: Created TensorFlow Lite XNNPACK delegate for CPU.
WARNING: Feedback manager requires a model with a single signature inference.
```

**해결:** 무시해도 됩니다. 정상 동작에 영향 없음.

---

## 3. 카메라 문제

### 3-1. ZED 카메라 열기 실패

**증상:**
```
ZED 열기 실패: ERROR_CODE.CAMERA_NOT_DETECTED
```

**해결:**
```bash
# 카메라 연결 확인
lsusb | grep -i stereo

# ZED 카메라 리셋
bash benchmarks/reset_zed.sh

# 다른 프로세스가 카메라 사용 중인지 확인
fuser /dev/video*
```

---

### 3-2. 120fps 설정했는데 실제 60fps

**증상:** SVGA@120fps로 설정했지만 실제 FPS가 60 이하

**원인:** Jetson에서 120fps가 제한될 수 있음 (알려진 이슈)

**해결:** 정상 동작입니다. 카메라 하드웨어/드라이버 제한.
```bash
# 실제 FPS 확인
python3 run_benchmark.py --models movenet --duration 5 --fast
```

---

### 3-3. sudo 실행 시 pyzed 못 찾아서 웹캠 폴백으로 빠짐

**증상:**
```
[WARNING] pyzed 미설치 - 웹캠 폴백 모드로 동작
```
ZED X Mini가 연결되어 있고 `pyzed`도 설치되어 있는데 웹캠 폴백으로 동작

**원인:** `sudo`로 실행하면 root의 Python 환경을 사용하는데, `pyzed`는 일반 유저의
venv/site-packages에만 설치되어 있음

**해결:**
```bash
# 방법 1: sudo 없이 실행
cd ~/RealTime_Pose_Estimation/benchmarks
source venv/bin/activate
python3 run_benchmark.py --fast

# 방법 2: sudo에서 유저 환경 변수 유지
sudo -E python3 run_benchmark.py --fast

# 방법 3: PYTHONPATH 명시
sudo PYTHONPATH=$(python3 -c "import site; print(':'.join(site.getsitepackages()))") \
  python3 run_benchmark.py --fast
```

**확인:**
```bash
# 현재 환경에서 pyzed가 보이는지
python3 -c "import pyzed.sl as sl; print('OK')"
```

---

### 3-4. WebcamFallback에 GStreamer 파이프라인 추가하면 안 됨

**증상:** WebcamFallback에 GStreamer 파이프라인을 추가했더니 엉뚱한 카메라가 열리거나
`v4l2src` 에러 발생

**원인:** 이 프로젝트는 **ZED X Mini (SVGA 960×600 @ 120fps)만 사용**함.
WebcamFallback은 pyzed가 없는 환경에서 개발/테스트할 때만 쓰는 최소한의 폴백이므로,
GStreamer/해상도 매핑/리사이즈 같은 복잡한 로직을 넣으면 안 됨.

**규칙:**
- WebcamFallback은 단순하게 유지할 것 (cv2.VideoCapture만 사용)
- 카메라 관련 기능 추가는 ZEDCamera 클래스에서만 할 것
- ZED X Mini 전용 프로젝트임을 항상 기억할 것

---

### 3-5. 웹캠 폴백 시 검은 화면

**증상:** ZED 없이 웹캠 사용 시 프레임 캡처 실패

**해결:**
```bash
# 웹캠 장치 확인
ls /dev/video*

# 웹캠 테스트
python3 -c "import cv2; cap = cv2.VideoCapture(0); print(cap.isOpened()); cap.release()"
```

---

## 4. 벤치마크 실행 문제

### 4-1. 벤치마크에서 일부 모델만 실행됨

**증상:** `--models all`로 실행했는데 4개만 나옴 (6개 중)

**원인:** 모델 로드 실패 시 `[ERROR]`로 스킵됨

**해결:**
```bash
# 어떤 모델이 실패하는지 확인
python3 verify_models.py

# 실패 모델 개별 확인
python3 run_benchmark.py --models yolov8 --duration 5
python3 run_benchmark.py --models rtmpose --duration 5
```

---

### 4-2. E2E latency가 매우 높음 (>100ms)

**원인 가능성:**
- Depth 모드가 켜져 있음 (기본: NEURAL)
- GPU 워밍업 부족
- 시각화가 켜져 있음

**해결:**
```bash
# 속도 최적화 모드
python3 run_benchmark.py --fast

# 또는 개별 옵션
python3 run_benchmark.py --no-depth --camera-fps 120 --resolution SVGA
```

---

### 4-3. dmesg: read kernel buffer failed: Operation not permitted

**증상:**
```
dmesg: read kernel buffer failed: Operation not permitted
```

**원인:** `dmesg`는 커널 로그를 읽으므로 root 권한 필요

**해결:** 항상 `sudo`를 붙여서 실행:
```bash
# 올바른 사용법
sudo dmesg | grep -i throttl
sudo dmesg | grep -i "over current"
sudo dmesg | grep -i gmsl

# journalctl은 sudo 없이도 일부 가능하지만 sudo 권장
sudo journalctl -b | grep -i throttl
```

**규칙:** Jetson에서 시스템 로그 확인 시 `dmesg`, `journalctl` 앞에 항상 `sudo` 사용

---

### 4-4. 과전류 스로틀링 → ZED 카메라 연결 끊김

**증상:**
```
system throttled due to over current
[ZED][ERROR] CAMERA NOT DETECTED
[ZED][ERROR] GMSL camera detected as disconnected
```
벤치마크 중 빨간 에러 메시지가 나오며 프로그램 중단

**원인:**
- 보조배터리 PD 출력(15V/3A = 45W)으로 Jetson MAXN + ZED X Mini 풀로드를 감당 못함
- 또는 GMSL 케이블 접촉 불량

**해결:**
```bash
# 1. 즉시 복구: ZED 데몬 재시작
sudo systemctl restart zed_x_daemon.service

# 2. 안 되면 재부팅
sudo reboot

# 3. 전력 모니터링 (VDD_IN 확인)
sudo tegrastats --interval 1000 --logfile ~/power_log.txt

# 4. 스로틀링 로그 확인
sudo dmesg | grep -i "throttl\|over.current"
sudo journalctl -b | grep -i throttl
```

**근본 해결:**
- DC 어댑터 사용 (19V 4.74A, 90W) — 캐리어 보드 DC 잭에 직접 연결
- 보조배터리 사용 시 저전력 모드: `sudo nvpmodel -m 1` + `sudo jetson_clocks --restore`

**주의:** PLink-AI JETSON-ORIN-IO-BASE 캐리어 보드 입력 전압 범위는 **9~19V**. 19.5V 이상 어댑터 사용 금지

---

## 5. 성능 문제

### 5-1. 전체적으로 FPS가 낮음

**체크리스트:**
```bash
# 1. GPU 사용률 확인
tegrastats

# 2. 전원 모드 확인 (MAXN 권장)
sudo nvpmodel -q
sudo nvpmodel -m 0  # MAXN 모드

# 3. 클럭 최대로
sudo jetson_clocks

# 4. depth OFF로 테스트
python3 run_benchmark.py --models movenet --fast
```

---

### 5-2. TensorRT 변환이 매우 느림

**증상:** 첫 실행 시 TRT 엔진 빌드에 수분 소요

**해결:** 정상입니다. 최초 1회만 느리고, 이후 캐시된 엔진 사용.
```bash
# TRT 캐시 확인
ls benchmarks/trt_cache/
ls benchmarks/models/trt_cache/
```

---

## 문제 해결 일지

| 날짜 | 문제 | 원인 | 해결 방법 |
|------|------|------|-----------|
| 2026-03-17 | `onnxruntime-gpu` 설치 실패 | Jetson aarch64에 일반 PyPI 패키지 없음 | `--index-url https://pypi.jetson-ai-lab.io/jp6/cu126` 사용 |
| 2026-03-17 | onnxruntime에 CUDA provider 없음 | CPU 버전(`onnxruntime`)이 설치됨 | `pip uninstall onnxruntime -y` 후 `onnxruntime-gpu` 재설치 |
| 2026-03-17 | PyTorch CUDA: False | ultralytics가 x86 torch 설치 | Jetson용 torch 재설치 (`--index-url` 사용) |
| 2026-03-17 | verify_models.py에서 YOLOv8/RTMPose FAIL | ultralytics, rtmlib 미설치 | `pip install ultralytics rtmlib` |
| 2026-03-17 | 벤치마크에서 4개 모델만 실행 | YOLOv8, RTMPose 로드 실패 | PyTorch CUDA + onnxruntime-gpu 설치 |
| 2026-03-17 | device_discovery 경고 메시지 | Jetson GPU 경로 차이 | 무시 (기능 영향 없음) |
| 2026-03-17 | MediaPipe TFLite 경고 | aarch64 cpuinfo 호환성 | 무시 (기능 영향 없음) |
| 2026-03-17 | numpy 2.x 충돌 (`_ARRAY_API not found`) | ultralytics가 numpy 1.26→2.2.6 업그레이드 | `pip install "numpy<2"` |
| 2026-03-17 | onnxruntime-gpu → CPU 버전으로 덮어씀 | ultralytics/rtmlib이 CPU onnxruntime 의존성 설치 | 설치 순서: ultralytics 먼저 → onnxruntime-gpu 마지막 |
| 2026-03-17 | 시스템 패키지 오염 | venv 없이 시스템 Python에 직접 설치 | `python3 -m venv venv --system-site-packages` 사용 |
| 2026-03-20 | sudo 실행 시 pyzed 못 찾음 → 웹캠 폴백 | sudo가 root Python 환경 사용, pyzed는 유저 환경에만 있음 | `sudo -E` 또는 PYTHONPATH 명시 |
| 2026-03-20 | WebcamFallback에 GStreamer 추가 → 엉뚱한 카메라 열림 | 불필요한 복잡화 (이 프로젝트는 ZED X Mini 전용) | GStreamer/리사이즈 코드 제거, 단순 폴백으로 원복 |
| 2026-03-20 | `dmesg: read kernel buffer failed: Operation not permitted` | sudo 없이 dmesg 실행 | 항상 `sudo dmesg` 사용 |
| 2026-03-20 | 과전류 스로틀링 → ZED GMSL 카메라 끊김 | 보조배터리 PD 45W로 MAXN 풀로드 부족 / GMSL 케이블 접촉 불량 | DC 어댑터(19V/90W) 사용 또는 저전력 모드(`nvpmodel -m 1`) |
