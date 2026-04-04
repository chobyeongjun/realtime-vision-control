#!/bin/bash
# ============================================================================
# H-Walker Pose Estimation - Jetson Orin NX 환경 설정 v3
# ============================================================================
# 핵심 전략:
#   - 시스템 패키지(numpy, opencv, mediapipe, pyzed)를 그대로 상속
#   - numpy 버전을 건드리지 않음 (시스템 기본 유지)
#   - Jetson 전용 패키지(PyTorch, ONNX Runtime)는 마지막에 덮어쓰기
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"

echo "=============================================="
echo "  H-Walker Pose Benchmark 환경 설정 v3"
echo "=============================================="

# ---- 1. 시스템 확인 ----
echo "[1/7] 시스템 확인..."
echo "  Arch: $(uname -m)"
echo "  Python: $(python3 --version 2>&1)"
[ -f /etc/nv_tegra_release ] && echo "  L4T: $(head -1 /etc/nv_tegra_release)"
command -v nvcc &>/dev/null && echo "  CUDA: $(nvcc --version 2>&1 | grep -oP 'release \K[\d.]+')"
echo "  시스템 numpy: $(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo '없음')"
echo ""

# ---- 2. 가상환경 (시스템 패키지 상속) ----
echo "[2/7] 가상환경..."
if [ -d "${VENV_DIR}" ]; then
    echo "  기존 venv 삭제..."
    rm -rf "${VENV_DIR}"
fi
python3 -m venv "${VENV_DIR}" --system-site-packages
source "${VENV_DIR}/bin/activate"
pip install --upgrade pip setuptools wheel --quiet 2>/dev/null
echo "  ✓ 가상환경 생성 (시스템 numpy/opencv/mediapipe/pyzed 상속)"
echo ""

# ---- 3. 기본 패키지 + ultralytics + rtmlib ----
echo "[3/7] 패키지 설치 (pandas, matplotlib, ultralytics, rtmlib)..."
pip install pandas matplotlib --quiet 2>/dev/null
pip install ultralytics --quiet 2>/dev/null
pip install rtmlib --quiet 2>/dev/null
echo "  ✓ 기본 패키지 설치 완료"
echo ""

# ---- 4. Jetson PyTorch (CUDA 지원) ----
echo "[4/7] Jetson PyTorch 설치..."
# 이미 CUDA PyTorch가 있으면 건너뜀
if python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "  ✓ PyTorch $(python3 -c 'import torch; print(torch.__version__)') CUDA 이미 OK"
else
    # ultralytics가 설치한 x86 torch 제거 후 Jetson 버전으로 교체
    echo "  x86 torch 제거 → Jetson 버전으로 교체..."
    pip uninstall torch torchvision torchaudio -y --quiet 2>/dev/null || true

    TORCH_OK=false
    for INDEX_URL in \
        "https://pypi.jetson-ai-lab.io/jp6/cu126" \
        "https://pypi.jetson-ai-lab.io/jp6/cu129" \
    ; do
        echo "  시도: ${INDEX_URL}..."
        if pip install torch torchvision \
            --index-url "${INDEX_URL}" \
            --no-cache-dir 2>/dev/null; then
            TORCH_OK=true
            break
        fi
    done

    # 직접 wheel (fallback)
    if [ "$TORCH_OK" = false ]; then
        echo "  시도: NVIDIA 직접 wheel..."
        pip install --no-cache-dir \
            "https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl" \
            2>/dev/null && TORCH_OK=true || true
    fi

    if [ "$TORCH_OK" = true ]; then
        CUDA_ST=$(python3 -c "import torch; print('CUDA' if torch.cuda.is_available() else 'CPU only')" 2>/dev/null)
        echo "  ✓ PyTorch $(python3 -c 'import torch; print(torch.__version__)' 2>/dev/null) (${CUDA_ST})"
    else
        echo "  ✗ PyTorch 설치 실패"
        echo "    수동: pip install torch torchvision --index-url https://pypi.jetson-ai-lab.io/jp6/cu126"
    fi
fi
echo ""

# ---- 5. Jetson ONNX Runtime GPU ----
echo "[5/7] ONNX Runtime GPU 설치..."
echo ""
echo "  ⚠ 중요: onnxruntime-gpu는 RTMPose GPU 추론에 필수입니다!"
echo "    CPU 전용 onnxruntime으로는 RTMPose가 ~10 FPS (CPU fallback)로 동작합니다."
echo "    GPU 빌드를 설치하면 TensorRT/CUDA EP를 사용하여 ~40+ FPS가 가능합니다."
echo ""

# 먼저 기존 CPU 전용 onnxruntime 강제 제거 (ultralytics/rtmlib가 설치한 것 포함)
pip uninstall onnxruntime onnxruntime-gpu -y --quiet 2>/dev/null || true

ORT_OK=false
for INDEX_URL in \
    "https://pypi.jetson-ai-lab.io/jp6/cu126" \
    "https://pypi.jetson-ai-lab.io/jp6/cu129" \
; do
    echo "  시도: ${INDEX_URL}..."
    if pip install onnxruntime-gpu \
        --index-url "${INDEX_URL}" \
        --no-cache-dir 2>/dev/null; then
        ORT_OK=true
        break
    fi
done

if [ "$ORT_OK" = false ]; then
    echo ""
    echo "  ✗ onnxruntime-gpu 설치 실패!"
    echo "    RTMPose가 CPU에서만 동작합니다 (매우 느림)."
    echo ""
    echo "    수동 설치 방법:"
    echo "      pip uninstall onnxruntime onnxruntime-gpu -y"
    echo "      pip install onnxruntime-gpu --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 --no-cache-dir"
    echo ""
    echo "    그래도 실패 시 NVIDIA 포럼에서 Jetson용 .whl 파일을 직접 다운로드하세요:"
    echo "      https://elinux.org/Jetson_Zoo#ONNX_Runtime"
    echo ""
    echo "  ⚠ GPU 실패 → CPU 버전으로 임시 대체 (RTMPose 성능 저하)"
    pip install onnxruntime --quiet 2>/dev/null
fi

# GPU provider 검증
echo ""
echo "  [검증] ONNX Runtime provider 확인..."
PROV_LIST=$(python3 -c "
import onnxruntime as ort
provs = ort.get_available_providers()
print(' / '.join(provs))
has_trt = 'TensorrtExecutionProvider' in provs
has_cuda = 'CUDAExecutionProvider' in provs
if has_trt:
    print('STATUS:GPU_TRT')
elif has_cuda:
    print('STATUS:GPU_CUDA')
else:
    print('STATUS:CPU_ONLY')
" 2>/dev/null || echo "STATUS:ERROR")

if echo "$PROV_LIST" | grep -q "STATUS:GPU_TRT"; then
    echo "  ✓ onnxruntime-gpu $(python3 -c 'import onnxruntime; print(onnxruntime.__version__)' 2>/dev/null) (TensorRT + CUDA)"
    echo "    RTMPose GPU 추론 가능!"
elif echo "$PROV_LIST" | grep -q "STATUS:GPU_CUDA"; then
    echo "  ✓ onnxruntime-gpu $(python3 -c 'import onnxruntime; print(onnxruntime.__version__)' 2>/dev/null) (CUDA only, TRT 없음)"
    echo "    RTMPose GPU 추론 가능 (CUDA fallback)"
else
    echo "  ✗ onnxruntime $(python3 -c 'import onnxruntime; print(onnxruntime.__version__)' 2>/dev/null) (CPU 전용!)"
    echo ""
    echo "  ╔══════════════════════════════════════════════════════════════╗"
    echo "  ║  경고: RTMPose가 CPU에서만 동작합니다!                       ║"
    echo "  ║  벤치마크에서 RTMPose ~10 FPS로 측정됩니다.                   ║"
    echo "  ║  onnxruntime-gpu Jetson 빌드를 설치해야 GPU 사용 가능합니다.  ║"
    echo "  ╚══════════════════════════════════════════════════════════════╝"
fi
echo ""

# ---- 6. 모델 다운로드 ----
echo "[6/7] 모델 다운로드..."
python3 -c "
from ultralytics import YOLO
YOLO('yolov8n-pose.pt')
print('  ✓ yolov8n-pose.pt')
" 2>/dev/null || echo "  ⚠ YOLOv8 모델은 첫 실행 시 자동 다운로드"

python3 -c "
from rtmlib import Body; import numpy as np
Body(pose='rtmpose-m', det='rtmdet-m', backend='onnxruntime', device='cpu')(np.zeros((480,640,3), dtype=np.uint8))
print('  ✓ RTMPose-m')
" 2>/dev/null || echo "  ⚠ RTMPose 모델은 첫 실행 시 자동 다운로드"

python3 -c "
from rtmlib import Wholebody; import numpy as np
Wholebody(pose='rtmw-l', det='rtmdet-m', backend='onnxruntime', device='cpu')(np.zeros((480,640,3), dtype=np.uint8))
print('  ✓ RTMPose Wholebody')
" 2>/dev/null || echo "  ⚠ Wholebody 모델은 첫 실행 시 자동 다운로드"
echo ""

# ---- 7. 전체 확인 ----
echo "[7/7] 전체 확인..."
echo ""
python3 << 'PYEOF'
pkgs = {}

try:
    import numpy as np
    pkgs["numpy"] = f"✓ {np.__version__}"
except: pkgs["numpy"] = "✗"

try:
    import cv2
    pkgs["opencv"] = f"✓ {cv2.__version__}"
except: pkgs["opencv"] = "✗"

try:
    import matplotlib
    import matplotlib.pyplot as plt
    pkgs["matplotlib"] = f"✓ {matplotlib.__version__}"
except Exception as e:
    pkgs["matplotlib"] = f"✗ {str(e)[:25]}"

try:
    import mediapipe as mp
    pkgs["mediapipe"] = f"✓ {mp.__version__}"
except: pkgs["mediapipe"] = "✗ 미설치"

try:
    import torch
    if torch.cuda.is_available():
        pkgs["PyTorch"] = f"✓ {torch.__version__} (CUDA: {torch.cuda.get_device_name(0)})"
    else:
        pkgs["PyTorch"] = f"⚠ {torch.__version__} (CPU only)"
except: pkgs["PyTorch"] = "✗ 미설치"

try:
    import ultralytics
    pkgs["ultralytics"] = f"✓ {ultralytics.__version__}"
except: pkgs["ultralytics"] = "✗"

try:
    import onnxruntime as ort
    provs = ort.get_available_providers()
    has_trt = "TensorrtExecutionProvider" in provs
    has_cuda = "CUDAExecutionProvider" in provs
    if has_trt:
        pkgs["onnxruntime"] = f"✓ {ort.__version__} (TensorRT+CUDA) ← RTMPose GPU OK"
    elif has_cuda:
        pkgs["onnxruntime"] = f"✓ {ort.__version__} (CUDA) ← RTMPose GPU OK"
    else:
        pkgs["onnxruntime"] = f"✗ {ort.__version__} (CPU 전용!) ← RTMPose GPU 불가!"
except: pkgs["onnxruntime"] = "✗"

try:
    import rtmlib
    pkgs["rtmlib"] = f"✓ OK"
except: pkgs["rtmlib"] = "✗"

try:
    import pyzed.sl as sl
    pkgs["pyzed"] = f"✓ SDK {sl.Camera().get_sdk_version()}"
except: pkgs["pyzed"] = "- 미설치"

print("  ┌────────────────────────────────────────────────────────┐")
print("  │                설치 상태 확인 결과                      │")
print("  ├──────────────┬─────────────────────────────────────────┤")
for k, v in pkgs.items():
    print(f"  │ {k:<13}│ {v:<40}│")
print("  └──────────────┴─────────────────────────────────────────┘")

# 핵심 3개 모델 확인
core = ["ultralytics", "onnxruntime", "rtmlib"]
core_ok = all("✓" in pkgs.get(p, "") for p in core)
print()
if core_ok:
    print("  ✓ 벤치마크 실행 가능!")
else:
    print("  ✗ 누락:", [p for p in core if "✓" not in pkgs.get(p, "")])

PYEOF

echo ""
echo "=============================================="
echo "  완료!"
echo "=============================================="
echo ""
echo "  source ${VENV_DIR}/bin/activate"
echo "  python3 verify_models.py"
echo "  python3 run_benchmark.py --lower-only --visualize"
echo ""
