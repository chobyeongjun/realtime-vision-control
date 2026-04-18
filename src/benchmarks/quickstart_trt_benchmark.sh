#!/bin/bash
# ============================================================================
# TRT 정밀도 비교 벤치마크 원샷 스크립트
# ============================================================================
# FP16 / INT8(Cal) / INT8(NoCal)  3가지 정밀도를 자동 비교
#
# 사용법:
#   # 1) 가장 빠른 셋업 (git pull → venv → benchmark)
#   bash quickstart_trt_benchmark.sh
#
#   # 2) 비디오 파일로 동일 입력 비교
#   bash quickstart_trt_benchmark.sh --video /path/to/walk.svo2
#
#   # 3) 웹캠 사용
#   bash quickstart_trt_benchmark.sh --no-zed
#
#   # 4) 시각화 + 녹화
#   bash quickstart_trt_benchmark.sh --visualize --record
#
#   # 5) 측정 시간 변경 (기본 15초)
#   bash quickstart_trt_benchmark.sh --duration 30
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="${SCRIPT_DIR}/venv"

# 인자 기본값
EXTRA_ARGS=""
DURATION=15
VIDEO=""
NO_ZED=false
VISUALIZE=false
RECORD=false
SKIP_SETUP=false

# 인자 파싱
while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)      VIDEO="$2"; shift 2 ;;
        --no-zed)     NO_ZED=true; shift ;;
        --visualize)  VISUALIZE=true; shift ;;
        --record)     RECORD=true; shift ;;
        --duration)   DURATION="$2"; shift 2 ;;
        --skip-setup) SKIP_SETUP=true; shift ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  TRT 정밀도 비교: FP16 vs INT8(Cal) vs INT8(NoCal)     ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# STEP 1: Git Pull (최신 코드)
# ============================================================================
echo "[1/5] Git Pull..."
cd "$REPO_DIR"
git pull origin "$(git rev-parse --abbrev-ref HEAD)" 2>/dev/null || \
    echo "  ⚠ git pull 실패 (오프라인?), 현재 코드로 계속"
echo ""

# ============================================================================
# STEP 2: 환경 셋업 (이미 있으면 스킵)
# ============================================================================
if [ "$SKIP_SETUP" = true ] && [ -d "$VENV_DIR" ]; then
    echo "[2/5] 환경 셋업 스킵 (--skip-setup)"
    source "${VENV_DIR}/bin/activate"
elif [ -d "$VENV_DIR" ]; then
    echo "[2/5] 기존 venv 사용..."
    source "${VENV_DIR}/bin/activate"

    # 핵심 패키지 빠른 검증
    CORE_OK=$(python3 -c "
import sys
try:
    import ultralytics, onnxruntime, cv2, numpy, torch
    assert torch.cuda.is_available(), 'no CUDA'
    print('OK')
except Exception as e:
    print(f'FAIL:{e}')
" 2>/dev/null)

    if echo "$CORE_OK" | grep -q "OK"; then
        echo "  ✓ 핵심 패키지 확인 완료"
    else
        echo "  ✗ 패키지 문제 ($CORE_OK), setup_jetson.sh 재실행..."
        cd "$SCRIPT_DIR"
        bash setup_jetson.sh
        source "${VENV_DIR}/bin/activate"
    fi
else
    echo "[2/5] 첫 실행 → setup_jetson.sh 실행..."
    cd "$SCRIPT_DIR"
    bash setup_jetson.sh
    source "${VENV_DIR}/bin/activate"
fi
echo ""

# ============================================================================
# STEP 3: TRT provider 확인
# ============================================================================
echo "[3/5] TensorRT 상태 확인..."
cd "$SCRIPT_DIR"
python3 -c "
import onnxruntime as ort
provs = ort.get_available_providers()
has_trt = 'TensorrtExecutionProvider' in provs
has_cuda = 'CUDAExecutionProvider' in provs

if has_trt:
    print('  ✓ TensorRT EP 사용 가능')
elif has_cuda:
    print('  ⚠ TensorRT 없음, CUDA fallback (YOLOv8은 PyTorch TRT 사용)')
else:
    print('  ✗ GPU provider 없음! CPU에서만 동작합니다.')

import torch
if torch.cuda.is_available():
    print(f'  ✓ PyTorch CUDA: {torch.cuda.get_device_name(0)}')
else:
    print('  ✗ PyTorch CUDA 없음')
" 2>/dev/null || echo "  ✗ 확인 실패"
echo ""

# ============================================================================
# STEP 4: 모델 다운로드 (필요시)
# ============================================================================
echo "[4/5] 모델 준비..."
python3 -c "
from ultralytics import YOLO
import os

# YOLOv8n (가장 빠른 nano)
m = YOLO('yolov8n-pose.pt')
print('  ✓ yolov8n-pose.pt')

# YOLOv8s (small)
m = YOLO('yolov8s-pose.pt')
print('  ✓ yolov8s-pose.pt')
" 2>/dev/null || echo "  모델은 벤치마크 시작 시 자동 다운로드됩니다"
echo ""

# ============================================================================
# STEP 5: 벤치마크 실행
# ============================================================================
echo "[5/5] 벤치마크 실행!"
echo ""

# 벤치마크 명령 구성
CMD="python3 run_int8_comparison.py --max-perf --duration ${DURATION}"

if [ -n "$VIDEO" ]; then
    CMD="$CMD --video $VIDEO"
fi

if [ "$NO_ZED" = true ]; then
    CMD="$CMD --no-zed"
fi

if [ "$VISUALIZE" = true ]; then
    CMD="$CMD --visualize"
fi

if [ "$RECORD" = true ]; then
    CMD="$CMD --record"
fi

CMD="$CMD $EXTRA_ARGS"

echo "  실행 명령:"
echo "  $ $CMD"
echo ""

cd "$SCRIPT_DIR"
eval $CMD
