#!/bin/bash
# =============================================================================
# YOLOv8 vs YOLO11 전체 성능 비교 스크립트
# =============================================================================
# Jetson Orin NX 16GB에서 최대 성능으로 테스트
#
# 테스트 항목:
#   1. YOLOv8n vs YOLO11n (PyTorch / TRT-FP16)
#   2. YOLOv8s vs YOLO11s (PyTorch / TRT-FP16)
#   3. INT8 비교: FP16 vs INT8-nocal vs INT8-cal (v8 + v11)
#
# 사용법:
#   chmod +x run_yolo_comparison.sh
#   sudo ./run_yolo_comparison.sh                    # 전체 테스트
#   sudo ./run_yolo_comparison.sh --quick             # 빠른 테스트 (10초/모델)
#   sudo ./run_yolo_comparison.sh --video walk.mp4    # 비디오 입력
# =============================================================================

set -e
cd "$(dirname "$0")"

DURATION=15
VIDEO_ARG=""
CALIB_ARG=""

# 인자 파싱
for arg in "$@"; do
    case $arg in
        --quick)
            DURATION=10
            ;;
        --video=*)
            VIDEO_ARG="--video ${arg#*=}"
            ;;
        --video)
            shift
            VIDEO_ARG="--video $1"
            ;;
        --calib-data=*)
            CALIB_ARG="--calib-data ${arg#*=}"
            ;;
    esac
    shift 2>/dev/null || true
done

echo "============================================================"
echo "  YOLOv8 vs YOLO11 성능 비교 (Jetson MAX Performance)"
echo "============================================================"
echo "  측정 시간: ${DURATION}초/모델"
echo "  비디오: ${VIDEO_ARG:-라이브 카메라}"
echo "============================================================"
echo ""

# ---- Step 1: PyTorch vs TRT FP16 비교 ----
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [1/3] PyTorch vs TensorRT FP16 비교"
echo "  (YOLOv8n, YOLOv8s, YOLO11n, YOLO11s)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 run_trt_comparison.py \
    --models yolov8 yolo11 \
    --max-perf \
    --duration $DURATION \
    $VIDEO_ARG

# ---- Step 2: INT8 비교 (캘리브레이션 없이) ----
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  [2/3] FP16 vs INT8 비교 (캘리브레이션 없음)"
echo "  (YOLOv8n/s + YOLO11n/s × FP16/INT8-nocal)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 run_int8_comparison.py \
    --max-perf \
    --skip-calibrated \
    --duration $DURATION \
    $VIDEO_ARG

# ---- Step 3: INT8 캘리브레이션 비교 (캘리브레이션 데이터 있을 때만) ----
CALIB_DIR="./calib_images/yolo"
if [ -d "$CALIB_DIR" ] && [ "$(ls -A $CALIB_DIR/*.jpg 2>/dev/null | head -1)" ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [3/3] INT8 전체 비교 (캘리브레이션 포함)"
    echo "  (FP16 / INT8-nocal / INT8-cal)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python3 run_int8_comparison.py \
        --max-perf \
        --calib-data "$CALIB_DIR" \
        --duration $DURATION \
        $VIDEO_ARG
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  [3/3] SKIP: 캘리브레이션 데이터 없음"
    echo "  캘리브레이션 비교를 하려면:"
    echo "    python3 calibrate_yolo.py --video walking.mp4 --build --mix-coco"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
fi

echo ""
echo "============================================================"
echo "  모든 테스트 완료!"
echo "  결과 파일: ./results/"
echo "============================================================"
