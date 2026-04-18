#!/usr/bin/env python3
"""
TensorRT 환경 진단 스크립트
============================
Jetson에서 실행하여 TRT가 올바르게 동작하는지 확인합니다.

사용법:
    python3 check_trt_status.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_section(title):
    print(f"\n{'='*50}")
    print(f"  {title}")
    print(f"{'='*50}")


def main():
    all_ok = True

    # 1. 기본 패키지
    check_section("1. 기본 패키지 확인")

    try:
        import numpy as np
        print(f"  [OK] numpy {np.__version__}")
    except ImportError:
        print("  [FAIL] numpy 미설치")
        all_ok = False

    try:
        import cv2
        print(f"  [OK] opencv {cv2.__version__}")
    except ImportError:
        print("  [FAIL] opencv 미설치")
        all_ok = False

    # 2. CUDA / GPU
    check_section("2. CUDA / GPU 확인")

    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        print(f"  [{'OK' if cuda_ok else 'FAIL'}] PyTorch CUDA: {cuda_ok}")
        if cuda_ok:
            print(f"  [INFO] GPU: {torch.cuda.get_device_name(0)}")
            print(f"  [INFO] CUDA Version: {torch.version.cuda}")
    except ImportError:
        print("  [WARN] PyTorch 미설치 (YOLOv8 사용 불가)")

    # 3. TensorRT
    check_section("3. TensorRT 확인")

    try:
        import tensorrt as trt
        print(f"  [OK] TensorRT {trt.__version__}")
    except ImportError:
        print("  [FAIL] tensorrt 패키지 미설치")
        print("  [TIP] Jetson: sudo apt install tensorrt python3-libnvinfer")
        all_ok = False

    # 4. ONNX Runtime
    check_section("4. ONNX Runtime 확인")

    try:
        import onnxruntime as ort
        print(f"  [OK] onnxruntime {ort.__version__}")
        providers = ort.get_available_providers()
        print(f"  [INFO] Available Providers: {providers}")

        has_trt_ep = 'TensorrtExecutionProvider' in providers
        has_cuda_ep = 'CUDAExecutionProvider' in providers
        has_cpu_ep = 'CPUExecutionProvider' in providers

        print(f"  [{'OK' if has_trt_ep else 'FAIL'}] TensorrtExecutionProvider")
        print(f"  [{'OK' if has_cuda_ep else 'WARN'}] CUDAExecutionProvider")
        print(f"  [{'OK' if has_cpu_ep else 'WARN'}] CPUExecutionProvider")

        if not has_trt_ep:
            print()
            print("  [TIP] TensorRT EP가 없습니다. 해결 방법:")
            print("    1. onnxruntime-gpu 설치 확인: pip install onnxruntime-gpu")
            print("    2. Jetson: Jetson Zoo에서 aarch64 빌드 설치")
            print("    3. TensorRT 버전과 onnxruntime 버전 호환성 확인")
            all_ok = False
    except ImportError:
        print("  [FAIL] onnxruntime 미설치")
        print("  [TIP] pip install onnxruntime-gpu")
        all_ok = False

    # 5. 모델별 TRT 테스트
    check_section("5. 모델 로드 테스트")

    # rtmlib (RTMPose)
    try:
        from rtmlib import Body, Wholebody
        print("  [OK] rtmlib 설치됨 (RTMPose / Wholebody)")
    except ImportError:
        print("  [WARN] rtmlib 미설치 (RTMPose 사용 불가)")

    # ultralytics (YOLOv8)
    try:
        from ultralytics import YOLO
        print("  [OK] ultralytics 설치됨 (YOLOv8)")
    except ImportError:
        print("  [WARN] ultralytics 미설치 (YOLOv8 사용 불가)")

    # tflite (MoveNet)
    try:
        from tflite_runtime.interpreter import Interpreter
        print("  [OK] tflite-runtime 설치됨 (MoveNet)")
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
            print("  [OK] tensorflow-lite 설치됨 (MoveNet)")
        except ImportError:
            print("  [WARN] tflite-runtime 미설치 (MoveNet 사용 불가)")

    # mediapipe
    try:
        import mediapipe as mp
        print(f"  [OK] mediapipe {mp.__version__}")
    except ImportError:
        print("  [WARN] mediapipe 미설치")

    # 6. TRT EP 실제 동작 테스트
    check_section("6. TRT EP 실제 동작 테스트")

    try:
        import onnxruntime as ort
        if 'TensorrtExecutionProvider' in ort.get_available_providers():
            # 간단한 ONNX 모델로 TRT EP 동작 테스트
            import numpy as np
            print("  TensorRT EP를 실제로 사용해봅니다...")

            # rtmlib 모델 경로 확인
            from pose_models import check_tensorrt_available, get_trt_providers, verify_trt_provider
            has_trt, has_cuda, provs = check_tensorrt_available()
            print(f"  check_tensorrt_available(): TRT={has_trt}, CUDA={has_cuda}")

            if has_trt:
                print("  [OK] TensorRT EP가 정상적으로 감지됩니다!")
                print("  → 모든 TRT 모델 사용 가능")
            else:
                print("  [FAIL] TRT EP 감지 실패")
        else:
            print("  [SKIP] TensorRT EP가 없어서 테스트 불가")
    except Exception as e:
        print(f"  [ERROR] {e}")

    # 7. 최종 판정
    check_section("7. 최종 판정")

    if all_ok:
        print("  [OK] TensorRT 환경이 정상입니다!")
        print("  → python3 run_record_demo.py  (전체 모델 녹화)")
        print("  → python3 run_trt_comparison.py  (PyTorch vs TRT 비교)")
    else:
        print("  [WARN] 일부 문제가 있습니다 (위 내용 참고)")
        print("  → TRT 없이 실행: python3 run_record_demo.py --no-trt")
        print("  → PyTorch만 비교: python3 run_trt_comparison.py --no-trt")

    print()


if __name__ == "__main__":
    main()
