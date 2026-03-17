#!/usr/bin/env python3
"""
모델 동작 확인 스크립트
======================
벤치마크 전에 각 모델이 실제로 로드되고 추론이 되는지 확인합니다.
카메라 없이 더미 이미지로 테스트합니다.

사용법:
    python3 verify_models.py              # 전체 모델 확인
    python3 verify_models.py --with-camera  # 카메라 연결해서 실제 테스트
"""

import sys
import os
import argparse
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def create_test_image():
    """테스트용 더미 이미지 (사람 형태는 아니지만 추론 파이프라인 확인용)"""
    img = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    return img


def verify_mediapipe():
    """MediaPipe Pose 동작 확인"""
    print("\n--- MediaPipe Pose ---")

    # 1. import 확인
    try:
        import mediapipe as mp
        print(f"  [1] import: OK (v{mp.__version__})")
    except ImportError as e:
        print(f"  [1] import: FAIL - {e}")
        return False

    # 2. 모델 로드
    try:
        from pose_models import MediaPipePose
        model = MediaPipePose(model_complexity=0)
        model.load()
        print(f"  [2] 모델 로드: OK")
    except Exception as e:
        print(f"  [2] 모델 로드: FAIL - {e}")
        return False

    # 3. 추론
    try:
        img = create_test_image()
        result = model.predict_with_timing(img)
        print(f"  [3] 추론: OK ({result.inference_time_ms:.1f}ms)")
        print(f"      detected={result.detected}, "
              f"keypoints={len(result.keypoints_2d)}")
    except Exception as e:
        print(f"  [3] 추론: FAIL - {e}")
        return False

    print(f"  ✓ MediaPipe Pose 정상 동작!")
    return True


def verify_yolov8():
    """YOLOv8-Pose 동작 확인"""
    print("\n--- YOLOv8-Pose ---")

    # 1. import 확인
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"  [1] import: OK (v{ultralytics.__version__})")
    except ImportError as e:
        print(f"  [1] import: FAIL - {e}")
        print(f"      → pip install ultralytics")
        return False

    # 2. 모델 로드 (자동 다운로드)
    try:
        from pose_models import YOLOv8Pose
        model = YOLOv8Pose(model_size="n")
        print(f"  [2] 모델 로드 중 (yolov8n-pose.pt 다운로드 포함)...")
        model.load()
        print(f"  [2] 모델 로드: OK")
    except Exception as e:
        print(f"  [2] 모델 로드: FAIL - {e}")
        return False

    # 3. 추론
    try:
        img = create_test_image()
        result = model.predict_with_timing(img)
        print(f"  [3] 추론: OK ({result.inference_time_ms:.1f}ms)")
        print(f"      detected={result.detected}, "
              f"keypoints={len(result.keypoints_2d)}")
    except Exception as e:
        print(f"  [3] 추론: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False

    # 4. TensorRT 확인 (선택)
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  [4] CUDA: OK ({torch.cuda.get_device_name(0)})")
        else:
            print(f"  [4] CUDA: 미감지 (CPU 모드)")
    except Exception:
        print(f"  [4] CUDA: 확인 불가")

    print(f"  ✓ YOLOv8-Pose 정상 동작!")
    return True


def verify_rtmpose():
    """RTMPose (rtmlib) 동작 확인"""
    print("\n--- RTMPose (rtmlib) ---")

    # 1. import 확인
    try:
        import rtmlib
        print(f"  [1] rtmlib import: OK")
    except ImportError as e:
        print(f"  [1] rtmlib import: FAIL - {e}")
        print(f"      → pip install rtmlib")
        return False

    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        gpu_ok = "CUDAExecutionProvider" in providers
        print(f"  [1] onnxruntime: OK (GPU={'yes' if gpu_ok else 'no'})")
        print(f"      providers: {providers}")
    except ImportError as e:
        print(f"  [1] onnxruntime: FAIL - {e}")
        print(f"      → pip install onnxruntime-gpu")
        return False

    # 2. Body 모델 로드 (RTMDet + RTMPose)
    try:
        from pose_models import RTMPoseModel
        # CPU로 먼저 테스트 (GPU 없어도 동작하도록)
        device = "cuda" if gpu_ok else "cpu"
        model = RTMPoseModel(
            mode="balanced",
            backend="onnxruntime",
            device=device,
        )
        print(f"  [2] 모델 로드 중 (자동 다운로드, 처음이면 시간 소요)...")
        model.load()
        print(f"  [2] 모델 로드: OK (device={device})")
    except Exception as e:
        print(f"  [2] 모델 로드: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False

    # 3. 추론
    try:
        img = create_test_image()
        result = model.predict_with_timing(img)
        print(f"  [3] 추론: OK ({result.inference_time_ms:.1f}ms)")
        print(f"      detected={result.detected}, "
              f"keypoints={len(result.keypoints_2d)}")
    except Exception as e:
        print(f"  [3] 추론: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"  ✓ RTMPose 정상 동작!")
    return True


def verify_rtmpose_wholebody():
    """RTMPose Wholebody (foot keypoints 포함) 동작 확인"""
    print("\n--- RTMPose Wholebody (133 keypoints) ---")

    try:
        import onnxruntime as ort
        gpu_ok = "CUDAExecutionProvider" in ort.get_available_providers()
    except ImportError:
        print(f"  [1] onnxruntime: FAIL")
        return False

    try:
        from rtmlib import Wholebody
        print(f"  [1] import: OK")
    except ImportError as e:
        print(f"  [1] Wholebody import: FAIL - {e}")
        return False

    try:
        from pose_models import RTMPoseWholebody
        device = "cuda" if gpu_ok else "cpu"
        model = RTMPoseWholebody(
            mode="balanced",
            backend="onnxruntime",
            device=device,
        )
        print(f"  [2] 모델 로드 중 (wholebody 모델 다운로드)...")
        model.load()
        print(f"  [2] 모델 로드: OK")
    except Exception as e:
        print(f"  [2] 모델 로드: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False

    try:
        img = create_test_image()
        result = model.predict_with_timing(img)
        print(f"  [3] 추론: OK ({result.inference_time_ms:.1f}ms)")
        print(f"      detected={result.detected}, "
              f"keypoints={len(result.keypoints_2d)}")

        # foot keypoint 확인
        foot_kps = [k for k in result.keypoints_2d.keys()
                    if "heel" in k or "toe" in k]
        print(f"      foot keypoints: {foot_kps if foot_kps else 'none detected (정상 - 더미 이미지)'}")
    except Exception as e:
        print(f"  [3] 추론: FAIL - {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"  ✓ RTMPose Wholebody 정상 동작!")
    return True


def verify_zed_bt():
    """ZED Body Tracking 동작 확인"""
    print("\n--- ZED Body Tracking ---")

    try:
        import pyzed.sl as sl
        cam = sl.Camera()
        ver = cam.get_sdk_version()
        print(f"  [1] import: OK (ZED SDK {ver})")
    except ImportError as e:
        print(f"  [1] import: FAIL - {e}")
        print(f"      ZED SDK가 설치되어 있어야 합니다")
        return False

    print(f"  [2] ZED Body Tracking은 카메라 연결 필요 → --with-camera 옵션으로 테스트")
    print(f"  ⚠ 카메라 없이는 확인 불가 (import만 확인됨)")
    return True


def verify_with_camera():
    """카메라를 사용한 실제 동작 확인"""
    print("\n" + "=" * 50)
    print("  카메라 실제 테스트")
    print("=" * 50)

    import cv2
    from zed_camera import create_camera, HAS_ZED

    # 카메라 열기
    camera = create_camera(use_zed=HAS_ZED)
    try:
        camera.open()
    except Exception as e:
        print(f"  카메라 열기 실패: {e}")
        print(f"  웹캠으로 대체 시도...")
        camera = create_camera(use_zed=False)
        camera.open()

    # 워밍업
    for _ in range(5):
        camera.grab()

    # 프레임 캡처
    if not camera.grab():
        print("  프레임 캡처 실패!")
        return

    rgb = camera.get_rgb()
    print(f"  캡처 성공: {rgb.shape}")

    # 각 모델로 추론
    models_to_test = []

    try:
        from pose_models import YOLOv8Pose
        models_to_test.append(("YOLOv8n-Pose", YOLOv8Pose(model_size="n")))
    except Exception:
        pass

    try:
        from pose_models import RTMPoseModel
        models_to_test.append(("RTMPose-m", RTMPoseModel()))
    except Exception:
        pass

    try:
        from pose_models import MediaPipePose
        models_to_test.append(("MediaPipe", MediaPipePose(model_complexity=0)))
    except Exception:
        pass

    for name, model in models_to_test:
        try:
            model.load()
            result = model.predict_with_timing(rgb)
            print(f"\n  {name}:")
            print(f"    Latency: {result.inference_time_ms:.1f}ms")
            print(f"    Detected: {result.detected}")
            if result.detected:
                print(f"    Lower limb conf: {result.get_lower_limb_confidence():.3f}")
                for kp_name, conf in sorted(result.confidences.items()):
                    pos = result.keypoints_2d.get(kp_name, (0, 0))
                    print(f"      {kp_name}: conf={conf:.3f} pos=({pos[0]:.0f},{pos[1]:.0f})")

            # 시각화 저장
            from pose_models import draw_pose
            vis = draw_pose(rgb, result, name)
            out_path = os.path.join(os.path.dirname(__file__),
                                     f"verify_{name.replace(' ', '_')}.jpg")
            cv2.imwrite(out_path, vis)
            print(f"    시각화 저장: {out_path}")
        except Exception as e:
            print(f"\n  {name}: FAIL - {e}")

    camera.close()


def main():
    parser = argparse.ArgumentParser(description="모델 동작 확인")
    parser.add_argument("--with-camera", action="store_true",
                        help="카메라 연결해서 실제 이미지로 테스트")
    args = parser.parse_args()

    print("=" * 50)
    print("  H-Walker Pose Model 동작 확인")
    print("=" * 50)

    results = {}
    results["MediaPipe"] = verify_mediapipe()
    results["YOLOv8-Pose"] = verify_yolov8()
    results["RTMPose"] = verify_rtmpose()
    results["RTMPose Wholebody"] = verify_rtmpose_wholebody()
    results["ZED BT"] = verify_zed_bt()

    # 결과 요약
    print("\n" + "=" * 50)
    print("  동작 확인 결과 요약")
    print("=" * 50)
    for name, ok in results.items():
        if ok is None:
            status = "- SKIP (미지원)"
        elif ok:
            status = "✓ OK"
        else:
            status = "✗ FAIL"
        print(f"  {name:<25} {status}")

    ok_count = sum(1 for v in results.values() if v is True)
    skip_count = sum(1 for v in results.values() if v is None)
    fail_count = sum(1 for v in results.values() if v is False)
    total = len(results)
    print(f"\n  {ok_count}/{total} 모델 사용 가능", end="")
    if skip_count > 0:
        print(f" ({skip_count}개 건너뜀)", end="")
    print()

    if ok_count >= 2:
        print(f"\n  → 벤치마크 실행 가능!")
        available = [n for n, v in results.items() if v is True]
        print(f"    사용 가능: {', '.join(available)}")
        print(f"    python3 run_benchmark.py --visualize")
    else:
        print(f"\n  → 설치를 먼저 완료하세요:")
        print(f"    bash setup_jetson.sh")

    if args.with_camera:
        verify_with_camera()


if __name__ == "__main__":
    main()
