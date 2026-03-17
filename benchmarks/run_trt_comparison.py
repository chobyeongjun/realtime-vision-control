#!/usr/bin/env python3
"""
PyTorch vs TensorRT 비교 벤치마크
==================================
각 모델을 기본 모드(PyTorch/ONNX)와 TensorRT FP16으로 두 번 돌려서
속도 차이를 비교합니다.

사용법:
    # 라이브 카메라로 전체 비교 (각 모델 15초)
    python3 run_trt_comparison.py

    # 특정 모델만
    python3 run_trt_comparison.py --models yolov8 rtmpose

    # SVO2 파일로 비교
    python3 run_trt_comparison.py --video test_data/walk_frontal.svo2

    # 120fps로 테스트
    python3 run_trt_comparison.py --camera-fps 120

    # 짧은 테스트 (각 모델 10초)
    python3 run_trt_comparison.py --duration 10
"""

import argparse
import time
import sys
import os
import json
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zed_camera import create_camera, HAS_ZED
from pose_models import (
    MediaPipePose, YOLOv8Pose, RTMPoseModel, RTMPoseWholebody,
    ZEDBodyTracking, PoseResult
)
from joint_angles import compute_lower_limb_angles


def benchmark_single_model(camera, model, duration=15, lower_only=False):
    """단일 모델 벤치마크 실행, 핵심 지표만 수집"""
    warmup_frames = 30

    # 워밍업
    print(f"    워밍업 ({warmup_frames} 프레임)...")
    for _ in range(warmup_frames):
        if camera.grab():
            rgb = camera.get_rgb()
            if lower_only:
                h = rgb.shape[0]
                rgb = rgb[h // 3:, :].copy()
            model.predict(rgb)

    # 측정
    print(f"    측정 시작 ({duration}초)...")
    grab_times = []
    infer_times = []
    e2e_times = []
    detected_count = 0
    lower_limb_count = 0
    frame_count = 0
    has_foot_kps_count = 0

    start_time = time.perf_counter()
    last_print = start_time

    while time.perf_counter() - start_time < duration:
        # Grab
        t0 = time.perf_counter()
        if not camera.grab():
            continue
        rgb = camera.get_rgb()
        t1 = time.perf_counter()

        if lower_only:
            h = rgb.shape[0]
            rgb = rgb[h // 3:, :].copy()

        # Inference
        t2 = time.perf_counter()
        result = model.predict(rgb)
        t3 = time.perf_counter()

        grab_ms = (t1 - t0) * 1000
        infer_ms = (t3 - t2) * 1000
        e2e_ms = (t3 - t0) * 1000

        grab_times.append(grab_ms)
        infer_times.append(infer_ms)
        e2e_times.append(e2e_ms)
        frame_count += 1

        if result.detected:
            detected_count += 1
        if result.has_lower_limb():
            lower_limb_count += 1

        # Foot keypoint 확인
        foot_keys = ["left_heel", "left_toe", "right_heel", "right_toe"]
        foot_detected = sum(1 for k in foot_keys if result.confidences.get(k, 0) > 0.3)
        if foot_detected >= 2:
            has_foot_kps_count += 1

        # 진행 상황
        now = time.perf_counter()
        if now - last_print >= 3.0:
            elapsed = now - start_time
            fps = frame_count / elapsed
            avg_infer = np.mean(infer_times[-50:])
            print(f"      {elapsed:.0f}s | {frame_count}f | FPS:{fps:.1f} | Infer:{avg_infer:.1f}ms")
            last_print = now

    total_time = time.perf_counter() - start_time

    if frame_count == 0:
        return {"error": "No frames captured"}

    return {
        "frame_count": frame_count,
        "avg_fps": frame_count / total_time,
        "grab_ms": {
            "avg": float(np.mean(grab_times)),
            "p95": float(np.percentile(grab_times, 95)),
            "min": float(np.min(grab_times)),
        },
        "infer_ms": {
            "avg": float(np.mean(infer_times)),
            "p95": float(np.percentile(infer_times, 95)),
            "p99": float(np.percentile(infer_times, 99)),
            "min": float(np.min(infer_times)),
            "max": float(np.max(infer_times)),
        },
        "e2e_ms": {
            "avg": float(np.mean(e2e_times)),
            "p95": float(np.percentile(e2e_times, 95)),
        },
        "e2e_under_50ms_pct": float(np.mean(np.array(e2e_times) < 50.0) * 100),
        "e2e_under_33ms_pct": float(np.mean(np.array(e2e_times) < 33.3) * 100),
        "detection_rate": detected_count / frame_count * 100,
        "lower_limb_rate": lower_limb_count / frame_count * 100,
        "foot_kp_rate": has_foot_kps_count / frame_count * 100,
    }


def create_model_pairs(selected_models, include_trt=True):
    """
    각 모델의 기본 버전과 TensorRT 버전 쌍을 만듦
    Returns: list of (label, model_instance)
    """
    pairs = []

    if "mediapipe" in selected_models or "all" in selected_models:
        pairs.append(("MediaPipe Lite", MediaPipePose(model_complexity=0)))
        # MediaPipe는 TensorRT 변환 없음

    if "yolov8" in selected_models or "all" in selected_models:
        pairs.append(("YOLOv8n", YOLOv8Pose(model_size="n", use_tensorrt=False)))
        if include_trt:
            pairs.append(("YOLOv8n (TRT)", YOLOv8Pose(model_size="n", use_tensorrt=True)))
        pairs.append(("YOLOv8s", YOLOv8Pose(model_size="s", use_tensorrt=False)))
        if include_trt:
            pairs.append(("YOLOv8s (TRT)", YOLOv8Pose(model_size="s", use_tensorrt=True)))

    if "rtmpose" in selected_models or "all" in selected_models:
        pairs.append(("RTMPose-bal", RTMPoseModel(mode="balanced", use_tensorrt=False)))
        if include_trt:
            pairs.append(("RTMPose-bal (TRT)", RTMPoseModel(mode="balanced", use_tensorrt=True)))
        pairs.append(("RTMPose-lite", RTMPoseModel(mode="lightweight", use_tensorrt=False)))
        if include_trt:
            pairs.append(("RTMPose-lite (TRT)", RTMPoseModel(mode="lightweight", use_tensorrt=True)))

    if "rtmpose_wb" in selected_models or "all" in selected_models:
        pairs.append(("RTMPose-WB-bal", RTMPoseWholebody(mode="balanced", use_tensorrt=False)))
        if include_trt:
            pairs.append(("RTMPose-WB-bal (TRT)", RTMPoseWholebody(mode="balanced", use_tensorrt=True)))
        pairs.append(("RTMPose-WB-lite", RTMPoseWholebody(mode="lightweight", use_tensorrt=False)))
        if include_trt:
            pairs.append(("RTMPose-WB-lite (TRT)", RTMPoseWholebody(mode="lightweight", use_tensorrt=True)))

    if ("zed_bt" in selected_models or "all" in selected_models) and HAS_ZED:
        pairs.append(("ZED BT (FAST)", ZEDBodyTracking(model="FAST")))
        # ZED BT는 자체 엔진, TensorRT 별도 불필요

    return pairs


def print_comparison_table(all_results):
    """비교 테이블 출력"""

    print("\n")
    print("=" * 110)
    print("  PyTorch vs TensorRT 비교 결과")
    print("=" * 110)

    header = f"  {'모델':<28} {'FPS':>6} {'Infer(avg)':>11} {'Infer(p95)':>11} {'E2E(avg)':>10} {'<50ms':>7} {'<33ms':>7} {'Det%':>6} {'Foot%':>6}"
    print(header)
    print("  " + "-" * 106)

    for label, stats in all_results.items():
        if "error" in stats:
            print(f"  {label:<28} ERROR: {stats['error']}")
            continue

        fps = stats["avg_fps"]
        infer_avg = stats["infer_ms"]["avg"]
        infer_p95 = stats["infer_ms"]["p95"]
        e2e_avg = stats["e2e_ms"]["avg"]
        under_50 = stats["e2e_under_50ms_pct"]
        under_33 = stats["e2e_under_33ms_pct"]
        det = stats["detection_rate"]
        foot = stats["foot_kp_rate"]

        # TRT 모델은 강조
        marker = "⚡" if "(TRT)" in label else "  "

        print(f"{marker}{label:<28} {fps:>5.1f} {infer_avg:>9.1f}ms {infer_p95:>9.1f}ms {e2e_avg:>8.1f}ms {under_50:>6.1f}% {under_33:>6.1f}% {det:>5.1f}% {foot:>5.1f}%")

    print("  " + "-" * 106)

    # 속도 향상 비교
    print("\n  [속도 향상 분석]")
    base_models = {}
    trt_models = {}
    for label, stats in all_results.items():
        if "error" in stats:
            continue
        if "(TRT)" in label:
            base_name = label.replace(" (TRT)", "")
            trt_models[base_name] = stats
        else:
            base_models[label] = stats

    for name, trt_stats in trt_models.items():
        if name in base_models:
            base_stats = base_models[name]
            base_infer = base_stats["infer_ms"]["avg"]
            trt_infer = trt_stats["infer_ms"]["avg"]
            speedup = base_infer / trt_infer if trt_infer > 0 else 0

            base_fps = base_stats["avg_fps"]
            trt_fps = trt_stats["avg_fps"]

            print(f"  {name}:")
            print(f"    Inference: {base_infer:.1f}ms → {trt_infer:.1f}ms ({speedup:.1f}x 빠름)")
            print(f"    FPS:       {base_fps:.1f} → {trt_fps:.1f}")
            print()

    # 최적 모델 추천
    print("  [최적 모델 추천]")

    # Foot keypoint 있는 모델 중 가장 빠른 것
    foot_models = {k: v for k, v in all_results.items()
                   if "error" not in v and v["foot_kp_rate"] > 50}
    if foot_models:
        best_foot = min(foot_models.items(), key=lambda x: x[1]["infer_ms"]["avg"])
        print(f"  🦶 Toe/Heel 포함 최고 속도: {best_foot[0]} ({best_foot[1]['infer_ms']['avg']:.1f}ms, {best_foot[1]['avg_fps']:.1f}FPS)")

    # 전체 최고 속도
    valid = {k: v for k, v in all_results.items() if "error" not in v}
    if valid:
        best_speed = min(valid.items(), key=lambda x: x[1]["infer_ms"]["avg"])
        print(f"  ⚡ 전체 최고 속도:           {best_speed[0]} ({best_speed[1]['infer_ms']['avg']:.1f}ms, {best_speed[1]['avg_fps']:.1f}FPS)")

    # E2E <50ms 최고
    if valid:
        best_e2e = max(valid.items(), key=lambda x: x[1]["e2e_under_50ms_pct"])
        print(f"  🎯 E2E <50ms 최고 달성률:   {best_e2e[0]} ({best_e2e[1]['e2e_under_50ms_pct']:.1f}%)")

    print()


def main():
    parser = argparse.ArgumentParser(description="PyTorch vs TensorRT 비교 벤치마크")
    parser.add_argument("--models", nargs="+",
                        default=["all"],
                        choices=["all", "mediapipe", "yolov8", "rtmpose", "rtmpose_wb", "zed_bt"],
                        help="비교할 모델 선택")
    parser.add_argument("--duration", type=int, default=15,
                        help="모델당 측정 시간 (초)")
    parser.add_argument("--no-zed", dest="use_zed", action="store_false",
                        help="ZED 카메라 대신 웹캠 사용")
    parser.add_argument("--resolution", default="SVGA",
                        choices=["SVGA", "VGA", "HD720", "HD1080", "HD1200"])
    parser.add_argument("--camera-fps", type=int, default=60,
                        help="카메라 FPS (ZED X Mini: 60 or 120)")
    parser.add_argument("--video", type=str, default=None,
                        help="동영상 파일 경로 (SVO2/MP4)")
    parser.add_argument("--lower-only", action="store_true",
                        help="하체만 보이는 상황 시뮬레이션")
    parser.add_argument("--no-trt", action="store_true",
                        help="TensorRT 비교 건너뛰기 (기본 모드만)")
    parser.add_argument("--trt-only", action="store_true",
                        help="TensorRT 모델만 실행")

    args = parser.parse_args()

    print("=" * 60)
    print("  PyTorch vs TensorRT 비교 벤치마크")
    print("=" * 60)
    print(f"  측정 시간: {args.duration}초/모델")
    print(f"  해상도: {args.resolution}")
    print(f"  카메라 FPS: {args.camera_fps}")
    if args.video:
        print(f"  입력: {args.video}")
    else:
        print(f"  카메라: {'ZED X Mini' if args.use_zed and HAS_ZED else 'Webcam'}")
    print(f"  TensorRT 비교: {'OFF' if args.no_trt else 'ON'}")
    print("=" * 60)
    print()

    # 카메라 초기화
    camera = create_camera(
        use_zed=args.use_zed,
        video_path=args.video,
        resolution=args.resolution,
        fps=args.camera_fps,
    )
    camera.open()

    # 카메라 워밍업
    print("[카메라 워밍업]")
    for _ in range(10):
        camera.grab()
    print()

    # 모델 쌍 생성
    include_trt = not args.no_trt
    pairs = create_model_pairs(args.models, include_trt=include_trt)

    # TRT-only 모드면 TRT 모델만 필터
    if args.trt_only:
        pairs = [(label, model) for label, model in pairs
                 if "(TRT)" in label or "MediaPipe" in label or "ZED" in label]

    all_results = {}
    total_models = len(pairs)

    for i, (label, model) in enumerate(pairs):
        print(f"\n{'=' * 50}")
        print(f"  [{i+1}/{total_models}] {label}")
        print(f"{'=' * 50}")

        try:
            # ZED BT는 카메라 객체 필요
            if isinstance(model, ZEDBodyTracking):
                model.load(camera)
            else:
                model.load()

            stats = benchmark_single_model(
                camera, model,
                duration=args.duration,
                lower_only=args.lower_only,
            )
            all_results[label] = stats

            # 간단 결과 출력
            if "error" not in stats:
                print(f"    → FPS: {stats['avg_fps']:.1f} | "
                      f"Infer: {stats['infer_ms']['avg']:.1f}ms | "
                      f"E2E: {stats['e2e_ms']['avg']:.1f}ms | "
                      f"Det: {stats['detection_rate']:.0f}%")

        except Exception as e:
            print(f"    [ERROR] {label} 실패: {e}")
            import traceback
            traceback.print_exc()
            all_results[label] = {"error": str(e)}

        # ZED BT 정리
        if isinstance(model, ZEDBodyTracking) and hasattr(model, 'close'):
            model.close()

    # 비교 테이블
    print_comparison_table(all_results)

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"trt_comparison_{timestamp}.json")

    save_data = {
        "timestamp": timestamp,
        "config": {
            "models": args.models,
            "duration": args.duration,
            "resolution": args.resolution,
            "camera_fps": args.camera_fps,
            "video": args.video,
            "trt_enabled": include_trt,
        },
        "results": {},
    }

    for label, stats in all_results.items():
        # numpy 값을 일반 Python으로 변환
        clean_stats = {}
        for k, v in stats.items():
            if isinstance(v, dict):
                clean_stats[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer)) else vv
                                  for kk, vv in v.items()}
            elif isinstance(v, (np.floating, np.integer)):
                clean_stats[k] = float(v)
            else:
                clean_stats[k] = v
        save_data["results"][label] = clean_stats

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, indent=2, ensure_ascii=False)

    print(f"  결과 저장: {output_path}")

    camera.close()
    print("\n완료!")


if __name__ == "__main__":
    main()
