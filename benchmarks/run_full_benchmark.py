#!/usr/bin/env python3
"""
전체 벤치마크 매트릭스 자동 실행기
====================================
모든 모델 x 시나리오 조합을 체계적으로 실행하고 분석 리포트를 생성합니다.

사용법:
    python3 run_full_benchmark.py                        # 전체 매트릭스
    python3 run_full_benchmark.py --duration 30           # 30초/구성
    python3 run_full_benchmark.py --skip-tensorrt         # TRT 변환 생략
    python3 run_full_benchmark.py --models rtmpose yolov8 # 특정 모델만
    python3 run_full_benchmark.py --scenarios full_body   # 특정 시나리오만
"""

import argparse
import subprocess
import sys
import os
import time
import json
import glob
from datetime import datetime


# ============================================================================
# 벤치마크 매트릭스 정의
# ============================================================================
MODEL_CONFIGS = {
    "mediapipe": [
        {"key": "mediapipe", "label": "MediaPipe Lite/Full"},
    ],
    "yolov8": [
        {"key": "yolov8", "label": "YOLOv8n/s PyTorch", "tensorrt": False},
    ],
    "yolov8_trt": [
        {"key": "yolov8", "label": "YOLOv8n/s TensorRT", "tensorrt": True},
    ],
    "rtmpose": [
        {"key": "rtmpose", "label": "RTMPose Lite/Balanced"},
    ],
    "rtmpose_wb": [
        {"key": "rtmpose_wb", "label": "RTMPose Wholebody Lite/Balanced"},
    ],
    "zed_bt": [
        {"key": "zed_bt", "label": "ZED Body Tracking Fast"},
    ],
}

SCENARIO_CONFIGS = {
    "full_body": {"lower_only": False, "label": "Full Body View"},
    "lower_only": {"lower_only": True, "label": "Lower Body Only"},
}


def run_benchmark(model_key, duration, use_zed, resolution, camera_fps,
                  tensorrt=False, lower_only=False, video=None):
    """단일 벤치마크 실행"""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_benchmark.py")

    cmd = [
        sys.executable, script,
        "--models", model_key,
        "--duration", str(duration),
        "--resolution", resolution,
        "--camera-fps", str(camera_fps),
    ]

    if not use_zed:
        cmd.append("--no-zed")
    if tensorrt:
        cmd.append("--tensorrt")
    if lower_only:
        cmd.append("--lower-only")
    if video:
        cmd.extend(["--video", video])

    print(f"\n  CMD: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="전체 벤치마크 매트릭스 실행")
    parser.add_argument("--duration", type=int, default=15,
                        help="모델당 측정 시간 (초, 기본: 15)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=list(MODEL_CONFIGS.keys()) + ["all"],
                        help="실행할 모델 그룹")
    parser.add_argument("--scenarios", nargs="+", default=None,
                        choices=list(SCENARIO_CONFIGS.keys()) + ["all"],
                        help="실행할 시나리오")
    parser.add_argument("--skip-tensorrt", action="store_true",
                        help="TensorRT 변환 생략 (시간 절약)")
    parser.add_argument("--no-zed", dest="use_zed", action="store_false",
                        help="ZED 없이 웹캠 사용")
    parser.add_argument("--resolution", default="SVGA")
    parser.add_argument("--camera-fps", type=int, default=60)
    parser.add_argument("--video", type=str, default=None,
                        help="동영상 파일 (카메라 대신)")
    parser.add_argument("--no-analysis", action="store_true",
                        help="완료 후 분석 스크립트 실행 안 함")
    args = parser.parse_args()

    # 모델 선택
    if args.models is None or "all" in args.models:
        selected_models = list(MODEL_CONFIGS.keys())
    else:
        selected_models = args.models

    if args.skip_tensorrt and "yolov8_trt" in selected_models:
        selected_models.remove("yolov8_trt")

    # 시나리오 선택
    if args.scenarios is None or "all" in args.scenarios:
        selected_scenarios = list(SCENARIO_CONFIGS.keys())
    else:
        selected_scenarios = args.scenarios

    # 매트릭스 구성
    tasks = []
    for model_name in selected_models:
        configs = MODEL_CONFIGS[model_name]
        for cfg in configs:
            for scenario_name in selected_scenarios:
                scenario = SCENARIO_CONFIGS[scenario_name]
                tasks.append({
                    "model_key": cfg["key"],
                    "model_label": cfg["label"],
                    "scenario": scenario_name,
                    "scenario_label": scenario["label"],
                    "tensorrt": cfg.get("tensorrt", False),
                    "lower_only": scenario["lower_only"],
                })

    # 총 소요 시간 추정
    total_configs = len(tasks)
    # 모델 내에서 여러 variant가 실행되므로 실제 시간은 더 김
    estimated_time = total_configs * (args.duration + 30)  # +30s 오버헤드
    print("=" * 60)
    print("  H-Walker 전체 벤치마크 매트릭스")
    print("=" * 60)
    print(f"  모델 그룹: {len(selected_models)}개")
    print(f"  시나리오: {len(selected_scenarios)}개")
    print(f"  총 실행 단위: {total_configs}개")
    print(f"  측정 시간: {args.duration}초/모델")
    print(f"  예상 소요: ~{estimated_time // 60}분 {estimated_time % 60}초")
    print("=" * 60)

    # 실행
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    success_count = 0
    fail_count = 0
    start_total = time.perf_counter()

    for i, task in enumerate(tasks, 1):
        print(f"\n{'=' * 60}")
        print(f"  [{i}/{total_configs}] {task['model_label']} / {task['scenario_label']}")
        trt_str = " + TensorRT" if task["tensorrt"] else ""
        print(f"  Model: {task['model_key']}{trt_str}")
        print(f"{'=' * 60}")

        try:
            ok = run_benchmark(
                model_key=task["model_key"],
                duration=args.duration,
                use_zed=args.use_zed,
                resolution=args.resolution,
                camera_fps=args.camera_fps,
                tensorrt=task["tensorrt"],
                lower_only=task["lower_only"],
                video=args.video,
            )
            if ok:
                success_count += 1
            else:
                fail_count += 1
                print(f"  [WARN] 벤치마크 실패")
        except Exception as e:
            fail_count += 1
            print(f"  [ERROR] {e}")

    elapsed = time.perf_counter() - start_total

    # 요약
    print(f"\n\n{'=' * 60}")
    print(f"  전체 벤치마크 완료!")
    print(f"  소요 시간: {elapsed:.0f}초 ({elapsed / 60:.1f}분)")
    print(f"  성공: {success_count}/{total_configs}")
    print(f"  실패: {fail_count}/{total_configs}")
    print(f"{'=' * 60}")

    # 분석 실행
    if not args.no_analysis:
        print("\n  분석 리포트 생성 중...")
        analysis_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze_results.py")
        subprocess.run([
            sys.executable, analysis_script,
            results_dir,
            "--output", os.path.join(results_dir, "analysis", "report.html"),
        ])
        print(f"  리포트: {os.path.join(results_dir, 'analysis', 'report.html')}")


if __name__ == "__main__":
    main()
