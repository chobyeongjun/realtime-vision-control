#!/usr/bin/env python3
"""
H-Walker Pose Estimation 모델 벤치마크
======================================
Jetson Orin NX에서 실행하여 각 모델의 FPS, Latency, 하체 인식 성능을 비교합니다.

사용법:
    python3 run_benchmark.py                    # 모든 모델 벤치마크
    python3 run_benchmark.py --models mediapipe yolov8  # 특정 모델만
    python3 run_benchmark.py --duration 30      # 30초간 측정
    python3 run_benchmark.py --no-zed           # 웹캠 사용 (ZED 없이)
    python3 run_benchmark.py --visualize        # 실시간 시각화
    python3 run_benchmark.py --lower-only       # 하체만 보이는 상황 시뮬레이션
"""

import argparse
import time
import sys
import os
import json
import numpy as np
import cv2
from datetime import datetime

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zed_camera import create_camera, HAS_ZED
from pose_models import (
    MediaPipePose, YOLOv8Pose, RTMPoseModel, RTMPoseWholebody,
    ZEDBodyTracking, draw_pose, PoseResult
)
from joint_angles import compute_lower_limb_angles
from metrics_3d import (
    compute_bone_lengths, compute_bone_length_stability,
    compute_symmetry_score, compute_depth_validity_rate,
    compute_anatomical_plausibility, aggregate_symmetry_scores,
)


class BenchmarkRunner:
    """모델 벤치마크 실행기"""

    def __init__(self, args):
        self.args = args
        self.results = {}

    def run(self):
        print("=" * 60)
        print("  H-Walker Pose Estimation 모델 벤치마크")
        print("=" * 60)
        print(f"  측정 시간: {self.args.duration}초/모델")
        if self.args.video:
            ext = os.path.splitext(self.args.video)[1].lower()
            depth_info = " (SVO2 - depth 포함)" if ext in ('.svo2', '.svo') else " (RGB only)"
            print(f"  입력: 동영상 파일 ({self.args.video}){depth_info}")
        else:
            print(f"  카메라: {'ZED X Mini' if self.args.use_zed and HAS_ZED else 'Webcam'}")
        print(f"  하체 전용 테스트: {self.args.lower_only}")
        print(f"  시각화: {self.args.visualize}")
        print("=" * 60)
        print()

        # 카메라 초기화
        camera = create_camera(
            use_zed=self.args.use_zed,
            video_path=self.args.video,
            resolution=self.args.resolution,
            fps=self.args.camera_fps,
        )
        camera.open()

        # 워밍업: 첫 몇 프레임 버리기
        print("[카메라 워밍업]")
        for _ in range(10):
            camera.grab()
        print()

        # 모델 목록 구성
        models = self._create_models(camera)

        # 각 모델 벤치마크
        for model_name, model in models.items():
            print(f"\n{'=' * 50}")
            print(f"  벤치마크: {model.name}")
            print(f"{'=' * 50}")

            try:
                model.load() if model_name != "zed_bt" else model.load(camera)
                stats = self._benchmark_model(camera, model)
                self.results[model.name] = stats
                self._print_stats(model.name, stats)
            except Exception as e:
                print(f"  [ERROR] {model.name} 실패: {e}")
                import traceback
                traceback.print_exc()
                self.results[model.name] = {"error": str(e)}

            # ZED BT 정리
            if model_name == "zed_bt" and hasattr(model, 'close'):
                model.close()

        # 최종 비교 결과
        self._print_comparison()
        self._save_results()

        camera.close()

    def _create_models(self, camera):
        """테스트할 모델 목록 생성"""
        models = {}
        selected = self.args.models

        if "mediapipe" in selected or "all" in selected:
            models["mediapipe_lite"] = MediaPipePose(model_complexity=0)
            models["mediapipe_full"] = MediaPipePose(model_complexity=1)

        if "yolov8" in selected or "all" in selected:
            models["yolov8n"] = YOLOv8Pose(model_size="n", use_tensorrt=self.args.tensorrt)
            models["yolov8s"] = YOLOv8Pose(model_size="s", use_tensorrt=self.args.tensorrt)

        if "rtmpose" in selected or "all" in selected:
            models["rtmpose_bal"] = RTMPoseModel(
                mode="balanced",
                device="cuda",
                use_tensorrt=self.args.tensorrt,
            )
            models["rtmpose_lite"] = RTMPoseModel(
                mode="lightweight",
                device="cuda",
                use_tensorrt=self.args.tensorrt,
            )

        if "rtmpose_wb" in selected or "all" in selected:
            models["rtmpose_wb_bal"] = RTMPoseWholebody(
                mode="balanced",
                device="cuda",
                use_tensorrt=self.args.tensorrt,
            )
            models["rtmpose_wb_lite"] = RTMPoseWholebody(
                mode="lightweight",
                device="cuda",
                use_tensorrt=self.args.tensorrt,
            )

        if ("zed_bt" in selected or "all" in selected) and HAS_ZED and self.args.use_zed:
            models["zed_bt"] = ZEDBodyTracking(model="FAST")

        if not models:
            print("[ERROR] 선택된 모델이 없습니다")
            sys.exit(1)

        return models

    def _benchmark_model(self, camera, model):
        """단일 모델 벤치마크 (E2E latency 분해 측정 + 3D 메트릭)"""
        stats = {
            "frame_count": 0,
            "detected_count": 0,
            "lower_limb_count": 0,
            # Latency 분해
            "grab_time_ms_list": [],
            "inference_time_ms_list": [],
            "postprocess_time_ms_list": [],
            "e2e_latency_ms_list": [],
            # 하체 confidence
            "lower_limb_conf_list": [],
            # 3D 메트릭
            "bone_lengths_per_frame": [],
            "symmetry_per_frame": [],
            "depth_validity_list": [],
            # 관절 각도
            "joint_angle_samples": {},
            # keypoint 개수
            "keypoint_3d_count_list": [],
        }

        duration = self.args.duration
        warmup_frames = 30

        # 워밍업
        print(f"  모델 워밍업 ({warmup_frames} 프레임)...")
        for i in range(warmup_frames):
            if camera.grab():
                rgb = camera.get_rgb()
                if self.args.lower_only:
                    rgb = self._crop_lower_half(rgb)
                model.predict(rgb)

        # 실제 측정
        print(f"  측정 시작 ({duration}초)...")
        start_time = time.perf_counter()
        frame_times = []
        last_fps_print = start_time

        while time.perf_counter() - start_time < duration:
            # ---- Phase A: Camera Grab ----
            t_grab_start = time.perf_counter()
            if not camera.grab():
                continue
            rgb = camera.get_rgb()
            depth = camera.get_depth() if hasattr(camera, 'get_depth') else None
            t_grab_end = time.perf_counter()

            if self.args.lower_only:
                rgb = self._crop_lower_half(rgb)

            # ---- Phase B: Model Inference ----
            t_infer_start = time.perf_counter()
            result = model.predict(rgb)
            t_infer_end = time.perf_counter()

            # ---- Phase C: Post-processing (2D→3D + 관절각도) ----
            t_post_start = time.perf_counter()

            # 2D → 3D 변환 (depth 있을 때)
            if depth is not None and result.detected:
                for name, (px, py) in result.keypoints_2d.items():
                    pt3d = camera.pixel_to_3d(px, py, depth)
                    if pt3d is not None:
                        result.keypoints_3d[name] = tuple(pt3d)

            # 관절 각도 계산
            use_3d = bool(result.keypoints_3d)
            result.joint_angles = compute_lower_limb_angles(result, use_3d=use_3d)

            t_post_end = time.perf_counter()

            # ---- Timing 기록 ----
            grab_ms = (t_grab_end - t_grab_start) * 1000
            infer_ms = (t_infer_end - t_infer_start) * 1000
            post_ms = (t_post_end - t_post_start) * 1000
            e2e_ms = (t_post_end - t_grab_start) * 1000

            result.grab_time_ms = grab_ms
            result.inference_time_ms = infer_ms
            result.postprocess_time_ms = post_ms
            result.e2e_latency_ms = e2e_ms

            # ---- 통계 수집 ----
            stats["frame_count"] += 1
            stats["grab_time_ms_list"].append(grab_ms)
            stats["inference_time_ms_list"].append(infer_ms)
            stats["postprocess_time_ms_list"].append(post_ms)
            stats["e2e_latency_ms_list"].append(e2e_ms)

            if result.detected:
                stats["detected_count"] += 1
            if result.has_lower_limb(min_conf=0.3):
                stats["lower_limb_count"] += 1
                stats["lower_limb_conf_list"].append(
                    result.get_lower_limb_confidence())

            # 3D 메트릭 수집
            if result.keypoints_3d:
                stats["keypoint_3d_count_list"].append(len(result.keypoints_3d))
                bone_lengths = compute_bone_lengths(result.keypoints_3d)
                if bone_lengths:
                    stats["bone_lengths_per_frame"].append(bone_lengths)
                    sym = compute_symmetry_score(bone_lengths)
                    if sym:
                        stats["symmetry_per_frame"].append(sym)
                dv = compute_depth_validity_rate(result.keypoints_3d, result.keypoints_2d)
                stats["depth_validity_list"].append(dv)

            # 관절 각도 수집
            for angle_name, angle_val in result.joint_angles.items():
                if angle_name not in stats["joint_angle_samples"]:
                    stats["joint_angle_samples"][angle_name] = []
                stats["joint_angle_samples"][angle_name].append(angle_val)

            frame_end = time.perf_counter()
            frame_times.append(frame_end - t_grab_start)

            # 시각화
            if self.args.visualize:
                vis = draw_pose(rgb, result, model.name)
                if len(frame_times) > 1:
                    fps = 1.0 / np.mean(frame_times[-30:])
                    cv2.putText(vis, f"FPS: {fps:.1f}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # E2E latency 표시
                cv2.putText(vis, f"E2E: {e2e_ms:.1f}ms (G:{grab_ms:.0f}+I:{infer_ms:.0f}+P:{post_ms:.0f})",
                           (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)
                cv2.imshow("Benchmark", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

            # 2초마다 진행 상황 출력
            now = time.perf_counter()
            if now - last_fps_print >= 2.0:
                elapsed = now - start_time
                fps = stats["frame_count"] / elapsed
                avg_e2e = np.mean(stats["e2e_latency_ms_list"][-100:])
                avg_infer = np.mean(stats["inference_time_ms_list"][-100:])
                print(f"    {elapsed:.0f}s | {stats['frame_count']} frames | "
                      f"FPS: {fps:.1f} | Infer: {avg_infer:.1f}ms | E2E: {avg_e2e:.1f}ms | "
                      f"Detection: {stats['detected_count']}/{stats['frame_count']}")
                last_fps_print = now

        if self.args.visualize:
            cv2.destroyAllWindows()

        # ---- 최종 통계 계산 ----
        total_time = time.perf_counter() - start_time
        n = stats["frame_count"]

        if n > 0:
            stats["avg_fps"] = n / total_time

            # Latency 분해 통계
            for prefix in ["grab_time_ms", "inference_time_ms", "postprocess_time_ms", "e2e_latency_ms"]:
                arr = stats[f"{prefix}_list"]
                stats[f"avg_{prefix}"] = float(np.mean(arr))
                stats[f"p95_{prefix}"] = float(np.percentile(arr, 95))
                stats[f"p99_{prefix}"] = float(np.percentile(arr, 99))
                stats[f"min_{prefix}"] = float(np.min(arr))
                stats[f"max_{prefix}"] = float(np.max(arr))

            # 하위 호환: 기존 필드명 유지
            stats["avg_latency_ms"] = stats["avg_inference_time_ms"]
            stats["p95_latency_ms"] = stats["p95_inference_time_ms"]
            stats["p99_latency_ms"] = stats["p99_inference_time_ms"]

            stats["detection_rate"] = stats["detected_count"] / n * 100
            stats["lower_limb_rate"] = stats["lower_limb_count"] / n * 100
            stats["avg_lower_limb_conf"] = (
                float(np.mean(stats["lower_limb_conf_list"]))
                if stats["lower_limb_conf_list"] else 0.0
            )

            # E2E <50ms 달성률
            e2e_arr = np.array(stats["e2e_latency_ms_list"])
            stats["e2e_under_50ms_rate"] = float(np.mean(e2e_arr < 50.0) * 100)

            # 3D 메트릭 집계
            if stats["bone_lengths_per_frame"]:
                stability = compute_bone_length_stability(stats["bone_lengths_per_frame"])
                stats["bone_length_mean"] = stability["mean"]
                stats["bone_length_cv"] = stability["cv"]
                # 마지막 프레임의 해부학적 타당성
                last_bones = stats["bone_lengths_per_frame"][-1]
                plausibility = compute_anatomical_plausibility(last_bones)
                stats["anatomical_plausibility"] = {
                    k: v["in_range"] for k, v in plausibility.items()
                }

            if stats["symmetry_per_frame"]:
                stats["symmetry_scores"] = aggregate_symmetry_scores(stats["symmetry_per_frame"])

            if stats["depth_validity_list"]:
                stats["avg_depth_validity"] = float(np.mean(stats["depth_validity_list"]))

            if stats["keypoint_3d_count_list"]:
                stats["avg_3d_keypoints"] = float(np.mean(stats["keypoint_3d_count_list"]))

            # 관절 각도 통계
            if stats["joint_angle_samples"]:
                stats["joint_angle_stats"] = {}
                for angle_name, values in stats["joint_angle_samples"].items():
                    arr = np.array(values)
                    stats["joint_angle_stats"][angle_name] = {
                        "mean": float(np.mean(arr)),
                        "std": float(np.std(arr)),
                        "min": float(np.min(arr)),
                        "max": float(np.max(arr)),
                        "count": len(values),
                    }

        # 큰 리스트 정리 (저장용 - 처음 100개만)
        for key in ["grab_time_ms_list", "inference_time_ms_list",
                     "postprocess_time_ms_list", "e2e_latency_ms_list"]:
            stats[key] = stats[key][:100]
        # 저장 불필요한 큰 데이터 제거
        stats.pop("bone_lengths_per_frame", None)
        stats.pop("symmetry_per_frame", None)
        stats.pop("depth_validity_list", None)
        stats.pop("lower_limb_conf_list", None)
        stats.pop("keypoint_3d_count_list", None)
        stats.pop("joint_angle_samples", None)

        return stats

    def _crop_lower_half(self, image):
        """하체만 보이도록 이미지 하반부만 크롭 (상체 가림 시뮬레이션)"""
        h, w = image.shape[:2]
        return image[h // 3:, :].copy()  # 상위 1/3 제거

    def _print_stats(self, model_name, stats):
        """단일 모델 결과 출력"""
        if "error" in stats:
            print(f"  {model_name}: ERROR - {stats['error']}")
            return

        print(f"\n  --- {model_name} 결과 ---")
        print(f"  총 프레임: {stats['frame_count']}")
        print(f"  평균 FPS: {stats.get('avg_fps', 0):.1f}")
        print()
        print(f"  Latency 분해 (ms):")
        print(f"    {'':>12} {'평균':>8} {'P95':>8} {'P99':>8} {'Min':>8} {'Max':>8}")
        for label, prefix in [("Grab", "grab_time_ms"),
                               ("Inference", "inference_time_ms"),
                               ("PostProc", "postprocess_time_ms"),
                               ("E2E Total", "e2e_latency_ms")]:
            print(f"    {label:>12} "
                  f"{stats.get(f'avg_{prefix}', 0):>8.1f} "
                  f"{stats.get(f'p95_{prefix}', 0):>8.1f} "
                  f"{stats.get(f'p99_{prefix}', 0):>8.1f} "
                  f"{stats.get(f'min_{prefix}', 0):>8.1f} "
                  f"{stats.get(f'max_{prefix}', 0):>8.1f}")
        e2e_ok = stats.get('e2e_under_50ms_rate', 0)
        print(f"  E2E <50ms 달성률: {e2e_ok:.1f}%  {'OK' if e2e_ok > 95 else 'FAIL'}")
        print()
        print(f"  사람 인식률: {stats.get('detection_rate', 0):.1f}%")
        print(f"  하체 인식률: {stats.get('lower_limb_rate', 0):.1f}%")
        print(f"  하체 평균 Confidence: {stats.get('avg_lower_limb_conf', 0):.3f}")

        # 3D 메트릭
        if stats.get("bone_length_cv"):
            print()
            print(f"  3D 안정성 (Bone Length CV):")
            for bone, cv in stats["bone_length_cv"].items():
                status = "OK" if cv < 0.05 else ("WARN" if cv < 0.10 else "BAD")
                mean_len = stats.get("bone_length_mean", {}).get(bone, 0)
                print(f"    {bone:<20} CV={cv:.4f} ({status})  mean={mean_len:.3f}m")

        if stats.get("symmetry_scores"):
            print(f"  좌우 대칭성:")
            for pair, sc in stats["symmetry_scores"].items():
                status = "OK" if sc["mean"] < 0.05 else "WARN"
                print(f"    {pair:<20} asymmetry={sc['mean']:.3f} ({status})")

        if stats.get("avg_depth_validity") is not None:
            print(f"  Depth 유효율: {stats['avg_depth_validity']:.1%}")

        # 관절 각도
        if stats.get("joint_angle_stats"):
            print()
            print(f"  관절 각도 (degrees):")
            for angle_name, angle_stats in stats["joint_angle_stats"].items():
                print(f"    {angle_name:<30} "
                      f"mean={angle_stats['mean']:>6.1f}  "
                      f"std={angle_stats['std']:>5.1f}  "
                      f"range=[{angle_stats['min']:.0f}, {angle_stats['max']:.0f}]  "
                      f"(n={angle_stats['count']})")

    def _print_comparison(self):
        """전체 비교 결과 테이블 출력"""
        print("\n")
        print("=" * 120)
        print("  모델 비교 결과")
        print("=" * 120)
        print(f"{'모델':<30} {'FPS':>6} {'Infer(ms)':>10} {'E2E(ms)':>9} {'P95 E2E':>8} "
              f"{'<50ms%':>7} {'인식률%':>7} {'하체%':>6} {'Conf':>5} {'Foot':>5}")
        print("-" * 120)

        for name, stats in self.results.items():
            if "error" in stats:
                print(f"{name:<30} {'ERROR':>6}")
                continue

            # foot keypoint 보유 여부
            has_foot = "Yes" if stats.get("joint_angle_stats", {}).get("left_ankle_dorsiflexion") else "No"

            print(f"{name:<30} "
                  f"{stats.get('avg_fps', 0):>6.1f} "
                  f"{stats.get('avg_inference_time_ms', stats.get('avg_latency_ms', 0)):>10.1f} "
                  f"{stats.get('avg_e2e_latency_ms', 0):>9.1f} "
                  f"{stats.get('p95_e2e_latency_ms', 0):>8.1f} "
                  f"{stats.get('e2e_under_50ms_rate', 0):>7.1f} "
                  f"{stats.get('detection_rate', 0):>7.1f} "
                  f"{stats.get('lower_limb_rate', 0):>6.1f} "
                  f"{stats.get('avg_lower_limb_conf', 0):>5.2f} "
                  f"{has_foot:>5}")

        print("-" * 120)
        print()

        # 추천
        valid = [(n, s) for n, s in self.results.items() if "error" not in s]
        if not valid:
            print("  유효한 결과가 없습니다.")
            return

        best_fps = max(valid, key=lambda x: x[1].get('avg_fps', 0))
        best_detect = max(valid, key=lambda x: x[1].get('lower_limb_rate', 0))
        best_e2e = min(valid, key=lambda x: x[1].get('avg_e2e_latency_ms', float('inf')))

        print(f"  최고 FPS:       {best_fps[0]} ({best_fps[1].get('avg_fps', 0):.1f} FPS)")
        print(f"  최저 E2E:       {best_e2e[0]} ({best_e2e[1].get('avg_e2e_latency_ms', 0):.1f} ms)")
        print(f"  최고 하체 인식:  {best_detect[0]} ({best_detect[1].get('lower_limb_rate', 0):.1f}%)")

        # 종합 추천 스코어
        print()
        print("  === 종합 추천 스코어 ===")
        scores = {}
        for name, stats in valid:
            e2e = stats.get('avg_e2e_latency_ms', 100)
            ll_rate = stats.get('lower_limb_rate', 0)
            conf = stats.get('avg_lower_limb_conf', 0)
            has_foot = 1.0 if stats.get("joint_angle_stats", {}).get("left_ankle_dorsiflexion") else 0.0
            fps = stats.get('avg_fps', 0)

            # 스코어 계산 (높을수록 좋음)
            latency_score = max(0, (50 - e2e) / 50) if e2e < 50 else -0.3  # <50ms 필수
            detect_score = ll_rate / 100.0
            conf_score = conf
            foot_score = has_foot
            fps_score = min(fps / 60.0, 1.0)

            # 3D 안정성 (bone CV 평균)
            bone_cvs = list(stats.get("bone_length_cv", {}).values())
            stability_score = max(0, 1.0 - np.mean(bone_cvs) * 10) if bone_cvs else 0.5

            total = (latency_score * 0.30 +
                     detect_score * 0.25 +
                     stability_score * 0.20 +
                     foot_score * 0.15 +
                     fps_score * 0.10)
            scores[name] = total

        for name, score in sorted(scores.items(), key=lambda x: -x[1]):
            marker = " <<<" if score == max(scores.values()) else ""
            print(f"    {name:<30} {score:.3f}{marker}")

    def _save_results(self):
        """결과를 JSON 파일로 저장"""
        output_dir = os.path.join(os.path.dirname(__file__), "results")
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"benchmark_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)

        # numpy 타입을 python 타입으로 변환
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        save_data = {
            "timestamp": timestamp,
            "config": {
                "duration": self.args.duration,
                "camera": "ZED" if self.args.use_zed else "Webcam",
                "resolution": self.args.resolution,
                "lower_only": self.args.lower_only,
                "tensorrt": self.args.tensorrt,
                "video": self.args.video,
                "has_depth": self.args.video is None or (
                    self.args.video and os.path.splitext(self.args.video)[1].lower() in ('.svo2', '.svo')
                ),
            },
            "results": {}
        }

        for name, stats in self.results.items():
            save_data["results"][name] = {
                k: convert(v) for k, v in stats.items()
            }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=convert)

        print(f"\n  결과 저장: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="H-Walker Pose Estimation 벤치마크")

    parser.add_argument("--models", nargs="+",
                        default=["all"],
                        choices=["all", "mediapipe", "yolov8", "rtmpose", "rtmpose_wb", "zed_bt"],
                        help="벤치마크할 모델 (기본: all)")
    parser.add_argument("--duration", type=int, default=15,
                        help="모델당 측정 시간 (초, 기본: 15)")
    parser.add_argument("--no-zed", dest="use_zed", action="store_false",
                        help="ZED 카메라 대신 웹캠 사용")
    parser.add_argument("--resolution", default="SVGA",
                        choices=["SVGA", "VGA", "HD720", "HD1080", "HD1200"],
                        help="카메라 해상도 (기본: SVGA, ZED X Mini 최적)")
    parser.add_argument("--camera-fps", type=int, default=60,
                        help="카메라 FPS (기본: 60, SVGA 최적)")
    parser.add_argument("--visualize", action="store_true",
                        help="실시간 시각화 (FPS 약간 감소)")
    parser.add_argument("--lower-only", action="store_true",
                        help="하체만 보이는 상황 시뮬레이션 (상위 1/3 크롭)")
    parser.add_argument("--tensorrt", action="store_true",
                        help="TensorRT 변환 사용 (YOLOv8, RTMPose)")
    parser.add_argument("--video", type=str, default=None,
                        help="동영상 파일 경로 (카메라 대신 영상 파일로 벤치마크)")

    args = parser.parse_args()

    # --video 사용 시: SVO2는 ZED SDK 유지 (depth 활용), 일반 영상은 ZED 비활성화
    if args.video:
        ext = os.path.splitext(args.video)[1].lower()
        if ext not in ('.svo2', '.svo'):
            args.use_zed = False

    runner = BenchmarkRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
