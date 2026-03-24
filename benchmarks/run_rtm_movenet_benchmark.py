#!/usr/bin/env python3
"""
RTMPose + MoveNet 최고 성능 벤치마크
=====================================
ONNX Runtime GPU 수정 후 RTMPose GPU 성능 + MoveNet 비교

모델:
  - RTMPose lightweight (TRT FP16) — GPU 추론
  - RTMPose lightweight (TRT INT8-nocal) — GPU 추론
  - RTMPose balanced (TRT FP16) — GPU 추론
  - MoveNet Lightning (TFLite) — 초경량
  - MoveNet Thunder (TFLite) — 정확
  - MoveNet Lightning (TRT) — TensorRT 가속

사용법:
  python run_rtm_movenet_benchmark.py --max-perf --video recordings/walking_20260319_160624.mp4 --record
"""

import argparse
import time
import sys
import os
import json
import numpy as np
import cv2
from datetime import datetime
from collections import OrderedDict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zed_camera import create_camera, HAS_ZED
from pose_models import (
    RTMPoseModel, MoveNetModel, draw_pose, PoseResult,
    check_tensorrt_available,
)
from joint_angles import compute_lower_limb_angles
from postprocess_accel import compute_lower_limb_angles_fast, HAS_CPP_EXT


def build_models():
    """RTMPose + MoveNet 모델 매트릭스"""
    models = OrderedDict()

    # RTMPose — GPU (onnxruntime-gpu 필수)
    models["RTMPose-LW / FP16"] = RTMPoseModel(
        mode="lightweight", use_tensorrt=True, precision="fp16")
    models["RTMPose-LW / INT8"] = RTMPoseModel(
        mode="lightweight", use_tensorrt=True, precision="int8", use_calib=False)
    models["RTMPose-Bal / FP16"] = RTMPoseModel(
        mode="balanced", use_tensorrt=True, precision="fp16")
    models["RTMPose-Bal / INT8"] = RTMPoseModel(
        mode="balanced", use_tensorrt=True, precision="int8", use_calib=False)

    # MoveNet — TFLite (CPU) + TRT (GPU)
    models["MoveNet Lightning"] = MoveNetModel(variant="lightning", use_tensorrt=False)
    models["MoveNet Thunder"] = MoveNetModel(variant="thunder", use_tensorrt=False)

    # MoveNet TRT — ONNX 파일이 있으면 추가
    onnx_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "models", "movenet_lightning.onnx")
    if os.path.exists(onnx_path):
        models["MoveNet Lightning (TRT)"] = MoveNetModel(
            variant="lightning", use_tensorrt=True)

    return models


class BenchmarkRunner:
    def __init__(self, args):
        self.args = args
        self.results = OrderedDict()
        self.system_info = {}
        self._jetson_optimized = False

    def run(self):
        models = build_models()

        # --max-perf
        if self.args.max_perf:
            try:
                from jetson_optimizer import optimize_jetson, get_system_info
                print()
                optimize_jetson()
                self.system_info = get_system_info()
                self._jetson_optimized = True
                print()
            except Exception as e:
                print(f"  [WARNING] Jetson 최적화 실패: {e}")

        # ONNX Runtime GPU 상태 확인
        has_trt, has_cuda, providers = check_tensorrt_available()
        print()
        print("=" * 70)
        print("  RTMPose + MoveNet 최고 성능 벤치마크")
        print("=" * 70)
        print(f"  ONNX Runtime providers: {providers}")
        if has_trt:
            print(f"  RTMPose GPU: TensorRT EP 활성")
        elif has_cuda:
            print(f"  RTMPose GPU: CUDA EP 활성 (TRT 없음)")
        else:
            print(f"  ╔═══════════════════════════════════════════════╗")
            print(f"  ║  경고: GPU provider 없음! CPU fallback!       ║")
            print(f"  ║  RTMPose 결과가 매우 느릴 수 있습니다.         ║")
            print(f"  ╚═══════════════════════════════════════════════╝")
        print(f"  모델: {len(models)}개")
        print(f"  측정: {self.args.duration}초/모델, 워밍업 {self.args.warmup_frames}프레임")
        print(f"  Max-Perf: {'ON' if self._jetson_optimized else 'OFF'}")
        if self.args.video:
            print(f"  입력: {self.args.video}")
        if self.system_info:
            print(f"  시스템: {self.system_info.get('model', 'unknown')}")
            print(f"  전력모드: {self.system_info.get('power_mode', 'unknown')}")
        print("=" * 70)

        # 실험 결과 폴더 (JSON + 녹화 모두)
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(
            os.path.dirname(__file__), "results", "rtm_movenet", self._timestamp)
        os.makedirs(self.run_dir, exist_ok=True)
        video_dir = self.run_dir if self.args.record else None
        print(f"  결과 폴더: {self.run_dir}")

        # 카메라
        camera = create_camera(
            use_zed=self.args.use_zed,
            video_path=self.args.video,
            resolution=self.args.resolution,
            fps=self.args.camera_fps,
            depth_mode=self.args.depth_mode,
        )
        camera.open()
        for _ in range(10):
            camera.grab()

        # 벤치마크 실행
        total = len(models)
        for idx, (name, model) in enumerate(models.items(), 1):
            print(f"\n{'─' * 60}")
            print(f"  [{idx}/{total}] {name}")
            print(f"{'─' * 60}")

            try:
                model.load()
            except Exception as e:
                print(f"  로드 실패: {e}")
                self.results[name] = {"error": str(e)}
                continue

            # 비디오면 처음부터 재생
            if self.args.video:
                camera.close()
                camera = create_camera(
                    use_zed=self.args.use_zed,
                    video_path=self.args.video,
                    resolution=self.args.resolution,
                    fps=self.args.camera_fps,
                    depth_mode=self.args.depth_mode,
                )
                camera.open()
                for _ in range(5):
                    camera.grab()

            stats = self._benchmark_model(camera, model, name, video_dir)
            self.results[name] = stats
            self._print_stats(name, stats)

        # 결과 비교
        self._print_comparison()
        self._save_results()
        camera.close()

        # Jetson 복원
        if self._jetson_optimized:
            try:
                from jetson_optimizer import restore_jetson
                restore_jetson()
            except Exception:
                pass

    def _benchmark_model(self, camera, model, name, video_dir):
        """단일 모델 벤치마크"""
        stats = {
            "frame_count": 0,
            "detected_count": 0,
            "lower_limb_count": 0,
            "inference_ms_list": [],
            "e2e_ms_list": [],
            "confidence_list": [],
            "lower_conf_list": [],
            "keypoint_count_list": [],
            "jitter_list": [],
        }

        prev_kps = None
        video_writer = None

        # 비디오 녹화 설정
        if video_dir:
            safe_name = name.replace("/", "_").replace(" ", "_")
            video_path = os.path.join(video_dir, f"{safe_name}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        # 워밍업 (FPS 추정 겸용)
        print(f"  워밍업 ({self.args.warmup_frames}프레임)...")
        warmup_start = time.perf_counter()
        warmup_count = 0
        for _ in range(self.args.warmup_frames):
            if camera.grab():
                rgb = camera.get_rgb()
                if rgb is not None:
                    model.predict(rgb)
                    warmup_count += 1
        warmup_elapsed = time.perf_counter() - warmup_start
        estimated_fps = warmup_count / warmup_elapsed if warmup_elapsed > 0 else 30.0

        # 측정
        print(f"  측정 중 ({self.args.duration}초)...")
        start = time.perf_counter()
        while time.perf_counter() - start < self.args.duration:
            t0 = time.perf_counter()
            if not camera.grab():
                # 비디오 끝나면 처음부터
                if self.args.video:
                    camera.close()
                    camera = create_camera(
                        use_zed=self.args.use_zed,
                        video_path=self.args.video,
                        resolution=self.args.resolution,
                        fps=self.args.camera_fps,
                        depth_mode=self.args.depth_mode,
                    )
                    camera.open()
                    if not camera.grab():
                        break
                else:
                    break
            rgb = camera.get_rgb()
            if rgb is None:
                continue

            t_infer = time.perf_counter()
            result = model.predict(rgb)
            t_done = time.perf_counter()

            infer_ms = (t_done - t_infer) * 1000
            e2e_ms = (t_done - t0) * 1000

            stats["frame_count"] += 1
            stats["inference_ms_list"].append(infer_ms)
            stats["e2e_ms_list"].append(e2e_ms)

            if result.detected:
                stats["detected_count"] += 1
                confs = list(result.confidences.values())
                if confs:
                    stats["confidence_list"].append(np.mean(confs))

                lower_keys = ["left_hip", "right_hip", "left_knee", "right_knee",
                              "left_ankle", "right_ankle"]
                lower_count = sum(1 for k in lower_keys if k in result.keypoints_2d)
                stats["keypoint_count_list"].append(lower_count)
                if lower_count >= 4:
                    stats["lower_limb_count"] += 1
                    lower_confs = [result.confidences.get(k, 0) for k in lower_keys
                                   if k in result.confidences]
                    if lower_confs:
                        stats["lower_conf_list"].append(np.mean(lower_confs))

                # Jitter 계산
                if prev_kps is not None:
                    dists = []
                    for k in result.keypoints_2d:
                        if k in prev_kps:
                            dx = result.keypoints_2d[k][0] - prev_kps[k][0]
                            dy = result.keypoints_2d[k][1] - prev_kps[k][1]
                            dists.append(np.sqrt(dx**2 + dy**2))
                    if dists:
                        stats["jitter_list"].append(np.mean(dists))
                prev_kps = dict(result.keypoints_2d)

            # 녹화
            if video_dir and rgb is not None:
                vis = rgb.copy()
                if result.detected:
                    vis = draw_pose(vis, result)

                elapsed = time.perf_counter() - start
                fps_now = stats["frame_count"] / elapsed if elapsed > 0 else 0
                cv2.putText(vis, f"{name} | {fps_now:.1f} FPS | {infer_ms:.1f}ms",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if video_writer is None:
                    h, w = vis.shape[:2]
                    video_writer = cv2.VideoWriter(video_path, fourcc, round(estimated_fps), (w, h))
                video_writer.write(vis)

        if video_writer is not None:
            video_writer.release()
            print(f"  녹화 저장: {video_path}")

        # 통계 계산
        frame_count = stats["frame_count"]
        if frame_count == 0:
            return {"error": "프레임 없음"}

        infer_arr = np.array(stats["inference_ms_list"])
        e2e_arr = np.array(stats["e2e_ms_list"])

        result_stats = {
            "frame_count": frame_count,
            "avg_fps": frame_count / (sum(stats["e2e_ms_list"]) / 1000) if stats["e2e_ms_list"] else 0,
            "avg_infer_ms": float(np.mean(infer_arr)),
            "p50_infer_ms": float(np.percentile(infer_arr, 50)),
            "p95_infer_ms": float(np.percentile(infer_arr, 95)),
            "p99_infer_ms": float(np.percentile(infer_arr, 99)),
            "avg_e2e_ms": float(np.mean(e2e_arr)),
            "p95_e2e_ms": float(np.percentile(e2e_arr, 95)),
            "detection_rate": 100 * stats["detected_count"] / frame_count,
            "lower_limb_rate": 100 * stats["lower_limb_count"] / frame_count,
            "avg_confidence": float(np.mean(stats["confidence_list"])) if stats["confidence_list"] else 0,
            "avg_jitter_px": float(np.mean(stats["jitter_list"])) if stats["jitter_list"] else 0,
        }

        return result_stats

    def _print_stats(self, name, stats):
        if "error" in stats:
            print(f"  ERROR: {stats['error']}")
            return
        print(f"  FPS: {stats.get('avg_fps', 0):.1f} | "
              f"Infer: {stats.get('avg_infer_ms', 0):.2f}ms (P95={stats.get('p95_infer_ms', 0):.2f}) | "
              f"인식: {stats.get('detection_rate', 0):.0f}% | "
              f"하체: {stats.get('lower_limb_rate', 0):.0f}% | "
              f"Conf: {stats.get('avg_confidence', 0):.4f}")

    def _print_comparison(self):
        valid = {n: s for n, s in self.results.items() if "error" not in s}
        if not valid:
            print("\n  유효한 결과 없음")
            return

        print(f"\n\n{'=' * 90}")
        print("  RTMPose + MoveNet 최고 성능 비교 결과")
        if self._jetson_optimized:
            print("  [Max-Perf ON]")
        print(f"{'=' * 90}")

        # 테이블
        col_fmt = "{:<28} {:>8} {:>10} {:>10} {:>8} {:>8} {:>10}"
        header = col_fmt.format(
            "모델", "FPS", "Infer(ms)", "P95(ms)", "인식%", "하체%", "Conf")
        print(f"\n  {header}")
        print(f"  {'─' * 88}")

        best_fps = 0
        best_name = ""
        for name, s in valid.items():
            row = col_fmt.format(
                name,
                f"{s['avg_fps']:.1f}",
                f"{s['avg_infer_ms']:.2f}",
                f"{s['p95_infer_ms']:.2f}",
                f"{s['detection_rate']:.0f}",
                f"{s['lower_limb_rate']:.0f}",
                f"{s['avg_confidence']:.4f}",
            )
            # 최고 FPS 표시
            marker = ""
            if s['avg_fps'] > best_fps:
                best_fps = s['avg_fps']
                best_name = name
            print(f"  {row}")

        print(f"\n  최고 속도: {best_name} ({best_fps:.1f} FPS)")

        # 120 FPS 도달 여부
        print(f"\n  {'─' * 88}")
        for name, s in valid.items():
            fps = s['avg_fps']
            infer = s['avg_infer_ms']
            if fps >= 120:
                print(f"  ✓ {name}: {fps:.1f} FPS — 120 FPS 달성!")
            elif fps >= 60:
                need_ms = 1000 / 120
                print(f"  △ {name}: {fps:.1f} FPS — 120 FPS 미달 "
                      f"(현재 {infer:.1f}ms, 필요 {need_ms:.1f}ms)")
            else:
                print(f"  ✗ {name}: {fps:.1f} FPS — 120 FPS 불가")
        print()

    def _save_results(self):
        filepath = os.path.join(self.run_dir, "results.json")

        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        save_data = {
            "experiment": "RTMPose + MoveNet Benchmark (GPU fixed)",
            "timestamp": self._timestamp,
            "config": {
                "duration": self.args.duration,
                "warmup_frames": self.args.warmup_frames,
                "video": self.args.video,
                "max_perf": self._jetson_optimized,
            },
            "system_info": self.system_info or None,
            "results": {
                name: {k: convert(v) for k, v in stats.items()}
                for name, stats in self.results.items()
            },
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=convert)
        print(f"  결과 저장: {filepath}")


def main():
    parser = argparse.ArgumentParser(
        description="RTMPose + MoveNet 최고 성능 벤치마크")
    parser.add_argument("--max-perf", action="store_true",
                        help="Jetson 최대 성능 모드")
    parser.add_argument("--duration", type=int, default=15,
                        help="모델당 측정 시간 (초, 기본: 15)")
    parser.add_argument("--warmup-frames", type=int, default=50,
                        help="워밍업 프레임 수 (기본: 50)")
    parser.add_argument("--video", type=str, default=None,
                        help="비디오 파일")
    parser.add_argument("--depth-mode", default="PERFORMANCE",
                        help="ZED 깊이 모드 (기본: PERFORMANCE)")
    parser.add_argument("--no-zed", dest="use_zed", action="store_false",
                        help="ZED 대신 웹캠 사용")
    parser.add_argument("--resolution", default="SVGA",
                        help="카메라 해상도 (기본: SVGA)")
    parser.add_argument("--camera-fps", type=int, default=120,
                        help="카메라 FPS (기본: 120)")
    parser.add_argument("--record", action="store_true",
                        help="모델별 추론 결과 비디오 녹화")

    args = parser.parse_args()
    runner = BenchmarkRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
