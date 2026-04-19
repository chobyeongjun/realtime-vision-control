#!/usr/bin/env python3
"""
INT8 Quantization 비교 실험
===========================
4개 모델 × 3개 정밀도 = 12 variants 비교

모델:
  - YOLOv8n-Pose (TRT) - 경량
  - YOLOv8s-Pose (TRT) - 중형
  - YOLO11n-Pose (TRT) - 경량
  - YOLO11s-Pose (TRT) - 중형

정밀도:
  - FP16 (기준선)
  - INT8 no-calibration (캘리브레이션 없이 INT8)
  - INT8 calibrated (실제 환경 영상으로 캘리브레이션)

측정 항목:
  - 속도: FPS, Inference Latency (ms), P95/P99
  - 정확도: 인식률(%), Keypoint Confidence, 하체 인식률
  - 안정성: Confidence 표준편차, 프레임 간 Keypoint Jitter

사용법:
  # 전체 9종 비교 (max-perf 권장)
  python run_int8_comparison.py --max-perf

  # GPU/CPU 모니터링 (별도 터미널에서 실행)
  sudo tegrastats --interval 1000

  # 캘리브레이션 데이터 지정 (걷기 영상에서 추출한 이미지)
  python run_int8_comparison.py --max-perf --calib-data ./calib_images/

  # FP16 vs INT8-nocal만 비교
  python run_int8_comparison.py --max-perf --skip-calibrated

  # 비디오 파일로 동일 입력 비교
  python run_int8_comparison.py --max-perf --video recording.mp4

  # 측정 시간 변경
  python run_int8_comparison.py --max-perf --duration 30

참고:
  - --max-perf: MAXN 전력모드, jetson_clocks, fan 최대, GPU 전력관리 OFF
  - INT8(cal)은 실제 환경 영상으로 캘리브레이션해야 정확합니다.
  - calibrate_yolo.py로 걷기 영상에서 캘리브레이션 이미지를 먼저 추출하세요.
  - 첫 실행 시 TRT 엔진 빌드로 수 분 소요됩니다. 이후 캐시 재사용.
"""

import argparse
import csv
import time
import sys
import os
import json
import numpy as np
import cv2
from datetime import datetime
from collections import OrderedDict

import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zed_camera import create_camera, GPUPreprocessor, HAS_ZED
from pose_models import YOLOv8Pose, draw_pose, PoseResult
from joint_angles import compute_lower_limb_angles
from postprocess_accel import compute_lower_limb_angles_fast, HAS_CPP_EXT


def center_crop_width(frame, target_width):
    """프레임 가로 중앙 크롭. target_width=0이면 원본 반환."""
    if target_width <= 0:
        return frame
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    offset = (w - target_width) // 2
    return frame[:, offset:offset + target_width]


# ============================================================================
# GPU/CPU/Thermal 모니터링 (Jetson + Desktop 호환)
# ============================================================================
class SystemMonitor:
    """백그라운드 스레드에서 GPU/CPU 사용률, 온도, 메모리를 주기적으로 기록"""

    def __init__(self, interval_s=0.5):
        self.interval_s = interval_s
        self._running = False
        self._thread = None
        self.samples = []  # [{timestamp, gpu_util, cpu_util, gpu_temp, gpu_mem_used_mb, gpu_mem_total_mb, cpu_freq_mhz}]
        self._is_jetson = os.path.exists("/sys/devices/gpu.0") or os.path.exists("/sys/devices/platform/gpu.0")

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        while self._running:
            sample = {"timestamp": time.perf_counter()}
            try:
                if self._is_jetson:
                    self._read_jetson(sample)
                else:
                    self._read_desktop(sample)
            except Exception:
                pass
            # CPU utilization (cross-platform)
            try:
                with open("/proc/stat", "r") as f:
                    line = f.readline()
                parts = line.split()
                idle = int(parts[4])
                total = sum(int(x) for x in parts[1:])
                if hasattr(self, "_prev_cpu"):
                    d_total = total - self._prev_cpu[0]
                    d_idle = idle - self._prev_cpu[1]
                    sample["cpu_util"] = (1.0 - d_idle / max(d_total, 1)) * 100
                self._prev_cpu = (total, idle)
            except Exception:
                pass
            self.samples.append(sample)
            time.sleep(self.interval_s)

    def _read_jetson(self, sample):
        """Jetson sysfs에서 GPU 사용률/온도/메모리 읽기"""
        # GPU utilization
        for path in ["/sys/devices/gpu.0/load", "/sys/devices/platform/gpu.0/load"]:
            try:
                with open(path) as f:
                    sample["gpu_util"] = int(f.read().strip()) / 10.0  # 0~1000 → 0~100%
                break
            except (FileNotFoundError, ValueError):
                continue
        # GPU temperature
        try:
            for tz_dir in sorted(os.listdir("/sys/class/thermal/")):
                if tz_dir.startswith("thermal_zone"):
                    tz_path = f"/sys/class/thermal/{tz_dir}"
                    with open(f"{tz_path}/type") as f:
                        tz_type = f.read().strip().lower()
                    if "gpu" in tz_type:
                        with open(f"{tz_path}/temp") as f:
                            sample["gpu_temp"] = int(f.read().strip()) / 1000.0
                        break
        except Exception:
            pass
        # CPU temperature
        try:
            for tz_dir in sorted(os.listdir("/sys/class/thermal/")):
                if tz_dir.startswith("thermal_zone"):
                    tz_path = f"/sys/class/thermal/{tz_dir}"
                    with open(f"{tz_path}/type") as f:
                        tz_type = f.read().strip().lower()
                    if "cpu" in tz_type:
                        with open(f"{tz_path}/temp") as f:
                            sample["cpu_temp"] = int(f.read().strip()) / 1000.0
                        break
        except Exception:
            pass
        # GPU memory (Jetson shared memory — use torch)
        try:
            import torch
            if torch.cuda.is_available():
                sample["gpu_mem_used_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
                sample["gpu_mem_total_mb"] = torch.cuda.get_device_properties(0).total_mem / 1024 / 1024
        except Exception:
            pass

    def _read_desktop(self, sample):
        """Desktop NVIDIA GPU — pynvml 사용"""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            sample["gpu_util"] = util.gpu
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            sample["gpu_temp"] = temp
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            sample["gpu_mem_used_mb"] = mem.used / 1024 / 1024
            sample["gpu_mem_total_mb"] = mem.total / 1024 / 1024
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                sample["gpu_mem_used_mb"] = torch.cuda.memory_allocated() / 1024 / 1024
        except Exception:
            pass

    def get_summary(self):
        """모니터링 요약 통계"""
        if not self.samples:
            return {}
        summary = {}
        for key in ["gpu_util", "cpu_util", "gpu_temp", "cpu_temp", "gpu_mem_used_mb"]:
            vals = [s[key] for s in self.samples if key in s]
            if vals:
                a = np.array(vals)
                summary[f"avg_{key}"] = float(np.mean(a))
                summary[f"max_{key}"] = float(np.max(a))
                summary[f"min_{key}"] = float(np.min(a))
        return summary

    def save_csv(self, path):
        """모니터링 샘플을 CSV로 저장"""
        if not self.samples:
            return
        keys = ["timestamp", "gpu_util", "cpu_util", "gpu_temp", "cpu_temp",
                "gpu_mem_used_mb", "gpu_mem_total_mb"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(keys)
            for s in self.samples:
                writer.writerow([f"{s.get(k, '')}" for k in keys])


# ============================================================================
# 실험 매트릭스 정의
# ============================================================================
PRECISIONS = ["FP16", "INT8-nocal", "INT8-cal"]

def build_model_matrix(skip_calibrated=False, calib_data=None):
    """
    4모델 × 3정밀도 매트릭스 생성 (YOLOv8n, YOLOv8s, YOLO11n, YOLO11s)

    Args:
        skip_calibrated: True면 INT8-cal 스킵
        calib_data: INT8 캘리브레이션 데이터 경로 (이미지 디렉토리)

    Returns:
        OrderedDict: { "YOLOv8n / FP16": model_instance, ... }
    """
    models = OrderedDict()

    precisions = PRECISIONS if not skip_calibrated else ["FP16", "INT8-nocal"]

    # YOLOv8 모델들
    for size in ["n", "s"]:
        for prec in precisions:
            key = f"YOLOv8{size} / {prec}"
            if prec == "FP16":
                models[key] = YOLOv8Pose(
                    model_size=size, use_tensorrt=True,
                    precision="fp16",
                )
            elif prec == "INT8-nocal":
                models[key] = YOLOv8Pose(
                    model_size=size, use_tensorrt=True,
                    precision="int8", use_calib=False,
                )
            elif prec == "INT8-cal":
                models[key] = YOLOv8Pose(
                    model_size=size, use_tensorrt=True,
                    precision="int8", use_calib=True,
                    calib_data=calib_data,
                )

    # YOLO11 모델들
    for size in ["n", "s"]:
        for prec in precisions:
            key = f"YOLO11{size} / {prec}"
            if prec == "FP16":
                models[key] = YOLOv8Pose(
                    model_size=size, use_tensorrt=True,
                    precision="fp16", yolo_version="v11",
                )
            elif prec == "INT8-nocal":
                models[key] = YOLOv8Pose(
                    model_size=size, use_tensorrt=True,
                    precision="int8", use_calib=False,
                    yolo_version="v11",
                )
            elif prec == "INT8-cal":
                models[key] = YOLOv8Pose(
                    model_size=size, use_tensorrt=True,
                    precision="int8", use_calib=True,
                    calib_data=calib_data, yolo_version="v11",
                )

    return models


# ============================================================================
# 벤치마크 실행
# ============================================================================
class Int8ComparisonRunner:
    """4모델 × 3정밀도 INT8 비교 실험 (YOLOv8n, YOLOv8s, YOLO11n, YOLO11s)"""

    MODEL_FAMILIES = ["YOLOv8n", "YOLOv8s", "YOLO11n", "YOLO11s"]

    def __init__(self, args):
        self.args = args
        self.results = OrderedDict()
        self.system_info = {}
        self._jetson_optimized = False

    def _make_run_dir(self):
        """실험 결과 폴더 생성: results/int8_comparison/YYYYMMDD_HHMMSS/"""
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(
            os.path.dirname(__file__), "results", "int8_comparison", self._timestamp)
        os.makedirs(run_dir, exist_ok=True)
        return run_dir

    def run(self):
        skip_cal = self.args.skip_calibrated
        precisions = PRECISIONS if not skip_cal else ["FP16", "INT8-nocal"]
        n_variants = len(self.MODEL_FAMILIES) * len(precisions)

        # --max-perf: Jetson 시스템 최적화
        if self.args.max_perf:
            try:
                from jetson_optimizer import optimize_jetson, get_system_info, restore_jetson
                print()
                optimize_jetson()
                self.system_info = get_system_info()
                self._jetson_optimized = True
                print()
            except Exception as e:
                print(f"  [WARNING] Jetson 최적화 실패 (sudo 필요?): {e}")
                print(f"  [WARNING] --max-perf 없이 계속 진행합니다.")
        else:
            print("\n  [INFO] --max-perf 미사용: GPU 클럭이 동적 조절되어 결과가 불안정할 수 있습니다.")
            print(f"  [INFO] 정확한 벤치마크를 위해 --max-perf 사용을 권장합니다.\n")

        # 실험 결과 폴더 (JSON + 녹화 영상 모두 여기에)
        self.run_dir = self._make_run_dir()
        self.video_dir = self.run_dir if self.args.record else None

        # GPU 전처리 + 비동기 캡처 (max-perf 모드)
        use_async = self.args.max_perf
        self._gpu_preproc = GPUPreprocessor() if self.args.max_perf else None

        print("=" * 70)
        print("  INT8 Quantization 비교 실험 (YOLOv8 + YOLO11)")
        print("=" * 70)
        print(f"  모델: YOLOv8n, YOLOv8s, YOLO11n, YOLO11s")
        print(f"  정밀도: {' / '.join(precisions)}")
        print(f"  총 {n_variants}개 variant 비교")
        print(f"  측정: {self.args.duration}초/모델, 워밍업 {self.args.warmup_frames}프레임")
        print(f"  Max-Perf: {'ON' if self._jetson_optimized else 'OFF'}")
        if use_async:
            print(f"  비동기 캡처: ON (파이프라인 병렬화)")
        if self._gpu_preproc:
            print(f"  GPU 전처리: ON")
        if self.args.video:
            print(f"  입력: {self.args.video}")
        else:
            print(f"  입력: {'ZED' if self.args.use_zed and HAS_ZED else 'Webcam'}")
        print(f"  결과 폴더: {self.run_dir}")
        if self.video_dir:
            print(f"  녹화: ON (스켈레톤 포함)")
        if self.system_info:
            print(f"  시스템: {self.system_info.get('model', 'unknown')}")
            print(f"  전력모드: {self.system_info.get('power_mode', 'unknown')}")
            gpu_freq = self.system_info.get('gpu_freq_mhz')
            if gpu_freq:
                print(f"  GPU 클럭: {gpu_freq} MHz")
        print("=" * 70)
        print()

        depth_mode = self.args.depth_mode

        # 카메라
        camera = create_camera(
            use_zed=self.args.use_zed,
            video_path=self.args.video,
            resolution=self.args.resolution,
            fps=self.args.camera_fps,
            depth_mode=depth_mode,
            async_capture=use_async,
        )
        camera.open()
        for _ in range(10):
            camera.grab()

        # 모델 매트릭스 생성
        calib_data = getattr(self.args, 'calib_data', None)
        models = build_model_matrix(skip_calibrated=skip_cal, calib_data=calib_data)

        # 벤치마크 실행
        total = len(models)
        for idx, (name, model) in enumerate(models.items(), 1):
            print(f"\n{'─'*60}")
            print(f"  [{idx}/{total}] {name}")
            print(f"{'─'*60}")

            try:
                model.load()
            except Exception as e:
                print(f"  로드 실패: {e}")
                self.results[name] = {"error": str(e)}
                continue

            # 비디오면 처음부터 다시 재생
            if self.args.video:
                camera.close()
                camera = create_camera(
                    use_zed=self.args.use_zed,
                    video_path=self.args.video,
                    resolution=self.args.resolution,
                    fps=self.args.camera_fps,
                    depth_mode=depth_mode,
                    async_capture=use_async,
                )
                camera.open()
                for _ in range(5):
                    camera.grab()

            stats = self._benchmark_model(camera, model, name)
            self.results[name] = stats
            self._print_stats(name, stats)

        # 비교 결과
        self._print_matrix_comparison()
        self._save_results()
        camera.close()

        # 녹화 리뷰 재생
        if self.video_dir and self.args.review:
            self._review_recordings()

        # Jetson 시스템 복원
        if self._jetson_optimized:
            try:
                from jetson_optimizer import restore_jetson
                restore_jetson()
            except Exception as e:
                print(f"  [WARNING] Jetson 복원 실패: {e}")

    def _benchmark_model(self, camera, model, name):
        """단일 모델 벤치마크 (상세 타이밍 로깅 + 시스템 모니터링 포함)"""
        # 시스템 모니터링 시작
        sysmon = SystemMonitor(interval_s=0.5)
        sysmon.start()
        stats = {
            "frame_count": 0,
            "detected_count": 0,
            "lower_limb_count": 0,
            "inference_ms_list": [],
            "e2e_ms_list": [],
            "confidence_list": [],
            "lower_conf_list": [],
            "keypoint_count_list": [],
            "prev_keypoints": None,
            "jitter_list": [],
            # 상세 타이밍 분해 리스트
            "grab_ms_list": [],
            "get_rgb_ms_list": [],
            "yolo_async_ms_list": [],
            "yolo_forward_ms_list": [],
            "gpu_sync_ms_list": [],
            "bbox_extract_ms_list": [],
            "filter_ms_list": [],
            "constraint_ms_list": [],
            "angle_ms_list": [],
            "total_predict_ms_list": [],
            "vis_ms_list": [],
        }

        duration = self.args.duration
        warmup_frames = self.args.warmup_frames

        # 워밍업 (FPS 추정 겸용)
        crop_w = getattr(self.args, 'center_crop', 0)
        print(f"  워밍업 ({warmup_frames}프레임)...")
        warmup_start = time.perf_counter()
        warmup_count = 0
        for _ in range(warmup_frames):
            if camera.grab():
                rgb = center_crop_width(camera.get_rgb(), crop_w)
                model.predict(rgb)
                warmup_count += 1
        warmup_elapsed = time.perf_counter() - warmup_start
        estimated_fps = warmup_count / warmup_elapsed if warmup_elapsed > 0 else 30.0

        # 비디오 녹화 설정
        video_writer = None
        if self.video_dir:
            safe_name = name.replace(" ", "_").replace("/", "_")
            video_path = os.path.join(self.video_dir, f"{safe_name}.mp4")

        # per-frame CSV 로깅 설정
        csv_writer = None
        csv_file = None
        if self.video_dir:
            safe_name = name.replace(" ", "_").replace("/", "_")
            csv_path = os.path.join(self.video_dir, f"{safe_name}_frames.csv")
            csv_file = open(csv_path, "w", newline="")
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                "frame", "timestamp_s",
                "grab_ms", "get_rgb_ms",
                "yolo_async_ms", "gpu_sync_ms", "yolo_forward_ms",
                "bbox_extract_ms", "filter_ms", "constraint_ms", "angle_ms",
                "total_predict_ms", "infer_ms", "e2e_ms",
                "vis_ms", "detected", "num_keypoints", "avg_conf",
            ])

        # 측정
        print(f"  측정 중 ({duration}초)...")
        start = time.perf_counter()

        try:
            while time.perf_counter() - start < duration:
                frame_no = stats["frame_count"]

                # ── Camera grab ──
                t0 = time.perf_counter()
                if not camera.grab():
                    if self.args.video:
                        break
                    continue
                t_grabbed = time.perf_counter()
                grab_ms = (t_grabbed - t0) * 1000

                # ── Camera get_rgb + center crop ──
                rgb = center_crop_width(camera.get_rgb(), crop_w)
                t_rgb = time.perf_counter()
                get_rgb_ms = (t_rgb - t_grabbed) * 1000

                # ── Model predict (내부 상세 타이밍 포함) ──
                t_infer = time.perf_counter()
                result = model.predict(rgb)
                t_done = time.perf_counter()

                infer_ms = (t_done - t_infer) * 1000
                e2e_ms = (t_done - t0) * 1000

                # ── 시각화 / 녹화 타이밍 ──
                t_vis_start = time.perf_counter()
                if self.args.visualize or self.video_dir:
                    vis = draw_pose(rgb.copy(), result)
                    fps_now = (stats["frame_count"] + 1) / max(time.perf_counter() - start, 0.001)
                    cv2.putText(vis, f"{name} | {infer_ms:.1f}ms | {fps_now:.1f}FPS",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                    if self.video_dir:
                        if video_writer is None:
                            h, w = vis.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            # 워밍업에서 측정한 실제 FPS로 저장 (정수로 반올림, MPEG4 timebase 제한)
                            video_writer = cv2.VideoWriter(video_path, fourcc, round(estimated_fps), (w, h))
                        video_writer.write(vis)

                    if self.args.visualize:
                        cv2.imshow("INT8 Comparison", vis)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                t_vis_end = time.perf_counter()
                vis_ms = (t_vis_end - t_vis_start) * 1000

                # ── 통계 수집 ──
                stats["frame_count"] += 1
                stats["inference_ms_list"].append(infer_ms)
                stats["e2e_ms_list"].append(e2e_ms)
                stats["grab_ms_list"].append(grab_ms)
                stats["get_rgb_ms_list"].append(get_rgb_ms)
                stats["vis_ms_list"].append(vis_ms)

                # predict() 내부 타이밍
                td = result.timing_detail
                stats["yolo_async_ms_list"].append(td["yolo_async_ms"])
                stats["yolo_forward_ms_list"].append(td["yolo_forward_ms"])
                stats["gpu_sync_ms_list"].append(td["gpu_sync_ms"])
                stats["bbox_extract_ms_list"].append(td["bbox_extract_ms"])
                stats["filter_ms_list"].append(td["filter_ms"])
                stats["constraint_ms_list"].append(td["constraint_ms"])
                stats["angle_ms_list"].append(td["angle_ms"])
                stats["total_predict_ms_list"].append(td["total_predict_ms"])

                avg_conf = 0.0
                kp_count = 0
                if result.detected:
                    stats["detected_count"] += 1
                    kp_count = len(result.keypoints_2d)
                    stats["keypoint_count_list"].append(kp_count)

                    if result.confidences:
                        confs = [c for c in result.confidences.values() if c > 0]
                        if confs:
                            avg_conf = np.mean(confs)
                            stats["confidence_list"].append(avg_conf)

                    lower_keys = ["left_hip", "right_hip", "left_knee", "right_knee",
                                  "left_ankle", "right_ankle"]
                    lower_confs = [result.confidences.get(k, 0) for k in lower_keys]
                    valid_lower = [c for c in lower_confs if c > 0]
                    if len(valid_lower) >= 4:
                        stats["lower_limb_count"] += 1
                        stats["lower_conf_list"].append(np.mean(valid_lower))

                    # Jitter
                    curr_kps = {k: np.array([x, y]) for k, (x, y) in result.keypoints_2d.items()}
                    if stats["prev_keypoints"] is not None:
                        dists = [np.linalg.norm(curr_kps[k] - stats["prev_keypoints"][k])
                                 for k in curr_kps if k in stats["prev_keypoints"]]
                        if dists:
                            stats["jitter_list"].append(np.mean(dists))
                    stats["prev_keypoints"] = curr_kps

                # ── per-frame CSV 기록 ──
                if csv_writer is not None:
                    csv_writer.writerow([
                        frame_no, f"{time.perf_counter() - start:.4f}",
                        f"{grab_ms:.3f}", f"{get_rgb_ms:.3f}",
                        f"{td['yolo_async_ms']:.3f}", f"{td['gpu_sync_ms']:.3f}", f"{td['yolo_forward_ms']:.3f}",
                        f"{td['bbox_extract_ms']:.3f}", f"{td['filter_ms']:.3f}",
                        f"{td['constraint_ms']:.3f}", f"{td['angle_ms']:.3f}",
                        f"{td['total_predict_ms']:.3f}", f"{infer_ms:.3f}", f"{e2e_ms:.3f}",
                        f"{vis_ms:.3f}", int(result.detected), kp_count, f"{avg_conf:.4f}",
                    ])

        finally:
            # 정리 (예외 발생해도 반드시 실행)
            if csv_file is not None:
                csv_file.close()
                print(f"  프레임 로그: {csv_path}")

            if video_writer is not None:
                video_writer.release()
                print(f"  녹화 저장: {video_path}")

            if self.args.visualize:
                cv2.destroyAllWindows()

        # 시스템 모니터링 중지 및 결과 수집
        sysmon.stop()
        stats["system_monitor"] = sysmon.get_summary()

        # 시스템 모니터링 CSV 저장
        if self.video_dir:
            safe_name = name.replace(" ", "_").replace("/", "_")
            sysmon_path = os.path.join(self.video_dir, f"{safe_name}_sysmon.csv")
            sysmon.save_csv(sysmon_path)
            print(f"  시스템 모니터: {sysmon_path}")

        stats.pop("prev_keypoints", None)
        self._compute_stats(stats)
        return stats

    def _compute_stats(self, stats):
        """통계 계산 (상세 타이밍 분해 포함)"""
        fc = stats["frame_count"]
        if fc == 0:
            return

        infer = np.array(stats["inference_ms_list"])
        e2e = np.array(stats["e2e_ms_list"])

        stats["avg_fps"] = fc / (np.sum(e2e) / 1000) if len(e2e) > 0 else 0
        stats["avg_infer_ms"] = float(np.mean(infer))
        stats["p50_infer_ms"] = float(np.percentile(infer, 50))
        stats["p95_infer_ms"] = float(np.percentile(infer, 95))
        stats["p99_infer_ms"] = float(np.percentile(infer, 99))
        stats["avg_e2e_ms"] = float(np.mean(e2e))
        stats["p95_e2e_ms"] = float(np.percentile(e2e, 95))

        stats["detection_rate"] = stats["detected_count"] / fc * 100
        stats["lower_limb_rate"] = stats["lower_limb_count"] / fc * 100

        stats["avg_confidence"] = float(np.mean(stats["confidence_list"])) if stats["confidence_list"] else 0.0
        stats["std_confidence"] = float(np.std(stats["confidence_list"])) if stats["confidence_list"] else 0.0
        stats["avg_lower_conf"] = float(np.mean(stats["lower_conf_list"])) if stats["lower_conf_list"] else 0.0
        stats["avg_keypoint_count"] = float(np.mean(stats["keypoint_count_list"])) if stats["keypoint_count_list"] else 0.0
        stats["avg_jitter_px"] = float(np.mean(stats["jitter_list"])) if stats["jitter_list"] else 0.0
        stats["p95_jitter_px"] = float(np.percentile(stats["jitter_list"], 95)) if stats["jitter_list"] else 0.0

        # ── 상세 타이밍 분해 통계 ──
        detail_keys = [
            "grab_ms_list", "get_rgb_ms_list",
            "yolo_async_ms_list", "yolo_forward_ms_list", "gpu_sync_ms_list",
            "bbox_extract_ms_list", "filter_ms_list",
            "constraint_ms_list", "angle_ms_list",
            "total_predict_ms_list", "vis_ms_list",
        ]
        for key in detail_keys:
            arr = stats.get(key, [])
            if arr:
                prefix = key.replace("_list", "")
                a = np.array(arr)
                stats[f"avg_{prefix}"] = float(np.mean(a))
                stats[f"p50_{prefix}"] = float(np.percentile(a, 50))
                stats[f"p95_{prefix}"] = float(np.percentile(a, 95))
                stats[f"max_{prefix}"] = float(np.max(a))

        # 리스트 축소 (JSON 크기 제한)
        all_list_keys = [
            "inference_ms_list", "e2e_ms_list", "confidence_list",
            "lower_conf_list", "keypoint_count_list", "jitter_list",
        ] + detail_keys
        for key in all_list_keys:
            if key in stats:
                stats[key] = stats[key][:100]

    def _print_stats(self, name, stats):
        """개별 결과 출력 (상세 타이밍 분해 포함)"""
        if "error" in stats:
            print(f"  ERROR: {stats['error']}")
            return

        print(f"  FPS: {stats.get('avg_fps', 0):.1f} | "
              f"Infer: {stats.get('avg_infer_ms', 0):.2f}ms (P95={stats.get('p95_infer_ms', 0):.2f}) | "
              f"인식: {stats.get('detection_rate', 0):.0f}% | "
              f"하체: {stats.get('lower_limb_rate', 0):.0f}% | "
              f"Conf: {stats.get('avg_confidence', 0):.4f}")

        # 상세 타이밍 분해 출력
        print(f"  ┌─ 타이밍 분해 (avg / P95 / max) ─────────────────")
        print(f"  │ camera.grab()     : {stats.get('avg_grab_ms', 0):6.2f} / {stats.get('p95_grab_ms', 0):6.2f} / {stats.get('max_grab_ms', 0):6.2f} ms")
        print(f"  │ camera.get_rgb()  : {stats.get('avg_get_rgb_ms', 0):6.2f} / {stats.get('p95_get_rgb_ms', 0):6.2f} / {stats.get('max_get_rgb_ms', 0):6.2f} ms")
        print(f"  │ YOLO async ret   : {stats.get('avg_yolo_async_ms', 0):6.2f} / {stats.get('p95_yolo_async_ms', 0):6.2f} / {stats.get('max_yolo_async_ms', 0):6.2f} ms  ← 비동기 리턴")
        print(f"  │ GPU sync wait    : {stats.get('avg_gpu_sync_ms', 0):6.2f} / {stats.get('p95_gpu_sync_ms', 0):6.2f} / {stats.get('max_gpu_sync_ms', 0):6.2f} ms  ← 추가 대기")
        print(f"  │ YOLO total       : {stats.get('avg_yolo_forward_ms', 0):6.2f} / {stats.get('p95_yolo_forward_ms', 0):6.2f} / {stats.get('max_yolo_forward_ms', 0):6.2f} ms  ← 실제 GPU 완료")
        print(f"  │ bbox/kp extract  : {stats.get('avg_bbox_extract_ms', 0):6.2f} / {stats.get('p95_bbox_extract_ms', 0):6.2f} / {stats.get('max_bbox_extract_ms', 0):6.2f} ms  ← CPU")
        print(f"  │ OneEuro filter   : {stats.get('avg_filter_ms', 0):6.2f} / {stats.get('p95_filter_ms', 0):6.2f} / {stats.get('max_filter_ms', 0):6.2f} ms  ← CPU")
        print(f"  │ segment constr.  : {stats.get('avg_constraint_ms', 0):6.2f} / {stats.get('p95_constraint_ms', 0):6.2f} / {stats.get('max_constraint_ms', 0):6.2f} ms  ← CPU")
        print(f"  │ angle calc       : {stats.get('avg_angle_ms', 0):6.2f} / {stats.get('p95_angle_ms', 0):6.2f} / {stats.get('max_angle_ms', 0):6.2f} ms  ← CPU")
        print(f"  │ predict() 전체   : {stats.get('avg_total_predict_ms', 0):6.2f} / {stats.get('p95_total_predict_ms', 0):6.2f} / {stats.get('max_total_predict_ms', 0):6.2f} ms")
        print(f"  │ 시각화/녹화      : {stats.get('avg_vis_ms', 0):6.2f} / {stats.get('p95_vis_ms', 0):6.2f} / {stats.get('max_vis_ms', 0):6.2f} ms")
        print(f"  │ E2E (grab→done) : {stats.get('avg_e2e_ms', 0):6.2f} / {stats.get('p95_e2e_ms', 0):6.2f} ms")
        print(f"  ├─ 시스템 리소스 ─────────────────────────────────")
        sm = stats.get("system_monitor", {})
        if sm:
            print(f"  │ GPU 사용률       : avg {sm.get('avg_gpu_util', 0):5.1f}% / max {sm.get('max_gpu_util', 0):5.1f}%")
            print(f"  │ CPU 사용률       : avg {sm.get('avg_cpu_util', 0):5.1f}% / max {sm.get('max_cpu_util', 0):5.1f}%")
            if "avg_gpu_temp" in sm:
                print(f"  │ GPU 온도        : avg {sm.get('avg_gpu_temp', 0):5.1f}C / max {sm.get('max_gpu_temp', 0):5.1f}C")
            if "avg_cpu_temp" in sm:
                print(f"  │ CPU 온도        : avg {sm.get('avg_cpu_temp', 0):5.1f}C / max {sm.get('max_cpu_temp', 0):5.1f}C")
            if "avg_gpu_mem_used_mb" in sm:
                print(f"  │ GPU 메모리      : avg {sm.get('avg_gpu_mem_used_mb', 0):5.0f}MB / max {sm.get('max_gpu_mem_used_mb', 0):5.0f}MB")
        else:
            print(f"  │ (시스템 모니터링 데이터 없음)")
        print(f"  └───────────────────────────────────────────────")

    # ========================================================================
    # 매트릭스 비교 출력 (핵심)
    # ========================================================================
    def _print_matrix_comparison(self):
        """3모델 × 3정밀도 매트릭스 형태로 비교 결과 출력"""
        valid = {n: s for n, s in self.results.items() if "error" not in s}
        if not valid:
            print("\n  유효한 결과가 없습니다.")
            return

        skip_cal = self.args.skip_calibrated
        precisions = PRECISIONS if not skip_cal else ["FP16", "INT8-nocal"]

        print(f"\n\n{'='*95}")
        print("  INT8 Quantization 비교 결과 (YOLOv8 × YOLO11 × 3정밀도)")
        if self._jetson_optimized:
            print("  [Max-Perf ON: MAXN, jetson_clocks, fan max, GPU power mgmt OFF]")
        print(f"{'='*95}")

        # ---- 속도 매트릭스 ----
        self._print_metric_matrix(
            "속도 - Inference Latency (ms)",
            valid, precisions,
            metric_key="avg_infer_ms",
            fmt=".2f",
            lower_is_better=True,
            show_speedup=True,
        )

        # ---- FPS 매트릭스 ----
        self._print_metric_matrix(
            "속도 - FPS",
            valid, precisions,
            metric_key="avg_fps",
            fmt=".1f",
            lower_is_better=False,
        )

        # ---- P95 Latency 매트릭스 ----
        self._print_metric_matrix(
            "속도 - P95 Inference Latency (ms)",
            valid, precisions,
            metric_key="p95_infer_ms",
            fmt=".2f",
            lower_is_better=True,
        )

        # ---- 인식률 매트릭스 ----
        self._print_metric_matrix(
            "정확도 - 인식률 (%)",
            valid, precisions,
            metric_key="detection_rate",
            fmt=".1f",
            lower_is_better=False,
        )

        # ---- 하체 인식률 매트릭스 ----
        self._print_metric_matrix(
            "정확도 - 하체 인식률 (%)",
            valid, precisions,
            metric_key="lower_limb_rate",
            fmt=".1f",
            lower_is_better=False,
        )

        # ---- Confidence 매트릭스 ----
        self._print_metric_matrix(
            "정확도 - Keypoint Confidence",
            valid, precisions,
            metric_key="avg_confidence",
            fmt=".4f",
            lower_is_better=False,
            show_delta=True,
        )

        # ---- Jitter 매트릭스 ----
        self._print_metric_matrix(
            "안정성 - Keypoint Jitter (px)",
            valid, precisions,
            metric_key="avg_jitter_px",
            fmt=".2f",
            lower_is_better=True,
        )

        # ---- 모델별 요약 ----
        self._print_summary(valid, precisions)

    def _print_metric_matrix(self, title, valid, precisions,
                              metric_key, fmt, lower_is_better,
                              show_speedup=False, show_delta=False):
        """단일 지표에 대한 매트릭스 테이블 출력"""
        col_w = 16
        prec_header = "".join(f"{p:>{col_w}}" for p in precisions)
        print(f"\n  [{title}]")
        print(f"  {'모델':<16}{prec_header}")
        print(f"  {'─' * (16 + col_w * len(precisions))}")

        for family in self.MODEL_FAMILIES:
            row = f"  {family:<16}"
            fp16_val = None

            for prec in precisions:
                key = f"{family} / {prec}"
                if key in valid:
                    val = valid[key].get(metric_key, 0)
                    cell = f"{val:{fmt}}"

                    if prec == "FP16":
                        fp16_val = val

                    # speedup 표시 (latency용)
                    if show_speedup and fp16_val and prec != "FP16" and val > 0:
                        ratio = fp16_val / val
                        cell += f" ({ratio:.1f}x)"

                    # delta 표시 (confidence용)
                    if show_delta and fp16_val is not None and prec != "FP16":
                        delta = val - fp16_val
                        cell += f" ({delta:+.4f})"

                    row += f"{cell:>{col_w}}"
                else:
                    row += f"{'N/A':>{col_w}}"

            print(row)

    def _print_summary(self, valid, precisions):
        """모델별 INT8 효과 요약"""
        print(f"\n  {'='*70}")
        print(f"  [모델별 INT8 효과 요약]")
        print(f"  {'='*70}")

        for family in self.MODEL_FAMILIES:
            fp16_key = f"{family} / FP16"
            nocal_key = f"{family} / INT8-nocal"
            cal_key = f"{family} / INT8-cal"

            if fp16_key not in valid or nocal_key not in valid:
                continue

            fp16 = valid[fp16_key]
            nocal = valid[nocal_key]

            fp16_ms = fp16.get("avg_infer_ms", 0)
            nocal_ms = nocal.get("avg_infer_ms", 0)
            speedup = fp16_ms / nocal_ms if nocal_ms > 0 else 0

            conf_drop = fp16.get("avg_confidence", 0) - nocal.get("avg_confidence", 0)
            detect_drop = fp16.get("detection_rate", 0) - nocal.get("detection_rate", 0)
            jitter_diff = nocal.get("avg_jitter_px", 0) - fp16.get("avg_jitter_px", 0)

            print(f"\n  {family}:")
            print(f"    FP16 → INT8(no-cal):")
            print(f"      속도:     {fp16_ms:.2f}ms → {nocal_ms:.2f}ms "
                  f"({speedup:.2f}x {'빠름' if speedup > 1 else '느림'})")
            print(f"      인식률:   {detect_drop:+.1f}%p "
                  f"{'OK' if abs(detect_drop) < 5 else 'WARNING'}")
            print(f"      Conf:     {conf_drop:+.4f} "
                  f"{'OK' if abs(conf_drop) < 0.02 else 'WARNING'}")
            print(f"      Jitter:   {jitter_diff:+.2f}px "
                  f"{'OK' if abs(jitter_diff) < 2.0 else 'WARNING'}")

            # 캘리브레이션 효과
            if cal_key in valid:
                cal = valid[cal_key]
                cal_ms = cal.get("avg_infer_ms", 0)
                cal_conf = cal.get("avg_confidence", 0)
                nocal_conf = nocal.get("avg_confidence", 0)
                cal_detect = cal.get("detection_rate", 0)
                nocal_detect = nocal.get("detection_rate", 0)

                print(f"    INT8(no-cal) → INT8(cal) [캘리브레이션 효과]:")
                print(f"      속도:     {nocal_ms:.2f}ms → {cal_ms:.2f}ms "
                      f"(차이 {cal_ms - nocal_ms:+.2f}ms)")
                print(f"      인식률:   {nocal_detect:.1f}% → {cal_detect:.1f}% "
                      f"({cal_detect - nocal_detect:+.1f}%p)")
                print(f"      Conf:     {nocal_conf:.4f} → {cal_conf:.4f} "
                      f"({cal_conf - nocal_conf:+.4f})")

        # 판단 기준
        print(f"\n  {'─'*70}")
        print(f"  [판단 기준]")
        print(f"    속도:    1.2x 이상 빨라지면 INT8 가치 있음")
        print(f"    인식률:  5%p 이상 떨어지면 사용 불가")
        print(f"    Conf:    0.02 이상 떨어지면 주의")
        print(f"    Jitter:  2px 이상 증가하면 주의")
        print(f"    캘리브:  no-cal vs cal 차이가 크면 → 캘리브레이션 필수")
        print(f"             차이가 작으면 → 캘리브레이션 불필요, no-cal로 충분")
        print()

    def _save_results(self):
        """결과 JSON 저장 — run_dir에 함께 저장"""
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
            "experiment": "INT8 Quantization YOLOv8 + YOLO11 Comparison",
            "timestamp": self._timestamp,
            "config": {
                "models": list(self.MODEL_FAMILIES),
                "precisions": PRECISIONS if not self.args.skip_calibrated
                              else ["FP16", "INT8-nocal"],
                "duration": self.args.duration,
                "warmup_frames": self.args.warmup_frames,
                "video": self.args.video,
                "max_perf": self._jetson_optimized,
            },
            "system_info": self.system_info if self.system_info else None,
            "results": {
                name: {k: convert(v) for k, v in stats.items()}
                for name, stats in self.results.items()
            },
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False, default=convert)

        print(f"  결과 저장: {filepath}")

    def _review_recordings(self):
        """벤치마크 종료 후 녹화 영상을 순차 재생하여 스켈레톤 추적 품질 리뷰

        조작:
            SPACE  - 일시정지/재생
            N      - 다음 모델 영상으로
            P      - 이전 모델 영상으로
            Q/ESC  - 리뷰 종료
            ←/→    - 5프레임 뒤로/앞으로
        """
        import glob as glob_mod

        video_files = sorted(glob_mod.glob(os.path.join(self.video_dir, "*.mp4")))
        if not video_files:
            print("  녹화 파일이 없습니다.")
            return

        print(f"\n{'='*70}")
        print(f"  녹화 리뷰 재생 ({len(video_files)}개 모델)")
        print(f"  조작: SPACE=일시정지  N=다음  P=이전  Q=종료  ←/→=5프레임 이동")
        print(f"{'='*70}\n")

        vid_idx = 0
        while 0 <= vid_idx < len(video_files):
            vpath = video_files[vid_idx]
            vname = os.path.splitext(os.path.basename(vpath))[0].replace("_", " ")

            # 이 모델의 벤치마크 결과 찾기
            stats = None
            for rname, rstat in self.results.items():
                safe = rname.replace(" ", "_").replace("/", "_")
                if safe in os.path.basename(vpath):
                    stats = rstat
                    break

            cap = cv2.VideoCapture(vpath)
            if not cap.isOpened():
                print(f"  ERROR: {vpath} 열기 실패")
                vid_idx += 1
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            delay = int(1000 / fps)

            print(f"  [{vid_idx + 1}/{len(video_files)}] {vname} "
                  f"({total_frames} frames, {fps:.0f}fps)")

            paused = False
            quit_all = False

            while True:
                if not paused:
                    ret, frame = cap.read()
                    if not ret:
                        # 영상 끝 → 3초 대기 후 다음
                        cv2.waitKey(2000)
                        break

                # HUD 오버레이
                frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                h, w = frame.shape[:2]

                # 상단 바
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, 70), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

                cv2.putText(frame, f"[{vid_idx+1}/{len(video_files)}] {vname}",
                            (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

                progress = f"Frame: {frame_num}/{total_frames}"
                if stats and "error" not in stats:
                    progress += (f"  |  FPS: {stats.get('avg_fps', 0):.1f}"
                                 f"  Infer: {stats.get('avg_infer_ms', 0):.1f}ms"
                                 f"  Det: {stats.get('detection_rate', 0):.0f}%"
                                 f"  Jitter: {stats.get('avg_jitter_px', 0):.1f}px")
                cv2.putText(frame, progress,
                            (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

                if paused:
                    cv2.putText(frame, "PAUSED",
                                (w - 120, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # 하단 진행 바
                bar_y = h - 8
                bar_fill = int(w * frame_num / max(total_frames, 1))
                cv2.rectangle(frame, (0, bar_y), (w, h), (40, 40, 40), -1)
                cv2.rectangle(frame, (0, bar_y), (bar_fill, h), (0, 200, 255), -1)

                cv2.imshow("Benchmark Review", frame)

                key = cv2.waitKey(0 if paused else delay) & 0xFF

                if key == ord('q') or key == 27:  # Q / ESC
                    quit_all = True
                    break
                elif key == ord(' '):  # SPACE
                    paused = not paused
                elif key == ord('n'):  # N - next
                    break
                elif key == ord('p'):  # P - previous
                    vid_idx = max(0, vid_idx - 2)  # -2 because +1 at end
                    break
                elif key == 81 or key == 2:  # ← (left arrow)
                    new_pos = max(0, frame_num - 5)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                elif key == 83 or key == 3:  # → (right arrow)
                    new_pos = min(total_frames - 1, frame_num + 5)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)

            cap.release()
            vid_idx += 1

            if quit_all:
                break

        cv2.destroyAllWindows()
        print(f"\n  리뷰 종료. 녹화 폴더: {self.video_dir}")


# ============================================================================
# 메인
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="INT8 Quantization 비교 실험 (YOLOv8n × YOLOv8s × RTMPose × 3정밀도)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python run_int8_comparison.py --max-perf                   # 전체 9종 비교 (권장)
  python run_int8_comparison.py --max-perf --calib-data ./calib_images/
  python run_int8_comparison.py --max-perf --skip-calibrated # FP16 vs INT8(no-cal)만
  python run_int8_comparison.py --max-perf --video rec.mp4   # 동일 영상으로 공정 비교
  python run_int8_comparison.py --max-perf --duration 30     # 30초간 측정
        """)

    parser.add_argument("--max-perf", action="store_true",
                        help="Jetson 최대 성능 모드 (MAXN, jetson_clocks, fan max)")
    parser.add_argument("--duration", type=int, default=15,
                        help="모델당 측정 시간 (초, 기본: 15)")
    parser.add_argument("--warmup-frames", type=int, default=50,
                        help="워밍업 프레임 수 (기본: 50)")
    parser.add_argument("--skip-calibrated", action="store_true",
                        help="캘리브레이션 INT8 스킵 (FP16 vs INT8-nocal만 비교)")
    parser.add_argument("--calib-data", type=str, default=None,
                        help="INT8 캘리브레이션 이미지 디렉토리 (걷기 영상에서 추출)")
    parser.add_argument("--video", type=str, default=None,
                        help="비디오 파일 (동일 입력으로 공정 비교)")
    parser.add_argument("--depth-mode", default="PERFORMANCE",
                        help="ZED 깊이 모드 (기본: PERFORMANCE, 끄려면 NONE)")
    parser.add_argument("--no-zed", dest="use_zed", action="store_false",
                        help="ZED 대신 웹캠 사용")
    parser.add_argument("--resolution", default="SVGA",
                        help="카메라 해상도 (기본: SVGA)")
    parser.add_argument("--center-crop", type=int, default=640, metavar="WIDTH",
                        help="프레임 가로 중앙 크롭 (기본: 640, 0=크롭 없음)")

    parser.add_argument("--camera-fps", type=int, default=120,
                        help="카메라 FPS (기본: 120)")
    parser.add_argument("--visualize", action="store_true",
                        help="실시간 시각화")
    parser.add_argument("--record", action="store_true", default=False,
                        help="모델별 추론 결과 비디오 녹화 (기본 OFF, 필요시 활성화)")
    parser.add_argument("--no-record", dest="record", action="store_false",
                        help="비디오 녹화 비활성화")
    parser.add_argument("--review", action="store_true", default=False,
                        help="벤치마크 후 녹화 영상 리뷰 재생")
    parser.add_argument("--seg-calib-reset", action="store_true",
                        help="세그먼트 캘리브레이션 초기화 (seg_calib.json 삭제 후 재측정)")

    args = parser.parse_args()

    # 세그먼트 캘리브레이션 리셋
    if args.seg_calib_reset:
        from pose_models import SegmentLengthConstraint
        calib_path = SegmentLengthConstraint.DEFAULT_CALIB_FILE
        if os.path.exists(calib_path):
            os.remove(calib_path)
            print(f"[seg-calib-reset] {calib_path} 삭제 → 재캘리브레이션 진행")

    runner = Int8ComparisonRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
