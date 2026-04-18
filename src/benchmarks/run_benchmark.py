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
    python3 run_benchmark.py --record           # 벤치마크 영상 녹화
    python3 run_benchmark.py --models yolo26_int8 --calib-data ./calib_images/yolo/ --rebuild-int8
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

from zed_camera import create_camera, HAS_ZED, GPUPreprocessor
from select_roi import load_crop_roi, apply_crop, auto_load_crop_roi, reset_crop_roi
from pose_models import (
    MediaPipePose, YOLOv8Pose, RTMPoseModel, RTMPoseWholebody,
    ZEDBodyTracking, MoveNetModel, draw_pose, PoseResult
)
from joint_angles import compute_lower_limb_angles
from postprocess_accel import (
    batch_2d_to_3d, create_joint_3d_filter, compute_lower_limb_angles_fast, HAS_CPP_EXT
)
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
        self.gpu_preproc = GPUPreprocessor() if getattr(args, 'max_perf', False) else None
        # ROI 크롭: --reset-crop이면 삭제, --no-crop이면 무시, 아니면 자동 로드
        if args.reset_crop:
            reset_crop_roi()
            self.crop_roi = None
        elif args.no_crop:
            self.crop_roi = None
        else:
            self.crop_roi = auto_load_crop_roi(args.crop)

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
        if self.crop_roi:
            print(f"  ROI 크롭: x={self.crop_roi['x']}, y={self.crop_roi['y']}, "
                  f"w={self.crop_roi['w']}, h={self.crop_roi['h']}")
        print(f"  시각화: {self.args.visualize}")
        if getattr(self.args, 'max_perf', False):
            print(f"  최대 성능 모드: ON (jetson_clocks + async + GPU preproc)")
        print("=" * 60)
        print()

        # 카메라 초기화
        if self.args.no_depth:
            depth_mode = "NONE"
        elif hasattr(self.args, 'depth_mode') and self.args.depth_mode:
            depth_mode = self.args.depth_mode
        else:
            depth_mode = "PERFORMANCE"
        use_async = getattr(self.args, 'max_perf', False)
        camera = create_camera(
            use_zed=self.args.use_zed,
            video_path=self.args.video,
            resolution=self.args.resolution,
            fps=self.args.camera_fps,
            depth_mode=depth_mode,
            async_capture=use_async,
        )
        camera.open()
        self._camera = camera  # 카메라 메타데이터 저장용 참조

        # Global Shutter 벤치마크 최적 설정 (ZED X Mini)
        if hasattr(camera, 'configure_for_benchmark'):
            camera.configure_for_benchmark()

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
        calib_data = getattr(self.args, 'calib_data', None)

        if "mediapipe" in selected or "all" in selected:
            models["mediapipe_lite"] = MediaPipePose(model_complexity=0)
            models["mediapipe_full"] = MediaPipePose(model_complexity=1)

        if "yolov8" in selected or "all" in selected:
            models["yolov8n"] = YOLOv8Pose(model_size="n", use_tensorrt=self.args.tensorrt)
            models["yolov8s"] = YOLOv8Pose(model_size="s", use_tensorrt=self.args.tensorrt)

        if "yolov8_int8" in selected:
            models["yolov8n_int8"] = YOLOv8Pose(model_size="n", use_tensorrt=True, precision="int8", use_calib=True, calib_data=calib_data)
            models["yolov8s_int8"] = YOLOv8Pose(model_size="s", use_tensorrt=True, precision="int8", use_calib=True, calib_data=calib_data)

        if "yolo11" in selected or "all" in selected:
            models["yolo11n"] = YOLOv8Pose(model_size="n", use_tensorrt=self.args.tensorrt, yolo_version="v11")
            models["yolo11s"] = YOLOv8Pose(model_size="s", use_tensorrt=self.args.tensorrt, yolo_version="v11")

        if "yolo11_int8" in selected:
            models["yolo11n_int8"] = YOLOv8Pose(model_size="n", use_tensorrt=True, precision="int8", use_calib=True, calib_data=calib_data, yolo_version="v11")
            models["yolo11s_int8"] = YOLOv8Pose(model_size="s", use_tensorrt=True, precision="int8", use_calib=True, calib_data=calib_data, yolo_version="v11")

        if "yolo26" in selected or "all" in selected:
            models["yolo26n"] = YOLOv8Pose(model_size="n", use_tensorrt=self.args.tensorrt, yolo_version="v26")
            models["yolo26s"] = YOLOv8Pose(model_size="s", use_tensorrt=self.args.tensorrt, yolo_version="v26")

        if "yolo26_int8" in selected:
            models["yolo26n_int8"] = YOLOv8Pose(model_size="n", use_tensorrt=True, precision="int8", use_calib=True, calib_data=calib_data, yolo_version="v26")
            models["yolo26s_int8"] = YOLOv8Pose(model_size="s", use_tensorrt=True, precision="int8", use_calib=True, calib_data=calib_data, yolo_version="v26")

        # yolo_all: YOLOv8/11/26 × n/s × FP16/INT8 TRT 전체 조합
        if "yolo_all" in selected:
            for ver, ver_label in [("v8", "yolov8"), ("v11", "yolo11"), ("v26", "yolo26")]:
                for size in ["n", "s"]:
                    # TRT FP16
                    key_fp16 = f"{ver_label}{size}_fp16"
                    models[key_fp16] = YOLOv8Pose(model_size=size, use_tensorrt=True, precision="fp16", yolo_version=ver)
                    # TRT INT8 (캘리브레이션)
                    key_int8 = f"{ver_label}{size}_int8"
                    models[key_int8] = YOLOv8Pose(model_size=size, use_tensorrt=True, precision="int8", use_calib=True, calib_data=calib_data, yolo_version=ver)

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

        if "rtmpose_int8" in selected:
            models["rtmpose_bal_int8"] = RTMPoseModel(
                mode="balanced",
                device="cuda",
                use_tensorrt=True,
                precision="int8",
                use_calib=True,
            )

        if "rtmpose_int8_nocal" in selected:
            models["rtmpose_bal_int8_nocal"] = RTMPoseModel(
                mode="balanced",
                device="cuda",
                use_tensorrt=True,
                precision="int8",
                use_calib=False,
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

        if "movenet" in selected or "all" in selected:
            models["movenet_lightning"] = MoveNetModel(
                variant="lightning", use_tensorrt=False)
            models["movenet_thunder"] = MoveNetModel(
                variant="thunder", use_tensorrt=False)

        if "movenet_trt" in selected:
            models["movenet_lightning_trt"] = MoveNetModel(
                variant="lightning", use_tensorrt=True)
            models["movenet_thunder_trt"] = MoveNetModel(
                variant="thunder", use_tensorrt=True)

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
            "grab_detail": {  # grab 병목 디버깅용
                "sdk_grab_ms": [],      # zed.grab() — 프레임 대기 + 스테레오 매칭
                "retrieve_rgb_ms": [],   # retrieve_image + get_data
                "retrieve_depth_ms": [], # retrieve_measure + get_data
            },
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
            # 프레임별 Joint 좌표
            "keypoints_2d_per_frame": [],
            "keypoints_3d_per_frame": [],
        }

        duration = self.args.duration
        warmup_frames = self.args.warmup_frames

        # 3D 필터 초기화 (Joint 위치 정확도 + depth 떨림 제거)
        # C++ 확장이 있으면 C++ 버전 사용 (후처리 4~6x 가속)
        joint_3d_filter = create_joint_3d_filter(min_cutoff=0.5, beta=0.01, max_missing=5)

        # 워밍업: 추론 + 후처리 파이프라인 전체를 워밍업
        # (Python JIT, NumPy 메모리 할당, OpenCV 함수 첫 호출 등 제거)
        print(f"  모델 워밍업 ({warmup_frames} 프레임, 전체 파이프라인)...")
        for i in range(warmup_frames):
            if camera.grab():
                # 워밍업도 측정 루프와 동일한 zero-copy 경로 사용
                try:
                    bgra_w = camera.get_rgb(raw_bgra=True)
                    depth_w = camera.get_depth() if hasattr(camera, 'get_depth') else None
                    h_w, w_w = bgra_w.shape[:2]
                    crop_w_val = 640
                    if w_w > crop_w_val:
                        cx = (w_w - crop_w_val) // 2
                        rgb_w = cv2.cvtColor(bgra_w[:, cx:cx + crop_w_val], cv2.COLOR_BGRA2BGR)
                    else:
                        cx = 0
                        rgb_w = cv2.cvtColor(bgra_w, cv2.COLOR_BGRA2BGR)
                except TypeError:
                    rgb_w = camera.get_rgb()
                    depth_w = camera.get_depth() if hasattr(camera, 'get_depth') else None
                    h_w, w_w = rgb_w.shape[:2]
                    crop_w_val = 640
                    if w_w > crop_w_val:
                        cx = (w_w - crop_w_val) // 2
                        rgb_w = rgb_w[:, cx:cx + crop_w_val].copy()
                    else:
                        cx = 0
                result_w = model.predict(rgb_w)
                # 후처리 워밍업: 3D 변환 + 관절각도 + draw_pose
                t_w = time.perf_counter()
                if depth_w is not None and result_w.detected:
                    raw_3d_w = batch_2d_to_3d(
                        result_w.keypoints_2d, depth_w, camera, crop_x=cx)
                    for name, pt3d_tuple in raw_3d_w.items():
                        pt3d_raw = np.array(pt3d_tuple, dtype=np.float32)
                        joint_3d_filter.filter(name, pt3d_raw, t_w)
                if result_w.detected:
                    compute_lower_limb_angles_fast(result_w, use_3d=bool(result_w.keypoints_3d))
                if i == warmup_frames - 1:
                    # 마지막 워밍업 프레임에서 draw_pose도 한번 호출
                    draw_pose(rgb_w, result_w, model.name)
        # 워밍업 후 3D 필터 리셋 (워밍업 데이터 잔류 방지)
        joint_3d_filter = create_joint_3d_filter(min_cutoff=0.5, beta=0.01, max_missing=5)

        # 녹화 설정
        video_writer = None
        record_path = None
        if getattr(self.args, 'record', False):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            record_dir = os.path.join(
                os.path.dirname(__file__), "results", "benchmark", timestamp)
            os.makedirs(record_dir, exist_ok=True)
            safe_name = model.name.replace(" ", "_").replace("/", "-")
            record_path = os.path.join(record_dir, f"{safe_name}.mp4")
            # VideoWriter는 첫 프레임에서 실제 크기로 초기화

        # 실제 측정
        print(f"  측정 시작 ({duration}초)...")
        start_time = time.perf_counter()
        frame_times = []
        last_fps_print = start_time

        # 센터 크롭 파라미터 사전 계산
        _crop_w = 640
        _use_raw_bgra = hasattr(camera, 'get_rgb') and callable(getattr(camera, 'get_rgb', None))

        while time.perf_counter() - start_time < duration:
            # ---- Phase A: Camera Grab (세부 타이밍 분해) ----
            t_grab_start = time.perf_counter()
            if not camera.grab():
                continue
            t_after_grab = time.perf_counter()
            # Zero-copy BGRA view → 크롭 → BGR 변환 (최소 메모리 복사)
            try:
                bgra = camera.get_rgb(raw_bgra=True)
                t_after_rgb = time.perf_counter()
            except TypeError:
                # raw_bgra 미지원 카메라 (Webcam 등)
                bgra = None
                rgb = camera.get_rgb(copy=False)
                t_after_rgb = time.perf_counter()
            depth = camera.get_depth() if hasattr(camera, 'get_depth') else None
            t_grab_end = time.perf_counter()

            # 센터 크롭 + BGRA→BGR 변환 (크롭된 영역만 변환 — 메모리 33% 절약)
            if bgra is not None:
                h_orig, w_orig = bgra.shape[:2]
                if w_orig > _crop_w:
                    crop_x = (w_orig - _crop_w) // 2
                    # BGRA 크롭 (view, zero-copy) → BGR 변환 (640x600만 처리)
                    bgra_crop = bgra[:, crop_x:crop_x + _crop_w]
                    rgb = cv2.cvtColor(bgra_crop, cv2.COLOR_BGRA2BGR)
                else:
                    crop_x = 0
                    rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            else:
                h_orig, w_orig = rgb.shape[:2]
                if w_orig > _crop_w:
                    crop_x = (w_orig - _crop_w) // 2
                    rgb = rgb[:, crop_x:crop_x + _crop_w].copy()
                else:
                    crop_x = 0

            # GPU 전처리 (max-perf 모드)
            if self.gpu_preproc is not None and hasattr(model, 'input_size'):
                size = model.input_size
                rgb = self.gpu_preproc.resize(rgb, (size, size))

            # ---- Phase B: Model Inference ----
            t_infer_start = time.perf_counter()
            result = model.predict(rgb)
            t_infer_end = time.perf_counter()

            # ---- Phase C: Post-processing (2D→3D + 관절각도) ----
            t_post_start = time.perf_counter()
            t_now = time.perf_counter()

            # 2D → 3D 변환 (depth 있을 때) + 3D 필터링
            if depth is not None and result.detected:
                # C++ 배치 변환으로 depth 패치 샘플링 + 역투영 일괄 처리
                raw_3d = batch_2d_to_3d(
                    result.keypoints_2d, depth, camera, crop_x=crop_x)
                for name, pt3d_tuple in raw_3d.items():
                    pt3d_raw = np.array(pt3d_tuple, dtype=np.float32)
                    pt3d = joint_3d_filter.filter(name, pt3d_raw, t_now)
                    if pt3d is not None:
                        result.keypoints_3d[name] = tuple(pt3d)
                # depth 무효 키포인트도 보간 시도
                for name in result.keypoints_2d:
                    if name not in result.keypoints_3d:
                        pt3d = joint_3d_filter.filter(name, None, t_now)
                        if pt3d is not None:
                            result.keypoints_3d[name] = tuple(pt3d)
            elif result.detected:
                # depth 없어도 이전 프레임 보간 시도
                for name in result.keypoints_2d:
                    pt3d = joint_3d_filter.filter(name, None, t_now)
                    if pt3d is not None:
                        result.keypoints_3d[name] = tuple(pt3d)

            # 3D 뼈 길이 제약 (정자세 캘리브레이션 → 보행 중 보정)
            if result.keypoints_3d:
                joint_3d_filter.apply_segment_constraint(result.keypoints_3d)

            # 관절 각도 계산 (C++ 가속)
            use_3d = bool(result.keypoints_3d)
            result.joint_angles = compute_lower_limb_angles_fast(result, use_3d=use_3d)

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
            stats["grab_detail"]["sdk_grab_ms"].append(
                (t_after_grab - t_grab_start) * 1000)
            stats["grab_detail"]["retrieve_rgb_ms"].append(
                (t_after_rgb - t_after_grab) * 1000)
            stats["grab_detail"]["retrieve_depth_ms"].append(
                (t_grab_end - t_after_rgb) * 1000)
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

            # 프레임별 Joint 좌표 수집
            if result.detected:
                # 2D: {joint_name: [x, y]} pixel 좌표
                kp2d = {k: [round(v[0], 1), round(v[1], 1)]
                        for k, v in result.keypoints_2d.items()}
                stats["keypoints_2d_per_frame"].append(kp2d)
                # 3D: {joint_name: [x, y, z]} meters
                if result.keypoints_3d:
                    kp3d = {k: [round(v[0], 4), round(v[1], 4), round(v[2], 4)]
                            for k, v in result.keypoints_3d.items()}
                    stats["keypoints_3d_per_frame"].append(kp3d)

            frame_end = time.perf_counter()
            frame_times.append(frame_end - t_grab_start)

            # 시각화 / 녹화
            if self.args.visualize or record_path:
                vis = draw_pose(rgb, result, model.name)
                if len(frame_times) > 1:
                    fps = 1.0 / np.mean(frame_times[-30:])
                    cv2.putText(vis, f"FPS: {fps:.1f}", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                # E2E latency 표시
                cv2.putText(vis, f"E2E: {e2e_ms:.1f}ms (G:{grab_ms:.0f}+I:{infer_ms:.0f}+P:{post_ms:.0f})",
                           (10, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 0), 1)

                # 녹화: 첫 프레임에서 VideoWriter 초기화
                if record_path and video_writer is None:
                    h_vis, w_vis = vis.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    est_fps = 30.0  # 초기 추정치, 실제 FPS는 가변
                    video_writer = cv2.VideoWriter(record_path, fourcc, est_fps, (w_vis, h_vis))
                    print(f"  녹화 시작: {record_path} ({w_vis}x{h_vis})")
                if video_writer is not None:
                    video_writer.write(vis)

                if self.args.visualize:
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

        # 녹화 종료
        if video_writer is not None:
            video_writer.release()
            n_frames = stats["frame_count"]
            print(f"  녹화 완료: {record_path} ({n_frames} 프레임)")
            stats["record_path"] = record_path
        elif record_path:
            print(f"  [{model.name}] WARNING: --record 설정했지만 VideoWriter 초기화 안 됨 (프레임 없음?)")
            stats["record_path"] = None

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

            # Grab 세부 통계
            gd = stats.get("grab_detail", {})
            for key in ["sdk_grab_ms", "retrieve_rgb_ms", "retrieve_depth_ms"]:
                arr = gd.get(key, [])
                if arr:
                    gd[f"avg_{key}"] = float(np.mean(arr))
                    gd[f"p95_{key}"] = float(np.percentile(arr, 95))

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
        # Grab 세부 분해
        gd = stats.get("grab_detail", {})
        if gd and gd.get("sdk_grab_ms"):
            print(f"  Grab 세부 분해 (ms):")
            for label, key in [("SDK grab()", "sdk_grab_ms"),
                                ("retrieve RGB", "retrieve_rgb_ms"),
                                ("retrieve Depth", "retrieve_depth_ms")]:
                arr = gd[key]
                if arr:
                    avg = sum(arr) / len(arr)
                    p95 = sorted(arr)[int(len(arr) * 0.95)]
                    print(f"    {label:>16} avg={avg:>6.2f}  p95={p95:>6.2f}")
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

        # 요약 (스코어 없이 raw 메트릭만)
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
        print()
        print("  (스코어링 제거됨 - 위 raw 메트릭으로 직접 비교하세요)")

    def _save_results(self):
        """결과를 JSON 파일로 저장 — results/benchmark/YYYYMMDD_HHMMSS/"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(
            os.path.dirname(__file__), "results", "benchmark", timestamp)
        os.makedirs(run_dir, exist_ok=True)
        filepath = os.path.join(run_dir, "results.json")

        # numpy 타입을 python 타입으로 변환
        def convert(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        # 카메라 메타데이터 수집 (Global Shutter 정보 등)
        camera_info = {}
        if hasattr(self, '_camera') and hasattr(self._camera, 'get_camera_info'):
            camera_info = self._camera.get_camera_info()

        save_data = {
            "timestamp": timestamp,
            "config": {
                "duration": self.args.duration,
                "camera": "ZED" if self.args.use_zed else "Webcam",
                "resolution": self.args.resolution,
                "lower_only": self.args.lower_only,
                "crop_roi": self.crop_roi,
                "tensorrt": self.args.tensorrt,
                "video": self.args.video,
                "has_depth": self.args.video is None or (
                    self.args.video and os.path.splitext(self.args.video)[1].lower() in ('.svo2', '.svo')
                ),
                "record": getattr(self.args, 'record', False),
                "calib_data": getattr(self.args, 'calib_data', None),
            },
            "camera_info": camera_info,
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
                        choices=["all", "mediapipe", "yolov8", "yolo11", "yolo26",
                                 "yolo_all",
                                 "yolov8_int8", "yolo11_int8", "yolo26_int8",
                                 "rtmpose",
                                 "rtmpose_int8", "rtmpose_int8_nocal",
                                 "rtmpose_wb", "movenet", "movenet_trt", "zed_bt"],
                        help="벤치마크할 모델 (기본: all). "
                             "yolo_all: YOLOv8/11 × n/s × FP16/INT8 전체 조합. "
                             "rtmpose_int8: INT8(캘리브레이션), "
                             "rtmpose_int8_nocal: INT8(캘리브레이션 없음)")
    parser.add_argument("--duration", type=int, default=15,
                        help="모델당 측정 시간 (초, 기본: 15)")
    parser.add_argument("--no-zed", dest="use_zed", action="store_false",
                        help="ZED 카메라 대신 웹캠 사용")
    parser.add_argument("--resolution", default="SVGA",
                        choices=["SVGA", "VGA", "HD720", "HD1080", "HD1200"],
                        help="카메라 해상도 (기본: SVGA, ZED X Mini 최적)")
    parser.add_argument("--camera-fps", type=int, default=120,
                        help="카메라 FPS (기본: 60, SVGA 최적)")
    parser.add_argument("--visualize", action="store_true",
                        help="실시간 시각화 (FPS 약간 감소)")
    parser.add_argument("--lower-only", action="store_true",
                        help="하체만 보이는 상황 시뮬레이션 (상위 1/3 크롭)")
    parser.add_argument("--crop", type=str, default=None,
                        help="ROI 크롭 JSON 파일 경로 (기본: crop_roi.json 자동 로드)")
    parser.add_argument("--no-crop", action="store_true",
                        help="저장된 크롭 설정 무시 (원본 프레임 사용)")
    parser.add_argument("--reset-crop", action="store_true",
                        help="저장된 크롭 설정 초기화 후 원본 프레임으로 실행")
    parser.add_argument("--tensorrt", action="store_true",
                        help="TensorRT 변환 사용 (YOLOv8, RTMPose)")
    parser.add_argument("--video", type=str, default=None,
                        help="동영상 파일 경로 (카메라 대신 영상 파일로 벤치마크)")
    parser.add_argument("--no-depth", action="store_true",
                        help="Depth OFF (속도 최적화, 3D 메트릭 비활성)")
    parser.add_argument("--depth-mode", default=None,
                        help="ZED 깊이 모드 (PERFORMANCE/NEURAL/NONE 등, 기본: PERFORMANCE)")
    parser.add_argument("--fast", action="store_true",
                        help="속도 최적화 모드: SVGA/120fps, depth OFF, 최소 후처리")
    parser.add_argument("--warmup-frames", type=int, default=30,
                        help="GPU 워밍업 프레임 수 (기본: 30)")
    parser.add_argument("--max-perf", action="store_true",
                        help="최대 성능 모드: jetson_clocks + nvpmodel MAXN + "
                             "비동기 캡처 + GPU 전처리 + depth PERFORMANCE")
    parser.add_argument("--record", action="store_true",
                        help="벤치마크 중 영상 녹화 (results/ 디렉토리에 모델별 mp4 저장)")
    parser.add_argument("--calib-data", type=str, default=None,
                        help="INT8 캘리브레이션 이미지 디렉토리 경로 (기본: calib_images/yolo/)")
    parser.add_argument("--rebuild-int8", action="store_true",
                        help="기존 INT8 엔진 삭제 후 재빌드 (캘리브레이션 데이터 변경 시)")
    parser.add_argument("--seg-calib-reset", action="store_true",
                        help="세그먼트 캘리브레이션 초기화 (seg_calib.json 삭제 후 재측정)")

    args = parser.parse_args()

    # INT8 엔진 재빌드: 기존 INT8 .engine 파일 삭제
    if args.rebuild_int8:
        import glob as _glob
        engine_dir = os.path.dirname(os.path.abspath(__file__))
        patterns = [
            os.path.join(engine_dir, "*-int8-*.engine"),
            os.path.join(engine_dir, "*_int8*.engine"),
        ]
        removed = []
        for pat in patterns:
            for f in _glob.glob(pat):
                os.remove(f)
                removed.append(os.path.basename(f))
        if removed:
            print(f"[rebuild-int8] 삭제된 엔진: {', '.join(removed)}")
            print(f"[rebuild-int8] 다음 실행 시 새 캘리브레이션으로 재빌드됩니다.")
        else:
            print("[rebuild-int8] 삭제할 INT8 엔진 파일이 없습니다.")

    # 세그먼트 캘리브레이션 리셋
    if args.seg_calib_reset:
        from pose_models import SegmentLengthConstraint
        calib_path = SegmentLengthConstraint.DEFAULT_CALIB_FILE
        if os.path.exists(calib_path):
            os.remove(calib_path)
            print(f"[seg-calib-reset] {calib_path} 삭제 → 재캘리브레이션 진행")

    # --max-perf 모드: 모든 최적화 자동 적용
    if args.max_perf:
        args.resolution = "SVGA"
        args.camera_fps = 120
        if not hasattr(args, 'depth_mode') or args.depth_mode is None:
            args.depth_mode = "PERFORMANCE"
        args.no_depth = False
        args.visualize = False
        args.tensorrt = True

        try:
            from jetson_optimizer import optimize_jetson, get_system_info
            optimize_jetson()
        except Exception as e:
            print(f"  [WARNING] Jetson 최적화 실패 (sudo 필요?): {e}")

    # --fast 모드: 속도 최적화 옵션 자동 설정
    if args.fast:
        args.resolution = "SVGA"
        args.camera_fps = 120
        if not hasattr(args, 'depth_mode') or args.depth_mode is None:
            args.depth_mode = "PERFORMANCE"
        args.no_depth = False
        args.visualize = False

    # --video 사용 시: SVO2는 ZED SDK 유지 (depth 활용), 일반 영상은 ZED 비활성화
    if args.video:
        ext = os.path.splitext(args.video)[1].lower()
        if ext not in ('.svo2', '.svo'):
            args.use_zed = False

    runner = BenchmarkRunner(args)
    runner.run()

    # max-perf 모드 종료 시 시스템 복원
    if args.max_perf:
        try:
            from jetson_optimizer import restore_jetson
            restore_jetson()
        except Exception:
            pass


if __name__ == "__main__":
    main()
