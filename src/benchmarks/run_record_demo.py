#!/usr/bin/env python3
"""
모델별 포즈 추정 데모 영상 녹화
=================================
각 모델(PyTorch + TRT)로 추론하면서 포즈 스켈레톤이 오버레이된 영상을 MP4로 저장합니다.
화면에 모델명, FPS, Latency가 실시간 표시됩니다.

사용법:
    # 모든 모델 녹화 (각 15초, TRT 포함)
    python3 run_record_demo.py

    # 특정 모델만
    python3 run_record_demo.py --models yolov8 rtmpose_wb

    # SVO2 파일로
    python3 run_record_demo.py --video test_data/walk_frontal.svo2

    # 녹화 시간 변경
    python3 run_record_demo.py --duration 20

    # TRT 없이
    python3 run_record_demo.py --no-trt

    # 출력 디렉토리 지정
    python3 run_record_demo.py --output-dir demo_videos/
"""

import argparse
import csv
import time
import sys
import os
import threading
import queue
import numpy as np
import cv2
from datetime import datetime
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zed_camera import create_camera, HAS_ZED
from select_roi import load_crop_roi, apply_crop, auto_load_crop_roi, reset_crop_roi
from pose_models import (
    MediaPipePose, YOLOv8Pose, RTMPoseModel, RTMPoseWholebody,
    ZEDBodyTracking, MoveNetModel, draw_pose, PoseResult,
    check_tensorrt_available, LOWER_LIMB_KEYPOINTS,
)
from postprocess_accel import (
    batch_2d_to_3d, create_joint_3d_filter, HAS_CPP_EXT
)
from joint_angles import compute_lower_limb_angles


def create_all_models(selected_models, include_trt=True, include_int8=False,
                      trt_only=False):
    """
    모든 모델 인스턴스를 생성
    trt_only=True: TRT/INT8 모델만 (PyTorch 건너뛰기)
    Returns: list of (label, model_instance)
    """
    models = []

    if "mediapipe" in selected_models or "all" in selected_models:
        if not trt_only:
            models.append(("MediaPipe_Lite", MediaPipePose(model_complexity=0)))
            models.append(("MediaPipe_Full", MediaPipePose(model_complexity=1)))

    for ver_key, ver_val, ver_prefix in [
        ("yolov8", "v8", "YOLOv8"), ("yolo11", "v11", "YOLO11"), ("yolo26", "v26", "YOLO26"),
    ]:
        if ver_key in selected_models or "all" in selected_models:
            for sz in ("n", "s"):
                if not trt_only:
                    models.append((f"{ver_prefix}{sz}", YOLOv8Pose(model_size=sz, use_tensorrt=False, yolo_version=ver_val)))
                if include_trt:
                    models.append((f"{ver_prefix}{sz}_TRT", YOLOv8Pose(model_size=sz, use_tensorrt=True, yolo_version=ver_val)))
                if include_int8:
                    models.append((f"{ver_prefix}{sz}_INT8", YOLOv8Pose(model_size=sz, use_tensorrt=True, yolo_version=ver_val, precision="int8")))

    if "rtmpose" in selected_models or "all" in selected_models:
        for mode in ("balanced", "lightweight"):
            if not trt_only:
                models.append((f"RTMPose_{mode}", RTMPoseModel(mode=mode, use_tensorrt=False)))
            if include_trt:
                models.append((f"RTMPose_{mode}_TRT", RTMPoseModel(mode=mode, use_tensorrt=True)))

    if "rtmpose_wb" in selected_models or "all" in selected_models:
        for mode in ("balanced", "lightweight"):
            if not trt_only:
                models.append((f"RTMPose_WB_{mode}", RTMPoseWholebody(mode=mode, use_tensorrt=False)))
            if include_trt:
                models.append((f"RTMPose_WB_{mode}_TRT", RTMPoseWholebody(mode=mode, use_tensorrt=True)))

    if "movenet" in selected_models or "all" in selected_models:
        for var in ("lightning", "thunder"):
            cap = var.capitalize()
            if not trt_only:
                models.append((f"MoveNet_{cap}", MoveNetModel(variant=var, use_tensorrt=False)))
            if include_trt:
                models.append((f"MoveNet_{cap}_TRT", MoveNetModel(variant=var, use_tensorrt=True)))

    if ("zed_bt" in selected_models or "all" in selected_models) and HAS_ZED:
        models.append(("ZED_BT_FAST", ZEDBodyTracking(model="FAST")))

    return models


def draw_demo_overlay(image, pose_result, model_name, fps, infer_ms, e2e_ms, frame_num, total_frames,
                      joint_angles=None):
    """데모용 오버레이: 포즈 + 모델명 + FPS + Latency + 진행률 + 관절 각도"""
    vis = draw_pose(image, pose_result, model_name)
    h, w = vis.shape[:2]

    # 상단 반투명 배경
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (w, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    # 모델명 (크게)
    cv2.putText(vis, model_name,
                (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # FPS & Latency
    cv2.putText(vis, f"FPS: {fps:.1f}",
                (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(vis, f"Infer: {infer_ms:.1f}ms",
                (200, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
    cv2.putText(vis, f"E2E: {e2e_ms:.1f}ms",
                (430, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 255), 2)

    # Detection 상태
    status = "Detected" if pose_result.detected else "No Detection"
    status_color = (0, 255, 0) if pose_result.detected else (0, 0, 255)
    cv2.putText(vis, status,
                (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # 하체 confidence
    ll_conf = pose_result.get_lower_limb_confidence()
    has_ll = pose_result.has_lower_limb()
    ll_text = f"Lower Limb: {'OK' if has_ll else 'FAIL'} (conf={ll_conf:.2f})"
    cv2.putText(vis, ll_text,
                (200, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    # 진행률 바
    progress = frame_num / max(total_frames, 1)
    bar_y = 115
    bar_w = w - 30
    cv2.rectangle(vis, (15, bar_y), (15 + bar_w, bar_y + 10), (80, 80, 80), -1)
    cv2.rectangle(vis, (15, bar_y), (15 + int(bar_w * progress), bar_y + 10), (0, 200, 0), -1)

    # 관절 각도 텍스트 (해당 관절 위치 근처에 표시)
    if joint_angles and pose_result.detected:
        # 각도명 → 표시할 관절 keypoint 매핑
        _angle_to_joint = {
            "left_knee_flexion": "left_knee",
            "right_knee_flexion": "right_knee",
            "left_hip_flexion": "left_hip",
            "right_hip_flexion": "right_hip",
            "left_ankle_dorsiflexion": "left_ankle",
            "right_ankle_dorsiflexion": "right_ankle",
        }
        # 각도명 → 짧은 라벨
        _angle_label = {
            "left_knee_flexion": "Knee",
            "right_knee_flexion": "Knee",
            "left_hip_flexion": "Hip",
            "right_hip_flexion": "Hip",
            "left_ankle_dorsiflexion": "Ankle",
            "right_ankle_dorsiflexion": "Ankle",
        }
        for angle_name, angle_deg in joint_angles.items():
            joint_name = _angle_to_joint.get(angle_name)
            if joint_name and joint_name in pose_result.keypoints_2d:
                kp = pose_result.keypoints_2d[joint_name]
                px, py = int(kp[0]), int(kp[1])
                label = _angle_label.get(angle_name, angle_name)
                text = f"{label}: {angle_deg:.1f}\xb0"
                # 오프셋: 텍스트가 관절과 겹치지 않도록
                cv2.putText(vis, text, (px + 10, py - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    return vis


def _writer_thread(write_queue, writer):
    """별도 스레드에서 프레임을 파일에 씀 (메인 루프 블로킹 방지)"""
    while True:
        frame = write_queue.get()
        if frame is None:  # 종료 신호
            break
        writer.write(frame)
    writer.release()


def record_model(camera, model, label, output_path, duration=15,
                 lower_only=False, crop_roi=None, target_fps=30, warmup_frames=30):
    """단일 모델로 데모 영상 녹화 (비동기 쓰기로 성능 영향 최소화)"""
    print(f"\n  녹화 시작: {label} → {output_path}")

    # 자동 센터 크롭 헬퍼
    def _center_crop(img, target_w=640):
        h, w = img.shape[:2]
        if w > target_w:
            x = (w - target_w) // 2
            return img[:, x:x + target_w].copy()
        return img

    # 워밍업
    print(f"    워밍업 ({warmup_frames} 프레임)...")
    for _ in range(warmup_frames):
        if camera.grab():
            rgb = _center_crop(camera.get_rgb())
            model.predict(rgb)

    # 첫 프레임으로 해상도 확인
    camera.grab()
    rgb = _center_crop(camera.get_rgb())
    frame_h, frame_w = rgb.shape[:2]

    # VideoWriter 초기화 + 비동기 쓰기 스레드
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, target_fps, (frame_w, frame_h))
    if not writer.isOpened():
        print(f"    [ERROR] VideoWriter 열기 실패: {output_path}")
        return None

    write_queue = queue.Queue(maxsize=60)
    writer_t = threading.Thread(target=_writer_thread, args=(write_queue, writer), daemon=True)
    writer_t.start()

    # 녹화 루프
    fps_window = deque(maxlen=30)
    total_est_frames = int(duration * target_fps)
    frame_count = 0
    infer_times = []
    e2e_times = []
    detected_count = 0

    # 세부 타이밍 기록 (단계별 ms)
    timing_log = []  # list of dicts per frame

    # CSV용 프레임별 상세 데이터 기록
    frame_data_log = []

    # 관절 각도 이름 목록
    _angle_names = [
        "left_knee_flexion", "right_knee_flexion",
        "left_hip_flexion", "right_hip_flexion",
        "left_ankle_dorsiflexion", "right_ankle_dorsiflexion",
    ]

    # 3D 필터 초기화 (C++ 확장이 있으면 C++ 버전 사용)
    joint_3d_filter = create_joint_3d_filter(min_cutoff=0.5, beta=0.01, max_missing=5)

    start_time = time.perf_counter()
    last_print = start_time

    while time.perf_counter() - start_time < duration:
        t0 = time.perf_counter()
        if not camera.grab():
            continue
        rgb = camera.get_rgb()
        depth = camera.get_depth() if hasattr(camera, 'get_depth') else None
        t_cam = time.perf_counter()

        # 자동 센터 크롭: 960x600 → 640x600
        h_orig, w_orig = rgb.shape[:2]
        crop_w = 640
        if w_orig > crop_w:
            crop_x = (w_orig - crop_w) // 2  # 160
            rgb = rgb[:, crop_x:crop_x + crop_w].copy()
        else:
            crop_x = 0
        t_preproc = time.perf_counter()

        # Inference (2D)
        result = model.predict(rgb)
        t_infer = time.perf_counter()

        # 2D → 3D 변환 (depth 있을 때) — C++ 배치 처리
        # 크롭 오프셋: 크롭된 좌표를 원본 좌표로 변환
        t_now = time.perf_counter()
        if depth is not None and result.detected:
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
            for name in result.keypoints_2d:
                pt3d = joint_3d_filter.filter(name, None, t_now)
                if pt3d is not None:
                    result.keypoints_3d[name] = tuple(pt3d)

        # 3D 뼈 길이 제약
        if result.keypoints_3d:
            joint_3d_filter.apply_segment_constraint(result.keypoints_3d)
        t_3d = time.perf_counter()

        infer_ms = (t_infer - t_preproc) * 1000
        threed_ms = (t_3d - t_infer) * 1000
        e2e_ms = (t_3d - t0) * 1000
        result.inference_time_ms = infer_ms
        result.e2e_latency_ms = e2e_ms

        infer_times.append(infer_ms)
        e2e_times.append(e2e_ms)
        frame_count += 1
        if result.detected:
            detected_count += 1

        # 관절 각도 계산 (3D 우선, 없으면 2D)
        angles = {}
        if result.detected:
            use_3d = bool(result.keypoints_3d)
            angles = compute_lower_limb_angles(result, use_3d=use_3d)

        # CSV용 프레임별 상세 데이터 수집
        timestamp_ms = (time.perf_counter() - start_time) * 1000
        row = {
            "frame_idx": frame_count,
            "timestamp_ms": round(timestamp_ms, 2),
            "detected": int(result.detected),
            "lower_limb_ok": int(result.has_lower_limb()),
            "inference_ms": round(infer_ms, 2),
            "e2e_ms": round(e2e_ms, 2),
            "3d_ms": round(threed_ms, 2),
        }
        # 2D keypoints
        for kp_name in LOWER_LIMB_KEYPOINTS:
            if kp_name in result.keypoints_2d:
                px, py = result.keypoints_2d[kp_name][:2]
                conf = result.confidences.get(kp_name, 0.0)
            else:
                px, py, conf = "", "", ""
            row[f"{kp_name}_x"] = px
            row[f"{kp_name}_y"] = py
            row[f"{kp_name}_conf"] = round(conf, 4) if conf != "" else ""
        # 3D keypoints
        for kp_name in LOWER_LIMB_KEYPOINTS:
            if kp_name in result.keypoints_3d:
                x3, y3, z3 = result.keypoints_3d[kp_name][:3]
            else:
                x3, y3, z3 = "", "", ""
            row[f"{kp_name}_3d_x"] = x3
            row[f"{kp_name}_3d_y"] = y3
            row[f"{kp_name}_3d_z"] = z3
        # 관절 각도
        for aname in _angle_names:
            row[aname] = round(angles[aname], 2) if aname in angles else ""
        frame_data_log.append(row)

        fps_window.append(time.perf_counter())
        if len(fps_window) >= 2:
            current_fps = (len(fps_window) - 1) / (fps_window[-1] - fps_window[0])
        else:
            current_fps = 0

        # 오버레이 그리기 (관절 각도 포함)
        vis = draw_demo_overlay(
            rgb, result, label,
            fps=current_fps,
            infer_ms=infer_ms,
            e2e_ms=e2e_ms,
            frame_num=frame_count,
            total_frames=total_est_frames,
            joint_angles=angles,
        )
        t_draw = time.perf_counter()

        # 비동기 쓰기 (큐가 가득 차면 오래된 프레임 버림)
        try:
            write_queue.put_nowait(vis.copy())
        except queue.Full:
            pass  # 프레임 드롭 (성능 우선)
        t_write = time.perf_counter()

        # 세부 타이밍 기록
        timing_log.append({
            "frame": frame_count,
            "cam_ms": (t_cam - t0) * 1000,
            "preproc_ms": (t_preproc - t_cam) * 1000,
            "infer_ms": infer_ms,
            "3d_ms": threed_ms,
            "draw_ms": (t_draw - t_3d) * 1000,
            "write_ms": (t_write - t_draw) * 1000,
            "total_ms": (t_write - t0) * 1000,
        })

        # 3초마다 진행 상황 (세부 타이밍 포함)
        now = time.perf_counter()
        if now - last_print >= 3.0:
            elapsed = now - start_time
            recent = timing_log[-50:]
            avg_cam = np.mean([t["cam_ms"] for t in recent])
            avg_pre = np.mean([t["preproc_ms"] for t in recent])
            avg_inf = np.mean([t["infer_ms"] for t in recent])
            avg_3d = np.mean([t["3d_ms"] for t in recent])
            avg_drw = np.mean([t["draw_ms"] for t in recent])
            avg_wrt = np.mean([t["write_ms"] for t in recent])
            avg_tot = np.mean([t["total_ms"] for t in recent])
            det_rate = detected_count / frame_count * 100 if frame_count > 0 else 0
            print(f"    {elapsed:.0f}s | {frame_count}f | FPS:{current_fps:.1f} | "
                  f"Cam:{avg_cam:.1f} Pre:{avg_pre:.1f} Inf:{avg_inf:.1f} "
                  f"3D:{avg_3d:.1f} Draw:{avg_drw:.1f} Write:{avg_wrt:.1f} "
                  f"Total:{avg_tot:.1f}ms | Det:{det_rate:.0f}%")
            last_print = now

    # 쓰기 스레드 종료 대기
    write_queue.put(None)
    writer_t.join(timeout=10)

    # 프레임별 상세 CSV 저장
    csv_path = output_path.replace(".mp4", ".csv")
    if frame_data_log:
        fieldnames = list(frame_data_log[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
            writer_csv = csv.DictWriter(csvf, fieldnames=fieldnames)
            writer_csv.writeheader()
            writer_csv.writerows(frame_data_log)
        print(f"    CSV 저장: {csv_path}")

    # 결과 요약
    total_time = time.perf_counter() - start_time
    avg_fps = frame_count / total_time if total_time > 0 else 0
    avg_infer = np.mean(infer_times) if infer_times else 0
    avg_e2e = np.mean(e2e_times) if e2e_times else 0
    p95_e2e = np.percentile(e2e_times, 95) if e2e_times else 0
    det_rate = detected_count / frame_count * 100 if frame_count > 0 else 0

    # 단계별 평균/P95 계산
    stage_names = ["cam_ms", "preproc_ms", "infer_ms", "3d_ms", "draw_ms", "write_ms", "total_ms"]
    stage_avg = {}
    stage_p95 = {}
    for s in stage_names:
        vals = [t[s] for t in timing_log]
        stage_avg[s] = float(np.mean(vals)) if vals else 0
        stage_p95[s] = float(np.percentile(vals, 95)) if vals else 0

    stats = {
        "label": label,
        "frames": frame_count,
        "avg_fps": avg_fps,
        "avg_infer_ms": avg_infer,
        "avg_e2e_ms": avg_e2e,
        "p95_e2e_ms": p95_e2e,
        "detection_rate": det_rate,
        "output_file": output_path,
        "timing_breakdown_avg": stage_avg,
        "timing_breakdown_p95": stage_p95,
    }

    # 세부 타이밍 JSON 저장
    timing_json_path = output_path.replace(".mp4", "_timing.json")
    import json
    with open(timing_json_path, "w") as f:
        json.dump({
            "label": label,
            "summary": {
                "frames": frame_count,
                "avg_fps": round(avg_fps, 2),
                "avg": {k: round(v, 2) for k, v in stage_avg.items()},
                "p95": {k: round(v, 2) for k, v in stage_p95.items()},
            },
            "per_frame": timing_log,
        }, f, indent=2)

    print(f"    완료: {frame_count}f | FPS:{avg_fps:.1f}")
    print(f"      Avg  → Cam:{stage_avg['cam_ms']:.1f} Pre:{stage_avg['preproc_ms']:.1f} "
          f"Inf:{stage_avg['infer_ms']:.1f} 3D:{stage_avg['3d_ms']:.1f} "
          f"Draw:{stage_avg['draw_ms']:.1f} Write:{stage_avg['write_ms']:.1f} "
          f"Total:{stage_avg['total_ms']:.1f}ms")
    print(f"      P95  → Cam:{stage_p95['cam_ms']:.1f} Pre:{stage_p95['preproc_ms']:.1f} "
          f"Inf:{stage_p95['infer_ms']:.1f} 3D:{stage_p95['3d_ms']:.1f} "
          f"Draw:{stage_p95['draw_ms']:.1f} Write:{stage_p95['write_ms']:.1f} "
          f"Total:{stage_p95['total_ms']:.1f}ms")
    print(f"      Det:{det_rate:.0f}%")
    print(f"    저장됨: {output_path}")
    print(f"    타이밍: {timing_json_path}")

    return stats


def print_summary_table(all_stats):
    """전체 녹화 결과 요약 테이블"""
    print("\n")
    print("=" * 130)
    print("  녹화 결과 요약 (단계별 타이밍 분석)")
    print("=" * 130)
    print(f"  {'모델':<25} {'FPS':>5} {'Cam':>6} {'Pre':>6} {'Infer':>7} {'3D':>6} {'Draw':>6} {'Write':>6} {'Total':>7} {'P95Tot':>7} {'Det%':>5}")
    print("  " + "-" * 136)

    for stats in all_stats:
        if stats is None:
            continue
        avg = stats.get('timing_breakdown_avg', {})
        p95 = stats.get('timing_breakdown_p95', {})
        print(f"  {stats['label']:<25} "
              f"{stats['avg_fps']:>5.1f} "
              f"{avg.get('cam_ms', 0):>5.1f} "
              f"{avg.get('preproc_ms', 0):>5.1f} "
              f"{avg.get('infer_ms', 0):>6.1f} "
              f"{avg.get('3d_ms', 0):>5.1f} "
              f"{avg.get('draw_ms', 0):>5.1f} "
              f"{avg.get('write_ms', 0):>5.1f} "
              f"{avg.get('total_ms', 0):>6.1f} "
              f"{p95.get('total_ms', 0):>6.1f} "
              f"{stats['detection_rate']:>4.0f}%")

    print("  " + "-" * 96)

    valid = [s for s in all_stats if s is not None]
    if valid:
        best_fps = max(valid, key=lambda s: s['avg_fps'])
        best_e2e = min(valid, key=lambda s: s['avg_e2e_ms'])
        best_det = max(valid, key=lambda s: s['detection_rate'])
        print(f"\n  최고 FPS:       {best_fps['label']} ({best_fps['avg_fps']:.1f})")
        print(f"  최저 E2E:       {best_e2e['label']} ({best_e2e['avg_e2e_ms']:.1f}ms)")
        print(f"  최고 인식률:    {best_det['label']} ({best_det['detection_rate']:.0f}%)")


def main():
    parser = argparse.ArgumentParser(description="모델별 포즈 추정 데모 영상 녹화")
    parser.add_argument("--models", nargs="+",
                        default=["all"],
                        choices=["all", "mediapipe", "yolov8", "yolo11", "yolo26",
                                 "rtmpose", "rtmpose_wb", "movenet", "zed_bt"],
                        help="녹화할 모델 (기본: all)")
    parser.add_argument("--duration", type=int, default=15,
                        help="모델당 녹화 시간 (초, 기본: 15)")
    parser.add_argument("--no-zed", dest="use_zed", action="store_false",
                        help="ZED 카메라 대신 웹캠 사용")
    parser.add_argument("--resolution", default="SVGA",
                        choices=["SVGA", "VGA", "HD720", "HD1080", "HD1200"],
                        help="카메라 해상도")
    parser.add_argument("--camera-fps", type=int, default=120,
                        help="카메라 FPS")
    parser.add_argument("--video", type=str, default=None,
                        help="동영상 파일 경로 (SVO2/MP4)")
    parser.add_argument("--lower-only", action="store_true",
                        help="하체만 보이는 상황 시뮬레이션")
    parser.add_argument("--crop", type=str, default=None,
                        help="ROI 크롭 JSON 파일 경로 (기본: crop_roi.json 자동 로드)")
    parser.add_argument("--no-crop", action="store_true",
                        help="저장된 크롭 설정 무시 (원본 프레임 사용)")
    parser.add_argument("--reset-crop", action="store_true",
                        help="저장된 크롭 설정 초기화 후 원본 프레임으로 실행")
    parser.add_argument("--no-trt", action="store_true",
                        help="TRT 모델 건너뛰기")
    parser.add_argument("--trt-only", action="store_true",
                        help="TRT 모델만 실행 (PyTorch 건너뛰기)")
    parser.add_argument("--int8", action="store_true",
                        help="INT8 TensorRT 모델도 포함 (FP16 대비 비교)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="출력 디렉토리 (기본: results/demo_videos/)")
    parser.add_argument("--target-fps", type=int, default=30,
                        help="출력 영상 FPS (기본: 30)")
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

    # 출력 디렉토리 — results/demo_videos/YYYYMMDD_HHMMSS/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "results", "demo_videos", timestamp)
    os.makedirs(output_dir, exist_ok=True)

    # ROI 크롭 로드 (자동 로드 / --no-crop / --reset-crop)
    crop_roi = None
    if args.reset_crop:
        reset_crop_roi()
    elif not args.no_crop:
        crop_roi = auto_load_crop_roi(args.crop)
        if crop_roi:
            print(f"  [ROI 크롭] x={crop_roi['x']}, y={crop_roi['y']}, "
                  f"w={crop_roi['w']}, h={crop_roi['h']}")

    # TRT 가용성 확인
    has_trt, has_cuda, avail_providers = check_tensorrt_available()
    include_trt = not args.no_trt

    print("=" * 60)
    print("  포즈 추정 데모 영상 녹화")
    print("=" * 60)
    print(f"  녹화 시간: {args.duration}초/모델")
    print(f"  해상도: {args.resolution}")
    print(f"  카메라 FPS: {args.camera_fps}")
    print(f"  출력 FPS: {args.target_fps}")
    print(f"  출력 디렉토리: {output_dir}")
    print(f"  TRT 포함: {'Yes' if include_trt else 'No'}")
    print(f"  TRT 가용: {has_trt} | CUDA: {has_cuda}")
    print(f"  Providers: {avail_providers}")
    if args.video:
        print(f"  입력: {args.video}")
    else:
        print(f"  카메라: {'ZED X Mini' if args.use_zed and HAS_ZED else 'Webcam'}")
    print("=" * 60)
    print()

    # 카메라 초기화
    if args.video:
        ext = os.path.splitext(args.video)[1].lower()
        if ext not in ('.svo2', '.svo'):
            args.use_zed = False

    camera = create_camera(
        use_zed=args.use_zed,
        video_path=args.video,
        resolution=args.resolution,
        fps=args.camera_fps,
        depth_mode="PERFORMANCE",
    )
    camera.open()

    # 카메라 워밍업
    print("[카메라 워밍업]")
    for _ in range(10):
        camera.grab()
    print()

    # 모델 목록 생성
    model_list = create_all_models(args.models, include_trt=include_trt,
                                    include_int8=getattr(args, 'int8', False),
                                    trt_only=getattr(args, 'trt_only', False))
    total_models = len(model_list)
    print(f"총 {total_models}개 모델 녹화 예정\n")

    all_stats = []

    for i, (label, model) in enumerate(model_list):
        print(f"\n{'=' * 50}")
        print(f"  [{i+1}/{total_models}] {label}")
        print(f"{'=' * 50}")

        output_path = os.path.join(output_dir, f"{label}.mp4")

        try:
            # 모델 로드
            if isinstance(model, ZEDBodyTracking):
                model.load(camera)
            else:
                model.load()

            # 비디오 파일 사용 시 처음으로 되감기
            if args.video and hasattr(camera, 'seek_to_start'):
                camera.seek_to_start()

            # 녹화
            stats = record_model(
                camera, model, label, output_path,
                duration=args.duration,
                lower_only=args.lower_only,
                crop_roi=crop_roi,
                target_fps=args.target_fps,
            )
            all_stats.append(stats)

        except Exception as e:
            print(f"  [ERROR] {label} 실패: {e}")
            import traceback
            tb_str = traceback.format_exc()
            traceback.print_exc()
            # 에러 로그 파일에 기록
            error_log_path = os.path.join(output_dir, "errors.log")
            with open(error_log_path, "a", encoding="utf-8") as ef:
                ef.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {label}\n")
                ef.write(tb_str)
                ef.write("\n")
            all_stats.append(None)

        # ZED BT 정리
        if isinstance(model, ZEDBodyTracking) and hasattr(model, 'close'):
            model.close()

    # 결과 요약
    print_summary_table(all_stats)

    # 결과 JSON 저장
    import json
    summary_path = os.path.join(output_dir, "results.json")
    summary_data = {
        "timestamp": timestamp,
        "config": {
            "duration": args.duration,
            "resolution": args.resolution,
            "camera_fps": args.camera_fps,
            "target_fps": args.target_fps,
            "video": args.video,
            "trt_included": include_trt,
            "trt_available": has_trt,
        },
        "results": [s for s in all_stats if s is not None],
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"\n  요약 저장: {summary_path}")

    camera.close()
    print(f"\n완료! 영상 {len([s for s in all_stats if s is not None])}개 저장됨 → {output_dir}/")


if __name__ == "__main__":
    main()
