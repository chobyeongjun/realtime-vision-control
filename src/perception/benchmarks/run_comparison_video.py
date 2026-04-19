#!/usr/bin/env python3
"""
모델 비교 영상 생성기
=====================
5개 포즈 추정 모델의 스켈레톤 인식 결과를 영상으로 녹화하여 시각적으로 비교합니다.

출력물:
  1. 모델별 개별 영상 (스켈레톤 오버레이)
  2. 전체 모델 비교 영상 (격자 배치)
  3. 프레임별 상세 메트릭 CSV

사용법:
    python3 run_comparison_video.py --video test_data/walk.mp4
    python3 run_comparison_video.py --video test_data/walk.svo2 --duration 30
    python3 run_comparison_video.py  # 라이브 카메라
"""

import argparse
import csv
import os
import sys
import time
import traceback
from datetime import datetime

import cv2
import numpy as np

# 현재 디렉토리를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zed_camera import create_camera, HAS_ZED
from pose_models import (
    MediaPipePose, YOLOv8Pose, RTMPoseModel, RTMPoseWholebody,
    ZEDBodyTracking, draw_pose, PoseResult, SKELETON_CONNECTIONS,
    KEYPOINT_COLORS, LOWER_LIMB_KEYPOINTS,
)
from joint_angles import compute_lower_limb_angles


# ============================================================================
# 상세 오버레이 그리기 (개별 영상용)
# ============================================================================
def draw_overlay_detailed(image, pose_result, model_name="",
                          fps=0.0, frame_idx=0, timestamp_ms=0.0,
                          min_conf=0.3):
    """
    상세 스켈레톤 오버레이 그리기
    - 스켈레톤 + confidence 색상 표시
    - 관절 각도 텍스트 오버레이
    - 좌측 상단 정보 패널 (모델명, FPS, 추론 시간, E2E, 상태, 프레임 번호)
    """
    vis = draw_pose(image, pose_result, model_name=model_name,
                    min_conf=min_conf, show_angles=True,
                    show_conf_colors=True)

    h, w = vis.shape[:2]

    # ---- 하단 정보 바 ----
    has_ll = pose_result.has_lower_limb(min_conf)
    avg_conf = pose_result.get_lower_limb_confidence()

    # 프레임/타임스탬프
    ts_str = f"Frame {frame_idx} | {timestamp_ms / 1000.0:.2f}s"
    cv2.putText(vis, ts_str, (10, h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # FPS (우측 상단)
    fps_str = f"FPS: {fps:.1f}"
    (tw, _), _ = cv2.getTextSize(fps_str, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(vis, fps_str, (w - tw - 10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # E2E latency (우측)
    if pose_result.e2e_latency_ms > 0:
        e2e_str = f"E2E: {pose_result.e2e_latency_ms:.1f}ms"
        (tw2, _), _ = cv2.getTextSize(e2e_str, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.putText(vis, e2e_str, (w - tw2 - 10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

    return vis


# ============================================================================
# 격자 비교 영상 생성
# ============================================================================
def create_grid_frame(cell_images, cell_labels, grid_cols=3, grid_rows=2,
                      total_w=1920, total_h=1080):
    """
    여러 모델의 오버레이 이미지를 격자 배치하여 하나의 비교 프레임 생성.

    Args:
        cell_images: 각 셀에 들어갈 이미지 리스트 (BGR numpy)
        cell_labels: 셀별 상단 라벨 문자열
        grid_cols, grid_rows: 격자 구성 (기본 3x2 = 6칸)
        total_w, total_h: 최종 출력 해상도

    Returns:
        grid_image: 격자 배치된 이미지 (BGR numpy)
    """
    cell_w = total_w // grid_cols
    cell_h = total_h // grid_rows
    grid = np.zeros((total_h, total_w, 3), dtype=np.uint8)

    for idx in range(grid_cols * grid_rows):
        row = idx // grid_cols
        col = idx % grid_cols
        x0 = col * cell_w
        y0 = row * cell_h

        if idx < len(cell_images) and cell_images[idx] is not None:
            # 셀 크기에 맞게 리사이즈
            resized = cv2.resize(cell_images[idx], (cell_w, cell_h))
            grid[y0:y0 + cell_h, x0:x0 + cell_w] = resized
        else:
            # 빈 셀: 검은 배경
            pass

        # 셀 상단 라벨
        if idx < len(cell_labels):
            label = cell_labels[idx]
        elif idx == len(cell_images):
            label = "Summary"
        else:
            label = ""

        if label:
            # 반투명 라벨 바
            overlay = grid[y0:y0 + 28, x0:x0 + cell_w].copy()
            cv2.rectangle(grid, (x0, y0), (x0 + cell_w, y0 + 28), (0, 0, 0), -1)
            cv2.addWeighted(grid[y0:y0 + 28, x0:x0 + cell_w], 0.7,
                            overlay, 0.3, 0,
                            grid[y0:y0 + 28, x0:x0 + cell_w])
            cv2.putText(grid, label, (x0 + 5, y0 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        # 셀 테두리
        cv2.rectangle(grid, (x0, y0), (x0 + cell_w - 1, y0 + cell_h - 1),
                      (80, 80, 80), 1)

    return grid


def draw_summary_cell(cell_w, cell_h, model_stats, frame_idx):
    """
    격자 마지막 셀에 표시할 요약 메트릭 패널.

    Args:
        cell_w, cell_h: 셀 크기
        model_stats: {model_name: {"avg_infer_ms", "avg_fps", "detect_rate", ...}}
        frame_idx: 현재 프레임 인덱스

    Returns:
        summary_image (BGR numpy)
    """
    panel = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 제목
    cv2.putText(panel, "=== Metrics Summary ===", (10, 30),
                font, 0.55, (0, 255, 255), 1)
    cv2.putText(panel, f"Frame: {frame_idx}", (10, 55),
                font, 0.45, (200, 200, 200), 1)

    # 테이블 헤더
    y = 85
    cv2.putText(panel, f"{'Model':<22} {'FPS':>5} {'Inf':>6} {'Det':>4}",
                (10, y), font, 0.38, (180, 180, 180), 1)
    y += 5
    cv2.line(panel, (10, y), (cell_w - 10, y), (100, 100, 100), 1)
    y += 18

    for name, st in model_stats.items():
        short_name = name[:20]
        fps_val = st.get("avg_fps", 0.0)
        infer_val = st.get("avg_infer_ms", 0.0)
        det_str = "Y" if st.get("last_detected", False) else "N"
        color = (0, 255, 0) if st.get("last_detected", False) else (0, 0, 255)

        line = f"{short_name:<22} {fps_val:>5.1f} {infer_val:>5.1f} {det_str:>4}"
        cv2.putText(panel, line, (10, y), font, 0.36, color, 1)
        y += 18
        if y > cell_h - 20:
            break

    return panel


# ============================================================================
# CSV 컬럼 정의
# ============================================================================
# 하체 keypoint 이름 목록 (CSV 컬럼용)
_KP_NAMES = [
    "left_hip", "left_knee", "left_ankle", "left_heel", "left_toe",
    "right_hip", "right_knee", "right_ankle", "right_heel", "right_toe",
]

# 관절 각도 이름 목록
_ANGLE_NAMES = [
    "left_knee_flexion", "right_knee_flexion",
    "left_hip_flexion", "right_hip_flexion",
    "left_ankle_dorsiflexion", "right_ankle_dorsiflexion",
]


def build_csv_header():
    """CSV 헤더 행 생성"""
    cols = [
        "frame_idx", "timestamp_ms", "model_name",
        "inference_ms", "grab_ms", "e2e_ms",
        "detected", "lower_limb_detected",
    ]
    # per-keypoint confidence
    for kp in _KP_NAMES:
        cols.append(f"{kp}_conf")
    # per-keypoint 2D 좌표
    for kp in _KP_NAMES:
        cols.append(f"{kp}_x")
        cols.append(f"{kp}_y")
    # 관절 각도
    for angle in _ANGLE_NAMES:
        cols.append(angle)
    return cols


def build_csv_row(frame_idx, timestamp_ms, model_name, pose_result):
    """PoseResult에서 CSV 한 행 생성"""
    row = {
        "frame_idx": frame_idx,
        "timestamp_ms": f"{timestamp_ms:.1f}",
        "model_name": model_name,
        "inference_ms": f"{pose_result.inference_time_ms:.2f}",
        "grab_ms": f"{pose_result.grab_time_ms:.2f}",
        "e2e_ms": f"{pose_result.e2e_latency_ms:.2f}",
        "detected": int(pose_result.detected),
        "lower_limb_detected": int(pose_result.has_lower_limb()),
    }
    for kp in _KP_NAMES:
        row[f"{kp}_conf"] = f"{pose_result.confidences.get(kp, 0.0):.4f}"
        if kp in pose_result.keypoints_2d:
            x, y = pose_result.keypoints_2d[kp]
            row[f"{kp}_x"] = f"{x:.1f}"
            row[f"{kp}_y"] = f"{y:.1f}"
        else:
            row[f"{kp}_x"] = ""
            row[f"{kp}_y"] = ""
    for angle in _ANGLE_NAMES:
        if angle in pose_result.joint_angles:
            row[angle] = f"{pose_result.joint_angles[angle]:.2f}"
        else:
            row[angle] = ""
    return row


# ============================================================================
# 모델 로드 헬퍼
# ============================================================================
def load_models(selected_models, use_zed=True, camera=None):
    """
    선택된 모델을 로드하고 {이름: 모델} 딕셔너리를 반환.
    로드 실패한 모델은 경고만 출력하고 건너뜀.
    """
    available = {}

    # 모델 정의: (키, 클래스, kwargs, zed_only 여부)
    model_defs = [
        ("mediapipe",    MediaPipePose,      {"model_complexity": 1}, False),
        ("yolo26",       YOLOv8Pose,         {"model_size": "n", "yolo_version": "v26"}, False),
        ("rtmpose",      RTMPoseModel,       {"mode": "balanced", "device": "cuda"}, False),
        ("rtmpose_wb",   RTMPoseWholebody,   {"mode": "balanced", "device": "cuda"}, False),
        ("zed_bt",       ZEDBodyTracking,    {"model": "FAST"},       True),
    ]

    for key, cls, kwargs, zed_only in model_defs:
        if selected_models and key not in selected_models:
            continue
        if zed_only and (not use_zed or not HAS_ZED):
            print(f"  [SKIP] {key}: ZED 카메라/SDK 필요 (--no-zed 또는 SDK 미설치)")
            continue

        try:
            model = cls(**kwargs)
            print(f"  [{key}] 모델 로드 중...")
            if key == "zed_bt":
                model.load(camera)
            else:
                model.load()
            available[model.name] = model
            print(f"  [{key}] 로드 성공: {model.name}")
        except Exception as e:
            print(f"  [WARNING] {key} 로드 실패: {e}")
            traceback.print_exc()

    return available


# ============================================================================
# 메인 비교 실행기
# ============================================================================
class ComparisonVideoRunner:
    """모든 모델의 비교 영상 생성"""

    def __init__(self, args):
        self.args = args
        self.output_dir = args.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # 타임스탬프 (출력 파일 접두사)
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ----------------------------------------------------------------
    # 진입점
    # ----------------------------------------------------------------
    def run(self):
        print("=" * 70)
        print("  모델 비교 영상 생성기")
        print("=" * 70)
        if self.args.video:
            print(f"  입력: {self.args.video}")
        else:
            cam_type = "ZED X Mini" if (not self.args.no_zed and HAS_ZED) else "Webcam"
            print(f"  카메라: {cam_type}")
        print(f"  최대 시간: {self.args.duration}초")
        print(f"  최대 프레임: {self.args.max_frames or '무제한'}")
        print(f"  출력 디렉토리: {self.output_dir}")
        print("=" * 70)
        print()

        # ---- 카메라/입력 소스 열기 ----
        use_zed = not self.args.no_zed
        camera = create_camera(
            use_zed=use_zed,
            video_path=self.args.video,
            resolution=self.args.resolution,
            fps=self.args.camera_fps,
        )
        camera.open()

        # 입력 FPS (영상 파일이면 원본 FPS 사용)
        cam_info = camera.get_camera_info_dict()
        source_fps = cam_info.get("fps", 30) or 30
        if source_fps <= 0:
            source_fps = 30

        # ---- 모델 로드 ----
        print("\n[모델 로드]")
        selected = self.args.models  # None이면 전체
        models = load_models(selected, use_zed=use_zed, camera=camera)

        if not models:
            print("[ERROR] 로드된 모델이 없습니다. 종료합니다.")
            camera.close()
            return

        model_names = list(models.keys())
        print(f"\n  로드 완료: {len(models)}개 모델")
        for mn in model_names:
            print(f"    - {mn}")
        print()

        # ---- 워밍업 ----
        print("[워밍업] 각 모델 10 프레임...")
        for _ in range(10):
            if not camera.grab():
                break
            rgb = camera.get_rgb()
            for model in models.values():
                try:
                    model.predict(rgb)
                except Exception:
                    pass
        print()

        # ---- VideoWriter 초기화 ----
        # 첫 프레임 해상도 확인
        camera.grab()
        sample_frame = camera.get_rgb()
        if sample_frame is None:
            print("[ERROR] 프레임을 읽을 수 없습니다.")
            camera.close()
            return
        fh, fw = sample_frame.shape[:2]
        print(f"  프레임 크기: {fw}x{fh}")

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_fps = min(source_fps, 30)  # 출력 FPS (최대 30)

        # 모델별 개별 VideoWriter
        individual_writers = {}
        for mn in model_names:
            safe_name = mn.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
            path = os.path.join(self.output_dir, f"{self.ts}_{safe_name}.mp4")
            writer = cv2.VideoWriter(path, fourcc, out_fps, (fw, fh))
            if not writer.isOpened():
                print(f"  [WARNING] VideoWriter 열기 실패: {path}")
                writer = None
            individual_writers[mn] = (writer, path)

        # 격자 비교 영상 VideoWriter
        grid_cols = 3
        grid_rows = 2
        grid_w = 1920
        grid_h = 1080
        grid_path = os.path.join(self.output_dir, f"{self.ts}_comparison_grid.mp4")
        grid_writer = cv2.VideoWriter(grid_path, fourcc, out_fps, (grid_w, grid_h))
        if not grid_writer.isOpened():
            print(f"  [WARNING] 격자 영상 VideoWriter 열기 실패: {grid_path}")
            grid_writer = None

        # ---- CSV 초기화 ----
        csv_path = os.path.join(self.output_dir, f"{self.ts}_metrics.csv")
        csv_header = build_csv_header()
        csv_file = open(csv_path, "w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_header)
        csv_writer.writeheader()

        # ---- 프레임 루프 ----
        print("[녹화 시작]")
        frame_idx = 0
        start_time = time.perf_counter()
        last_progress_time = start_time

        # 모델별 누적 통계 (요약 패널용)
        model_accum = {mn: {
            "infer_sum": 0.0, "frame_count": 0, "detect_count": 0,
            "avg_fps": 0.0, "avg_infer_ms": 0.0, "last_detected": False,
        } for mn in model_names}

        # FPS 계산용
        frame_times = []

        try:
            while True:
                elapsed = time.perf_counter() - start_time

                # 종료 조건: 시간 초과
                if elapsed >= self.args.duration:
                    print(f"\n  시간 제한 도달 ({self.args.duration}초)")
                    break

                # 종료 조건: 프레임 수 제한
                if self.args.max_frames and frame_idx >= self.args.max_frames:
                    print(f"\n  프레임 제한 도달 ({self.args.max_frames})")
                    break

                # 프레임 가져오기
                t_grab_start = time.perf_counter()
                if not camera.grab():
                    print("\n  입력 종료 (더 이상 프레임 없음)")
                    break
                rgb = camera.get_rgb()
                t_grab_end = time.perf_counter()
                grab_ms = (t_grab_end - t_grab_start) * 1000

                if rgb is None:
                    continue

                timestamp_ms = elapsed * 1000.0

                # depth (있으면)
                depth = None
                if hasattr(camera, "get_depth"):
                    depth = camera.get_depth()

                # ---- 각 모델 추론 ----
                cell_images = []
                cell_labels = []

                for mn in model_names:
                    model = models[mn]
                    result = PoseResult()

                    try:
                        t_infer = time.perf_counter()
                        result = model.predict(rgb)
                        infer_ms = (time.perf_counter() - t_infer) * 1000
                    except Exception:
                        infer_ms = 0.0

                    result.inference_time_ms = infer_ms
                    result.grab_time_ms = grab_ms
                    result.e2e_latency_ms = grab_ms + infer_ms

                    # 2D→3D 변환 (depth 있을 때)
                    if depth is not None and result.detected:
                        for kp_name, (px, py) in result.keypoints_2d.items():
                            pt3d = camera.pixel_to_3d(px, py, depth)
                            if pt3d is not None:
                                result.keypoints_3d[kp_name] = tuple(pt3d)

                    # 관절 각도 계산
                    use_3d = bool(result.keypoints_3d)
                    result.joint_angles = compute_lower_limb_angles(
                        result, use_3d=use_3d)

                    # 누적 통계 갱신
                    acc = model_accum[mn]
                    acc["infer_sum"] += infer_ms
                    acc["frame_count"] += 1
                    if result.detected:
                        acc["detect_count"] += 1
                    acc["last_detected"] = result.detected
                    acc["avg_infer_ms"] = acc["infer_sum"] / acc["frame_count"]

                    # FPS (최근 30프레임 기준)
                    current_fps = 0.0
                    if len(frame_times) > 0:
                        recent = frame_times[-30:]
                        current_fps = len(recent) / sum(recent) if sum(recent) > 0 else 0
                    acc["avg_fps"] = current_fps

                    # 개별 오버레이 이미지 생성
                    overlay = draw_overlay_detailed(
                        rgb, result, model_name=mn,
                        fps=current_fps, frame_idx=frame_idx,
                        timestamp_ms=timestamp_ms, min_conf=0.3)

                    # 개별 영상 저장
                    w_pair = individual_writers.get(mn)
                    if w_pair and w_pair[0] is not None:
                        w_pair[0].write(overlay)

                    # 격자용 이미지/라벨 수집
                    cell_images.append(overlay)
                    short = mn[:25]
                    cell_labels.append(
                        f"{short} | {infer_ms:.1f}ms | "
                        f"{'OK' if result.has_lower_limb() else 'FAIL'}")

                    # CSV 행 기록
                    csv_row = build_csv_row(frame_idx, timestamp_ms, mn, result)
                    csv_writer.writerow(csv_row)

                # ---- 격자 비교 영상 프레임 생성 ----
                if grid_writer is not None:
                    # 6번째 셀: 요약 패널 (5개 모델 + 1 요약)
                    cell_w = grid_w // grid_cols
                    cell_h = grid_h // grid_rows
                    summary_img = draw_summary_cell(
                        cell_w, cell_h, model_accum, frame_idx)
                    cell_images_grid = list(cell_images)
                    cell_labels_grid = list(cell_labels)
                    # 모델 수가 격자 칸보다 적으면 요약 패널 추가
                    while len(cell_images_grid) < grid_cols * grid_rows - 1:
                        cell_images_grid.append(None)
                        cell_labels_grid.append("")
                    cell_images_grid.append(summary_img)
                    cell_labels_grid.append("Summary")

                    grid_frame = create_grid_frame(
                        cell_images_grid, cell_labels_grid,
                        grid_cols=grid_cols, grid_rows=grid_rows,
                        total_w=grid_w, total_h=grid_h)
                    grid_writer.write(grid_frame)

                # 프레임 시간 기록
                frame_end = time.perf_counter()
                frame_dt = frame_end - t_grab_start
                frame_times.append(frame_dt)

                frame_idx += 1

                # ---- 2초마다 진행 상황 출력 ----
                now = time.perf_counter()
                if now - last_progress_time >= 2.0:
                    overall_fps = frame_idx / (now - start_time) if (now - start_time) > 0 else 0
                    parts = []
                    for mn in model_names:
                        acc = model_accum[mn]
                        short = mn[:15]
                        parts.append(f"{short}={acc['avg_infer_ms']:.1f}ms")
                    infer_info = ", ".join(parts)
                    print(f"  {elapsed:.0f}s | {frame_idx} frames | "
                          f"Pipeline FPS: {overall_fps:.1f} | {infer_info}")
                    last_progress_time = now

        except KeyboardInterrupt:
            print("\n  [Ctrl+C] 녹화 중단")

        # ---- 정리 ----
        total_elapsed = time.perf_counter() - start_time
        print(f"\n[녹화 완료] {frame_idx} 프레임, {total_elapsed:.1f}초")

        # VideoWriter 닫기
        for mn, (writer, path) in individual_writers.items():
            if writer is not None:
                writer.release()
                print(f"  개별 영상: {path}")
        if grid_writer is not None:
            grid_writer.release()
            print(f"  격자 비교 영상: {grid_path}")

        # CSV 닫기
        csv_file.close()
        print(f"  메트릭 CSV: {csv_path}")

        # ZED BT 정리
        for mn, model in models.items():
            if hasattr(model, "close"):
                try:
                    model.close()
                except Exception:
                    pass

        camera.close()

        # ---- 최종 요약 테이블 ----
        self._print_summary(model_accum, frame_idx, total_elapsed)

    # ----------------------------------------------------------------
    # 요약 출력
    # ----------------------------------------------------------------
    def _print_summary(self, model_accum, total_frames, total_elapsed):
        """모든 모델의 평균 메트릭 비교 테이블 출력"""
        print()
        print("=" * 80)
        print("  모델 비교 요약")
        print("=" * 80)
        print(f"  총 프레임: {total_frames}, 총 시간: {total_elapsed:.1f}초")
        print()
        print(f"  {'모델':<35} {'Avg Infer(ms)':>14} {'인식률(%)':>10} {'Pipeline FPS':>13}")
        print(f"  {'-' * 72}")

        for mn, acc in model_accum.items():
            n = acc["frame_count"]
            if n == 0:
                print(f"  {mn:<35} {'N/A':>14} {'N/A':>10} {'N/A':>13}")
                continue
            avg_infer = acc["infer_sum"] / n
            det_rate = acc["detect_count"] / n * 100 if n > 0 else 0
            fps = n / total_elapsed if total_elapsed > 0 else 0
            print(f"  {mn:<35} {avg_infer:>13.1f} {det_rate:>9.1f} {fps:>12.1f}")

        print(f"  {'-' * 72}")
        print()


# ============================================================================
# CLI
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="포즈 추정 모델 비교 영상 생성기")

    parser.add_argument("--video", type=str, default=None,
                        help="입력 영상 경로 (mp4, svo2). 미지정 시 라이브 카메라")
    parser.add_argument("--duration", type=float, default=30,
                        help="최대 녹화 시간 (초, 기본: 30)")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(
                            os.path.dirname(os.path.abspath(__file__)),
                            "results", "comparison_videos"),
                        help="출력 디렉토리 (기본: results/comparison_videos/)")
    parser.add_argument("--models", nargs="+", default=None,
                        choices=["mediapipe", "yolo26", "rtmpose",
                                 "rtmpose_wb", "zed_bt"],
                        help="특정 모델만 선택 (기본: 전체)")
    parser.add_argument("--resolution", default="SVGA",
                        choices=["SVGA", "VGA", "HD720", "HD1080", "HD1200"],
                        help="카메라 해상도 (기본: SVGA)")
    parser.add_argument("--camera-fps", type=int, default=30,
                        help="카메라 FPS (기본: 30)")
    parser.add_argument("--no-zed", action="store_true",
                        help="ZED 카메라 대신 웹캠 사용")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="최대 처리 프레임 수 (기본: 무제한)")

    args = parser.parse_args()

    # --video 사용 시: SVO2는 ZED SDK 유지, 일반 영상은 ZED 비활성화
    if args.video:
        ext = os.path.splitext(args.video)[1].lower()
        if ext not in (".svo2", ".svo"):
            args.no_zed = True

    runner = ComparisonVideoRunner(args)
    runner.run()


if __name__ == "__main__":
    main()
