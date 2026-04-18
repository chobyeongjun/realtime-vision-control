#!/usr/bin/env python3
"""
YOLOv8 포즈 추정 실시간 테스트 (크롭 + 추론 파라미터 제어)
==========================================================

사용법:
    # 실시간 ZED 카메라 (기본)
    python3 run_cropped_demo.py

    # 실시간 웹캠
    python3 run_cropped_demo.py --no-zed

    # 동영상 파일
    python3 run_cropped_demo.py --video recordings/walking.mp4

    # YOLOv8s + TRT + One Euro Filter 스무딩
    python3 run_cropped_demo.py --model-size s --trt --smoothing 0.3

    # 해상도 높여서 정확도 올리기
    python3 run_cropped_demo.py --imgsz 640

    # 민감하게 (낮은 conf로 약한 keypoint도 잡기)
    python3 run_cropped_demo.py --conf 0.15

조작:
    q / ESC : 종료
    c       : 크롭 토글 (ON/OFF)
    s       : 스크린샷 저장
    +/-     : confidence threshold 조절
    [/]     : One Euro Filter min_cutoff 조절 (떨림 제거 강도)
    1/2     : 입력 해상도 변경 (640/480/320 순환)
    a       : 관절 각도 표시 토글
    h       : conf 히트맵 색상 토글
    r       : 파라미터 리셋
    i       : 현재 설정 출력
"""

import argparse
import time
import sys
import os
import cv2
import numpy as np
from collections import deque

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zed_camera import create_camera, HAS_ZED
from select_roi import auto_load_crop_roi, apply_crop
from pose_models import YOLOv8Pose, draw_pose


IMGSZ_OPTIONS = [640, 480, 320]


def print_settings(model, show_angles, show_conf_colors, crop_enabled):
    """현재 설정 출력"""
    print(f"\n  === 현재 설정 ===")
    print(f"  conf:      {model.conf:.2f}")
    print(f"  iou:       {model.iou:.2f}")
    print(f"  imgsz:     {model.imgsz}")
    filter_status = "ON" if model.smoothing > 0 else "OFF"
    print(f"  filter:    {filter_status} (One Euro Filter)")
    if model.smoothing > 0:
        print(f"    min_cutoff: {model.filter_min_cutoff:.1f}")
        print(f"    beta:       {model.filter_beta:.3f}")
    print(f"  half:      {model.half}")
    print(f"  max_det:   {model.max_det}")
    print(f"  classes:   {model.classes}")
    print(f"  crop:      {'ON' if crop_enabled else 'OFF'}")
    print(f"  angles:    {'ON' if show_angles else 'OFF'}")
    print(f"  heatmap:   {'ON' if show_conf_colors else 'OFF'}")
    print()


def draw_settings_panel(vis, model, show_angles, show_conf_colors, crop_enabled, fps, infer_ms):
    """좌측 하단에 실시간 설정 패널 표시"""
    h, w = vis.shape[:2]

    filter_status = "ON" if model.smoothing > 0 else "OFF"
    lines = [
        f"conf: {model.conf:.2f} (+/-)",
        f"1EF: {filter_status} ([/])",
        f"imgsz: {model.imgsz} (1/2)",
        f"angles: {'ON' if show_angles else 'OFF'} (a)",
        f"heatmap: {'ON' if show_conf_colors else 'OFF'} (h)",
    ]
    if model.smoothing > 0:
        lines.insert(2, f"  mc={model.filter_min_cutoff:.1f} b={model.filter_beta:.3f}")

    # 반투명 배경
    panel_h = len(lines) * 22 + 10
    panel_w = 220
    py = h - panel_h - 5
    overlay = vis.copy()
    cv2.rectangle(overlay, (5, py), (5 + panel_w, h - 5), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, vis, 0.3, 0, vis)

    for i, line in enumerate(lines):
        cv2.putText(vis, line, (10, py + 18 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    return vis


def main():
    parser = argparse.ArgumentParser(
        description="YOLOv8 포즈 추정 실시간 테스트 (크롭 + 파라미터 제어)",
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # 소스
    parser.add_argument("--video", type=str, default=None,
                        help="동영상 파일 경로 (생략하면 실시간 카메라)")
    parser.add_argument("--no-zed", dest="use_zed", action="store_false",
                        help="ZED 카메라 대신 웹캠 사용")
    parser.add_argument("--resolution", default="SVGA",
                        choices=["SVGA", "VGA", "HD720", "HD1080"])

    # 모델
    parser.add_argument("--model-size", type=str, default="n", choices=["n", "s"])
    parser.add_argument("--yolo-version", type=str, default="v8", choices=["v8", "v11"],
                        help="YOLO 버전 (v8=YOLOv8, v11=YOLO11, 기본: v8)")
    parser.add_argument("--trt", action="store_true", help="TensorRT 엔진 사용")

    # 크롭
    parser.add_argument("--crop", type=str, default=None,
                        help="ROI 크롭 JSON 경로")
    parser.add_argument("--no-crop", action="store_true")
    parser.add_argument("--center-crop", type=int, default=640, metavar="WIDTH",
                        help="프레임 가로 중앙 크롭 (기본: 640, 0=크롭 없음)")

    # 추론 파라미터
    parser.add_argument("--conf", type=float, default=0.15,
                        help="confidence threshold (기본: 0.15)")
    parser.add_argument("--iou", type=float, default=0.7,
                        help="NMS IoU threshold (기본: 0.7)")
    parser.add_argument("--imgsz", type=int, default=640,
                        choices=[640, 480, 320],
                        help="입력 해상도 (기본: 640)")
    parser.add_argument("--max-det", type=int, default=1,
                        help="최대 검출 인원 (기본: 1)")
    parser.add_argument("--smoothing", type=float, default=0.3,
                        help="One Euro Filter ON/OFF (0=OFF, >0=ON, 기본 0.3)")
    parser.add_argument("--filter-min-cutoff", type=float, default=1.0,
                        help="One Euro Filter min_cutoff: 떨림 제거 강도 (낮을수록 강하게, 기본 1.0)")
    parser.add_argument("--filter-beta", type=float, default=0.007,
                        help="One Euro Filter beta: 지연 감소 (높을수록 빠른 동작 추종, 기본 0.007)")

    # 세그먼트 길이 제약
    parser.add_argument("--no-segment-constraint", dest="segment_constraint",
                        action="store_false",
                        help="세그먼트 길이 제약 비활성화")
    parser.add_argument("--seg-calib-frames", type=int, default=30,
                        help="세그먼트 캘리브레이션 프레임 수 (기본 30, ~1초@30fps)")
    parser.add_argument("--seg-tolerance", type=float, default=0.15,
                        help="세그먼트 길이 허용 오차 (0.15 = ±15%%)")

    # 출력
    parser.add_argument("--save-video", type=str, default=None,
                        help="결과 영상 저장 경로 (MP4)")
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

    # ROI 로드
    crop_roi = None
    if not args.no_crop:
        crop_roi = auto_load_crop_roi(args.crop)

    if crop_roi:
        print(f"[ROI 크롭] x={crop_roi['x']}, y={crop_roi['y']}, "
              f"w={crop_roi['w']}, h={crop_roi['h']}")
    else:
        print("[크롭 없음] 전체 프레임 사용")

    # 카메라/비디오 열기
    if args.video:
        use_zed = os.path.splitext(args.video)[1].lower() in ('.svo2', '.svo')
    else:
        use_zed = args.use_zed

    camera = create_camera(
        use_zed=use_zed,
        video_path=args.video,
        resolution=args.resolution,
        depth_mode="NONE",
    )
    camera.open()

    # 모델 로드
    trt_label = "_TRT" if args.trt else ""
    version_label = "YOLO11" if args.yolo_version == "v11" else "YOLOv8"
    model_label = f"{version_label}{args.model_size}{trt_label}"
    print(f"[모델 로드] {model_label}...")
    model = YOLOv8Pose(
        model_size=args.model_size,
        use_tensorrt=args.trt,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        max_det=args.max_det,
        smoothing=args.smoothing,
        yolo_version=args.yolo_version,
        filter_min_cutoff=args.filter_min_cutoff,
        filter_beta=args.filter_beta,
        segment_constraint=args.segment_constraint,
        seg_calib_frames=args.seg_calib_frames,
        seg_tolerance=args.seg_tolerance,
    )
    model.load()
    print(f"[모델 준비 완료]")

    # 초기 설정값 저장 (리셋용)
    init_conf = args.conf
    init_smoothing = args.smoothing
    init_imgsz = args.imgsz
    init_min_cutoff = args.filter_min_cutoff
    init_beta = args.filter_beta

    # VideoWriter
    writer = None

    # 런타임 변수
    fps_window = deque(maxlen=30)
    crop_enabled = crop_roi is not None
    show_angles = True
    show_conf_colors = True
    frame_count = 0
    imgsz_idx = IMGSZ_OPTIONS.index(args.imgsz) if args.imgsz in IMGSZ_OPTIONS else 0

    print("\n  === 실시간 포즈 추정 ===")
    print("  q/ESC: 종료 | c: 크롭 | +/-: conf | [/]: 필터 강도 | 1/2: 해상도")
    print("  a: 각도 | h: 히트맵 | s: 스크린샷 | i: 설정확인 | r: 리셋")
    print()

    while True:
        if not camera.grab():
            if args.video:
                if hasattr(camera, 'seek_to_start'):
                    camera.seek_to_start()
                    continue
                break
            continue

        rgb = camera.get_rgb()

        # 가로 중앙 크롭 (960→640)
        if args.center_crop > 0:
            h, w = rgb.shape[:2]
            if w > args.center_crop:
                offset = (w - args.center_crop) // 2
                rgb = rgb[:, offset:offset + args.center_crop]

        # ROI 크롭 적용
        if crop_enabled and crop_roi:
            input_frame = apply_crop(rgb, crop_roi)
        else:
            input_frame = rgb

        # 추론
        t1 = time.perf_counter()
        result = model.predict(input_frame)
        t2 = time.perf_counter()
        infer_ms = (t2 - t1) * 1000

        # FPS 계산
        fps_window.append(time.perf_counter())
        if len(fps_window) >= 2:
            current_fps = (len(fps_window) - 1) / (fps_window[-1] - fps_window[0])
        else:
            current_fps = 0

        # 시각화
        vis = draw_pose(input_frame, result, model_label,
                        show_angles=show_angles,
                        show_conf_colors=show_conf_colors)
        h, w = vis.shape[:2]

        # 상단 HUD
        overlay = vis.copy()
        cv2.rectangle(overlay, (0, 0), (w, 85), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

        cv2.putText(vis, f"{model_label}  FPS: {current_fps:.1f}  Infer: {infer_ms:.1f}ms",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        crop_status = "CROP: ON" if crop_enabled else "CROP: OFF"
        crop_color = (0, 255, 0) if crop_enabled else (0, 0, 255)
        cv2.putText(vis, crop_status,
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, crop_color, 2)

        detect_status = "Detected" if result.detected else "No Detection"
        detect_color = (0, 255, 0) if result.detected else (0, 0, 255)
        cv2.putText(vis, detect_status,
                    (200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, detect_color, 2)

        ll_conf = result.get_lower_limb_confidence()
        cv2.putText(vis, f"Lower Limb conf: {ll_conf:.2f}",
                    (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)

        # 세그먼트 캘리브레이션 상태 표시
        seg = getattr(model, '_seg_constraint', None)
        if seg is not None:
            if not seg.calibrated:
                pct = int(seg.progress * 100)
                # 프로그레스 바 표시
                bar_w = 150
                bar_x = w - bar_w - 15
                bar_y = 10
                cv2.rectangle(vis, (bar_x, bar_y), (bar_x + bar_w, bar_y + 18),
                              (50, 50, 50), -1)
                fill_w = int(bar_w * seg.progress)
                cv2.rectangle(vis, (bar_x, bar_y), (bar_x + fill_w, bar_y + 18),
                              (0, 200, 255), -1)
                cv2.putText(vis, f"Calib: {pct}%",
                            (bar_x + 5, bar_y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)
            else:
                cv2.putText(vis, "Bone OK",
                            (w - 80, 24),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 150), 1)

        # 관절 각도 상단 표시
        if show_angles and result.joint_angles:
            angle_text = "  ".join(f"{k.replace('_angle','')}: {v:.0f}\u00b0"
                                   for k, v in result.joint_angles.items())
            cv2.putText(vis, angle_text,
                        (300, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 100), 1)

        # 설정 패널
        vis = draw_settings_panel(vis, model, show_angles, show_conf_colors,
                                  crop_enabled, current_fps, infer_ms)

        cv2.imshow("YOLOv8 Pose Control", vis)

        # VideoWriter (실시간 FPS 기반)
        if args.save_video and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(args.save_video, fourcc, round(max(current_fps, 15)), (w, h))
        if writer:
            writer.write(vis)

        frame_count += 1

        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord('c'):
            crop_enabled = not crop_enabled
            print(f"  크롭: {'ON' if crop_enabled else 'OFF'}")
        elif key == ord('+') or key == ord('='):
            model.conf = min(0.9, model.conf + 0.05)
            print(f"  conf: {model.conf:.2f}")
        elif key == ord('-') or key == ord('_'):
            model.conf = max(0.05, model.conf - 0.05)
            print(f"  conf: {model.conf:.2f}")
        elif key == ord(']'):
            if model.smoothing > 0:
                model.filter_min_cutoff = max(0.1, model.filter_min_cutoff - 0.2)
                if model._kp_filter is not None:
                    model._kp_filter.min_cutoff = model.filter_min_cutoff
                    for fx, fy in model._kp_filter._filters.values():
                        fx.min_cutoff = model.filter_min_cutoff
                        fy.min_cutoff = model.filter_min_cutoff
                print(f"  filter min_cutoff: {model.filter_min_cutoff:.1f} (떨림 제거 강하게)")
        elif key == ord('['):
            if model.smoothing > 0:
                model.filter_min_cutoff = min(5.0, model.filter_min_cutoff + 0.2)
                if model._kp_filter is not None:
                    model._kp_filter.min_cutoff = model.filter_min_cutoff
                    for fx, fy in model._kp_filter._filters.values():
                        fx.min_cutoff = model.filter_min_cutoff
                        fy.min_cutoff = model.filter_min_cutoff
                print(f"  filter min_cutoff: {model.filter_min_cutoff:.1f} (원본에 가깝게)")
        elif key == ord('1'):
            imgsz_idx = (imgsz_idx + 1) % len(IMGSZ_OPTIONS)
            model.imgsz = IMGSZ_OPTIONS[imgsz_idx]
            print(f"  imgsz: {model.imgsz}")
        elif key == ord('2'):
            imgsz_idx = (imgsz_idx - 1) % len(IMGSZ_OPTIONS)
            model.imgsz = IMGSZ_OPTIONS[imgsz_idx]
            print(f"  imgsz: {model.imgsz}")
        elif key == ord('a'):
            show_angles = not show_angles
            print(f"  관절 각도: {'ON' if show_angles else 'OFF'}")
        elif key == ord('h'):
            show_conf_colors = not show_conf_colors
            print(f"  히트맵 색상: {'ON' if show_conf_colors else 'OFF'}")
        elif key == ord('r'):
            model.conf = init_conf
            model.smoothing = init_smoothing
            model.filter_min_cutoff = init_min_cutoff
            model.filter_beta = init_beta
            if model._kp_filter is not None:
                model._kp_filter.min_cutoff = init_min_cutoff
                model._kp_filter.beta = init_beta
                for fx, fy in model._kp_filter._filters.values():
                    fx.min_cutoff = init_min_cutoff
                    fx.beta = init_beta
                    fy.min_cutoff = init_min_cutoff
                    fy.beta = init_beta
            model.imgsz = init_imgsz
            imgsz_idx = IMGSZ_OPTIONS.index(init_imgsz) if init_imgsz in IMGSZ_OPTIONS else 0
            show_angles = True
            show_conf_colors = True
            print(f"  설정 리셋 완료")
        elif key == ord('i'):
            print_settings(model, show_angles, show_conf_colors, crop_enabled)
        elif key == ord('s'):
            ss_path = f"screenshot_{frame_count:05d}.jpg"
            cv2.imwrite(ss_path, vis)
            print(f"  스크린샷: {ss_path}")

    # 정리
    cv2.destroyAllWindows()
    if writer:
        writer.release()
        print(f"  영상 저장: {args.save_video}")
    camera.close()
    print(f"\n  총 {frame_count} 프레임 처리 완료")


if __name__ == "__main__":
    main()
