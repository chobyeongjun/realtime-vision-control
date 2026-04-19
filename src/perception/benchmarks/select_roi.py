#!/usr/bin/env python3
"""
인터랙티브 ROI(관심 영역) 선택 도구
====================================
카메라 또는 동영상에서 첫 프레임을 띄우고,
마우스로 크롭 영역을 드래그하여 선택합니다.

선택 결과는 crop_roi.json 으로 저장되며,
run_benchmark.py 와 run_record_demo.py 에서 자동으로 로드됩니다.
(--no-crop으로 일시 비활성, --reset-crop으로 초기화 가능)

사용법:
    # ZED 카메라에서 ROI 선택 (한 번만 하면 됨)
    python3 select_roi.py

    # 녹화된 영상에서 ROI 선택
    python3 select_roi.py --video recordings/walking_20260319_160624.mp4

    # 크롭 초기화 (삭제)
    python3 select_roi.py --reset

    # 저장 경로 지정
    python3 select_roi.py --output my_roi.json

조작:
    - 마우스 드래그: ROI 사각형 그리기
    - ENTER / SPACE: 확정
    - R: 다시 그리기 (리셋)
    - ESC / Q: 취소
"""

import argparse
import json
import os
import sys
import cv2

# 같은 디렉토리의 zed_camera 임포트
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zed_camera import create_camera


def select_roi_from_frame(frame, window_name="Select ROI - drag, then ENTER to confirm"):
    """OpenCV selectROI 를 이용한 인터랙티브 ROI 선택"""
    print("\n  [ROI 선택]")
    print("    마우스로 크롭 영역을 드래그하세요.")
    print("    ENTER/SPACE: 확정  |  R: 리셋  |  ESC/Q: 취소\n")

    roi = cv2.selectROI(window_name, frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)

    x, y, w, h = roi
    if w == 0 or h == 0:
        print("  [취소] ROI가 선택되지 않았습니다.")
        return None

    return {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}


def default_crop_path():
    """기본 crop_roi.json 경로 반환"""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "crop_roi.json")


def load_crop_roi(path):
    """crop_roi.json 파일에서 ROI 로드"""
    with open(path, "r") as f:
        roi = json.load(f)
    # 유효성 검사
    for key in ("x", "y", "w", "h"):
        if key not in roi:
            raise ValueError(f"crop_roi.json에 '{key}' 키가 없습니다.")
    return roi


def auto_load_crop_roi(explicit_path=None):
    """크롭 ROI 자동 로드. explicit_path가 없으면 기본 경로에서 시도.
    파일이 없으면 None 반환."""
    path = explicit_path or default_crop_path()
    if not os.path.exists(path):
        return None
    try:
        return load_crop_roi(path)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"  [WARNING] crop_roi.json 로드 실패: {e}")
        return None


def reset_crop_roi(path=None):
    """저장된 crop_roi.json 삭제 (초기화)"""
    path = path or default_crop_path()
    if os.path.exists(path):
        os.remove(path)
        print(f"  [초기화] 크롭 설정 삭제됨: {path}")
    else:
        print(f"  [초기화] 크롭 설정 파일이 없습니다: {path}")


def apply_crop(frame, roi):
    """프레임에 ROI 크롭 적용. roi = {"x", "y", "w", "h"}"""
    x, y, w, h = roi["x"], roi["y"], roi["w"], roi["h"]
    return frame[y:y+h, x:x+w].copy()


def main():
    parser = argparse.ArgumentParser(description="인터랙티브 ROI(크롭 영역) 선택")
    parser.add_argument("--video", type=str, default=None,
                        help="동영상 파일 경로 (MP4/SVO2)")
    parser.add_argument("--no-zed", dest="use_zed", action="store_false",
                        help="ZED 카메라 대신 웹캠 사용")
    parser.add_argument("--resolution", default="SVGA",
                        choices=["SVGA", "VGA", "HD720", "HD1080", "HD1200"],
                        help="카메라 해상도")
    parser.add_argument("--output", type=str, default=None,
                        help="ROI 저장 경로 (기본: crop_roi.json)")
    parser.add_argument("--width", type=int, default=None,
                        help="가로 폭 고정 (예: --width 480). 클릭한 위치를 중심으로 지정 폭, 세로 전체로 자동 설정")
    parser.add_argument("--reset", action="store_true",
                        help="저장된 크롭 설정 초기화 (삭제)")
    args = parser.parse_args()

    # 출력 경로 설정
    output_path = args.output or default_crop_path()

    # 초기화 모드
    if args.reset:
        reset_crop_roi(output_path)
        return

    # 카메라/비디오 열기
    if args.video:
        ext = os.path.splitext(args.video)[1].lower()
        use_zed = ext in ('.svo2', '.svo')
    else:
        use_zed = args.use_zed

    camera = create_camera(
        use_zed=use_zed,
        video_path=args.video,
        resolution=args.resolution,
    )
    camera.open()

    # 첫 프레임 캡처
    print("프레임 캡처 중...")
    for _ in range(5):  # 몇 프레임 버리고 안정화
        camera.grab()

    if not camera.grab():
        print("[ERROR] 프레임을 가져올 수 없습니다.")
        camera.close()
        return

    frame = camera.get_rgb()
    camera.close()

    # BGR로 변환 (OpenCV 표시용)
    if len(frame.shape) == 3 and frame.shape[2] == 4:
        display = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    elif len(frame.shape) == 3 and frame.shape[2] == 3:
        display = frame.copy()
    else:
        display = frame.copy()

    frame_h, frame_w = display.shape[:2]
    print(f"프레임 크기: {frame_w}x{frame_h}")

    # ROI 선택
    if args.width:
        # --width 모드: 프레임 중앙 기준 고정 폭, 세로 전체
        fixed_w = min(args.width, frame_w)
        x = (frame_w - fixed_w) // 2
        roi = {"x": x, "y": 0, "w": fixed_w, "h": frame_h}
        print(f"\n  [고정 폭 모드] 가로={fixed_w}px, 세로={frame_h}px")
        print(f"  x={x} (중앙 정렬)")
    else:
        roi = select_roi_from_frame(display)

    if roi is None:
        print("ROI 선택이 취소되었습니다.")
        return

    # 미리보기
    cropped = apply_crop(display, roi)
    print(f"\n  선택된 ROI: x={roi['x']}, y={roi['y']}, "
          f"w={roi['w']}, h={roi['h']}")
    print(f"  크롭 결과: {roi['w']}x{roi['h']}")

    cv2.imshow("Cropped Preview - ENTER to save, ESC to cancel", cropped)
    key = cv2.waitKey(0) & 0xFF
    cv2.destroyAllWindows()

    if key == 27 or key == ord('q'):  # ESC or Q
        print("저장 취소.")
        return

    # JSON 저장
    roi["source_width"] = frame_w
    roi["source_height"] = frame_h
    with open(output_path, "w") as f:
        json.dump(roi, f, indent=2)

    print(f"\n  ROI 저장 완료: {output_path}")
    print(f"  이후 run_benchmark.py / run_record_demo.py 실행 시 자동 적용됩니다.")
    print(f"  초기화: python3 select_roi.py --reset")


if __name__ == "__main__":
    main()
