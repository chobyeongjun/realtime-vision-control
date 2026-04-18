#!/usr/bin/env python3
"""
벤치마크용 테스트 데이터 준비
==============================
사람이 카메라 앞에 없어도 반복 테스트가 가능하도록
공개 보행 영상을 다운로드하거나, ZED 녹화 영상을 사용합니다.

사용법:
    python3 prepare_test_data.py --download         # 공개 보행 영상 다운로드
    python3 prepare_test_data.py --list              # 사용 가능한 테스트 영상 목록
    python3 prepare_test_data.py --info video.mp4    # 영상 정보 확인

다운로드 후 벤치마크 실행:
    python3 run_benchmark.py --video test_data/gait_frontal.mp4 --models all
    python3 run_full_benchmark.py --video test_data/gait_lateral.mp4
"""

import argparse
import os
import sys
import subprocess

TEST_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data")

# ============================================================================
# 공개 보행 데이터셋 정보
# ============================================================================
PUBLIC_DATASETS = {
    "cmu_mocap_walk": {
        "description": "CMU MoCap - 보행 시퀀스 (렌더링 영상)",
        "url": "http://mocap.cs.cmu.edu/",
        "note": "직접 다운로드 필요. 웹사이트에서 Subject #2 (walking) 다운로드",
        "auto_download": False,
    },
    "human36m": {
        "description": "Human3.6M - 실내 보행 (MoCap + 영상 동시 수집)",
        "url": "http://vision.imar.ro/human3.6m/",
        "note": "학술 목적 등록 필요. Walking, WalkDog, WalkTogether 카테고리 사용",
        "auto_download": False,
    },
    "gait_in_the_wild": {
        "description": "Gait in the Wild - 다양한 환경 보행 영상",
        "url": "https://github.com/mdshopon/Gait-Recognition-Using-Human-Pose-Estimation",
        "note": "GitHub에서 다운로드 가능",
        "auto_download": False,
    },
}

# ============================================================================
# 자체 테스트 영상 생성 (사람 없이도 기본 동작 확인용)
# ============================================================================
def create_synthetic_test_video(output_path, duration=10, fps=30):
    """
    합성 테스트 영상 생성.
    실제 사람은 없지만 모델의 기본 동작(로드, 추론, 타이밍)을 확인할 수 있음.
    Detection rate는 0%가 나오지만, FPS/Latency 측정은 유효함.
    """
    import cv2
    import numpy as np

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    w, h = 960, 600  # SVGA (ZED X Mini 기본)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = duration * fps
    print(f"  합성 테스트 영상 생성: {output_path}")
    print(f"  {w}x{h} @ {fps}fps, {duration}초 ({total_frames} 프레임)")

    for i in range(total_frames):
        # 배경 + 움직이는 도형 (모델이 사람으로 오인하지는 않지만 타이밍 측정에 유효)
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = (40, 40, 40)  # 어두운 배경

        # 격자 패턴 (depth estimation 테스트에 유용)
        for x in range(0, w, 50):
            cv2.line(frame, (x, 0), (x, h), (60, 60, 60), 1)
        for y in range(0, h, 50):
            cv2.line(frame, (0, y), (w, y), (60, 60, 60), 1)

        # 프레임 번호 표시
        cv2.putText(frame, f"SYNTHETIC TEST - Frame {i}/{total_frames}",
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(frame, "No person detected = EXPECTED",
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)
        cv2.putText(frame, "This tests: model load, inference timing, pipeline flow",
                   (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 100), 1)

        out.write(frame)

    out.release()
    print(f"  생성 완료: {output_path}")
    return output_path


def create_stick_figure_video(output_path, duration=10, fps=30):
    """
    스틱 피겨 보행 애니메이션 생성.
    실제 사람은 아니지만 보행 패턴의 시각적 확인용.
    일부 모델(특히 YOLOv8)은 이것도 감지할 수 있음.
    """
    import cv2
    import numpy as np

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    w, h = 960, 600
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    total_frames = duration * fps
    print(f"  스틱 피겨 보행 영상 생성: {output_path}")

    stride_period = fps  # 1초에 한 걸음
    cx, cy = w // 2, h // 2  # 중심

    for i in range(total_frames):
        frame = np.ones((h, w, 3), dtype=np.uint8) * 200  # 밝은 배경

        t = i / fps
        phase = (i % stride_period) / stride_period  # 0~1 보행 사이클

        # 간단한 보행 모델 (sagittal plane)
        hip_y = cy + 20
        # 왼다리
        l_hip_angle = np.sin(phase * 2 * np.pi) * 30  # -30 ~ +30 deg
        l_knee_angle = max(0, np.sin(phase * 2 * np.pi - 0.5) * 40)  # 0 ~ 40 deg
        # 오른다리 (반대 위상)
        r_hip_angle = np.sin((phase + 0.5) * 2 * np.pi) * 30
        r_knee_angle = max(0, np.sin((phase + 0.5) * 2 * np.pi - 0.5) * 40)

        thigh_len = 120
        shank_len = 110

        def draw_leg(hip_x, hip_angle_deg, knee_angle_deg, color):
            ha = np.radians(hip_angle_deg + 90)  # 90 = 수직 아래
            knee_x = int(hip_x + thigh_len * np.cos(ha))
            knee_y = int(hip_y + thigh_len * np.sin(ha))

            ka = ha + np.radians(knee_angle_deg)
            ankle_x = int(knee_x + shank_len * np.cos(ka))
            ankle_y = int(knee_y + shank_len * np.sin(ka))

            cv2.line(frame, (hip_x, hip_y), (knee_x, knee_y), color, 4)
            cv2.line(frame, (knee_x, knee_y), (ankle_x, ankle_y), color, 4)
            cv2.circle(frame, (hip_x, hip_y), 6, color, -1)
            cv2.circle(frame, (knee_x, knee_y), 6, color, -1)
            cv2.circle(frame, (ankle_x, ankle_y), 6, color, -1)

        # 몸통
        head_y = hip_y - 180
        cv2.line(frame, (cx, hip_y), (cx, head_y), (50, 50, 50), 4)
        cv2.circle(frame, (cx, head_y - 20), 20, (50, 50, 50), 3)

        # 골반
        pelvis_w = 40
        cv2.line(frame, (cx - pelvis_w, hip_y), (cx + pelvis_w, hip_y), (50, 50, 50), 4)

        # 다리 그리기
        draw_leg(cx - pelvis_w, l_hip_angle, l_knee_angle, (0, 150, 0))   # 왼쪽 = 초록
        draw_leg(cx + pelvis_w, r_hip_angle, r_knee_angle, (150, 0, 0))   # 오른쪽 = 빨강(BGR)

        cv2.putText(frame, f"Stick Figure Gait | t={t:.1f}s | phase={phase:.0%}",
                   (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        out.write(frame)

    out.release()
    print(f"  생성 완료: {output_path}")
    return output_path


# ============================================================================
# ZED X Mini 녹화 안내
# ============================================================================
ZED_RECORD_GUIDE = """
═══════════════════════════════════════════════════════════
  ZED X Mini 테스트 영상 녹화 가이드
═══════════════════════════════════════════════════════════

1. 녹화 스크립트 실행:
   python3 record_zed.py --output test_data/my_gait_test.svo2 --duration 60

2. SVO2 → MP4 변환 (필요 시):
   python3 record_zed.py --convert test_data/my_gait_test.svo2

3. 벤치마크 실행:
   python3 run_benchmark.py --video test_data/my_gait_test.mp4 --models all

═══════════════════════════════════════════════════════════
  녹화 시 주의사항
═══════════════════════════════════════════════════════════

카메라 배치:
  - 환자로부터 1.5~3m 거리
  - 카메라 높이: 허리~가슴 높이 (하체가 잘 보이도록)
  - 정면(frontal) + 측면(lateral) 두 각도 녹화 권장

촬영 시나리오 (각 30초 이상):
  ① 전신 보행 (full_body): 사람이 정면에서 앞뒤로 걸음
  ② 하체만 보임 (lower_only): 상체를 가리고 걸음 (워커 시뮬레이션)
  ③ 측면 보행 (lateral): 옆에서 촬영 (관절 각도 관찰 용이)
  ④ 제자리 걸음 (stationary): 제자리에서 걸음 (bone length CV 측정에 최적)

해상도: SVGA (960x600) @ 60fps (ZED X Mini 기본, 가장 빠른 설정)
"""


def list_test_data():
    """사용 가능한 테스트 영상 목록"""
    import cv2

    print("\n═══════════════════════════════════════════════")
    print("  사용 가능한 테스트 영상")
    print("═══════════════════════════════════════════════")

    if not os.path.exists(TEST_DATA_DIR):
        print(f"  test_data 디렉토리 없음: {TEST_DATA_DIR}")
        print(f"  --download 로 테스트 영상을 준비하세요.")
        return

    files = []
    for ext in ['*.mp4', '*.avi', '*.mkv', '*.svo2', '*.svo']:
        import glob
        files.extend(glob.glob(os.path.join(TEST_DATA_DIR, ext)))
        files.extend(glob.glob(os.path.join(TEST_DATA_DIR, "**", ext), recursive=True))

    if not files:
        print("  테스트 영상 없음. --download 로 준비하세요.")
        return

    for f in sorted(files):
        name = os.path.relpath(f, TEST_DATA_DIR)
        size_mb = os.path.getsize(f) / (1024 * 1024)

        if f.endswith(('.svo2', '.svo')):
            print(f"  {name:<40} {size_mb:>7.1f} MB  [ZED SVO - depth 포함]")
        else:
            cap = cv2.VideoCapture(f)
            if cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frames / fps if fps > 0 else 0
                cap.release()
                print(f"  {name:<40} {size_mb:>7.1f} MB  "
                      f"{w}x{h} @ {fps:.0f}fps  {duration:.1f}s")
            else:
                print(f"  {name:<40} {size_mb:>7.1f} MB  [열기 실패]")

    print()
    print("  사용법:")
    print("    python3 run_benchmark.py --video test_data/<파일명> --models all")


def show_video_info(path):
    """영상 파일 상세 정보"""
    import cv2

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"  열기 실패: {path}")
        return

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frames / fps if fps > 0 else 0
    codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec_str = "".join([chr((codec >> 8 * i) & 0xFF) for i in range(4)])

    print(f"\n  파일: {path}")
    print(f"  해상도: {w}x{h}")
    print(f"  FPS: {fps:.1f}")
    print(f"  프레임 수: {frames}")
    print(f"  길이: {duration:.1f}초 ({duration/60:.1f}분)")
    print(f"  코덱: {codec_str}")
    print(f"  파일 크기: {os.path.getsize(path) / (1024*1024):.1f} MB")

    # 첫 프레임에서 사람 감지 가능성 확인
    ret, frame = cap.read()
    if ret:
        brightness = frame.mean()
        print(f"  평균 밝기: {brightness:.0f}/255")
        if brightness < 30:
            print(f"  [WARN] 영상이 너무 어두움 - 인식률 저하 가능")

    cap.release()


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="벤치마크용 테스트 데이터 준비")
    parser.add_argument("--download", action="store_true",
                        help="합성 테스트 영상 생성 + 공개 데이터셋 안내")
    parser.add_argument("--list", action="store_true",
                        help="사용 가능한 테스트 영상 목록")
    parser.add_argument("--info", type=str,
                        help="영상 파일 정보 확인")
    parser.add_argument("--record-guide", action="store_true",
                        help="ZED X Mini 녹화 가이드")
    args = parser.parse_args()

    if args.list:
        list_test_data()

    elif args.info:
        show_video_info(args.info)

    elif args.record_guide:
        print(ZED_RECORD_GUIDE)

    elif args.download:
        os.makedirs(TEST_DATA_DIR, exist_ok=True)

        # 1. 합성 테스트 영상 (사람 없이 파이프라인 동작 확인)
        print("\n[1/2] 합성 테스트 영상 생성 (파이프라인 동작 확인용)...")
        create_synthetic_test_video(
            os.path.join(TEST_DATA_DIR, "synthetic_pipeline_test.mp4"),
            duration=10, fps=30)

        # 2. 스틱 피겨 보행 영상
        print("\n[2/2] 스틱 피겨 보행 영상 생성 (보행 패턴 시각화용)...")
        create_stick_figure_video(
            os.path.join(TEST_DATA_DIR, "stick_figure_gait.mp4"),
            duration=10, fps=30)

        print("\n" + "=" * 60)
        print("  합성 영상 준비 완료!")
        print("=" * 60)
        print()
        print("  즉시 테스트 가능:")
        print("    python3 run_benchmark.py --video test_data/synthetic_pipeline_test.mp4 --models mediapipe --duration 5")
        print()
        print("  실제 사람 영상이 필요하면:")
        print("    1. 공개 데이터셋:")

        for name, info in PUBLIC_DATASETS.items():
            print(f"       - {info['description']}")
            print(f"         {info['url']}")
            print(f"         {info['note']}")
            print()

        print("    2. 직접 녹화:")
        print("       python3 record_zed.py --output test_data/my_walk.svo2 --duration 60")
        print("       (또는 스마트폰으로 걷는 영상 촬영 → test_data/ 에 복사)")
        print()
        print("  가장 간단한 방법:")
        print("    스마트폰으로 걷는 영상 30초 촬영 → test_data/ 에 복사 → --video로 사용")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
