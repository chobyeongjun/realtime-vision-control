#!/usr/bin/env python3
"""
ZED X Mini 테스트 영상 녹화
============================
벤치마크용 보행 영상을 ZED X Mini로 녹화합니다.
SVO2 형식(depth 포함)과 MP4 형식(RGB만) 모두 지원.

사용법:
    # SVO2 녹화 (depth 포함 - 3D 벤치마크에 필수)
    python3 record_zed.py --output test_data/walk_frontal.svo2 --duration 60

    # MP4 녹화 (RGB만 - 가볍고 범용)
    python3 record_zed.py --output test_data/walk_frontal.mp4 --duration 60

    # SVO2 → MP4 변환
    python3 record_zed.py --convert test_data/walk_frontal.svo2

    # 녹화 + 실시간 미리보기
    python3 record_zed.py --output test_data/walk.svo2 --duration 30 --preview
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def record_svo2(output_path, duration, resolution="SVGA", fps=60, preview=False):
    """ZED SVO2 녹화 (depth 포함)"""
    try:
        import pyzed.sl as sl
    except ImportError:
        print("pyzed 미설치. ZED SDK 필요.")
        return False

    import cv2

    zed = sl.Camera()

    init = sl.InitParameters()
    res_map = {
        "SVGA": sl.RESOLUTION.SVGA,
        "HD720": sl.RESOLUTION.HD720,
        "HD1080": sl.RESOLUTION.HD1080,
        "HD1200": sl.RESOLUTION.HD1200,
    }
    init.camera_resolution = res_map.get(resolution, sl.RESOLUTION.SVGA)
    init.camera_fps = fps
    init.depth_mode = sl.DEPTH_MODE.NEURAL
    init.coordinate_units = sl.UNIT.METER

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ZED 열기 실패: {err}")
        return False

    info = zed.get_camera_information()
    w = info.camera_configuration.resolution.width
    h = info.camera_configuration.resolution.height
    actual_fps = info.camera_configuration.fps

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    # SVO2 녹화 설정
    rec_params = sl.RecordingParameters()
    rec_params.video_filename = output_path
    rec_params.compression_mode = sl.SVO_COMPRESSION_MODE.H264

    err = zed.enable_recording(rec_params)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"녹화 시작 실패: {err}")
        zed.close()
        return False

    print(f"\n  녹화 시작: {output_path}")
    print(f"  {w}x{h} @ {actual_fps}fps, {duration}초")
    print(f"  종료: Ctrl+C 또는 {duration}초 후 자동 종료")
    if preview:
        print(f"  미리보기: 'q'로 종료")
    print()

    image = sl.Mat()
    runtime = sl.RuntimeParameters()
    start = time.perf_counter()
    frame_count = 0

    try:
        while time.perf_counter() - start < duration:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                frame_count += 1

                elapsed = time.perf_counter() - start
                if frame_count % (actual_fps * 2) == 0:
                    rec_fps = frame_count / elapsed
                    remaining = duration - elapsed
                    print(f"  {elapsed:.0f}s / {duration}s | "
                          f"{frame_count} frames | {rec_fps:.1f} fps | "
                          f"남은 시간: {remaining:.0f}s")

                if preview:
                    zed.retrieve_image(image, sl.VIEW.LEFT)
                    frame = image.get_data()[:, :, :3].copy()
                    cv2.putText(frame, f"REC {elapsed:.1f}s | {frame_count} frames",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("ZED Recording", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    except KeyboardInterrupt:
        print("\n  녹화 중단 (사용자)")

    zed.disable_recording()
    zed.close()
    if preview:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0

    print(f"\n  녹화 완료!")
    print(f"  파일: {output_path}")
    print(f"  프레임: {frame_count}")
    print(f"  길이: {elapsed:.1f}초")
    print(f"  크기: {size_mb:.1f} MB")
    print(f"\n  벤치마크 실행:")
    print(f"    python3 run_benchmark.py --video {output_path} --models all")
    return True


def record_mp4(output_path, duration, resolution="SVGA", fps=60, preview=False):
    """ZED RGB를 MP4로 녹화 (depth 없음, 가볍고 범용)"""
    try:
        import pyzed.sl as sl
    except ImportError:
        print("pyzed 미설치. ZED SDK 필요.")
        return False

    import cv2

    zed = sl.Camera()
    init = sl.InitParameters()
    res_map = {
        "SVGA": sl.RESOLUTION.SVGA,
        "HD720": sl.RESOLUTION.HD720,
        "HD1080": sl.RESOLUTION.HD1080,
    }
    init.camera_resolution = res_map.get(resolution, sl.RESOLUTION.SVGA)
    init.camera_fps = fps
    init.depth_mode = sl.DEPTH_MODE.NONE  # depth 불필요

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"ZED 열기 실패: {err}")
        return False

    info = zed.get_camera_information()
    w = info.camera_configuration.resolution.width
    h = info.camera_configuration.resolution.height
    actual_fps = info.camera_configuration.fps

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, actual_fps, (w, h))

    print(f"\n  MP4 녹화 시작: {output_path}")
    print(f"  {w}x{h} @ {actual_fps}fps, {duration}초")

    image = sl.Mat()
    runtime = sl.RuntimeParameters()
    start = time.perf_counter()
    frame_count = 0

    try:
        while time.perf_counter() - start < duration:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()[:, :, :3].copy()
                out.write(frame)
                frame_count += 1

                if preview and frame_count % 2 == 0:
                    elapsed = time.perf_counter() - start
                    cv2.putText(frame, f"REC {elapsed:.1f}s",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.imshow("ZED Recording", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

    except KeyboardInterrupt:
        print("\n  녹화 중단 (사용자)")

    out.release()
    zed.close()
    if preview:
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    size_mb = os.path.getsize(output_path) / (1024 * 1024) if os.path.exists(output_path) else 0

    print(f"\n  녹화 완료!")
    print(f"  파일: {output_path}")
    print(f"  프레임: {frame_count}")
    print(f"  길이: {elapsed:.1f}초")
    print(f"  크기: {size_mb:.1f} MB")
    return True


def convert_svo2_to_mp4(svo_path):
    """SVO2 파일을 MP4로 변환"""
    try:
        import pyzed.sl as sl
    except ImportError:
        print("pyzed 미설치")
        return

    import cv2

    mp4_path = svo_path.rsplit('.', 1)[0] + '.mp4'

    zed = sl.Camera()
    init = sl.InitParameters()
    init.set_from_svo_file(svo_path)
    init.svo_real_time_mode = False

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"SVO 열기 실패: {err}")
        return

    info = zed.get_camera_information()
    w = info.camera_configuration.resolution.width
    h = info.camera_configuration.resolution.height
    fps = info.camera_configuration.fps

    total = zed.get_svo_number_of_frames()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mp4_path, fourcc, fps, (w, h))

    image = sl.Mat()
    runtime = sl.RuntimeParameters()

    print(f"  변환: {svo_path} → {mp4_path}")
    print(f"  {total} 프레임, {w}x{h} @ {fps}fps")

    frame_count = 0
    while True:
        err = zed.grab(runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            break
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()[:, :, :3].copy()
        out.write(frame)
        frame_count += 1

        if frame_count % 100 == 0:
            print(f"  {frame_count}/{total} ({frame_count/total*100:.0f}%)")

    out.release()
    zed.close()
    print(f"  변환 완료: {mp4_path}")


def main():
    parser = argparse.ArgumentParser(description="ZED X Mini 테스트 영상 녹화")
    parser.add_argument("--output", "-o", type=str, default="test_data/recording.svo2",
                        help="출력 파일 (.svo2 또는 .mp4)")
    parser.add_argument("--duration", "-d", type=int, default=30,
                        help="녹화 시간 (초, 기본: 30)")
    parser.add_argument("--resolution", default="SVGA",
                        choices=["SVGA", "HD720", "HD1080", "HD1200"])
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--preview", action="store_true",
                        help="실시간 미리보기")
    parser.add_argument("--convert", type=str,
                        help="SVO2 파일을 MP4로 변환")
    args = parser.parse_args()

    if args.convert:
        convert_svo2_to_mp4(args.convert)
    elif args.output.endswith(('.svo2', '.svo')):
        record_svo2(args.output, args.duration, args.resolution, args.fps, args.preview)
    elif args.output.endswith('.mp4'):
        record_mp4(args.output, args.duration, args.resolution, args.fps, args.preview)
    else:
        print(f"  지원하지 않는 형식: {args.output}")
        print(f"  .svo2 (depth 포함) 또는 .mp4 (RGB만) 사용")


if __name__ == "__main__":
    main()
