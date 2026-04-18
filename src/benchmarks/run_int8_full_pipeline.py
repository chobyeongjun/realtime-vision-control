#!/usr/bin/env python3
"""
INT8 전체 파이프라인: 촬영 → 캘리브레이션 → 엔진 빌드 → 벤치마크
================================================================

Jetson에서 이 스크립트 하나로 모든 과정을 자동 실행합니다.

순서:
  1단계: ZED 카메라로 걷는 모습 촬영 (30초)
  2단계: 촬영 영상에서 캘리브레이션 이미지 추출 (200장)
  3단계: 기존 INT8 엔진 전부 삭제
  4단계: half=True + 실제 캘리브레이션 데이터로 INT8 엔진 재빌드
  5단계: YOLOv8n/s × FP16/INT8-nocal/INT8-cal 벤치마크 실행

사용법:
  # 전체 파이프라인 (촬영부터 벤치마크까지)
  python run_int8_full_pipeline.py

  # 이미 녹화된 영상이 있으면 촬영 스킵
  python run_int8_full_pipeline.py --video walking.mp4

  # 촬영 시간/캘리브레이션 이미지 수 조정
  python run_int8_full_pipeline.py --record-duration 60 --num-calib 300

  # 특정 단계만 실행
  python run_int8_full_pipeline.py --skip-record --skip-build  # 벤치마크만
  python run_int8_full_pipeline.py --video walking.mp4 --skip-benchmark  # 빌드까지만
"""

import os
import sys
import argparse
import time
import glob
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CALIB_DIR = os.path.join(SCRIPT_DIR, "calib_images", "yolo")
RECORDING_DIR = os.path.join(SCRIPT_DIR, "recordings")


# ============================================================================
# 1단계: 걷기 영상 촬영
# ============================================================================
def step1_record(args):
    """ZED 카메라로 걷기 영상 촬영"""
    if args.video:
        print(f"  기존 영상 사용: {args.video}")
        if not os.path.exists(args.video):
            print(f"  ERROR: 파일을 찾을 수 없습니다: {args.video}")
            return None
        return args.video

    if args.skip_record:
        # 기존 캘리브레이션 이미지 확인
        if os.path.isdir(CALIB_DIR) and len(glob.glob(os.path.join(CALIB_DIR, "*.jpg"))) > 0:
            existing = len(glob.glob(os.path.join(CALIB_DIR, "*.jpg")))
            print(f"  기존 캘리브레이션 이미지 사용: {existing}장 ({CALIB_DIR})")
            return "skip"
        print("  ERROR: --skip-record인데 캘리브레이션 이미지가 없습니다.")
        print(f"  {CALIB_DIR}에 이미지를 준비하거나, --video 옵션을 사용하세요.")
        return None

    # ZED로 녹화
    os.makedirs(RECORDING_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = os.path.join(RECORDING_DIR, f"walking_{timestamp}.mp4")

    duration = args.record_duration

    try:
        import pyzed.sl as sl
        import cv2
    except ImportError:
        print("  ERROR: pyzed 또는 cv2 미설치")
        return None

    zed = sl.Camera()
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.SVGA
    init.camera_fps = 30  # 캘리브레이션용이므로 30fps면 충분
    init.depth_mode = sl.DEPTH_MODE.NONE

    err = zed.open(init)
    if err != sl.ERROR_CODE.SUCCESS:
        print(f"  ERROR: ZED 열기 실패: {err}")
        return None

    info = zed.get_camera_information()
    w = info.camera_configuration.resolution.width
    h = info.camera_configuration.resolution.height

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, 30.0, (w, h))

    image = sl.Mat()
    runtime = sl.RuntimeParameters()

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  ZED 카메라 앞에서 자연스럽게 걸어 주세요  │")
    print(f"  │                                           │")
    print(f"  │  - 다양한 거리 (가까이/멀리)               │")
    print(f"  │  - 다양한 각도 (정면/측면)                 │")
    print(f"  │  - 자연스러운 보행 동작                    │")
    print(f"  │                                           │")
    print(f"  │  녹화 시간: {duration}초                        │")
    print(f"  │  'q' 키로 조기 종료                       │")
    print(f"  └─────────────────────────────────────────┘\n")

    # 카메라 안정화 대기
    for _ in range(30):
        zed.grab(runtime)

    start = time.perf_counter()
    frame_count = 0

    try:
        while time.perf_counter() - start < duration:
            if zed.grab(runtime) == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                frame = image.get_data()[:, :, :3].copy()
                out.write(frame)
                frame_count += 1

                # 화면에 진행 상황 표시
                elapsed = time.perf_counter() - start
                remaining = duration - elapsed
                display = frame.copy()
                cv2.putText(display,
                            f"REC {elapsed:.0f}s / {duration}s | {frame_count} frames | q=stop",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(display,
                            f"Walk naturally at various distances",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                # 남은 시간 프로그레스 바
                bar_w = int(w * 0.8)
                bar_x = int(w * 0.1)
                bar_y = h - 40
                progress = elapsed / duration
                cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20),
                              (100, 100, 100), -1)
                cv2.rectangle(display, (bar_x, bar_y),
                              (bar_x + int(bar_w * progress), bar_y + 20),
                              (0, 0, 255), -1)
                cv2.putText(display, f"{remaining:.0f}s left",
                            (bar_x + bar_w + 10, bar_y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("Walking Calibration Recording", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("  사용자가 녹화를 종료했습니다.")
                    break

    except KeyboardInterrupt:
        print("  Ctrl+C로 녹화 종료")

    out.release()
    zed.close()
    cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start
    size_mb = os.path.getsize(video_path) / (1024 * 1024)

    print(f"  녹화 완료: {video_path}")
    print(f"  {frame_count}프레임, {elapsed:.1f}초, {size_mb:.1f}MB")
    return video_path


# ============================================================================
# 2단계: 캘리브레이션 이미지 추출
# ============================================================================
def step2_extract_calib(video_path, args):
    """녹화 영상에서 캘리브레이션 이미지 추출"""
    if video_path == "skip":
        return True

    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: 비디오를 열 수 없습니다: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_images = args.num_calib

    print(f"  영상: {video_path}")
    print(f"  총 프레임: {total_frames}, FPS: {fps:.1f}")

    # 균일 간격으로 추출
    skip = max(1, total_frames // num_images)
    print(f"  매 {skip}프레임마다 추출 → 목표 {num_images}장")

    # 기존 캘리브레이션 이미지 정리
    os.makedirs(CALIB_DIR, exist_ok=True)
    old_files = glob.glob(os.path.join(CALIB_DIR, "calib_*.jpg"))
    if old_files:
        for f in old_files:
            os.remove(f)
        print(f"  기존 이미지 {len(old_files)}장 삭제")

    count = 0
    frame_idx = 0
    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            path = os.path.join(CALIB_DIR, f"calib_{count:04d}.jpg")
            cv2.imwrite(path, frame)
            count += 1
        frame_idx += 1

    cap.release()
    print(f"  {count}장 추출 완료 → {CALIB_DIR}")
    return count > 0


# ============================================================================
# 3단계: 기존 엔진 삭제
# ============================================================================
def step3_clean_engines():
    """기존 TRT 엔진 파일 전부 삭제"""
    patterns = [
        os.path.join(SCRIPT_DIR, "*.engine"),
        os.path.join(SCRIPT_DIR, "trt_cache", "**", "*"),
    ]

    deleted = 0
    for pattern in patterns:
        for f in glob.glob(pattern, recursive=True):
            if os.path.isfile(f):
                os.remove(f)
                print(f"  삭제: {os.path.basename(f)}")
                deleted += 1

    # Ultralytics가 만든 엔진도 확인
    home = os.path.expanduser("~")
    ultra_cache = os.path.join(home, ".config", "Ultralytics")
    if os.path.isdir(ultra_cache):
        for f in glob.glob(os.path.join(ultra_cache, "**", "*.engine"), recursive=True):
            os.remove(f)
            print(f"  삭제: {f}")
            deleted += 1

    if deleted == 0:
        print(f"  삭제할 엔진 파일 없음")
    else:
        print(f"  총 {deleted}개 삭제")


# ============================================================================
# 4단계: INT8 엔진 빌드
# ============================================================================
def step4_build_engines(args):
    """FP16 + INT8-nocal + INT8-cal 엔진 빌드"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ERROR: ultralytics 미설치. pip install ultralytics")
        return False

    model_sizes = ["n", "s"]

    for size in model_sizes:
        model_name = f"yolov8{size}-pose.pt"
        print(f"\n  모델: {model_name}")

        model = YOLO(model_name)

        # --- FP16 엔진 ---
        fp16_engine = f"yolov8{size}-pose.engine"
        if not os.path.exists(fp16_engine):
            print(f"    [FP16] 빌드 중...")
            t0 = time.time()
            model.export(format='engine', half=True)
            print(f"    [FP16] 완료 ({time.time()-t0:.0f}초)")
        else:
            print(f"    [FP16] 기존 엔진 사용: {fp16_engine}")

        # --- INT8-nocal 엔진 ---
        nocal_engine = f"yolov8{size}-pose-int8-nocal.engine"
        print(f"    [INT8-nocal] 빌드 중...")
        t0 = time.time()
        _build_int8_subprocess(model_name, nocal_engine, calib_data=None)
        print(f"    [INT8-nocal] 완료 ({time.time()-t0:.0f}초)")

        # --- INT8-cal 엔진 ---
        cal_engine = f"yolov8{size}-pose-int8-cal.engine"
        num_calib = len(glob.glob(os.path.join(CALIB_DIR, "*.jpg")))
        print(f"    [INT8-cal] 빌드 중... (캘리브레이션 이미지: {num_calib}장)")
        t0 = time.time()
        _build_int8_subprocess(model_name, cal_engine, calib_data=CALIB_DIR)
        print(f"    [INT8-cal] 완료 ({time.time()-t0:.0f}초)")

    return True


def _create_calib_yaml(calib_dir):
    """캘리브레이션 이미지 디렉토리를 Ultralytics YAML 형식으로 변환"""
    yaml_path = os.path.join(calib_dir, "calib_dataset.yaml")
    # Ultralytics는 data YAML에서 train/val 이미지 경로를 읽어 캘리브레이션에 사용
    with open(yaml_path, "w") as f:
        f.write(f"path: {os.path.abspath(calib_dir)}\n")
        f.write("train: .\n")
        f.write("val: .\n")
        f.write("names:\n")
        f.write("  0: person\n")
        f.write("nc: 1\n")
        f.write("kpt_shape: [17, 3]\n")
    return yaml_path


def _build_int8_subprocess(model_name, engine_name, calib_data=None):
    """INT8 엔진 빌드를 subprocess로 격리 (TRT crash 방지)"""
    if calib_data:
        # 디렉토리면 YAML 래퍼 생성
        if os.path.isdir(calib_data):
            calib_yaml = _create_calib_yaml(calib_data)
        else:
            calib_yaml = calib_data
        data_arg = f", data={calib_yaml!r}"
    else:
        data_arg = ""

    script = f"""
import os, sys
# 시스템 matplotlib과 numpy 충돌 방지: /usr/lib 경로 제거
sys.path = [p for p in sys.path if not p.startswith('/usr/lib/python3/dist-packages')]
os.environ['MPLBACKEND'] = 'Agg'
os.chdir({SCRIPT_DIR!r})
from ultralytics import YOLO
model = YOLO({model_name!r})
path = model.export(format='engine', half=True, int8=True{data_arg})
default_engine = {model_name.replace('.pt', '.engine')!r}
target = {engine_name!r}
if os.path.exists(default_engine) and target != default_engine:
    os.rename(default_engine, target)
    path = target
print('__ENGINE_PATH__=' + str(path))
"""
    try:
        # 시스템 matplotlib/numpy 충돌 방지를 위한 환경 변수 설정
        env = os.environ.copy()
        env["MPLBACKEND"] = "Agg"
        if "PYTHONPATH" in env:
            env["PYTHONPATH"] = ":".join(
                p for p in env["PYTHONPATH"].split(":")
                if not p.startswith("/usr/lib/python3/dist-packages")
            )
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True, text=True, timeout=3600,
            cwd=SCRIPT_DIR,
            encoding="utf-8", errors="replace",
            env=env,
        )
        for line in result.stdout.splitlines():
            if line.startswith("__ENGINE_PATH__="):
                engine_path = line.split("=", 1)[1].strip()
                if os.path.exists(engine_path):
                    size_mb = os.path.getsize(engine_path) / (1024 * 1024)
                    print(f"      엔진: {engine_path} ({size_mb:.1f}MB)")
                    return True

        print(f"      빌드 실패 (returncode={result.returncode})")
        if result.stderr:
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"        {line}")
        return False
    except subprocess.TimeoutExpired:
        print(f"      타임아웃 (60분 초과)")
        return False


# ============================================================================
# 5단계: 벤치마크 실행
# ============================================================================
def step5_benchmark(args):
    """YOLOv8 n/s × 3정밀도 벤치마크"""
    cmd = [
        sys.executable, os.path.join(SCRIPT_DIR, "run_int8_comparison.py"),
        "--calib-data", CALIB_DIR,
        "--duration", str(args.benchmark_duration),
    ]

    if args.video:
        cmd.extend(["--video", args.video])

    if args.record:
        cmd.append("--record")

    print(f"  실행: {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=SCRIPT_DIR)


# ============================================================================
# 메인
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="INT8 전체 파이프라인: 촬영 → 캘리브레이션 → 엔진 빌드 → 벤치마크",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 전체 실행 (ZED 촬영부터)
  python run_int8_full_pipeline.py

  # 이미 녹화된 영상 사용
  python run_int8_full_pipeline.py --video walking.mp4

  # 촬영 시간 조정
  python run_int8_full_pipeline.py --record-duration 60

  # 빌드까지만 (벤치마크 스킵)
  python run_int8_full_pipeline.py --skip-benchmark

  # 기존 캘리브레이션으로 벤치마크만
  python run_int8_full_pipeline.py --skip-record --skip-build
        """)

    # 입력
    parser.add_argument("--video", type=str, default=None,
                        help="기존 걷기 영상 (없으면 ZED로 촬영)")
    parser.add_argument("--record-duration", type=int, default=30,
                        help="녹화 시간, 초 (기본: 30)")
    parser.add_argument("--num-calib", type=int, default=200,
                        help="캘리브레이션 이미지 수 (기본: 200)")

    # 단계 스킵
    parser.add_argument("--skip-record", action="store_true",
                        help="촬영 스킵 (기존 캘리브레이션 이미지 사용)")
    parser.add_argument("--skip-build", action="store_true",
                        help="엔진 빌드 스킵")
    parser.add_argument("--skip-benchmark", action="store_true",
                        help="벤치마크 스킵")

    # 벤치마크 옵션
    parser.add_argument("--benchmark-duration", type=int, default=20,
                        help="모델당 벤치마크 시간, 초 (기본: 20)")
    parser.add_argument("--record", action="store_true",
                        help="벤치마크 영상 녹화")

    args = parser.parse_args()

    print("=" * 60)
    print("  INT8 전체 파이프라인")
    print("  촬영 → 캘리브레이션 → 엔진 빌드 → 벤치마크")
    print("=" * 60)
    print(f"  모델: YOLOv8n-Pose, YOLOv8s-Pose")
    print(f"  정밀도: FP16, INT8-nocal, INT8-cal")
    print(f"  캘리브레이션: 실제 보행 영상 {args.num_calib}장")
    print("=" * 60)

    total_start = time.time()

    # ── 1단계: 촬영 ──
    print(f"\n\n{'━'*60}")
    print(f"  [1/5] 걷기 영상 촬영")
    print(f"{'━'*60}")
    video_path = step1_record(args)
    if video_path is None:
        print("  FAILED: 영상 촬영/로드 실패. 중단합니다.")
        sys.exit(1)

    # ── 2단계: 캘리브레이션 이미지 추출 ──
    print(f"\n\n{'━'*60}")
    print(f"  [2/5] 캘리브레이션 이미지 추출")
    print(f"{'━'*60}")
    if not step2_extract_calib(video_path, args):
        print("  FAILED: 이미지 추출 실패. 중단합니다.")
        sys.exit(1)

    # ── 3단계: 기존 엔진 삭제 ──
    print(f"\n\n{'━'*60}")
    print(f"  [3/5] 기존 TRT 엔진 삭제")
    print(f"{'━'*60}")
    if not args.skip_build:
        step3_clean_engines()
    else:
        print("  스킵 (--skip-build)")

    # ── 4단계: 엔진 빌드 ──
    print(f"\n\n{'━'*60}")
    print(f"  [4/5] INT8 엔진 빌드 (half=True + 실제 캘리브레이션)")
    print(f"{'━'*60}")
    if not args.skip_build:
        if not step4_build_engines(args):
            print("  WARNING: 일부 엔진 빌드 실패. 벤치마크에서 fallback됩니다.")
    else:
        print("  스킵 (--skip-build)")

    # ── 5단계: 벤치마크 ──
    print(f"\n\n{'━'*60}")
    print(f"  [5/5] YOLOv8 n/s × 3정밀도 벤치마크")
    print(f"{'━'*60}")
    if not args.skip_benchmark:
        step5_benchmark(args)
    else:
        print("  스킵 (--skip-benchmark)")

    # ── 완료 ──
    total_elapsed = time.time() - total_start
    minutes = int(total_elapsed // 60)
    seconds = int(total_elapsed % 60)
    print(f"\n\n{'='*60}")
    print(f"  전체 파이프라인 완료! ({minutes}분 {seconds}초)")
    print(f"{'='*60}")
    print(f"  캘리브레이션 이미지: {CALIB_DIR}")
    print(f"  결과: {os.path.join(SCRIPT_DIR, 'results')}/")
    if video_path and video_path != "skip":
        print(f"  녹화 영상: {video_path}")


if __name__ == "__main__":
    main()
