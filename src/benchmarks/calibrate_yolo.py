#!/usr/bin/env python3
"""
YOLOv8 INT8 캘리브레이션 이미지 준비
=====================================
걷기 영상(ZED/웹캠 녹화)에서 프레임을 추출하여
YOLOv8 INT8 캘리브레이션에 사용할 이미지를 준비합니다.

Ultralytics의 model.export(int8=True, data=...) 에서
data 파라미터로 이 이미지 디렉토리를 지정합니다.

사용법:
  # 걷기 영상에서 캘리브레이션 이미지 추출
  python calibrate_yolo.py --video walking.mp4

  # ZED 카메라로 실시간 촬영 (걸으면서 200프레임 캡처)
  python calibrate_yolo.py --zed --num-images 200

  # 웹캠으로 촬영
  python calibrate_yolo.py --webcam --num-images 200

  # 추출 후 바로 INT8 엔진 빌드까지
  python calibrate_yolo.py --video walking.mp4 --build

  # 이후 벤치마크에서 사용
  python run_int8_comparison.py --calib-data ./calib_images/yolo/
"""

import os
import sys
import argparse
import glob
import time
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_roi import auto_load_crop_roi, apply_crop


def augment_frame(frame, rng):
    """캘리브레이션용 데이터 증강 — 실사용 환경의 다양성을 반영

    INT8 캘리브레이션의 핵심: activation 범위의 다양성 확보
    → 밝기/대비/색상 변화로 다양한 activation 분포를 커버해야
      캘리브레이션이 과적합되지 않음
    """
    h, w = frame.shape[:2]
    aug = frame.copy()

    # 1. 밝기/대비 변화 (조명 조건 다양화) — 넓은 범위
    alpha = rng.uniform(0.5, 1.5)   # 대비 (이전: 0.7~1.3)
    beta = rng.randint(-50, 51)      # 밝기 (이전: -30~30)
    aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    # 2. 좌우 반전 (50% 확률)
    if rng.random() > 0.5:
        aug = cv2.flip(aug, 1)

    # 3. 가우시안 블러 (30% 확률, 모션블러 시뮬레이션)
    if rng.random() > 0.7:
        ksize = rng.choice([3, 5, 7])
        aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

    # 4. 스케일 지터 — 거리 변화 시뮬레이션 (넓은 범위)
    if rng.random() > 0.4:
        scale = rng.uniform(0.7, 1.3)  # 이전: 0.85~1.15
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(aug, (new_w, new_h))
        # 중앙 크롭 또는 패딩으로 원래 크기 복원
        if scale > 1.0:
            cx, cy = new_w // 2, new_h // 2
            aug = resized[cy - h//2:cy - h//2 + h, cx - w//2:cx - w//2 + w]
        else:
            canvas = np.zeros_like(frame)
            y_off = (h - new_h) // 2
            x_off = (w - new_w) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
            aug = canvas

    # 5. 색상 지터 (HSV 공간, 40% 확률) — 확대
    if rng.random() > 0.6:
        hsv = cv2.cvtColor(aug, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] = (hsv[:, :, 0] + rng.randint(-15, 16)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.7, 1.3), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.7, 1.3), 0, 255)
        aug = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # 6. 부분 가림 (occlusion) 시뮬레이션 (20% 확률)
    #    → 가려진 keypoint에 대한 activation 범위도 캘리브레이션에 반영
    if rng.random() > 0.8:
        occ_x = rng.randint(0, w - w // 4)
        occ_y = rng.randint(0, h - h // 4)
        occ_w = rng.randint(w // 8, w // 3)
        occ_h = rng.randint(h // 8, h // 3)
        occ_color = int(rng.randint(0, 200))
        cv2.rectangle(aug, (occ_x, occ_y),
                      (occ_x + occ_w, occ_y + occ_h),
                      (occ_color, occ_color, occ_color), -1)

    # 7. 가우시안 노이즈 (15% 확률) — 센서 노이즈 시뮬레이션
    if rng.random() > 0.85:
        noise = rng.normal(0, rng.uniform(5, 15), aug.shape).astype(np.int16)
        aug = np.clip(aug.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return aug


def compute_frame_diff(prev_gray, curr_gray):
    """프레임 간 차이 점수 (장면 변화 감지용)"""
    diff = cv2.absdiff(prev_gray, curr_gray)
    return np.mean(diff)


def extract_from_video(video_path, output_dir, max_images=300, skip_frames=3,
                       crop_roi=None, augment=True, diversity_threshold=5.0,
                       min_resolution=480, save_format="png"):
    """비디오에서 다양성 기반 프레임 추출 + 증강

    Args:
        min_resolution: 캘리브레이션 이미지 최소 해상도 (짧은 변 기준, 기본 480px)
                        소스가 이보다 작으면 업스케일하여 품질 보장
        save_format: 저장 포맷 ("png"=무손실 권장, "jpg"=JPEG95)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: 비디오를 열 수 없습니다: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  비디오: {video_path}")
    print(f"  총 프레임: {total_frames}, FPS: {fps:.1f}, 해상도: {src_w}x{src_h}")
    print(f"  저장 포맷: {save_format.upper()} ({'무손실' if save_format == 'png' else 'JPEG 95'})")

    if crop_roi:
        print(f"  ROI 크롭 적용: {crop_roi['w']}x{crop_roi['h']}")

    # 해상도 보장 — 짧은 변이 min_resolution 미만이면 업스케일 계수 계산
    upscale_factor = 1.0
    effective_h = crop_roi['h'] if crop_roi else src_h
    effective_w = crop_roi['w'] if crop_roi else src_w
    short_side = min(effective_h, effective_w)
    if short_side < min_resolution and short_side > 0:
        upscale_factor = min_resolution / short_side
        new_w = int(effective_w * upscale_factor)
        new_h = int(effective_h * upscale_factor)
        print(f"  ⚠ 소스 해상도({effective_w}x{effective_h})가 낮음 → "
              f"{new_w}x{new_h}로 업스케일 (INTER_CUBIC)")

    # augment 비율: 원본 60% + 증강 40%
    if augment:
        base_images = int(max_images * 0.6)
        aug_images = max_images - base_images
        print(f"  원본 {base_images}장 + 증강 {aug_images}장 = 총 {max_images}장 목표")
    else:
        base_images = max_images
        aug_images = 0

    # 균일 샘플링
    if total_frames > 0 and total_frames // skip_frames > base_images:
        skip_frames = max(1, total_frames // base_images)
    print(f"  매 {skip_frames}프레임마다 추출")

    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.RandomState(42)
    count = 0
    frame_idx = 0
    prev_gray = None
    skipped_similar = 0
    collected_frames = []  # 증강용 원본 보관

    while count < base_images:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_frames == 0:
            # 크롭 적용
            if crop_roi:
                frame = apply_crop(frame, crop_roi)

            # 장면 다양성 체크 — 너무 비슷한 프레임 건너뛰기
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = compute_frame_diff(prev_gray, gray)
                if diff < diversity_threshold:
                    skipped_similar += 1
                    frame_idx += 1
                    continue
            prev_gray = gray

            # 해상도 업스케일 (필요시)
            save_frame = frame
            if upscale_factor > 1.0:
                save_frame = cv2.resize(
                    frame, None,
                    fx=upscale_factor, fy=upscale_factor,
                    interpolation=cv2.INTER_CUBIC)

            ext = "png" if save_format == "png" else "jpg"
            path = os.path.join(output_dir, f"calib_{count:04d}.{ext}")
            if save_format == "png":
                cv2.imwrite(path, save_frame)
            else:
                cv2.imwrite(path, save_frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            collected_frames.append(save_frame)
            count += 1

        frame_idx += 1

    cap.release()

    if skipped_similar > 0:
        print(f"  유사 프레임 {skipped_similar}장 건너뜀 (다양성 필터)")

    # 증강 이미지 생성
    if augment and collected_frames and aug_images > 0:
        print(f"  증강 이미지 {aug_images}장 생성 중...")
        for i in range(aug_images):
            src = collected_frames[rng.randint(0, len(collected_frames))]
            aug_frame = augment_frame(src, rng)
            ext = "png" if save_format == "png" else "jpg"
            path = os.path.join(output_dir, f"calib_aug_{i:04d}.{ext}")
            if save_format == "png":
                cv2.imwrite(path, aug_frame)
            else:
                cv2.imwrite(path, aug_frame,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
            count += 1

    # 통계 리포트
    print(f"\n  === 캘리브레이션 데이터 리포트 ===")
    print(f"  총 이미지: {count}장 → {output_dir}")
    if collected_frames:
        sample = collected_frames[0]
        print(f"  이미지 크기: {sample.shape[1]}x{sample.shape[0]}")
    print(f"  원본: {len(collected_frames)}장, 증강: {count - len(collected_frames)}장")
    return count


def capture_from_camera(output_dir, max_images=200, use_zed=False, crop_roi=None):
    """ZED 또는 웹캠에서 실시간 캡처 (크롭 지원)"""
    if use_zed:
        try:
            from zed_camera import create_camera
            camera = create_camera(use_zed=True, resolution="SVGA",
                                   fps=30, depth_mode="NONE")
            camera.open()
            print("  ZED 카메라 열림")
        except Exception as e:
            print(f"  ZED 카메라 실패: {e}, 웹캠으로 전환")
            use_zed = False

    if not use_zed:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("ERROR: 웹캠을 열 수 없습니다")
            return 0

    if crop_roi:
        print(f"  ROI 크롭 적용: {crop_roi['w']}x{crop_roi['h']}")

    os.makedirs(output_dir, exist_ok=True)
    count = 0
    capture_interval = 5  # 매 5프레임마다 캡처

    print(f"\n  === 캘리브레이션 이미지 캡처 ===")
    print(f"  카메라 앞에서 자연스럽게 걸어 주세요.")
    print(f"  다양한 거리/각도로 걷는 모습을 포함하면 좋습니다.")
    print(f"  목표: {max_images}장 | 'q'로 조기 종료")
    print()

    frame_count = 0
    prev_gray = None

    while count < max_images:
        if use_zed:
            if not camera.grab():
                continue
            rgb = camera.get_rgb()
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            ret, frame = cap.read()
            if not ret:
                break

        # 크롭 적용
        if crop_roi:
            frame = apply_crop(frame, crop_roi)

        frame_count += 1
        if frame_count % capture_interval == 0:
            # 다양성 체크
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                diff = compute_frame_diff(prev_gray, gray)
                if diff < 5.0:
                    continue
            prev_gray = gray

            path = os.path.join(output_dir, f"calib_{count:04d}.png")
            cv2.imwrite(path, frame)
            count += 1

        # 진행 표시
        display = frame.copy()
        cv2.putText(display, f"Captured: {count}/{max_images}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(display, "Walk naturally. Press 'q' to stop.",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.imshow("Calibration Capture", display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if use_zed:
        camera.close()
    else:
        cap.release()
    cv2.destroyAllWindows()

    print(f"  {count}장 캡처 완료 → {output_dir}")
    return count


def download_coco_calib_images(output_dir, num_images=100):
    """
    COCO val2017 이미지를 다운로드하여 캘리브레이션 데이터 다양성 보강

    COCO 이미지가 포함하는 다양성:
    - 실내/실외, 다양한 조명, 다양한 포즈
    - 부분 가림(occlusion), 다양한 거리/각도
    - 이미 person 클래스가 포함된 이미지가 다수

    걷기 영상(도메인 특화) + COCO(범용) 혼합으로
    Entropy 캘리브레이션의 과적합 문제를 완화합니다.
    """
    import urllib.request
    import zipfile

    coco_dir = os.path.join(output_dir, "coco_calib")

    # 이미 다운로드된 경우 재사용
    existing = glob.glob(os.path.join(coco_dir, "*.png")) + \
        glob.glob(os.path.join(coco_dir, "*.jpg"))
    if len(existing) >= num_images:
        print(f"  COCO 캘리브레이션 이미지 재사용: {len(existing)}장")
        return coco_dir

    os.makedirs(coco_dir, exist_ok=True)

    # Ultralytics coco8-pose 데이터셋 사용 (작고 빠름, person 포함)
    # 대안: COCO val2017에서 person 이미지만 추출
    print(f"  COCO 캘리브레이션 이미지 준비 중...")

    # 방법 1: ultralytics의 coco8-pose 자동 다운로드 활용
    try:
        from ultralytics.data.utils import check_det_dataset
        dataset_info = check_det_dataset('coco8-pose.yaml')
        coco8_dir = dataset_info.get('path', '')
        if coco8_dir:
            # coco8-pose 이미지를 캘리브레이션 디렉토리로 복사
            for subdir in ['train', 'val']:
                img_dir = os.path.join(coco8_dir, 'images', subdir)
                if os.path.isdir(img_dir):
                    for f in glob.glob(os.path.join(img_dir, '*.jpg')):
                        dst = os.path.join(coco_dir, os.path.basename(f))
                        if not os.path.exists(dst):
                            import shutil
                            shutil.copy2(f, dst)
            existing = glob.glob(os.path.join(coco_dir, "*.png")) + \
        glob.glob(os.path.join(coco_dir, "*.jpg"))
            if existing:
                print(f"  COCO (coco8-pose) 이미지: {len(existing)}장")
                return coco_dir
    except Exception as e:
        print(f"  coco8-pose 자동 다운로드 실패: {e}")

    # 방법 2: 합성 다양성 이미지 생성 (네트워크 없이)
    print(f"  COCO 다운로드 불가 → 다양성 합성 이미지 {num_images}장 생성")
    rng = np.random.RandomState(12345)
    for i in range(num_images):
        # 다양한 배경 + 사람 형태 합성
        bg_val = rng.randint(20, 230)
        img = np.full((640, 640, 3), bg_val, dtype=np.uint8)

        # 랜덤 그래디언트 배경
        for c in range(3):
            gradient = np.linspace(rng.randint(0, 128), rng.randint(128, 255), 640)
            if rng.random() > 0.5:
                img[:, :, c] = gradient[np.newaxis, :].astype(np.uint8)
            else:
                img[:, :, c] = gradient[:, np.newaxis].astype(np.uint8)

        # 사람 형태 (다양한 크기/위치)
        cx = rng.randint(100, 540)
        cy = rng.randint(100, 540)
        body_w = rng.randint(40, 200)
        body_h = rng.randint(100, 400)
        color = tuple(int(x) for x in rng.randint(0, 255, 3))
        cv2.ellipse(img, (cx, cy), (body_w // 2, body_h // 2), 0, 0, 360, color, -1)

        # 다리
        for dx in [-20, 20]:
            leg_len = rng.randint(60, 200)
            leg_color = tuple(int(x) for x in rng.randint(0, 255, 3))
            cv2.line(img, (cx + dx, cy + body_h // 3),
                     (cx + dx + rng.randint(-30, 30), cy + body_h // 3 + leg_len),
                     leg_color, rng.randint(4, 12))

        # 노이즈 추가
        noise = rng.randint(-20, 21, img.shape, dtype=np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        cv2.imwrite(os.path.join(coco_dir, f"coco_synth_{i:04d}.png"), img)

    print(f"  합성 다양성 이미지: {num_images}장")
    return coco_dir


def mix_calib_directories(walking_dir, coco_dir, output_dir, walking_ratio=0.7):
    """
    걷기 데이터 + COCO 데이터를 혼합하여 최종 캘리브레이션 디렉토리 생성

    Args:
        walking_ratio: 걷기 데이터 비율 (0.7 = 70% 걷기 + 30% COCO)

    혼합 근거:
    - 걷기 데이터: 타겟 도메인 최적화 (실제 사용 환경)
    - COCO 데이터: 범용 분포 커버 (activation 범위 과적합 방지)
    """
    walking_images = sorted(
        glob.glob(os.path.join(walking_dir, "*.png")) +
        glob.glob(os.path.join(walking_dir, "*.jpg")))
    coco_images = sorted(
        glob.glob(os.path.join(coco_dir, "*.png")) +
        glob.glob(os.path.join(coco_dir, "*.jpg")))

    if not walking_images:
        print(f"  WARNING: 걷기 이미지 없음 ({walking_dir})")
        return walking_dir
    if not coco_images:
        print(f"  WARNING: COCO 이미지 없음 ({coco_dir})")
        return walking_dir

    total = len(walking_images) + len(coco_images)
    n_walking = int(total * walking_ratio)
    n_coco = total - n_walking

    # 비율에 맞게 샘플링
    rng = np.random.RandomState(42)
    if len(walking_images) > n_walking:
        walking_sample = list(rng.choice(walking_images, n_walking, replace=False))
    else:
        walking_sample = walking_images

    if len(coco_images) > n_coco:
        coco_sample = list(rng.choice(coco_images, n_coco, replace=False))
    else:
        coco_sample = coco_images

    # 혼합 디렉토리 생성
    mixed_dir = os.path.join(output_dir, "mixed")
    os.makedirs(mixed_dir, exist_ok=True)

    import shutil
    idx = 0
    for src in walking_sample:
        dst = os.path.join(mixed_dir, f"mixed_{idx:04d}.jpg")
        shutil.copy2(src, dst)
        idx += 1
    for src in coco_sample:
        dst = os.path.join(mixed_dir, f"mixed_{idx:04d}.jpg")
        shutil.copy2(src, dst)
        idx += 1

    print(f"  혼합 캘리브레이션 데이터: {idx}장 "
          f"(걷기 {len(walking_sample)} + COCO {len(coco_sample)})")
    return mixed_dir


def create_calib_yaml(calib_dir):
    """
    Ultralytics INT8 캘리브레이션용 YAML 파일 생성

    Ultralytics export(data=...)는 YAML 파일을 요구함.
    캘리브레이션 이미지 디렉토리를 train/val로 참조하는 YAML을 생성.
    """
    import yaml

    calib_dir = os.path.abspath(calib_dir)
    yaml_path = os.path.join(calib_dir, "calib_dataset.yaml")

    data = {
        'path': calib_dir,
        'train': '.',        # 캘리브레이션 이미지가 있는 디렉토리
        'val': '.',
        'names': {0: 'person'},  # pose 모델이므로 person 클래스
        'nc': 1,
        'kpt_shape': [17, 3],
    }

    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)

    print(f"  캘리브레이션 YAML 생성: {yaml_path}")
    return yaml_path


def build_int8_engines(calib_dir, model_sizes=None, yolo_versions=None):
    """캘리브레이션 이미지로 YOLO INT8 엔진 빌드 (YOLOv8 + YOLO11)"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("ERROR: ultralytics 미설치. pip install ultralytics")
        return

    if model_sizes is None:
        model_sizes = ["n", "s"]

    if yolo_versions is None:
        yolo_versions = ["yolov8", "yolo11", "yolo26"]

    # YAML 파일 생성 (Ultralytics가 디렉토리 직접 지원 안 함)
    calib_yaml = create_calib_yaml(calib_dir)

    for prefix in yolo_versions:
        version_label = {"yolov8": "YOLOv8", "yolo11": "YOLO11", "yolo26": "YOLO26"}.get(prefix, prefix)
        for size in model_sizes:
            model_name = f"{prefix}{size}-pose.pt"
            engine_name = f"{prefix}{size}-pose-int8-cal.engine"

            print(f"\n{'─'*50}")
            print(f"  {version_label}{size} INT8 엔진 빌드")
            print(f"  모델: {model_name}")
            print(f"  캘리브레이션: {calib_yaml}")
            print(f"{'─'*50}")

            # 기존 INT8 엔진 삭제
            for old in [engine_name, f"{prefix}{size}-pose.engine"]:
                if os.path.exists(old):
                    print(f"  기존 엔진 삭제: {old}")
                    os.remove(old)

            model = YOLO(model_name)
            t0 = time.time()
            path = model.export(
                format='engine',
                half=True,       # FP16 폴백 활성화 (중요!)
                int8=True,
                data=calib_yaml,
                imgsz=640,       # 추론 시 사용하는 해상도와 동일하게 맞춤
            )
            elapsed = time.time() - t0

            if path and os.path.exists(str(path)):
                # 정밀도별 이름으로 저장
                default_engine = f"{prefix}{size}-pose.engine"
                if os.path.exists(default_engine) and engine_name != default_engine:
                    os.rename(default_engine, engine_name)
                    path = engine_name
                print(f"  빌드 완료: {path} ({elapsed:.0f}초)")
            else:
                print(f"  빌드 실패!")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO INT8 캘리브레이션 이미지 준비 (YOLOv8 + YOLO11, 증강 + 다양성 필터)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  python calibrate_yolo.py --video walking.mp4             # 영상에서 추출 (증강 포함)
  python calibrate_yolo.py --video walking.mp4 --no-augment  # 증강 없이 원본만
  python calibrate_yolo.py --zed --num-images 200          # ZED 촬영
  python calibrate_yolo.py --webcam --num-images 200       # 웹캠 촬영
  python calibrate_yolo.py --video walking.mp4 --build     # 추출 + 엔진 빌드

  # crop_roi.json이 있으면 자동으로 크롭된 영역에서 캘리브레이션
  python select_roi.py --video walking.mp4    # 먼저 ROI 선택
  python calibrate_yolo.py --video walking.mp4 --build   # 크롭 적용됨
        """)

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=str, help="걷기 영상 파일")
    source.add_argument("--zed", action="store_true", help="ZED 카메라로 촬영")
    source.add_argument("--webcam", action="store_true", help="웹캠으로 촬영")

    parser.add_argument("--num-images", type=int, default=300,
                        help="캘리브레이션 이미지 수 (기본: 300)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="출력 디렉토리 (기본: ./calib_images/yolo/)")
    parser.add_argument("--build", action="store_true",
                        help="캘리브레이션 후 INT8 엔진도 빌드")
    parser.add_argument("--models", type=str, nargs='+', default=["n", "s"],
                        choices=["n", "s", "m", "l", "x"],
                        help="빌드할 모델 사이즈 (기본: n s)")
    parser.add_argument("--yolo-versions", type=str, nargs='+',
                        default=["yolov8", "yolo11", "yolo26"],
                        choices=["yolov8", "yolo11", "yolo26"],
                        help="빌드할 YOLO 버전 (기본: yolov8 yolo11 yolo26)")

    # 새 옵션들
    parser.add_argument("--no-augment", action="store_true",
                        help="데이터 증강 비활성화 (원본만 사용)")
    parser.add_argument("--crop", type=str, default=None,
                        help="ROI 크롭 JSON 경로 (기본: crop_roi.json 자동 로드)")
    parser.add_argument("--no-crop", action="store_true",
                        help="크롭 비활성화")
    parser.add_argument("--diversity", type=float, default=5.0,
                        help="프레임 다양성 임계값 (낮을수록 유사 프레임 허용, 기본: 5.0)")
    parser.add_argument("--mix-coco", action="store_true",
                        help="COCO 이미지를 혼합하여 캘리브레이션 다양성 보강 "
                             "(과적합 방지, INT8 정확도 향상)")
    parser.add_argument("--walking-ratio", type=float, default=0.7,
                        help="COCO 혼합 시 걷기 데이터 비율 (기본: 0.7 = 70%% 걷기 + 30%% COCO)")
    parser.add_argument("--save-format", type=str, default="png",
                        choices=["png", "jpg"],
                        help="캘리브레이션 이미지 저장 포맷 (png=무손실 권장, jpg=JPEG95, 기본: png)")
    parser.add_argument("--min-resolution", type=int, default=480,
                        help="최소 해상도 (짧은 변 기준, 기본: 480px, 미만 시 업스케일)")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "calib_images", "yolo")

    # ROI 크롭 로드
    crop_roi = None
    if not args.no_crop:
        crop_roi = auto_load_crop_roi(args.crop)

    print("=" * 50)
    print("  YOLO INT8 캘리브레이션 이미지 준비")
    print("=" * 50)
    if crop_roi:
        print(f"  [ROI 크롭] {crop_roi['w']}x{crop_roi['h']} "
              f"(x={crop_roi['x']}, y={crop_roi['y']})")
    print(f"  증강: {'ON (밝기/대비/반전/블러/스케일)' if not args.no_augment else 'OFF'}")
    print(f"  다양성 필터: {args.diversity}")
    if args.mix_coco:
        print(f"  COCO 혼합: ON (걷기 {args.walking_ratio*100:.0f}% + COCO {(1-args.walking_ratio)*100:.0f}%)")
    print()

    # 이미지 수집
    if args.video:
        count = extract_from_video(
            args.video, output_dir, args.num_images,
            crop_roi=crop_roi,
            augment=not args.no_augment,
            diversity_threshold=args.diversity,
            min_resolution=args.min_resolution,
            save_format=args.save_format,
        )
    elif args.zed:
        count = capture_from_camera(output_dir, args.num_images,
                                     use_zed=True, crop_roi=crop_roi)
    else:
        count = capture_from_camera(output_dir, args.num_images,
                                     use_zed=False, crop_roi=crop_roi)

    if count == 0:
        print("ERROR: 캘리브레이션 이미지를 수집하지 못했습니다.")
        sys.exit(1)

    print(f"\n  캘리브레이션 이미지: {count}장")
    print(f"  저장 위치: {output_dir}")

    # COCO 데이터 혼합
    build_dir = output_dir
    if args.mix_coco:
        print(f"\n{'='*50}")
        print(f"  COCO 데이터 혼합 (과적합 방지)")
        print(f"{'='*50}")
        coco_dir = download_coco_calib_images(output_dir)
        build_dir = mix_calib_directories(
            output_dir, coco_dir, output_dir,
            walking_ratio=args.walking_ratio,
        )
        print(f"  최종 캘리브레이션 디렉토리: {build_dir}")

    # 엔진 빌드
    if args.build:
        print(f"\n{'='*50}")
        print(f"  INT8 엔진 빌드 시작")
        print(f"{'='*50}")
        build_int8_engines(build_dir, args.models, args.yolo_versions)
    else:
        print(f"\n  사용법:")
        print(f"    # 벤치마크에서 사용")
        print(f"    python run_int8_comparison.py --calib-data {build_dir}")
        print(f"    # 또는 엔진만 빌드")
        print(f"    python calibrate_yolo.py --video {args.video or 'walking.mp4'} --build")


if __name__ == "__main__":
    main()
