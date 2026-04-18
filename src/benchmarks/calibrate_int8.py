#!/usr/bin/env python3
"""
INT8 캘리브레이션 데이터 생성 스크립트
======================================
TensorRT INT8 엔진 빌드를 위한 캘리브레이션 캐시를 생성합니다.

캘리브레이션이란?
  - INT8은 FP32 대비 값의 범위가 매우 작음 (-128 ~ 127)
  - 각 레이어의 activation 값 분포를 측정하여 최적의 스케일링 팩터를 결정
  - 이를 위해 대표적인 입력 이미지 100~500장을 모델에 통과시킴
  - 결과로 calibration cache 파일 생성 → TRT 엔진 빌드 시 사용

사용법:
  # 이미지 폴더로 캘리브레이션 (권장)
  python calibrate_int8.py --images ./calib_images/ --model rtmpose

  # 비디오 파일에서 프레임 추출하여 캘리브레이션
  python calibrate_int8.py --video recording.mp4 --model rtmpose

  # 합성 데이터로 빠른 캘리브레이션 (정확도 다소 낮음)
  python calibrate_int8.py --synthetic --model rtmpose

  # 캘리브레이션 이미지 수집 (웹캠에서 캡처)
  python calibrate_int8.py --capture --num-images 200
"""

import os
import sys
import argparse
import glob
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from select_roi import auto_load_crop_roi, apply_crop


# ============================================================================
# 캘리브레이션 이미지 수집
# ============================================================================
def collect_from_images(image_dir, max_images=500):
    """이미지 디렉토리에서 캘리브레이션 이미지 로드"""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
    paths = []
    for ext in extensions:
        paths.extend(glob.glob(os.path.join(image_dir, ext)))
        paths.extend(glob.glob(os.path.join(image_dir, "**", ext), recursive=True))

    paths = sorted(set(paths))[:max_images]
    if not paths:
        print(f"ERROR: {image_dir}에서 이미지를 찾을 수 없습니다.")
        return []

    images = []
    for p in paths:
        img = cv2.imread(p)
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    print(f"  이미지 {len(images)}장 로드 완료 (from {image_dir})")
    return images


def augment_image(image, rng):
    """캘리브레이션용 이미지 증강 (RGB 입력)"""
    aug = image.copy()
    h, w = aug.shape[:2]

    # 밝기/대비
    alpha = rng.uniform(0.7, 1.3)
    beta = rng.randint(-30, 31)
    aug = cv2.convertScaleAbs(aug, alpha=alpha, beta=beta)

    # 좌우 반전 (50%)
    if rng.random() > 0.5:
        aug = cv2.flip(aug, 1)

    # 가우시안 블러 (30%)
    if rng.random() > 0.7:
        ksize = rng.choice([3, 5])
        aug = cv2.GaussianBlur(aug, (ksize, ksize), 0)

    # 스케일 지터 (50%)
    if rng.random() > 0.5:
        scale = rng.uniform(0.85, 1.15)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(aug, (new_w, new_h))
        if scale > 1.0:
            cx, cy = new_w // 2, new_h // 2
            aug = resized[cy - h//2:cy - h//2 + h, cx - w//2:cx - w//2 + w]
        else:
            canvas = np.zeros_like(image)
            y_off = (h - new_h) // 2
            x_off = (w - new_w) // 2
            canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
            aug = canvas

    return aug


def collect_from_video(video_path, max_images=300, skip_frames=5,
                       crop_roi=None, augment=True):
    """비디오에서 프레임 추출 + 다양성 필터 + 증강"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: 비디오를 열 수 없습니다: {video_path}")
        return []

    if crop_roi:
        print(f"  ROI 크롭 적용: {crop_roi['w']}x{crop_roi['h']}")

    if augment:
        base_count = int(max_images * 0.6)
        aug_count = max_images - base_count
    else:
        base_count = max_images
        aug_count = 0

    images = []
    frame_idx = 0
    prev_gray = None
    rng = np.random.RandomState(42)

    while len(images) < base_count:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip_frames == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if crop_roi:
                rgb = apply_crop(rgb, crop_roi)

            # 다양성 필터
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            if prev_gray is not None:
                diff = np.mean(cv2.absdiff(prev_gray, gray))
                if diff < 5.0:
                    frame_idx += 1
                    continue
            prev_gray = gray

            images.append(rgb)
        frame_idx += 1

    cap.release()
    print(f"  비디오에서 {len(images)}프레임 추출 완료 (from {video_path})")

    # 증강
    if augment and images and aug_count > 0:
        print(f"  증강 이미지 {aug_count}장 생성 중...")
        base_images = list(images)
        for i in range(aug_count):
            src = base_images[rng.randint(0, len(base_images))]
            images.append(augment_image(src, rng))
        print(f"  총 {len(images)}장 (원본 {len(base_images)} + 증강 {aug_count})")

    return images


def collect_from_webcam(num_images=200, camera_id=0):
    """웹캠에서 캘리브레이션 이미지 캡처"""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print(f"ERROR: 카메라 {camera_id}를 열 수 없습니다.")
        return []

    print(f"  웹캠에서 {num_images}장 캡처합니다.")
    print("  다양한 포즈를 취하면서 자연스럽게 움직여 주세요.")
    print("  'q'를 누르면 조기 종료합니다.")

    images = []
    frame_count = 0
    capture_interval = 3  # 매 3프레임마다 캡처

    while len(images) < num_images:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % capture_interval == 0:
            images.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # 진행 상황 표시
            display = frame.copy()
            cv2.putText(display, f"Captured: {len(images)}/{num_images}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Calibration Capture", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"  {len(images)}장 캡처 완료")
    return images


def generate_synthetic(num_images=200, width=640, height=480):
    """합성 캘리브레이션 데이터 생성 (빠르지만 정확도 다소 낮음)"""
    print(f"  합성 캘리브레이션 이미지 {num_images}장 생성 중...")
    images = []
    rng = np.random.RandomState(42)

    for i in range(num_images):
        # 사람 형태를 모방한 합성 이미지
        img = rng.randint(50, 200, (height, width, 3), dtype=np.uint8)

        # 사람 형태의 직사각형/타원 그리기
        cx = rng.randint(width // 4, 3 * width // 4)
        cy = rng.randint(height // 4, 3 * height // 4)
        body_w = rng.randint(60, 150)
        body_h = rng.randint(150, 350)

        cv2.ellipse(img, (cx, cy), (body_w // 2, body_h // 2), 0, 0, 360,
                    (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)), -1)
        # 다리 추가
        leg_len = rng.randint(80, 180)
        cv2.line(img, (cx - 20, cy + body_h // 3), (cx - 30, cy + body_h // 3 + leg_len),
                 (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)), 8)
        cv2.line(img, (cx + 20, cy + body_h // 3), (cx + 30, cy + body_h // 3 + leg_len),
                 (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255)), 8)

        images.append(img)

    print(f"  합성 이미지 {len(images)}장 생성 완료")
    return images


def save_calib_images(images, output_dir):
    """캘리브레이션 이미지를 디렉토리에 저장 (PNG 무손실)"""
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(output_dir, f"calib_{i:04d}.png"), bgr)
    print(f"  {len(images)}장을 {output_dir}에 저장 (PNG 무손실)")


# ============================================================================
# TensorRT INT8 캘리브레이터
# ============================================================================
class OnnxInt8Calibrator:
    """
    ONNX 모델용 TensorRT INT8 캘리브레이터

    작동 방식:
      1. ONNX 모델의 입력 형식 확인 (shape, dtype)
      2. 캘리브레이션 이미지를 모델 입력 형식으로 전처리
      3. TensorRT가 각 레이어의 activation 분포를 측정
      4. 최적의 INT8 스케일링 팩터를 결정하여 캐시에 저장
    """

    def __init__(self, onnx_path, images, cache_file, batch_size=1):
        import tensorrt as trt

        self.cache_file = cache_file
        self.batch_size = batch_size
        self.images = images
        self.current_idx = 0

        # ONNX 모델에서 입력 shape 추출
        import onnxruntime as ort
        sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        inp = sess.get_inputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape  # e.g., [1, 3, 256, 192] or [1, 3, 640, 640]
        self.dtype = np.float32

        # 동적 shape 처리
        self.input_h = self.input_shape[2] if isinstance(self.input_shape[2], int) else 256
        self.input_w = self.input_shape[3] if isinstance(self.input_shape[3], int) else 192

        print(f"  캘리브레이터 초기화: input={self.input_name}, "
              f"shape=[1,3,{self.input_h},{self.input_w}], images={len(images)}")
        del sess

    def preprocess(self, image):
        """이미지를 모델 입력 형식으로 변환"""
        # Resize
        img = cv2.resize(image, (self.input_w, self.input_h))
        # Normalize to [0, 1] then standardize
        img = img.astype(np.float32) / 255.0
        # BGR→RGB는 이미 되어있다고 가정
        # HWC → CHW
        img = np.transpose(img, (2, 0, 1))
        # Add batch dimension
        img = np.expand_dims(img, axis=0)
        return np.ascontiguousarray(img)

    def get_batch(self):
        """다음 배치의 캘리브레이션 데이터 반환"""
        if self.current_idx >= len(self.images):
            return None
        img = self.images[self.current_idx]
        self.current_idx += 1
        return self.preprocess(img)


def run_calibration_tensorrt(onnx_path, images, cache_file, calib_algo="minmax"):
    """
    TensorRT Python API로 INT8 캘리브레이션 수행

    이 함수가 하는 일:
    1. ONNX 모델을 TensorRT 네트워크로 파싱
    2. INT8 캘리브레이션 모드로 빌더 설정
    3. 캘리브레이션 이미지를 순차적으로 통과시켜 분포 측정
    4. 캘리브레이션 캐시를 파일로 저장

    Args:
        calib_algo: "minmax" (안정적, 범용) 또는 "entropy" (데이터 의존적)
            - minmax: activation의 min/max로 범위 결정, 과적합 위험 낮음
            - entropy: KL-divergence 최소화, 캘리브레이션 데이터에 최적화되지만
                       데이터 분포가 편향되면 오히려 정확도 저하
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("ERROR: tensorrt 패키지가 설치되어 있지 않습니다.")
        print("  Jetson: sudo apt-get install python3-libnvinfer")
        return False

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    algo_label = calib_algo.upper()
    print(f"\n=== TensorRT INT8 캘리브레이션 시작 ({algo_label}) ===")
    print(f"  ONNX 모델: {onnx_path}")
    print(f"  캘리브레이션 이미지: {len(images)}장")
    print(f"  알고리즘: {algo_label}")
    print(f"  캐시 파일: {cache_file}")

    calibrator = OnnxInt8Calibrator(onnx_path, images, cache_file)

    # 캘리브레이션 알고리즘 선택
    if calib_algo == "minmax":
        # MinMax: activation의 절대 min/max로 범위 결정
        # - 과적합 위험 낮음, 범용적
        # - Pose estimation처럼 keypoint 좌표가 중요한 경우 안정적
        BaseCalibrator = trt.IInt8MinMaxCalibrator
        print(f"  → MinMax: 보수적 범위 설정, 과적합 방지")
    else:
        # Entropy: KL-divergence 최소화로 최적 범위 결정
        # - 캘리브레이션 데이터와 유사한 입력에서 최적
        # - 데이터 분포가 편향되면 오히려 정확도 저하 가능
        BaseCalibrator = trt.IInt8EntropyCalibrator2
        print(f"  → Entropy: 데이터 기반 최적화 (과적합 주의)")

    class TrtInt8Calibrator(BaseCalibrator):
        def __init__(self, calib):
            super().__init__()
            self.calib = calib
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
            # GPU 메모리 할당
            dummy = calib.preprocess(calib.images[0])
            self.device_input = cuda.mem_alloc(dummy.nbytes)
            self.batch_size = 1

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            batch = self.calib.get_batch()
            if batch is None:
                return None
            import pycuda.driver as cuda
            cuda.memcpy_htod(self.device_input, batch)
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if os.path.exists(self.calib.cache_file):
                with open(self.calib.cache_file, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.calib.cache_file, "wb") as f:
                f.write(cache)
            print(f"  캘리브레이션 캐시 저장 완료: {self.calib.cache_file}")

    try:
        int8_calibrator = TrtInt8Calibrator(calibrator)
    except ImportError:
        print("WARNING: pycuda 미설치. ONNX Runtime 기반 캘리브레이션으로 전환합니다.")
        return run_calibration_ort(onnx_path, images, cache_file, calib_algo)

    # TRT 빌더로 캘리브레이션 수행
    builder = trt.Builder(TRT_LOGGER)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  ONNX 파싱 에러: {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)
    config.set_flag(trt.BuilderFlag.INT8)
    config.int8_calibrator = int8_calibrator

    # FP16도 함께 활성화 (INT8 불가 레이어는 FP16 사용)
    config.set_flag(trt.BuilderFlag.FP16)

    print("  캘리브레이션 진행 중... (첫 실행 시 수분 소요)")
    engine = builder.build_serialized_network(network, config)

    if engine is None:
        print("ERROR: TRT 엔진 빌드 실패")
        return False

    print(f"  INT8 캘리브레이션 완료! (알고리즘: {algo_label})")
    print(f"  캐시 파일: {cache_file}")
    print(f"  캐시 크기: {os.path.getsize(cache_file)} bytes")
    return True


def run_calibration_ort(onnx_path, images, cache_file, calib_algo="minmax"):
    """
    ONNX Runtime 기반 간이 캘리브레이션

    pycuda가 없는 환경에서의 대안:
    - ONNX Runtime quantization 도구를 사용하여 캘리브레이션
    - 결과를 TRT가 인식하는 형식으로 변환
    """
    try:
        from onnxruntime.quantization import CalibrationDataReader, create_calibrator
        from onnxruntime.quantization import CalibrationMethod
    except ImportError:
        print("ERROR: onnxruntime.quantization을 사용할 수 없습니다.")
        print("  pip install onnxruntime 또는 onnxruntime-gpu 설치 필요")
        return False

    import onnxruntime as ort

    # 입력 shape 확인
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    input_name = inp.name
    input_shape = inp.shape
    input_h = input_shape[2] if isinstance(input_shape[2], int) else 256
    input_w = input_shape[3] if isinstance(input_shape[3], int) else 192
    del sess

    class PoseCalibrationReader(CalibrationDataReader):
        def __init__(self, images, input_name, input_h, input_w):
            self.images = images
            self.input_name = input_name
            self.input_h = input_h
            self.input_w = input_w
            self.idx = 0

        def get_next(self):
            if self.idx >= len(self.images):
                return None
            img = self.images[self.idx]
            self.idx += 1
            # Preprocess
            img = cv2.resize(img, (self.input_w, self.input_h))
            img = img.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = np.expand_dims(img, axis=0)
            return {self.input_name: img}

    # 알고리즘 선택
    if calib_algo == "minmax":
        ort_method = CalibrationMethod.MinMax
    else:
        ort_method = CalibrationMethod.Entropy

    algo_label = calib_algo.upper()
    print(f"\n=== ONNX Runtime 캘리브레이션 시작 ({algo_label}) ===")
    print(f"  ONNX 모델: {onnx_path}")
    print(f"  입력 shape: [1, 3, {input_h}, {input_w}]")
    print(f"  알고리즘: {algo_label}")
    print(f"  캘리브레이션 이미지: {len(images)}장")

    reader = PoseCalibrationReader(images, input_name, input_h, input_w)

    calibrator = create_calibrator(
        onnx_path,
        calibrate_method=ort_method,
    )
    calibrator.set_execution_providers(['CUDAExecutionProvider', 'CPUExecutionProvider'])

    print("  캘리브레이션 데이터 수집 중...")
    calibrator.collect_data(reader)

    print("  동적 범위 계산 중...")
    tensors_range = calibrator.compute_data()

    # TensorRT 캘리브레이션 캐시 형식으로 변환
    write_trt_calibration_cache(tensors_range, cache_file, calib_algo)

    print(f"  캘리브레이션 완료!")
    print(f"  캐시 파일: {cache_file}")
    return True


def write_trt_calibration_cache(tensors_range, cache_file, calib_algo="minmax"):
    """
    ONNX Runtime quantization 결과를 TensorRT 캘리브레이션 캐시 형식으로 변환

    TRT 캘리브레이션 캐시 형식:
      TRT-{version}-{CalibrationAlgorithm}
      tensor_name: hex_encoded_scale
    """
    import struct

    if calib_algo == "minmax":
        header = "TRT-8XXX-MinMaxCalibration"
    else:
        header = "TRT-8XXX-EntropyCalibration2"
    lines = [header]
    for name, (min_val, max_val) in tensors_range.items():
        # 동적 범위에서 스케일 계산
        abs_max = max(abs(float(min_val)), abs(float(max_val)))
        if abs_max == 0:
            abs_max = 1e-8
        # TRT 형식: tensor_name: hex_float(scale)
        scale_bytes = struct.pack("f", abs_max)
        hex_scale = scale_bytes.hex()
        lines.append(f"{name}: {hex_scale}")

    with open(cache_file, "w") as f:
        f.write("\n".join(lines))

    print(f"  TRT 캘리브레이션 캐시 변환 완료 ({len(lines)-1}개 텐서)")


# ============================================================================
# 모델별 ONNX 경로 탐색
# ============================================================================
def find_rtmpose_onnx(model_type="pose"):
    """rtmlib이 다운로드한 RTMPose ONNX 모델 경로 탐색"""
    # rtmlib 기본 캐시 위치
    search_dirs = [
        os.path.expanduser("~/.cache/rtmlib"),
        os.path.expanduser("~/rtmlib_models"),
        "/tmp/rtmlib",
    ]

    # 추가: 현재 디렉토리와 benchmarks 디렉토리
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs.extend([
        script_dir,
        os.path.join(script_dir, "models"),
    ])

    onnx_files = []
    for d in search_dirs:
        if os.path.isdir(d):
            for root, dirs, files in os.walk(d):
                for f in files:
                    if f.endswith(".onnx"):
                        full_path = os.path.join(root, f)
                        onnx_files.append(full_path)

    if not onnx_files:
        return None

    # 모델 타입에 따라 필터링
    if model_type == "det":
        # detection 모델 (rtmdet)
        candidates = [f for f in onnx_files if "det" in os.path.basename(f).lower()]
    else:
        # pose 모델 (rtmpose)
        candidates = [f for f in onnx_files if "pose" in os.path.basename(f).lower()
                       or "rtmpose" in os.path.basename(f).lower()]

    if not candidates:
        candidates = onnx_files

    return candidates


def find_onnx_from_rtmlib_session(mode="balanced"):
    """rtmlib Body를 로드하여 실제 ONNX 경로를 가져옴"""
    try:
        from rtmlib import Body
        body = Body(mode=mode, to_openpose=False,
                    backend="onnxruntime", device="cpu")

        paths = {}
        if hasattr(body, 'det_model'):
            det_path = getattr(body.det_model, 'model_path', None)
            if det_path and os.path.exists(det_path):
                paths['det'] = det_path
        if hasattr(body, 'pose_model'):
            pose_path = getattr(body.pose_model, 'model_path', None)
            if pose_path and os.path.exists(pose_path):
                paths['pose'] = pose_path

        return paths
    except Exception as e:
        print(f"  rtmlib에서 모델 경로를 가져올 수 없음: {e}")
        return {}


# ============================================================================
# 메인
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="TensorRT INT8 캘리브레이션 캐시 생성",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
예제:
  # 이미지 폴더로 캘리브레이션 (가장 정확)
  python calibrate_int8.py --images ./calib_images/ --model rtmpose

  # 비디오에서 프레임 추출
  python calibrate_int8.py --video recording.mp4 --model rtmpose

  # 합성 데이터로 빠른 캘리브레이션
  python calibrate_int8.py --synthetic --model rtmpose

  # 웹캠으로 캘리브레이션 이미지 캡처만
  python calibrate_int8.py --capture --output-dir ./calib_images/
        """)

    # 데이터 소스
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=str, help="캘리브레이션 이미지 디렉토리")
    source.add_argument("--video", type=str, help="비디오 파일에서 프레임 추출")
    source.add_argument("--capture", action="store_true", help="웹캠에서 이미지 캡처")
    source.add_argument("--synthetic", action="store_true",
                        help="합성 데이터 사용 (빠르지만 정확도 낮음)")

    # 모델
    parser.add_argument("--model", type=str, default="rtmpose",
                        choices=["rtmpose", "rtmpose_wb"],
                        help="캘리브레이션할 모델 (default: rtmpose)")
    parser.add_argument("--mode", type=str, default="balanced",
                        choices=["lightweight", "balanced", "performance"],
                        help="RTMPose 모드 (default: balanced)")
    parser.add_argument("--onnx", type=str, default=None,
                        help="ONNX 모델 경로 직접 지정 (자동 탐색 대신)")

    # 옵션
    parser.add_argument("--num-images", type=int, default=300,
                        help="캘리브레이션 이미지 수 (default: 300)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="캘리브레이션 캐시 출력 디렉토리")
    parser.add_argument("--save-images", action="store_true",
                        help="캘리브레이션 이미지를 디스크에 저장")
    parser.add_argument("--camera-id", type=int, default=0,
                        help="웹캠 ID (--capture 시)")
    parser.add_argument("--crop", type=str, default=None,
                        help="ROI 크롭 JSON 경로 (기본: crop_roi.json 자동 로드)")
    parser.add_argument("--no-crop", action="store_true",
                        help="크롭 비활성화")
    parser.add_argument("--no-augment", action="store_true",
                        help="데이터 증강 비활성화")
    parser.add_argument("--calib-algo", type=str, default="minmax",
                        choices=["minmax", "entropy"],
                        help="캘리브레이션 알고리즘 (기본: minmax). "
                             "minmax: 안정적/범용, entropy: 데이터 의존적/과적합 위험")

    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = args.output_dir or os.path.join(script_dir, "int8_calib_cache")
    os.makedirs(output_dir, exist_ok=True)

    # ROI 크롭 로드
    crop_roi = None
    if not args.no_crop:
        crop_roi = auto_load_crop_roi(args.crop)

    augment = not args.no_augment

    # ---- 1. 캘리브레이션 이미지 수집 ----
    print("\n[1/3] 캘리브레이션 이미지 수집")
    if crop_roi:
        print(f"  [ROI 크롭] {crop_roi['w']}x{crop_roi['h']}")
    if augment:
        print(f"  [증강] ON (밝기/대비/반전/블러/스케일)")

    if args.images:
        images = collect_from_images(args.images, args.num_images)
    elif args.video:
        images = collect_from_video(args.video, args.num_images,
                                     crop_roi=crop_roi, augment=augment)
    elif args.capture:
        images = collect_from_webcam(args.num_images, args.camera_id)
        # 캡처한 이미지 저장
        save_dir = os.path.join(script_dir, "calib_images")
        save_calib_images(images, save_dir)
        print(f"  캡처된 이미지가 {save_dir}에 저장되었습니다.")
        if not args.onnx and not args.model:
            print("  이 이미지로 캘리브레이션하려면:")
            print(f"    python calibrate_int8.py --images {save_dir} --model rtmpose")
            return
    else:
        images = generate_synthetic(args.num_images)

    if not images:
        print("ERROR: 캘리브레이션 이미지를 수집하지 못했습니다.")
        sys.exit(1)

    if args.save_images and not args.capture:
        save_dir = os.path.join(script_dir, "calib_images")
        save_calib_images(images, save_dir)

    # ---- 2. ONNX 모델 경로 찾기 ----
    print("\n[2/3] ONNX 모델 탐색")
    if args.onnx:
        onnx_paths = {"pose": args.onnx}
        print(f"  지정된 ONNX 경로: {args.onnx}")
    else:
        print(f"  rtmlib에서 {args.model} ({args.mode}) 모델 경로 탐색 중...")
        onnx_paths = find_onnx_from_rtmlib_session(args.mode)
        if not onnx_paths:
            print("  rtmlib 세션에서 경로를 가져올 수 없음. 파일 시스템 탐색 중...")
            det_candidates = find_rtmpose_onnx("det")
            pose_candidates = find_rtmpose_onnx("pose")
            if det_candidates:
                onnx_paths["det"] = det_candidates[0]
            if pose_candidates:
                onnx_paths["pose"] = pose_candidates[0]

    if not onnx_paths:
        print("ERROR: ONNX 모델을 찾을 수 없습니다.")
        print("  --onnx 옵션으로 경로를 직접 지정하거나,")
        print("  먼저 rtmlib으로 모델을 한번 실행하여 자동 다운로드해주세요.")
        sys.exit(1)

    for name, path in onnx_paths.items():
        print(f"  {name}: {path}")

    # ---- 3. 캘리브레이션 실행 ----
    print("\n[3/3] INT8 캘리브레이션 수행")
    success_count = 0
    for model_name, onnx_path in onnx_paths.items():
        cache_file = os.path.join(output_dir, f"{args.model}_{model_name}_int8.cache")
        print(f"\n--- {model_name} 모델 캘리브레이션 ---")

        ok = run_calibration_tensorrt(onnx_path, images, cache_file, args.calib_algo)
        if ok:
            success_count += 1

    # ---- 결과 요약 ----
    print(f"\n{'='*50}")
    print(f"캘리브레이션 완료!")
    print(f"  성공: {success_count}/{len(onnx_paths)}")
    print(f"  캐시 위치: {output_dir}")
    print(f"\n사용법:")
    print(f"  run_benchmark.py에서 --int8 옵션으로 사용 가능:")
    print(f"    python run_benchmark.py --model rtmpose --tensorrt --int8")
    print(f"\n  또는 pose_models.py에서:")
    print(f"    model = RTMPoseModel(mode='balanced', use_tensorrt=True, precision='int8')")


if __name__ == "__main__":
    main()
