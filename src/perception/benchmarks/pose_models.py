"""
Pose Estimation 모델 래퍼
=============================
각 모델의 통일된 인터페이스를 제공합니다.
하체 keypoint 추출에 초점을 맞추고 있습니다.

지원 모델:
  1. MediaPipe Pose (BlazePose) - Google, 경량, 33 keypoints
  2. YOLOv8-Pose - Ultralytics, detection+pose 통합, 17 COCO keypoints
  3. RTMPose (rtmlib) - OpenMMLab, SOTA 속도/정확도, 17 COCO keypoints
  4. RTMPose Wholebody - 133 keypoints (foot 포함!)
  5. ZED Body Tracking - Stereolabs, 38 keypoints (기준선)
  6. MoveNet - Google, TFLite 기반, Lightning(빠름)/Thunder(정확), 17 COCO keypoints

하체 전용 인식 관련 참고:
  - Top-down 방식 (RTMPose, MediaPipe): 사람 검출 → 크롭 → 포즈 추정
    → 상체 안 보이면 사람 검출 실패 가능!
  - Bottom-up/Single-shot 방식 (YOLOv8-Pose): keypoint 직접 검출
    → 부분 인식에 상대적으로 강함
  - 이 벤치마크로 실제로 어떤 모델이 하체만 보일 때 동작하는지 확인
"""

import json
import math
import numpy as np
import time
import os
import cv2
from abc import ABC, abstractmethod


# ============================================================================
# One Euro Filter — 적응형 저역 통과 필터
# ============================================================================
# 참고: Casiez et al., "1€ Filter: A Simple Speed-based Low-pass Filter
#        for Noisy Input in Interactive Systems", CHI 2012
#
# 워커(이동식 보행기) 하지 추적 최적화:
#   - 보행 시 관절 속도가 낮으므로 min_cutoff을 낮게 설정 → 떨림 강하게 제거
#   - beta를 적당히 설정 → 보행 속도 변화에는 지연 없이 반응
#   - d_cutoff은 속도 추정 자체의 노이즈 제거용 (기본값 유지)

class OneEuroFilter:
    """1D One Euro Filter (하나의 스칼라 값용)"""

    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_prev = x0
        self.dx_prev = 0.0
        self.t_prev = t0

    @staticmethod
    def _smoothing_factor(te, cutoff):
        r = 2 * math.pi * cutoff * te
        return r / (r + 1)

    def __call__(self, t, x):
        te = t - self.t_prev
        if te <= 0:
            return x

        # 속도 추정 (미분)의 저역 통과
        a_d = self._smoothing_factor(te, self.d_cutoff)
        dx = (x - self.x_prev) / te
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # 적응형 cutoff: 속도가 빠르면 cutoff을 올려서 지연 최소화
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)

        # 위치의 저역 통과
        a = self._smoothing_factor(te, cutoff)
        x_hat = a * x + (1 - a) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat


class KeypointOneEuroFilter:
    """모든 키포인트에 대한 One Euro Filter 관리자

    각 키포인트의 (x, y) 좌표에 독립적인 1D One Euro Filter 적용.

    보행 분석 최적 파라미터:
        min_cutoff=1.0: 떨림 제거 강도 (낮을수록 강하게 필터링)
        beta=0.007: 지연 감소 (높을수록 빠른 동작에 민감)
        d_cutoff=1.0: 속도 추정 노이즈 필터링

    워커 환경에서 하지만 추적하므로:
    - 팔 휘두르기 같은 급격한 동작 없음
    - 보행 속도는 느림 (0.5~1.5 m/s)
    - min_cutoff을 낮추면 떨림 제거 극대화
    """

    def __init__(self, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._filters = {}  # name → (OneEuroFilter_x, OneEuroFilter_y)

    def filter(self, name, x, y, t):
        """키포인트 좌표를 필터링하여 반환

        Args:
            name: 키포인트 이름 (예: "left_knee")
            x, y: 원시 좌표
            t: 현재 타임스탬프 (초)

        Returns:
            (filtered_x, filtered_y)
        """
        if name not in self._filters:
            self._filters[name] = (
                OneEuroFilter(t, x, self.min_cutoff, self.beta, self.d_cutoff),
                OneEuroFilter(t, y, self.min_cutoff, self.beta, self.d_cutoff),
            )
            return x, y

        fx, fy = self._filters[name]
        return fx(t, x), fy(t, y)

    def reset(self):
        """모든 필터 상태 초기화"""
        self._filters.clear()

    @property
    def has_state(self):
        return len(self._filters) > 0

    def get_last_positions(self):
        """각 키포인트의 마지막 필터 출력값 반환"""
        return {name: (fx.x_prev, fy.x_prev)
                for name, (fx, fy) in self._filters.items()}


# ============================================================================
# Segment Length Constraint — 뼈 길이 제약
# ============================================================================
# 사람의 뼈 길이는 변하지 않으므로, 초기 정지 상태에서 세그먼트 길이를 측정한 뒤
# 이후 프레임에서 해당 길이를 벗어나는 keypoint를 보정합니다.
#
# 워커 환경: 처음에 환자가 서 있는 상태에서 자동 캘리브레이션 후 보행 시작

class SegmentLengthConstraint:
    """초기 정지 프레임에서 세그먼트 길이를 학습하고 이후 프레임에 제약 적용

    세그먼트 정의 (하지):
        left_hip → left_knee (좌 대퇴)
        left_knee → left_ankle (좌 경골)
        right_hip → right_knee (우 대퇴)
        right_knee → right_ankle (우 경골)

    동작 방식:
        1. 캘리브레이션 단계: calib_frames 프레임 동안 세그먼트 길이 수집
        2. 캘리브레이션 완료 후: 중앙값을 기준 길이로 설정
        3. 이후 프레임: 세그먼트 길이가 기준 대비 tolerance를 벗어나면
           자식 키포인트를 기준 길이에 맞게 투영(projection)
    """

    SEGMENTS = [
        ("left_hip", "left_knee"),
        ("left_knee", "left_ankle"),
        ("right_hip", "right_knee"),
        ("right_knee", "right_ankle"),
    ]

    DEFAULT_CALIB_FILE = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "seg_calib.json")

    def __init__(self, calib_frames=30, tolerance=0.20, calib_file=None):
        """
        Args:
            calib_frames: 캘리브레이션에 사용할 프레임 수 (기본 30, ~1초@30fps)
            tolerance: 허용 오차 비율 (0.20 = ±20%)
            calib_file: 캘리브레이션 결과 저장/로드 파일 경로
                        (기본: benchmarks/seg_calib.json)
        """
        self.calib_frames = calib_frames
        self.tolerance = tolerance
        self._calib_file = calib_file or self.DEFAULT_CALIB_FILE

        self._samples = {}       # (parent, child) → [길이 리스트]
        self._ref_lengths = {}   # (parent, child) → 기준 길이 (중앙값)
        self._frame_count = 0
        self._calibrated = False

        # 저장된 캘리브레이션 파일이 있으면 로드
        self._try_load()

    @property
    def calibrated(self):
        return self._calibrated

    @property
    def progress(self):
        """캘리브레이션 진행률 (0.0 ~ 1.0)"""
        if self._calibrated:
            return 1.0
        return min(1.0, self._frame_count / self.calib_frames)

    def _segment_length(self, p1, p2):
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def update(self, keypoints_2d, confidences, min_conf=0.3):
        """캘리브레이션 중이면 샘플 수집, 완료 후이면 제약 적용

        Args:
            keypoints_2d: dict {name → (x, y)} — 수정됨 (in-place)
            confidences: dict {name → float}
            min_conf: 샘플 수집/제약 적용 시 최소 confidence

        Returns:
            True if 제약이 적용됨 (캘리브레이션 완료 상태)
        """
        if not self._calibrated:
            # 다른 모델 인스턴스가 이미 저장했을 수 있으므로 재시도
            if self._frame_count == 0:
                self._try_load()
            if self._calibrated:
                return self._apply_constraint(keypoints_2d, confidences, min_conf)
            return self._collect_sample(keypoints_2d, confidences, min_conf)
        else:
            return self._apply_constraint(keypoints_2d, confidences, min_conf)

    def _collect_sample(self, keypoints_2d, confidences, min_conf):
        """캘리브레이션 샘플 수집"""
        valid_segments = 0
        for parent, child in self.SEGMENTS:
            if (parent in keypoints_2d and child in keypoints_2d and
                    confidences.get(parent, 0) >= min_conf and
                    confidences.get(child, 0) >= min_conf):
                length = self._segment_length(
                    keypoints_2d[parent], keypoints_2d[child])
                if length > 5:  # 최소 5px (너무 짧으면 노이즈)
                    key = (parent, child)
                    if key not in self._samples:
                        self._samples[key] = []
                    self._samples[key].append(length)
                    valid_segments += 1

        # 4개 세그먼트 중 3개 이상 유효해야 프레임 카운트
        if valid_segments >= 3:
            self._frame_count += 1

        # 캘리브레이션 완료 체크
        if self._frame_count >= self.calib_frames:
            self._finalize_calibration()

        return False

    def _finalize_calibration(self):
        """수집된 샘플에서 중앙값으로 기준 길이 설정"""
        for key, lengths in self._samples.items():
            if len(lengths) >= 5:  # 최소 5개 샘플 필요
                sorted_lengths = sorted(lengths)
                mid = len(sorted_lengths) // 2
                self._ref_lengths[key] = sorted_lengths[mid]

        if len(self._ref_lengths) >= 3:  # 4개 중 3개 이상 기준 확보
            self._calibrated = True
            parent_child_strs = [f"  {p}→{c}: {l:.1f}px"
                                 for (p, c), l in self._ref_lengths.items()]
            print(f"[SegmentConstraint] 캘리브레이션 완료 "
                  f"({self._frame_count} 프레임)")
            for s in parent_child_strs:
                print(s)
            self.save()
        else:
            # 샘플 부족 — 프레임 더 수집
            self._frame_count = max(0, self._frame_count - 10)

        # 샘플 메모리 해제
        self._samples.clear()

    def _apply_constraint(self, keypoints_2d, confidences, min_conf):
        """세그먼트 길이 제약 적용 — 자식 키포인트를 보정"""
        applied = False
        for parent, child in self.SEGMENTS:
            key = (parent, child)
            if key not in self._ref_lengths:
                continue
            if (parent not in keypoints_2d or child not in keypoints_2d):
                continue
            if (confidences.get(parent, 0) < min_conf or
                    confidences.get(child, 0) < min_conf):
                continue

            ref_len = self._ref_lengths[key]
            px, py = keypoints_2d[parent]
            cx, cy = keypoints_2d[child]
            cur_len = self._segment_length((px, py), (cx, cy))

            if cur_len < 1e-6:
                continue

            # 허용 범위: ref_len * (1 ± tolerance)
            min_len = ref_len * (1 - self.tolerance)
            max_len = ref_len * (1 + self.tolerance)

            if min_len <= cur_len <= max_len:
                continue  # 범위 내 — 보정 불필요

            # 자식 키포인트를 parent→child 방향으로 기준 길이에 투영
            target_len = max(min_len, min(max_len, cur_len))
            scale = target_len / cur_len
            new_cx = px + (cx - px) * scale
            new_cy = py + (cy - py) * scale
            keypoints_2d[child] = (new_cx, new_cy)
            applied = True

        return applied

    def get_ref_lengths(self):
        """기준 세그먼트 길이 반환 (디버그/표시용)"""
        return {f"{p}→{c}": length
                for (p, c), length in self._ref_lengths.items()}

    def save(self, path=None):
        """캘리브레이션 결과를 JSON 파일로 저장"""
        path = path or self._calib_file
        data = {
            "tolerance": self.tolerance,
            "ref_lengths": {f"{p}|{c}": length
                           for (p, c), length in self._ref_lengths.items()},
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"[SegmentConstraint] 캘리브레이션 저장 → {path}")

    def _try_load(self):
        """저장된 캘리브레이션 파일이 있으면 로드"""
        if not os.path.exists(self._calib_file):
            return
        try:
            with open(self._calib_file, "r") as f:
                data = json.load(f)
            ref = data.get("ref_lengths", {})
            if not ref:
                return
            self._ref_lengths = {
                tuple(k.split("|")): v for k, v in ref.items()
            }
            if len(self._ref_lengths) >= 3:
                self._calibrated = True
                print(f"[SegmentConstraint] 저장된 캘리브레이션 로드 ← {self._calib_file}")
                for (p, c), length in self._ref_lengths.items():
                    print(f"  {p}→{c}: {length:.1f}px")
        except Exception as e:
            print(f"[SegmentConstraint] 캘리브레이션 로드 실패: {e}")

    def reset(self):
        """캘리브레이션 초기화"""
        self._samples.clear()
        self._ref_lengths.clear()
        self._frame_count = 0
        self._calibrated = False


# ============================================================================
# TensorRT 가용성 체크 유틸리티
# ============================================================================
def check_tensorrt_available():
    """TensorRT EP가 사용 가능한지 확인"""
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        has_trt = 'TensorrtExecutionProvider' in providers
        has_cuda = 'CUDAExecutionProvider' in providers
        return has_trt, has_cuda, providers
    except ImportError:
        return False, False, []


def get_trt_providers(cache_dir="./trt_cache", precision="fp16", int8_calib_cache=None):
    """TensorRT + CUDA fallback provider 목록 생성

    Args:
        cache_dir: TRT 엔진 캐시 디렉토리
        precision: "fp16" 또는 "int8"
        int8_calib_cache: INT8 캘리브레이션 캐시 파일 경로 (precision="int8" 시 필요)
    """
    os.makedirs(cache_dir, exist_ok=True)

    trt_options = {
        'device_id': 0,
        'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,
        'trt_fp16_enable': True,
        'trt_engine_cache_enable': True,
        'trt_engine_cache_path': cache_dir,
    }

    if precision == "int8":
        trt_options['trt_int8_enable'] = True
        if int8_calib_cache and os.path.exists(int8_calib_cache):
            trt_options['trt_int8_calibration_table_name'] = int8_calib_cache
            print(f"  [TRT] INT8 캘리브레이션 캐시 사용: {int8_calib_cache}")
        else:
            # 캘리브레이션 캐시 없이 INT8 시도 (TRT가 자체적으로 처리)
            print(f"  [TRT] WARNING: INT8 캘리브레이션 캐시 없음. 정확도가 저하될 수 있습니다.")
            print(f"  [TRT] 정확한 INT8을 위해 calibrate_int8.py로 캐시를 먼저 생성하세요.")

    return [
        ('TensorrtExecutionProvider', trt_options),
        ('CUDAExecutionProvider', {'device_id': 0}),
        'CPUExecutionProvider',
    ]


def find_int8_calib_cache(model_name, model_type="pose"):
    """INT8 캘리브레이션 캐시 파일 자동 탐색"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cache_dir = os.path.join(script_dir, "int8_calib_cache")

    candidates = [
        os.path.join(cache_dir, f"{model_name}_{model_type}_int8.cache"),
        os.path.join(cache_dir, f"{model_name}_int8.cache"),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path
    return None


def setup_trt_env(cache_dir, precision="fp16"):
    """TensorRT 환경변수를 미리 설정 (모델 로드 전에 호출해야 함)"""
    os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"
    os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
    os.environ["ORT_TENSORRT_CACHE_PATH"] = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    if precision == "int8":
        os.environ["ORT_TENSORRT_INT8_ENABLE"] = "1"


def verify_trt_provider(session, model_name=""):
    """ONNX Runtime 세션이 실제로 TRT를 사용하는지 확인"""
    actual = session.get_providers()
    using_trt = 'TensorrtExecutionProvider' in actual
    prefix = f"[{model_name}] " if model_name else ""
    if using_trt:
        print(f"  {prefix}TensorRT EP 활성 확인: {actual}")
    else:
        print(f"  {prefix}WARNING: TensorRT 미사용, 실제 providers: {actual}")
    return using_trt


# ============================================================================
# 하체 Keypoint 정의 (모든 모델에서 통일)
# ============================================================================
LOWER_LIMB_KEYPOINTS = [
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
    "left_heel", "left_toe",
    "right_heel", "right_toe",
]


class PoseResult:
    """통일된 Pose 결과"""

    def __init__(self):
        self.keypoints_2d = {}   # name → (x, y) pixel
        self.confidences = {}    # name → float (0~1)
        self.keypoints_3d = {}   # name → (x, y, z) meters (depth 있을 때)
        self.inference_time_ms = 0.0
        self.detected = False
        # E2E latency 분해
        self.grab_time_ms = 0.0
        self.postprocess_time_ms = 0.0
        self.e2e_latency_ms = 0.0
        # 상세 타이밍 분해 (디버깅용)
        self.timing_detail = {
            "yolo_async_ms": 0.0,        # YOLO model() 비동기 리턴까지 (GPU 미완료 가능)
            "gpu_sync_ms": 0.0,          # torch.cuda.synchronize() 추가 대기
            "yolo_forward_ms": 0.0,      # 실제 YOLO 전체 (async + sync, GPU 완료 보장)
            "bbox_extract_ms": 0.0,      # bbox/keypoint CPU 추출
            "filter_ms": 0.0,            # One Euro Filter 스무딩
            "constraint_ms": 0.0,        # 세그먼트 길이 제약
            "angle_ms": 0.0,             # 관절 각도 계산
            "total_predict_ms": 0.0,     # predict() 전체
        }
        # 관절 각도
        self.joint_angles = {}   # name → angle_degrees

    def get_lower_limb_confidence(self):
        """하체 주요 keypoint 평균 confidence"""
        keys = ["left_hip", "left_knee", "left_ankle",
                "right_hip", "right_knee", "right_ankle"]
        confs = [self.confidences.get(k, 0.0) for k in keys]
        return float(np.mean(confs)) if confs else 0.0

    def has_lower_limb(self, min_conf=0.3):
        """하체 keypoint가 충분히 인식되었는지"""
        required = ["left_hip", "left_knee", "left_ankle",
                     "right_hip", "right_knee", "right_ankle"]
        count = sum(1 for k in required if self.confidences.get(k, 0) >= min_conf)
        return count >= 4  # 6개 중 4개 이상


class PoseModel(ABC):
    """Pose Estimation 모델 베이스 클래스"""

    def __init__(self, name: str):
        self.name = name
        self.is_loaded = False

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def predict(self, rgb_image: np.ndarray) -> PoseResult:
        pass

    def predict_with_timing(self, rgb_image: np.ndarray) -> PoseResult:
        t0 = time.perf_counter()
        result = self.predict(rgb_image)
        t1 = time.perf_counter()
        result.inference_time_ms = (t1 - t0) * 1000
        return result


# ============================================================================
# 1. MediaPipe Pose (BlazePose)
# ============================================================================
class MediaPipePose(PoseModel):
    """
    Google MediaPipe Pose
    - 33 keypoints (상체+하체+얼굴 일부)
    - Heel, Toe (foot_index) 포함!
    - 경량, CPU에서도 동작
    - 단점: 전신이 어느 정도 보여야 detection 성공
    """

    KEYPOINT_MAP = {
        23: "left_hip", 24: "right_hip",
        25: "left_knee", 26: "right_knee",
        27: "left_ankle", 28: "right_ankle",
        29: "left_heel", 30: "right_heel",
        31: "left_toe", 32: "right_toe",
    }

    def __init__(self, model_complexity=1):
        name = f"MediaPipe (complexity={model_complexity})"
        super().__init__(name)
        self.model_complexity = model_complexity

    def load(self):
        try:
            import mediapipe as mp
        except ImportError:
            raise ImportError(
                "mediapipe 미설치.\n"
                "  pip install mediapipe\n"
                "  Jetson aarch64: https://github.com/google/mediapipe 참고"
            )

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=self.model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.is_loaded = True
        print(f"  [{self.name}] 로드 완료")

    def predict(self, rgb_image: np.ndarray) -> PoseResult:
        result = PoseResult()
        h, w = rgb_image.shape[:2]

        rgb = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        mp_result = self.pose.process(rgb)

        if mp_result.pose_landmarks:
            result.detected = True
            for mp_idx, our_name in self.KEYPOINT_MAP.items():
                lm = mp_result.pose_landmarks.landmark[mp_idx]
                result.keypoints_2d[our_name] = (lm.x * w, lm.y * h)
                result.confidences[our_name] = lm.visibility
        return result


# ============================================================================
# 2. YOLOv8-Pose
# ============================================================================
class YOLOv8Pose(PoseModel):
    """
    Ultralytics YOLO Pose (YOLOv8 / YOLO11)
    - Detection + Pose를 한번에 (single-shot)
    - 17 COCO keypoints (heel/toe 없음)
    - 부분 인식에 상대적으로 강함 (YOLO가 partial object 감지 가능)
    - TensorRT 변환 지원
    - 추론 파라미터 세밀 제어 지원
    - yolo_version="v8" (기본) 또는 "v11" 선택 가능
    """

    KEYPOINT_MAP = {
        11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee",
        15: "left_ankle", 16: "right_ankle",
    }

    # 관절 각도 계산용 트리플릿: (부모, 관절, 자식) → 관절에서의 각도
    JOINT_ANGLE_TRIPLETS = {
        "left_knee_angle":  ("left_hip", "left_knee", "left_ankle"),
        "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    }

    def __init__(self, model_size="n", use_tensorrt=False,
                 precision="fp16", use_calib=True, calib_data=None,
                 conf=0.25, iou=0.7, imgsz=640, max_det=1,
                 agnostic_nms=True, half=True,
                 smoothing=0.0, classes=None, yolo_version="v8",
                 filter_min_cutoff=1.0, filter_beta=0.007,
                 segment_constraint=True, seg_calib_frames=30, seg_tolerance=0.15,
                 seg_calib_file=None):
        """
        Args:
            model_size: "n" (nano) or "s" (small)
            use_tensorrt: TensorRT 엔진 사용 여부
            precision: "fp16" 또는 "int8" (use_tensorrt=True일 때만 의미)
            use_calib: INT8 시 캘리브레이션 데이터 사용 여부
            calib_data: 캘리브레이션 데이터 경로
            --- 추론 파라미터 ---
            conf: confidence threshold (0~1, 낮을수록 민감, 기본 0.25)
            iou: NMS IoU threshold (0~1, 높을수록 중복 허용, 기본 0.7)
            imgsz: 입력 해상도 (640/480/320, 기본 640)
            max_det: 최대 검출 인원 (기본 1=가장 가까운 1명만)
            agnostic_nms: class-agnostic NMS (단일 클래스이므로 True 권장)
            half: FP16 추론 사용 (TRT일 때 이미 적용됨, PyTorch에서도 사용)
            smoothing: One Euro Filter ON/OFF (0=OFF, >0=ON)
            classes: 검출할 클래스 (None=전체, [0]=person만)
            yolo_version: "v8" (YOLOv8) 또는 "v11" (YOLO11)
            --- One Euro Filter 파라미터 (smoothing > 0일 때 사용) ---
            filter_min_cutoff: 떨림 제거 강도 (낮을수록 강하게, 기본 1.0)
            filter_beta: 지연 감소 (높을수록 빠른 동작 추종, 기본 0.007)
            --- 세그먼트 길이 제약 ---
            segment_constraint: 뼈 길이 제약 ON/OFF (기본 True)
            seg_calib_frames: 캘리브레이션 프레임 수 (기본 30, ~1초@30fps)
            seg_tolerance: 허용 오차 비율 (0.15 = ±15%)
            seg_calib_file: 세그먼트 캘리브레이션 저장/로드 파일 경로
        """
        # YOLO 버전별 모델 이름 프리픽스
        self.yolo_version = yolo_version
        if yolo_version == "v26":
            self._model_prefix = "yolo26"
            version_label = "YOLO26"
        elif yolo_version == "v11":
            self._model_prefix = "yolo11"
            version_label = "YOLO11"
        else:
            self._model_prefix = "yolov8"
            version_label = "YOLOv8"

        name = f"{version_label}{model_size}-Pose"
        if use_tensorrt:
            prec_label = precision.upper()
            if precision == "int8":
                calib_label = "cal" if use_calib else "nocal"
                name += f" (TRT-{prec_label}-{calib_label})"
            else:
                name += f" (TRT-{prec_label})"
        super().__init__(name)
        self.model_size = model_size
        self.use_tensorrt = use_tensorrt
        self.precision = precision
        self.use_calib = use_calib
        # calib_data를 즉시 절대경로로 변환 (나중에 cwd가 바뀔 수 있으므로)
        self.calib_data = os.path.abspath(calib_data) if calib_data else calib_data
        self.model = None

        # 추론 파라미터
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.max_det = max_det
        self.agnostic_nms = agnostic_nms
        self.half = half
        self.classes = classes if classes is not None else [0]  # person만

        # One Euro Filter 스무딩
        self.smoothing = smoothing
        self.filter_min_cutoff = filter_min_cutoff
        self.filter_beta = filter_beta
        self._kp_filter = KeypointOneEuroFilter(
            min_cutoff=filter_min_cutoff, beta=filter_beta
        ) if smoothing > 0 else None
        self._prev_confidences = {}  # name → float (confidence EMA용)
        self._lost_frames = 0  # 연속 미검출 프레임 수

        # 세그먼트 길이 제약
        self._seg_constraint = SegmentLengthConstraint(
            calib_frames=seg_calib_frames, tolerance=seg_tolerance,
            calib_file=seg_calib_file,
        ) if segment_constraint else None

    def _export_int8_safe(self, model_name, engine_name):
        """INT8 엔진 빌드를 subprocess로 격리 (TRT crash 방지)

        TRT 10.3.0에서 INT8 빌드 시 'solve error'로 프로세스가
        크래시할 수 있으므로 별도 프로세스에서 실행합니다.
        """
        import subprocess
        import sys

        if self.use_calib:
            calib_source = self.calib_data or 'coco8-pose.yaml'
            # self.calib_data는 생성자에서 이미 절대경로로 변환됨
            # 디렉토리면 Ultralytics YAML 래퍼 생성
            if os.path.isdir(calib_source):
                yaml_path = os.path.join(calib_source, "calib_dataset.yaml")
                with open(yaml_path, "w") as f:
                    f.write(f"path: {os.path.abspath(calib_source)}\n")
                    f.write("train: .\nval: .\n")
                    f.write("names:\n  0: person\n")
                    f.write("nc: 1\n")
                    f.write("kpt_shape: [17, 3]\n")
                calib_source = yaml_path
            data_arg = f", data={calib_source!r}"
            print(f"  [{self.name}] INT8 캘리브레이션 데이터 사용: {calib_source}")
            # YAML 검증: kpt_shape 필수 필드 확인
            if os.path.isfile(calib_source):
                with open(calib_source, "r") as _yf:
                    _yaml_content = _yf.read()
                if "kpt_shape" not in _yaml_content:
                    print(f"  [{self.name}] WARNING: calib YAML에 kpt_shape 없음 → 자동 추가")
                    with open(calib_source, "a") as _yf:
                        _yf.write("kpt_shape: [17, 3]\n")
                if "nc:" not in _yaml_content:
                    with open(calib_source, "a") as _yf:
                        _yf.write("nc: 1\n")
        else:
            data_arg = ""
            print(f"  [{self.name}] INT8 캘리브레이션 없이 변환")

        script = f"""
import os, sys
os.environ['MPLBACKEND'] = 'Agg'
os.chdir({os.getcwd()!r})
from ultralytics import YOLO
model = YOLO({model_name!r})
path = model.export(format='engine', half=False, int8=True, imgsz={self.imgsz}{data_arg})
# 엔진 파일을 정밀도별 이름으로 저장
default_engine = '{self._model_prefix}{self.model_size}-pose.engine'
if os.path.exists(default_engine) and {engine_name!r} != default_engine:
    os.rename(default_engine, {engine_name!r})
    path = {engine_name!r}
print('__ENGINE_PATH__=' + str(path))
"""
        print(f"  [{self.name}] subprocess로 INT8 엔진 빌드 시작...")
        try:
            # 시스템 matplotlib/numpy 충돌 방지를 위한 환경 변수 설정
            env = os.environ.copy()
            env["MPLBACKEND"] = "Agg"
            # matplotlib 충돌 방지: 시스템 경로는 유지 (Pillow 등 필요)
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=3600,
                encoding="utf-8", errors="replace",
                env=env,
            )
            # stdout에서 엔진 경로 추출
            for line in result.stdout.splitlines():
                if line.startswith("__ENGINE_PATH__="):
                    engine_path = line.split("=", 1)[1].strip()
                    if os.path.exists(engine_path):
                        print(f"  [{self.name}] INT8 엔진 빌드 성공: {engine_path}")
                        return engine_path

            # 실패 시 로그 출력 + 파일 기록
            print(f"  [{self.name}] INT8 엔진 빌드 실패 (returncode={result.returncode})")
            err_msg = ""
            if result.stderr:
                err_lines = result.stderr.strip().splitlines()[-10:]
                for line in err_lines:
                    print(f"    {line}")
                err_msg = result.stderr
            self._log_error("INT8 엔진 빌드 실패", err_msg,
                            stdout=result.stdout, calib_source=calib_source)
            print(f"  [{self.name}] 상세 에러 → benchmarks/errors.log")
            return None
        except subprocess.TimeoutExpired:
            self._log_error("INT8 엔진 빌드 타임아웃", "10분 초과",
                            calib_source=calib_source)
            print(f"  [{self.name}] INT8 엔진 빌드 타임아웃 (10분 초과)")
            return None
        except Exception as e:
            self._log_error("INT8 엔진 빌드 subprocess 에러", str(e),
                            calib_source=calib_source)
            print(f"  [{self.name}] INT8 엔진 빌드 subprocess 에러: {e}")
            return None

    def _log_error(self, context, detail, stdout=None, calib_source=None):
        """에러를 errors.log 파일에 상세 기록 (디버깅용)"""
        from datetime import datetime
        log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "errors.log")
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                f.write(f"{'=' * 70}\n")
                f.write(f"[{ts}] {self.name} - {context}\n")
                f.write(f"{'=' * 70}\n")

                # 모델 설정
                f.write(f"  model_prefix: {self._model_prefix}\n")
                f.write(f"  model_size: {self.model_size}\n")
                f.write(f"  yolo_version: {self.yolo_version}\n")
                f.write(f"  precision: {self.precision}\n")
                f.write(f"  use_tensorrt: {self.use_tensorrt}\n")
                f.write(f"  use_calib: {self.use_calib}\n")
                f.write(f"  calib_data: {self.calib_data}\n")
                f.write(f"  imgsz: {self.imgsz}\n")

                # 캘리브레이션 YAML 내용 (있으면)
                if calib_source and os.path.isfile(calib_source):
                    f.write(f"\n  --- calib YAML: {calib_source} ---\n")
                    try:
                        with open(calib_source, "r") as yf:
                            for line in yf:
                                f.write(f"    {line}")
                        f.write("\n")
                    except Exception:
                        f.write("    (읽기 실패)\n")

                # stdout
                if stdout:
                    f.write(f"\n  --- stdout (last 30 lines) ---\n")
                    for line in stdout.strip().splitlines()[-30:]:
                        f.write(f"    {line}\n")

                # stderr / detail
                if detail:
                    f.write(f"\n  --- stderr / detail ---\n")
                    for line in str(detail).strip().splitlines()[-50:]:
                        f.write(f"    {line}\n")

                f.write(f"\n\n")
        except Exception:
            pass

    def load(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics 미설치. pip install ultralytics")

        model_name = f"{self._model_prefix}{self.model_size}-pose.pt"
        print(f"  [{self.name}] 모델 로드 중: {model_name}")
        self.model = YOLO(model_name)

        if self.use_tensorrt:
            # TensorRT 사전 체크
            try:
                import tensorrt
                print(f"  [{self.name}] TensorRT {tensorrt.__version__} 감지됨")
            except ImportError:
                print(f"  [{self.name}] WARNING: tensorrt 패키지 미설치, 엔진 변환 실패할 수 있음")

            # 정밀도·입력크기별 엔진 파일명 구분
            sz = self.imgsz
            if self.precision == "int8":
                calib_suffix = "cal" if self.use_calib else "nocal"
                engine_name = f"{self._model_prefix}{self.model_size}-pose-{sz}-int8-{calib_suffix}.engine"
            else:
                engine_name = f"{self._model_prefix}{self.model_size}-pose-{sz}.engine"
            engine_path = os.path.join(os.path.dirname(model_name) or ".", engine_name)

            try:
                if os.path.exists(engine_path):
                    print(f"  [{self.name}] 기존 TRT 엔진 재사용: {engine_path}")
                    self.model = YOLO(engine_path)
                else:
                    print(f"  [{self.name}] TensorRT 엔진 변환 중 (최초 1회, 수분 소요)...")
                    if self.precision == "int8":
                        exported_path = self._export_int8_safe(model_name, engine_name)
                        if exported_path is None:
                            raise RuntimeError("INT8 엔진 빌드 실패 (subprocess)")
                    else:
                        exported_path = self.model.export(
                            format="engine", half=True, int8=False, imgsz=self.imgsz)

                        # 엔진 파일을 정밀도별 이름으로 저장
                        default_engine = f"{self._model_prefix}{self.model_size}-pose.engine"
                        if os.path.exists(default_engine) and engine_name != default_engine:
                            os.rename(default_engine, engine_name)
                            exported_path = engine_name

                    self.model = YOLO(exported_path)
                print(f"  [{self.name}] TensorRT 엔진 로드 성공")
            except Exception as e:
                print(f"  [{self.name}] TensorRT 변환/로드 실패: {e}")
                print(f"  [{self.name}] PyTorch 모드로 fallback")
                self.model = YOLO(model_name)
                self.use_tensorrt = False

        # 워밍업 (첫 추론은 느림)
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model(dummy, verbose=False, imgsz=self.imgsz)

        self.is_loaded = True
        if self.use_tensorrt:
            mode_str = f"TensorRT {self.precision.upper()}"
        else:
            mode_str = "PyTorch"
        print(f"  [{self.name}] 로드 완료 ({mode_str})")

    @staticmethod
    def _calc_angle(p1, p2, p3):
        """세 점 사이의 각도 계산 (p2가 관절점, 도 단위)"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))

    def predict(self, rgb_image: np.ndarray) -> PoseResult:
        result = PoseResult()
        t_predict_start = time.perf_counter()

        # ── Step 1: YOLO 추론 (전처리 + GPU 추론 + NMS) ──
        t0 = time.perf_counter()
        results = self.model(
            rgb_image,
            verbose=False,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            agnostic_nms=self.agnostic_nms,
            half=self.half,
            classes=self.classes,
        )
        t_async = time.perf_counter()
        result.timing_detail["yolo_async_ms"] = (t_async - t0) * 1000  # GPU 비동기 리턴까지

        # ── Step 1b: GPU 동기화 (실제 GPU 완료 대기) ──
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except ImportError:
            pass
        t_synced = time.perf_counter()
        result.timing_detail["gpu_sync_ms"] = (t_synced - t_async) * 1000   # sync 대기 시간
        result.timing_detail["yolo_forward_ms"] = (t_synced - t0) * 1000    # 실제 GPU 포함 전체

        if not results or len(results) == 0:
            self._lost_frames += 1
            result.timing_detail["total_predict_ms"] = (time.perf_counter() - t_predict_start) * 1000
            return self._apply_smoothing_fallback(result)

        r = results[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            self._lost_frames += 1
            result.timing_detail["total_predict_ms"] = (time.perf_counter() - t_predict_start) * 1000
            return self._apply_smoothing_fallback(result)

        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            self._lost_frames += 1
            result.timing_detail["total_predict_ms"] = (time.perf_counter() - t_predict_start) * 1000
            return self._apply_smoothing_fallback(result)

        # ── Step 2: bbox/keypoint CPU 추출 ──
        t2 = time.perf_counter()
        xyxy = boxes.xyxy.cpu().numpy()
        areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
        best_idx = int(areas.argmax())

        kps = r.keypoints[best_idx]
        xy = kps.xy[0].cpu().numpy()   # (17, 2)
        conf = kps.conf
        if conf is not None:
            conf = conf[0].cpu().numpy()  # (17,)
        else:
            conf = np.ones(17, dtype=np.float32)
        t3 = time.perf_counter()
        result.timing_detail["bbox_extract_ms"] = (t3 - t2) * 1000

        result.detected = True
        self._lost_frames = 0

        # ── Step 3: One Euro Filter 스무딩 ──
        t4 = time.perf_counter()
        for coco_idx, our_name in self.KEYPOINT_MAP.items():
            if coco_idx < len(xy):
                x, y = float(xy[coco_idx, 0]), float(xy[coco_idx, 1])
                c = float(conf[coco_idx])
                # (0,0)은 미감지 keypoint
                if x > 0 or y > 0:
                    # One Euro Filter 스무딩 적용
                    if self._kp_filter is not None:
                        t_now = time.perf_counter()
                        x, y = self._kp_filter.filter(our_name, x, y, t_now)
                        # confidence는 단순 EMA (안정성)
                        prev_c = self._prev_confidences.get(our_name, c)
                        c = 0.3 * prev_c + 0.7 * c

                    result.keypoints_2d[our_name] = (x, y)
                    result.confidences[our_name] = c
                    self._prev_confidences[our_name] = c
                else:
                    result.confidences[our_name] = 0.0
        t5 = time.perf_counter()
        result.timing_detail["filter_ms"] = (t5 - t4) * 1000

        # ── Step 4: 세그먼트 길이 제약 ──
        t6 = time.perf_counter()
        if self._seg_constraint is not None:
            self._seg_constraint.update(
                result.keypoints_2d, result.confidences)
        t7 = time.perf_counter()
        result.timing_detail["constraint_ms"] = (t7 - t6) * 1000

        # ── Step 5: 관절 각도 계산 ──
        t8 = time.perf_counter()
        for angle_name, (p1_name, p2_name, p3_name) in self.JOINT_ANGLE_TRIPLETS.items():
            if (p1_name in result.keypoints_2d and
                p2_name in result.keypoints_2d and
                p3_name in result.keypoints_2d and
                result.confidences.get(p1_name, 0) > 0.3 and
                result.confidences.get(p2_name, 0) > 0.3 and
                result.confidences.get(p3_name, 0) > 0.3):
                angle = self._calc_angle(
                    result.keypoints_2d[p1_name],
                    result.keypoints_2d[p2_name],
                    result.keypoints_2d[p3_name],
                )
                result.joint_angles[angle_name] = angle
        t9 = time.perf_counter()
        result.timing_detail["angle_ms"] = (t9 - t8) * 1000

        result.timing_detail["total_predict_ms"] = (time.perf_counter() - t_predict_start) * 1000
        return result

    def _apply_smoothing_fallback(self, result):
        """검출 실패 시 이전 프레임 keypoint를 감쇠하여 반환 (떨림 방지)"""
        if self._kp_filter is None or not self._kp_filter.has_state:
            return result
        # 최대 15프레임까지 fallback (약 0.5초 @ 30fps)
        if self._lost_frames > 15:
            self._kp_filter.reset()
            self._prev_confidences.clear()
            return result

        # 이전 값을 감쇠하여 반환 (처음 5프레임은 천천히, 이후 빠르게 감쇠)
        if self._lost_frames <= 5:
            decay = max(0, 1.0 - self._lost_frames * 0.08)
        else:
            decay = max(0, 0.6 - (self._lost_frames - 5) * 0.06)
        result.detected = True
        for name, (x, y) in self._kp_filter.get_last_positions().items():
            c = self._prev_confidences.get(name, 0) * decay
            if c > 0.1:
                result.keypoints_2d[name] = (x, y)
                result.confidences[name] = c
        return result


# ============================================================================
# 3. RTMPose (via rtmlib)
# ============================================================================
class RTMPoseModel(PoseModel):
    """
    RTMPose - OpenMMLab Real-Time Pose Estimation
    - rtmlib 사용 (공식 경량 배포 라이브러리)
    - RTMDet (사람 검출) + RTMPose (포즈 추정) 2-stage
    - 17 COCO keypoints
    - ONNX Runtime / OpenCV DNN 백엔드 지원
    - TensorRT 지원 (ONNX Runtime Provider)

    설치: pip install rtmlib onnxruntime-gpu
    """

    KEYPOINT_MAP = {
        11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee",
        15: "left_ankle", 16: "right_ankle",
    }

    def __init__(self, mode="balanced",
                 backend="onnxruntime", device="cuda",
                 use_tensorrt=False, precision="fp16",
                 use_calib=True,
                 smoothing=0.0, filter_min_cutoff=1.0, filter_beta=0.007,
                 segment_constraint=True, seg_calib_frames=30, seg_tolerance=0.15,
                 seg_calib_file=None):
        """
        Args:
            mode: RTMPose 모드 ("lightweight", "balanced", "performance")
            backend: 추론 백엔드
            device: 디바이스 ("cuda" or "cpu")
            use_tensorrt: TensorRT EP 사용 여부
            precision: TRT 정밀도 - "fp16" 또는 "int8"
            use_calib: INT8 시 캘리브레이션 캐시 사용 여부 (False면 캐시 없이 INT8)
            --- One Euro Filter ---
            smoothing: 스무딩 ON/OFF (0=OFF, >0=ON)
            filter_min_cutoff: 떨림 제거 강도 (낮을수록 강하게, 기본 1.0)
            filter_beta: 지연 감소 (높을수록 빠른 동작 추종, 기본 0.007)
            --- 세그먼트 길이 제약 ---
            segment_constraint: 뼈 길이 제약 ON/OFF (기본 True)
            seg_calib_frames: 캘리브레이션 프레임 수 (기본 30)
            seg_tolerance: 허용 오차 비율 (0.15 = ±15%)
            seg_calib_file: 세그먼트 캘리브레이션 저장/로드 파일 경로
        """
        name = f"RTMPose ({mode})"
        if use_tensorrt:
            prec_label = precision.upper()
            if precision == "int8":
                calib_label = "cal" if use_calib else "nocal"
                name += f" (TRT-{prec_label}-{calib_label})"
            else:
                name += f" (TRT-{prec_label})"
        super().__init__(name)
        self.mode = mode
        self.backend = backend
        self.device = device
        self.use_tensorrt = use_tensorrt
        self.precision = precision
        self.use_calib = use_calib
        self.body = None

        # One Euro Filter 스무딩
        self.smoothing = smoothing
        self._kp_filter = KeypointOneEuroFilter(
            min_cutoff=filter_min_cutoff, beta=filter_beta
        ) if smoothing > 0 else None
        self._prev_confidences = {}

        # 세그먼트 길이 제약
        self._seg_constraint = SegmentLengthConstraint(
            calib_frames=seg_calib_frames, tolerance=seg_tolerance,
            calib_file=seg_calib_file,
        ) if segment_constraint else None

    def _auto_calibrate(self):
        """캘리브레이션 캐시가 없을 때 합성 데이터로 자동 생성"""
        try:
            from calibrate_int8 import generate_synthetic, run_calibration_tensorrt, find_rtmpose_onnx

            images = generate_synthetic(num_images=100)
            script_dir = os.path.dirname(os.path.abspath(__file__))
            cache_dir = os.path.join(script_dir, "int8_calib_cache")
            os.makedirs(cache_dir, exist_ok=True)

            # ONNX 모델 경로 탐색
            det_candidates = find_rtmpose_onnx("det")
            pose_candidates = find_rtmpose_onnx("pose")

            det_cache = None
            pose_cache = None

            if det_candidates:
                det_cache_path = os.path.join(cache_dir, "rtmpose_det_int8.cache")
                if run_calibration_tensorrt(det_candidates[0], images, det_cache_path):
                    det_cache = det_cache_path

            if pose_candidates:
                pose_cache_path = os.path.join(cache_dir, "rtmpose_pose_int8.cache")
                if run_calibration_tensorrt(pose_candidates[0], images, pose_cache_path):
                    pose_cache = pose_cache_path

            if det_cache or pose_cache:
                print(f"  [{self.name}] 캘리브레이션 캐시 자동 생성 완료")
            else:
                print(f"  [{self.name}] 캘리브레이션 캐시 자동 생성 실패 → 캐시 없이 진행")
            return det_cache, pose_cache
        except Exception as e:
            print(f"  [{self.name}] 자동 캘리브레이션 실패: {e}")
            return None, None

    def load(self):
        try:
            from rtmlib import Body
        except ImportError:
            raise ImportError(
                "rtmlib 미설치.\n"
                "  pip install rtmlib\n"
                "  + pip install onnxruntime-gpu  (Jetson: Jetson Zoo에서 설치)"
            )

        backend = self.backend
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trt_cache")

        if self.use_tensorrt:
            # TRT 가용성 사전 체크
            has_trt, has_cuda, providers = check_tensorrt_available()
            if not has_trt:
                print(f"  [{self.name}] WARNING: TensorrtExecutionProvider 미감지")
                print(f"  [{self.name}] 사용 가능: {providers}")
                if has_cuda:
                    print(f"  [{self.name}] CUDA EP로 fallback")
                else:
                    print(f"  [{self.name}] ╔═══════════════════════════════════════════════════════╗")
                    print(f"  [{self.name}] ║  GPU 없음! CPU fallback → 성능 10배 이상 저하됨       ║")
                    print(f"  [{self.name}] ║                                                       ║")
                    print(f"  [{self.name}] ║  원인: onnxruntime이 CPU 전용 빌드입니다.              ║")
                    print(f"  [{self.name}] ║  해결: Jetson용 onnxruntime-gpu를 설치하세요:          ║")
                    print(f"  [{self.name}] ║    pip uninstall onnxruntime onnxruntime-gpu -y        ║")
                    print(f"  [{self.name}] ║    pip install onnxruntime-gpu \\                      ║")
                    print(f"  [{self.name}] ║      --index-url https://pypi.jetson-ai-lab.io/jp6/cu126 ║")
                    print(f"  [{self.name}] ║      --no-cache-dir                                   ║")
                    print(f"  [{self.name}] ╚═══════════════════════════════════════════════════════╝")

            backend = "onnxruntime"
            # 환경변수를 모델 로드 전에 설정 (중요!)
            setup_trt_env(cache_dir, precision=self.precision)

            prec_str = self.precision.upper()
            print(f"  [{self.name}] TensorRT EP 활성화 ({prec_str}, 최초 빌드 시 수분 소요)")
            if self.precision == "int8":
                print(f"  [{self.name}] INT8 모드: 캘리브레이션 캐시를 탐색합니다...")

        print(f"  [{self.name}] 로드 중... (최초 실행 시 모델 자동 다운로드)")
        self.body = Body(
            mode=self.mode,
            to_openpose=False,
            backend=backend,
            device=self.device,
        )

        # TensorRT EP 설정 (rtmlib 내부 ONNX session에 적용)
        if self.use_tensorrt:
            self._apply_tensorrt_provider(cache_dir)

        # 워밍업 (TensorRT 첫 실행 시 엔진 빌드됨)
        print(f"  [{self.name}] 워밍업 중...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            self.body(dummy)

        self.is_loaded = True
        print(f"  [{self.name}] 로드 완료 (backend={backend}, device={self.device}, "
              f"TRT={self.use_tensorrt}, precision={self.precision})")

    def _apply_tensorrt_provider(self, cache_dir="./trt_cache"):
        """rtmlib 내부 ONNX session에 TensorRT provider 적용"""
        try:
            import onnxruntime as ort

            # INT8일 경우 캘리브레이션 캐시 탐색 (use_calib=True일 때만)
            det_calib = None
            pose_calib = None
            if self.precision == "int8" and self.use_calib:
                det_calib = find_int8_calib_cache("rtmpose", "det")
                pose_calib = find_int8_calib_cache("rtmpose", "pose")
                if not det_calib and not pose_calib:
                    print(f"  [{self.name}] 캘리브레이션 캐시 없음 → 합성 데이터로 자동 생성 시도")
                    det_calib, pose_calib = self._auto_calibrate()
            elif self.precision == "int8" and not self.use_calib:
                print(f"  [{self.name}] 캘리브레이션 없이 INT8 모드 (비교 실험용)")

            # rtmlib Body 내부의 det_model과 pose_model 세션 교체
            replaced = 0
            if hasattr(self.body, 'det_model') and hasattr(self.body.det_model, 'session'):
                det_model_path = getattr(self.body.det_model, 'model_path', None)
                if det_model_path and os.path.exists(det_model_path):
                    trt_providers = get_trt_providers(
                        cache_dir, precision=self.precision,
                        int8_calib_cache=det_calib)
                    self.body.det_model.session = ort.InferenceSession(
                        det_model_path, providers=trt_providers)
                    verify_trt_provider(self.body.det_model.session, f"{self.name}/det")
                    replaced += 1
            if hasattr(self.body, 'pose_model') and hasattr(self.body.pose_model, 'session'):
                pose_model_path = getattr(self.body.pose_model, 'model_path', None)
                if pose_model_path and os.path.exists(pose_model_path):
                    trt_providers = get_trt_providers(
                        cache_dir, precision=self.precision,
                        int8_calib_cache=pose_calib)
                    self.body.pose_model.session = ort.InferenceSession(
                        pose_model_path, providers=trt_providers)
                    verify_trt_provider(self.body.pose_model.session, f"{self.name}/pose")
                    replaced += 1

            if replaced > 0:
                print(f"  [{self.name}] TensorRT provider 적용 완료 "
                      f"({replaced}개 세션, precision={self.precision})")
            else:
                print(f"  [{self.name}] WARNING: 교체 가능한 ONNX 세션을 찾지 못함")
        except Exception as e:
            print(f"  [{self.name}] TensorRT provider 적용 실패, CUDA fallback: {e}")

    def predict(self, rgb_image: np.ndarray) -> PoseResult:
        result = PoseResult()

        keypoints, scores = self.body(rgb_image)
        # keypoints: (N, 17, 2) - N명의 사람, 17 COCO keypoints
        # scores: (N, 17)

        if keypoints is None or len(keypoints) == 0:
            return result

        # 가장 큰 사람 선택 (keypoint 범위로 판단)
        if len(keypoints) > 1:
            areas = []
            for kp in keypoints:
                valid = kp[kp.sum(axis=1) > 0]
                if len(valid) > 0:
                    x_range = valid[:, 0].max() - valid[:, 0].min()
                    y_range = valid[:, 1].max() - valid[:, 1].min()
                    areas.append(x_range * y_range)
                else:
                    areas.append(0)
            best_idx = int(np.argmax(areas))
        else:
            best_idx = 0

        kps = keypoints[best_idx]   # (17, 2)
        scrs = scores[best_idx]     # (17,)

        result.detected = True
        for coco_idx, our_name in self.KEYPOINT_MAP.items():
            if coco_idx < len(kps):
                x, y = float(kps[coco_idx, 0]), float(kps[coco_idx, 1])
                c = float(scrs[coco_idx])
                if x > 0 or y > 0:
                    # One Euro Filter 스무딩 적용
                    if self._kp_filter is not None:
                        t_now = time.perf_counter()
                        x, y = self._kp_filter.filter(our_name, x, y, t_now)
                        prev_c = self._prev_confidences.get(our_name, c)
                        c = 0.3 * prev_c + 0.7 * c
                    result.keypoints_2d[our_name] = (x, y)
                    result.confidences[our_name] = c
                    self._prev_confidences[our_name] = c
                else:
                    result.confidences[our_name] = 0.0

        # 세그먼트 길이 제약 적용
        if self._seg_constraint is not None:
            self._seg_constraint.update(
                result.keypoints_2d, result.confidences)

        return result


# ============================================================================
# 4. RTMPose Wholebody (133 keypoints - FOOT 포함!)
# ============================================================================
class RTMPoseWholebody(PoseModel):
    """
    RTMPose Wholebody - 133 keypoints
    - Body(17) + Foot(6x2=12) + Face(68) + Hand(21x2=42)
    - Foot keypoints 포함! → heel, toe 직접 추출 가능
    - rtmlib의 Wholebody 클래스 사용

    Foot keypoint indices (COCO-Wholebody):
      17~22: left foot (ankle 근처 6개 포인트)
      23~28: right foot (위와 동일, 일부 모델에서는 다름)

    참고: Wholebody 모델은 Body-only보다 무겁지만
          Walker에 필수적인 foot keypoint를 제공
    """

    # Wholebody에서 foot keypoints의 인덱스는 모델마다 다를 수 있음
    # COCO-Wholebody 기준: body(0-16), foot(17-22 left, 23-28 right) 형태
    # rtmlib은 133 keypoints: body(17) + foot(6) + face(68) + hand_l(21) + hand_r(21)
    # foot 인덱스: 17~22 (left_big_toe, left_small_toe, left_heel, ...)

    # Body keypoints (COCO 17)
    BODY_MAP = {
        11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee",
        15: "left_ankle", 16: "right_ankle",
    }

    # Foot keypoints (COCO-Wholebody, indices 17-22)
    # 17: left_big_toe, 18: left_small_toe, 19: left_heel
    # 20: right_big_toe, 21: right_small_toe, 22: right_heel
    FOOT_MAP = {
        17: "left_toe",    # left_big_toe (대표 toe)
        19: "left_heel",   # left_heel
        20: "right_toe",   # right_big_toe (대표 toe)
        22: "right_heel",  # right_heel
    }
    # 소발가락 (선택적 - 추가 정밀도용)
    FOOT_MAP_EXTRA = {
        18: "left_small_toe",
        21: "right_small_toe",
    }

    def __init__(self, mode="balanced",
                 backend="onnxruntime", device="cuda",
                 use_tensorrt=False):
        name = f"RTMPose Wholebody ({mode})"
        if use_tensorrt:
            name += " (TRT)"
        super().__init__(name)
        self.mode = mode
        self.backend = backend
        self.device = device
        self.use_tensorrt = use_tensorrt

    def load(self):
        try:
            from rtmlib import Wholebody
        except ImportError:
            raise ImportError(
                "rtmlib 미설치.\n"
                "  pip install rtmlib\n"
                "  + pip install onnxruntime-gpu"
            )

        backend = self.backend
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trt_cache")

        if self.use_tensorrt:
            # TRT 가용성 사전 체크
            has_trt, has_cuda, providers = check_tensorrt_available()
            if not has_trt:
                print(f"  [{self.name}] WARNING: TensorrtExecutionProvider 미감지")
                print(f"  [{self.name}] 사용 가능: {providers}")
                if has_cuda:
                    print(f"  [{self.name}] CUDA EP로 fallback")

            backend = "onnxruntime"
            # 환경변수를 모델 로드 전에 설정 (중요!)
            setup_trt_env(cache_dir)
            print(f"  [{self.name}] TensorRT EP 활성화 (FP16, 최초 빌드 시 수분 소요)")

        print(f"  [{self.name}] 로드 중... (최초 실행 시 모델 자동 다운로드)")
        self.wholebody = Wholebody(
            mode=self.mode,
            to_openpose=False,
            backend=backend,
            device=self.device,
        )

        # TensorRT EP 설정
        if self.use_tensorrt:
            self._apply_tensorrt_provider(cache_dir)

        # 워밍업 (TensorRT 첫 실행 시 엔진 빌드됨)
        print(f"  [{self.name}] 워밍업 중...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            self.wholebody(dummy)

        self.is_loaded = True
        print(f"  [{self.name}] 로드 완료 (TRT={self.use_tensorrt})")

    def _apply_tensorrt_provider(self, cache_dir="./trt_cache"):
        """rtmlib 내부 ONNX session에 TensorRT provider 적용"""
        try:
            import onnxruntime as ort
            trt_providers = get_trt_providers(cache_dir)

            replaced = 0
            if hasattr(self.wholebody, 'det_model') and hasattr(self.wholebody.det_model, 'session'):
                det_path = getattr(self.wholebody.det_model, 'model_path', None)
                if det_path and os.path.exists(det_path):
                    self.wholebody.det_model.session = ort.InferenceSession(
                        det_path, providers=trt_providers)
                    verify_trt_provider(self.wholebody.det_model.session, f"{self.name}/det")
                    replaced += 1
            if hasattr(self.wholebody, 'pose_model') and hasattr(self.wholebody.pose_model, 'session'):
                pose_path = getattr(self.wholebody.pose_model, 'model_path', None)
                if pose_path and os.path.exists(pose_path):
                    self.wholebody.pose_model.session = ort.InferenceSession(
                        pose_path, providers=trt_providers)
                    verify_trt_provider(self.wholebody.pose_model.session, f"{self.name}/pose")
                    replaced += 1

            if replaced > 0:
                print(f"  [{self.name}] TensorRT provider 적용 완료 ({replaced}개 세션)")
            else:
                print(f"  [{self.name}] WARNING: 교체 가능한 ONNX 세션을 찾지 못함")
        except Exception as e:
            print(f"  [{self.name}] TensorRT provider 적용 실패, CUDA fallback: {e}")

    def predict(self, rgb_image: np.ndarray) -> PoseResult:
        result = PoseResult()

        keypoints, scores = self.wholebody(rgb_image)

        if keypoints is None or len(keypoints) == 0:
            return result

        # 가장 큰 사람 선택
        if len(keypoints) > 1:
            areas = []
            for kp in keypoints:
                valid = kp[kp.sum(axis=1) > 0]
                if len(valid) > 0:
                    x_range = valid[:, 0].max() - valid[:, 0].min()
                    y_range = valid[:, 1].max() - valid[:, 1].min()
                    areas.append(x_range * y_range)
                else:
                    areas.append(0)
            best_idx = int(np.argmax(areas))
        else:
            best_idx = 0

        kps = keypoints[best_idx]
        scrs = scores[best_idx]
        num_kps = len(kps)

        result.detected = True

        # Body keypoints
        for coco_idx, our_name in self.BODY_MAP.items():
            if coco_idx < num_kps:
                x, y = float(kps[coco_idx, 0]), float(kps[coco_idx, 1])
                s = float(scrs[coco_idx])
                if x > 0 or y > 0:
                    result.keypoints_2d[our_name] = (x, y)
                    result.confidences[our_name] = s

        # Foot keypoints (primary: big_toe, heel)
        for wb_idx, our_name in self.FOOT_MAP.items():
            if wb_idx < num_kps:
                x, y = float(kps[wb_idx, 0]), float(kps[wb_idx, 1])
                s = float(scrs[wb_idx])
                if x > 0 or y > 0:
                    result.keypoints_2d[our_name] = (x, y)
                    result.confidences[our_name] = s

        # Foot keypoints (extra: small_toe)
        for wb_idx, our_name in self.FOOT_MAP_EXTRA.items():
            if wb_idx < num_kps:
                x, y = float(kps[wb_idx, 0]), float(kps[wb_idx, 1])
                s = float(scrs[wb_idx])
                if x > 0 or y > 0:
                    result.keypoints_2d[our_name] = (x, y)
                    result.confidences[our_name] = s

        return result


# ============================================================================
# 5. ZED Body Tracking (기준선)
# ============================================================================
class ZEDBodyTracking(PoseModel):
    """
    ZED SDK 내장 Body Tracking
    - 38 keypoints (BODY_38 format)
    - Heel, Toe 포함
    - 3D keypoint 직접 제공 (depth 불필요)
    - 단점: 전신이 보여야 인식, GPU 부하 높음, latency 높음
    """

    KEYPOINT_MAP_38 = {
        22: "left_hip", 25: "right_hip",
        23: "left_knee", 26: "right_knee",
        24: "left_ankle", 27: "right_ankle",
        33: "left_heel", 34: "right_heel",
        35: "left_toe", 37: "right_toe",
    }

    def __init__(self, model="FAST"):
        super().__init__(f"ZED BT ({model})")
        self.model_type = model
        self.zed = None
        self.body_runtime = None

    def load(self, zed_camera=None):
        try:
            import pyzed.sl as sl
        except ImportError:
            raise ImportError("pyzed 미설치. ZED SDK 설치 필요")

        if zed_camera is None:
            raise ValueError("ZED 카메라 객체가 필요합니다")

        self.zed = zed_camera.zed

        body_params = sl.BodyTrackingParameters()
        body_params.enable_tracking = True
        body_params.enable_body_fitting = True
        body_params.body_format = sl.BODY_FORMAT.BODY_38

        model_map = {
            "FAST": sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST,
            "MEDIUM": sl.BODY_TRACKING_MODEL.HUMAN_BODY_MEDIUM,
            "ACCURATE": sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE,
        }
        body_params.detection_model = model_map.get(
            self.model_type, sl.BODY_TRACKING_MODEL.HUMAN_BODY_FAST)

        err = self.zed.enable_body_tracking(body_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Body Tracking 활성화 실패: {err}")

        self.body_runtime = sl.BodyTrackingRuntimeParameters()
        self.body_runtime.detection_confidence_threshold = 40
        self.bodies = sl.Bodies()

        self.is_loaded = True
        print(f"  [{self.name}] 로드 완료")

    def predict(self, rgb_image: np.ndarray) -> PoseResult:
        result = PoseResult()

        import pyzed.sl as sl
        err = self.zed.retrieve_bodies(self.bodies, self.body_runtime)
        if err != sl.ERROR_CODE.SUCCESS:
            return result

        if len(self.bodies.body_list) == 0:
            return result

        # 가장 가까운 사람
        body = min(self.bodies.body_list, key=lambda b: b.position[2])
        result.detected = True

        kps_3d = body.keypoint
        kps_2d = body.keypoint_2d
        confs = body.keypoint_confidence

        for zed_idx, our_name in self.KEYPOINT_MAP_38.items():
            if zed_idx < len(kps_2d):
                result.keypoints_2d[our_name] = (
                    float(kps_2d[zed_idx][0]),
                    float(kps_2d[zed_idx][1]))
                result.confidences[our_name] = float(confs[zed_idx]) / 100.0
            if zed_idx < len(kps_3d):
                kp3 = kps_3d[zed_idx]
                result.keypoints_3d[our_name] = (
                    float(kp3[0]), float(kp3[1]), float(kp3[2]))

        return result

    def close(self):
        if self.zed:
            self.zed.disable_body_tracking()


# ============================================================================
# 6. MoveNet (Google, TFLite)
# ============================================================================
class MoveNetModel(PoseModel):
    """
    Google MoveNet - 초경량 포즈 추정
    - TFLite runtime 사용 (tflite_runtime.interpreter)
    - Lightning: 192x192, 초고속 (~3ms)
    - Thunder: 256x256, 더 정확
    - 17 COCO keypoints (heel/toe 없음)
    - TensorRT: ONNX Runtime TRT EP 지원 (선택적)

    설치: pip install tflite-runtime
    모델 다운로드: python3 download_movenet.py
    """

    KEYPOINT_MAP = {
        11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee",
        15: "left_ankle", 16: "right_ankle",
    }

    # MoveNet 모델 파일 정보
    MODEL_INFO = {
        "lightning": {
            "input_size": 192,
            "tflite_file": "movenet_lightning.tflite",
            "onnx_file": "movenet_lightning.onnx",
        },
        "thunder": {
            "input_size": 256,
            "tflite_file": "movenet_thunder.tflite",
            "onnx_file": "movenet_thunder.onnx",
        },
    }

    def __init__(self, variant="lightning", use_tensorrt=False, model_dir=None):
        name = f"MoveNet ({variant})"
        if use_tensorrt:
            name += " (TRT)"
        super().__init__(name)
        self.variant = variant
        self.use_tensorrt = use_tensorrt
        self.model_dir = model_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "models")
        self.interpreter = None
        self.ort_session = None
        self.info = self.MODEL_INFO[variant]
        self.input_size = self.info["input_size"]

    def load(self):
        import os

        if self.use_tensorrt:
            self._load_onnx_trt()
        else:
            self._load_tflite()

        # 워밍업
        dummy = np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8)
        for _ in range(5):
            self._run_inference(dummy)

        self.is_loaded = True
        mode_str = "TRT" if self.use_tensorrt else "TFLite"
        print(f"  [{self.name}] 로드 완료 ({mode_str})")

    def _load_tflite(self):
        """TFLite 모델 로드"""
        try:
            from tflite_runtime.interpreter import Interpreter
        except ImportError:
            try:
                from tensorflow.lite.python.interpreter import Interpreter
            except ImportError:
                raise ImportError(
                    "tflite-runtime 미설치.\n"
                    "  pip install tflite-runtime\n"
                    "  또는: pip install tensorflow"
                )

        model_path = os.path.join(self.model_dir, self.info["tflite_file"])
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"MoveNet 모델 파일 없음: {model_path}\n"
                f"  python3 download_movenet.py 로 다운로드하세요"
            )

        print(f"  [{self.name}] TFLite 모델 로드: {model_path}")
        self.interpreter = Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self._input_details = self.interpreter.get_input_details()
        self._output_details = self.interpreter.get_output_details()

    def _load_onnx_trt(self):
        """ONNX + TensorRT EP 로드"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("onnxruntime 미설치. pip install onnxruntime-gpu")

        # TRT 가용성 사전 체크
        has_trt, has_cuda, avail_providers = check_tensorrt_available()
        if not has_trt:
            print(f"  [{self.name}] WARNING: TensorrtExecutionProvider 미감지")
            print(f"  [{self.name}] 사용 가능: {avail_providers}")

        onnx_path = os.path.join(self.model_dir, self.info["onnx_file"])
        if not os.path.exists(onnx_path):
            raise FileNotFoundError(
                f"MoveNet ONNX 모델 없음: {onnx_path}\n"
                f"  python3 download_movenet.py 로 다운로드하세요"
            )

        cache_dir = os.path.join(self.model_dir, "trt_cache")
        providers = get_trt_providers(cache_dir)

        print(f"  [{self.name}] ONNX+TRT 로드: {onnx_path}")
        print(f"  [{self.name}] TensorRT 최초 빌드 시 수분 소요...")
        self.ort_session = ort.InferenceSession(onnx_path, providers=providers)
        verify_trt_provider(self.ort_session, self.name)

    def _run_inference(self, input_image):
        """전처리된 이미지로 추론 실행, (1, 1, 17, 3) 형태 반환"""
        if self.interpreter is not None:
            # TFLite
            input_tensor = np.expand_dims(input_image, axis=0).astype(
                self._input_details[0]['dtype'])
            self.interpreter.set_tensor(self._input_details[0]['index'], input_tensor)
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(self._output_details[0]['index'])
            return output  # (1, 1, 17, 3) - y, x, confidence
        elif self.ort_session is not None:
            # ONNX Runtime
            input_name = self.ort_session.get_inputs()[0].name
            input_tensor = np.expand_dims(input_image, axis=0).astype(np.int32)
            output = self.ort_session.run(None, {input_name: input_tensor})
            return output[0]  # (1, 1, 17, 3)

    def predict(self, rgb_image: np.ndarray) -> PoseResult:
        result = PoseResult()
        h, w = rgb_image.shape[:2]

        # 전처리: 리사이즈 + RGB 변환
        input_img = cv2.resize(rgb_image, (self.input_size, self.input_size))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        # 추론
        output = self._run_inference(input_img)

        # 출력 파싱: (1, 1, 17, 3) → keypoints
        kps = output[0, 0]  # (17, 3) - y, x, confidence

        # confidence 체크
        valid_count = np.sum(kps[:, 2] > 0.1)
        if valid_count >= 3:
            result.detected = True

        for coco_idx, our_name in self.KEYPOINT_MAP.items():
            if coco_idx < len(kps):
                y_norm, x_norm, conf = kps[coco_idx]
                px = float(x_norm * w)
                py = float(y_norm * h)
                if conf > 0.01:
                    result.keypoints_2d[our_name] = (px, py)
                    result.confidences[our_name] = float(conf)
                else:
                    result.confidences[our_name] = 0.0

        return result


# ============================================================================
# 7. LowerBodyPoseModel — 하체 전용 6kpt 커스텀 모델
# ============================================================================
class LowerBodyPoseModel(PoseModel):
    """
    하체 전용 YOLO-Pose 모델 (6 keypoints: hip/knee/ankle)

    YOLO26s-pose를 하체 전용으로 Fine-Tuning한 커스텀 모델.
    원본 YOLO26s-pose (17kpt)와 완전히 분리된 별도 모델입니다.

    지원 모드:
      - 1-Stage: 전체 프레임에서 직접 하체 검출+포즈 추론
      - 2-Stage: 기존 YOLO(17kpt)로 사람 검출 → 하체 crop → 하체 모델로 정밀 추론

    명명 규칙:
      - 모델 파일: yolo26s-lower6-416.pt / .onnx / .engine
      - Registry 키: "lower_body", "lower_body_2stage"
    """

    KEYPOINT_MAP = {
        0: "left_hip",
        1: "right_hip",
        2: "left_knee",
        3: "right_knee",
        4: "left_ankle",
        5: "right_ankle",
    }

    JOINT_ANGLE_TRIPLETS = {
        "left_knee_angle":  ("left_hip", "left_knee", "left_ankle"),
        "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
    }

    def __init__(self, model_path=None, use_tensorrt=False,
                 imgsz=640, two_stage=False, stage1_model=None,
                 conf=0.25, iou=0.7, max_det=1, half=True,
                 smoothing=0.0, filter_min_cutoff=1.0, filter_beta=0.007,
                 segment_constraint=True, seg_calib_frames=30, seg_tolerance=0.15,
                 seg_calib_file=None):
        """
        Args:
            model_path: 커스텀 학습된 모델 (.pt/.onnx/.engine)
            use_tensorrt: TRT 엔진 사용 여부
            imgsz: 입력 해상도 (학습 시와 동일해야 함)
            two_stage: 2-Stage 파이프라인 사용
            stage1_model: 2-Stage 시 Stage1 전신 모델 (YOLOv8Pose 인스턴스)
            conf, iou, max_det, half: YOLO 추론 파라미터
            smoothing: One Euro Filter (0=OFF)
            filter_min_cutoff, filter_beta: 필터 파라미터
            segment_constraint: 뼈 길이 제약 ON/OFF
        """
        mode_str = "2-Stage" if two_stage else "1-Stage"
        trt_str = " TRT-FP16" if use_tensorrt else ""
        super().__init__(f"LowerBody-6kpt ({mode_str}{trt_str})")

        self.model_path = model_path
        self.use_tensorrt = use_tensorrt
        self.imgsz = imgsz
        self.two_stage = two_stage
        self.stage1_model = stage1_model
        self.model = None

        # 추론 파라미터
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.half = half

        # One Euro Filter
        self.smoothing = smoothing
        self._kp_filter = KeypointOneEuroFilter(
            min_cutoff=filter_min_cutoff, beta=filter_beta
        ) if smoothing > 0 else None

        # 세그먼트 길이 제약
        self._seg_constraint = SegmentLengthConstraint(
            calib_frames=seg_calib_frames, tolerance=seg_tolerance,
            calib_file=seg_calib_file,
        ) if segment_constraint else None

    def load(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics 미설치. pip install ultralytics")

        if self.model_path is None:
            print(f"  [{self.name}] WARNING: model_path 미지정. "
                  f"학습된 모델 경로를 지정하세요.")
            print(f"  예: LowerBodyPoseModel(model_path='yolo26s-lower6-416.pt')")
            return

        print(f"  [{self.name}] 모델 로드: {self.model_path}")

        if self.use_tensorrt:
            # TRT 엔진 직접 로드
            engine_path = self.model_path
            if engine_path.endswith(".pt"):
                # .pt → .engine 변환 시도
                engine_name = engine_path.replace(".pt", f"-{self.imgsz}.engine")
                if os.path.exists(engine_name):
                    engine_path = engine_name
                else:
                    print(f"  [{self.name}] TRT 엔진 빌드 중 (최초 1회)...")
                    model = YOLO(self.model_path)
                    exported = model.export(
                        format="engine", half=True, imgsz=self.imgsz)
                    engine_path = exported
            self.model = YOLO(engine_path, task="pose")
            print(f"  [{self.name}] TRT 엔진 로드: {engine_path}")
        else:
            self.model = YOLO(self.model_path, task="pose")

        # 2-Stage: Stage1 모델도 로드
        if self.two_stage and self.stage1_model is not None:
            if not self.stage1_model.is_loaded:
                print(f"  [{self.name}] Stage1 모델 로드 중...")
                self.stage1_model.load()

        # 워밍업
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model(dummy, verbose=False, imgsz=self.imgsz)

        self.is_loaded = True
        print(f"  [{self.name}] 로드 완료")

    def _crop_lower_body(self, rgb_image, stage1_result):
        """
        Stage1 결과로부터 하체 영역 crop.

        Returns:
            (cropped_image, crop_info) where crop_info = (x_off, y_off, crop_w, crop_h)
            or (None, None) if crop 불가
        """
        h, w = rgb_image.shape[:2]

        # Stage1에서 hip/ankle 위치 가져오기
        kps = stage1_result.keypoints_2d
        ys = []
        xs = []
        for name in ["left_hip", "right_hip", "left_knee", "right_knee",
                      "left_ankle", "right_ankle"]:
            if name in kps:
                px, py = kps[name]
                if stage1_result.confidences.get(name, 0) > 0.3:
                    xs.append(px)
                    ys.append(py)

        if len(xs) < 2:
            return None, None

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        bw = x_max - x_min
        bh = y_max - y_min

        # 패딩 (여유)
        pad = 0.3
        x1 = max(0, int(x_min - bw * pad))
        y1 = max(0, int(y_min - bh * (pad + 0.1)))  # hip 위로 여유
        x2 = min(w, int(x_max + bw * pad))
        y2 = min(h, int(y_max + bh * (pad + 0.05)))  # ankle 아래 여유

        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w < 20 or crop_h < 20:
            return None, None

        cropped = rgb_image[y1:y2, x1:x2]
        return cropped, (x1, y1, crop_w, crop_h)

    def _back_project_keypoints(self, result, crop_info):
        """crop 좌표 → 원본 이미지 좌표로 변환"""
        x_off, y_off, crop_w, crop_h = crop_info
        new_kps = {}
        for name, (px, py) in result.keypoints_2d.items():
            # crop 내 비율 계산 후 offset 적용
            new_kps[name] = (px + x_off, py + y_off)
        result.keypoints_2d = new_kps

    def predict(self, rgb_image: np.ndarray) -> PoseResult:
        result = PoseResult()
        t0 = time.perf_counter()

        input_image = rgb_image
        crop_info = None

        # 2-Stage: Stage1 → crop
        if self.two_stage and self.stage1_model is not None:
            stage1_result = self.stage1_model.predict(rgb_image)
            if stage1_result.detected:
                cropped, crop_info = self._crop_lower_body(rgb_image, stage1_result)
                if cropped is not None:
                    input_image = cropped

        # YOLO 추론
        results = self.model(
            input_image,
            verbose=False,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            max_det=self.max_det,
            half=self.half,
        )

        t_infer = time.perf_counter()
        result.inference_time_ms = (t_infer - t0) * 1000

        # 결과 파싱
        if results and len(results) > 0 and results[0].keypoints is not None:
            kps_data = results[0].keypoints.data
            if len(kps_data) > 0:
                person_kps = kps_data[0].cpu().numpy()
                ih, iw = input_image.shape[:2]

                valid_count = 0
                for idx, our_name in self.KEYPOINT_MAP.items():
                    if idx < len(person_kps):
                        kp = person_kps[idx]
                        x, y = float(kp[0]), float(kp[1])
                        conf = float(kp[2]) if len(kp) > 2 else 1.0

                        if conf > 0.01 and x > 0 and y > 0:
                            result.keypoints_2d[our_name] = (x, y)
                            result.confidences[our_name] = conf
                            if conf > 0.3:
                                valid_count += 1
                        else:
                            result.confidences[our_name] = 0.0

                if valid_count >= 3:
                    result.detected = True

        # 2-Stage: crop 좌표 → 원본 좌표
        if crop_info is not None and result.detected:
            self._back_project_keypoints(result, crop_info)

        # One Euro Filter (떨림 감소)
        if self._kp_filter is not None and result.detected:
            t_now = time.perf_counter()
            for name in list(result.keypoints_2d.keys()):
                x, y = result.keypoints_2d[name]
                fx, fy = self._kp_filter.filter(name, t_now, x, y)
                result.keypoints_2d[name] = (fx, fy)

        # 세그먼트 길이 제약 (in-place로 keypoints_2d 수정)
        if self._seg_constraint is not None and result.detected:
            self._seg_constraint.update(
                result.keypoints_2d, result.confidences)

        # 관절 각도 계산
        if result.detected:
            for angle_name, (p, j, c) in self.JOINT_ANGLE_TRIPLETS.items():
                if (p in result.keypoints_2d and
                    j in result.keypoints_2d and
                    c in result.keypoints_2d):
                    min_conf = min(
                        result.confidences.get(p, 0),
                        result.confidences.get(j, 0),
                        result.confidences.get(c, 0),
                    )
                    if min_conf >= 0.3:
                        angle = self._calc_angle(
                            result.keypoints_2d[p],
                            result.keypoints_2d[j],
                            result.keypoints_2d[c],
                        )
                        result.joint_angles[angle_name] = angle

        result.timing_detail["total_predict_ms"] = (time.perf_counter() - t0) * 1000
        return result

    @staticmethod
    def _calc_angle(p1, p2, p3):
        """세 점 사이의 각도 계산 (p2가 관절점, 도 단위)"""
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        return float(np.degrees(np.arccos(cos_angle)))


# MoveNet을 포함한 모델 레지스트리
MODEL_REGISTRY = {
    "mediapipe": lambda **kw: MediaPipePose(model_complexity=kw.get("complexity", 1)),
    "yolov8": lambda **kw: YOLOv8Pose(model_size=kw.get("size", "n"),
                                        use_tensorrt=kw.get("tensorrt", False)),
    "yolov8_int8": lambda **kw: YOLOv8Pose(model_size=kw.get("size", "n"),
                                             use_tensorrt=True,
                                             precision="int8", use_calib=True),
    "yolov8_int8_nocal": lambda **kw: YOLOv8Pose(model_size=kw.get("size", "n"),
                                                    use_tensorrt=True,
                                                    precision="int8", use_calib=False),
    "yolo11": lambda **kw: YOLOv8Pose(model_size=kw.get("size", "n"),
                                        use_tensorrt=kw.get("tensorrt", False),
                                        yolo_version="v11"),
    "yolo11_int8": lambda **kw: YOLOv8Pose(model_size=kw.get("size", "n"),
                                             use_tensorrt=True,
                                             precision="int8", use_calib=True,
                                             yolo_version="v11"),
    "yolo11_int8_nocal": lambda **kw: YOLOv8Pose(model_size=kw.get("size", "n"),
                                                    use_tensorrt=True,
                                                    precision="int8", use_calib=False,
                                                    yolo_version="v11"),
    "yolo26": lambda **kw: YOLOv8Pose(model_size=kw.get("size", "n"),
                                        use_tensorrt=kw.get("tensorrt", False),
                                        yolo_version="v26"),
    "yolo26_int8": lambda **kw: YOLOv8Pose(model_size=kw.get("size", "n"),
                                             use_tensorrt=True,
                                             precision="int8", use_calib=True,
                                             yolo_version="v26"),
    "yolo26_int8_nocal": lambda **kw: YOLOv8Pose(model_size=kw.get("size", "n"),
                                                    use_tensorrt=True,
                                                    precision="int8", use_calib=False,
                                                    yolo_version="v26"),
    "rtmpose": lambda **kw: RTMPoseModel(mode=kw.get("mode", "balanced"),
                                          use_tensorrt=kw.get("tensorrt", False),
                                          precision=kw.get("precision", "fp16")),
    "rtmpose_int8": lambda **kw: RTMPoseModel(mode=kw.get("mode", "balanced"),
                                               use_tensorrt=True,
                                               precision="int8",
                                               use_calib=True),
    "rtmpose_int8_nocal": lambda **kw: RTMPoseModel(mode=kw.get("mode", "balanced"),
                                                      use_tensorrt=True,
                                                      precision="int8",
                                                      use_calib=False),
    "rtmpose_wb": lambda **kw: RTMPoseWholebody(mode=kw.get("mode", "balanced"),
                                                  use_tensorrt=kw.get("tensorrt", False)),
    "movenet": lambda **kw: MoveNetModel(variant=kw.get("variant", "lightning"),
                                          use_tensorrt=False),
    "movenet_trt": lambda **kw: MoveNetModel(variant=kw.get("variant", "lightning"),
                                              use_tensorrt=True),
    # 하체 전용 커스텀 모델 (Fine-Tuned, 원본 YOLO26s와 완전 분리)
    "lower_body": lambda **kw: LowerBodyPoseModel(
        model_path=kw.get("model_path"),
        use_tensorrt=kw.get("tensorrt", False),
        imgsz=kw.get("imgsz", 640),
        two_stage=False,
        smoothing=kw.get("smoothing", 0.0),
    ),
    "lower_body_2stage": lambda **kw: LowerBodyPoseModel(
        model_path=kw.get("model_path"),
        use_tensorrt=kw.get("tensorrt", False),
        imgsz=kw.get("imgsz", 640),
        two_stage=True,
        stage1_model=YOLOv8Pose(
            model_size="s", use_tensorrt=True, yolo_version="v26"),
        smoothing=kw.get("smoothing", 0.0),
    ),
}


# ============================================================================
# 시각화 유틸리티
# ============================================================================
SKELETON_CONNECTIONS = [
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_ankle", "left_toe"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_ankle", "right_toe"),
    ("left_hip", "right_hip"),
]

KEYPOINT_COLORS = {
    "left_hip": (0, 255, 0), "left_knee": (0, 200, 0),
    "left_ankle": (0, 150, 0), "left_heel": (0, 100, 0), "left_toe": (0, 80, 0),
    "right_hip": (0, 0, 255), "right_knee": (0, 0, 200),
    "right_ankle": (0, 0, 150), "right_heel": (0, 0, 100), "right_toe": (0, 0, 80),
}


def _conf_to_color(conf):
    """confidence 값을 색상으로 변환 (빨강 → 노랑 → 초록)"""
    if conf >= 0.7:
        return (0, 255, 0)       # 초록
    elif conf >= 0.4:
        ratio = (conf - 0.4) / 0.3
        return (0, int(255 * ratio), int(255 * (1 - ratio)))  # 노랑 → 초록
    else:
        return (0, 0, 255)       # 빨강


def draw_pose(image, pose_result: PoseResult, model_name="", min_conf=0.3,
              show_angles=True, show_conf_colors=True):
    """이미지에 하체 pose 그리기 (관절 각도 + confidence 히트맵)"""
    vis = image.copy()

    # Skeleton 연결선 — confidence 기반 색상
    for kp1_name, kp2_name in SKELETON_CONNECTIONS:
        if (kp1_name in pose_result.keypoints_2d and
            kp2_name in pose_result.keypoints_2d and
            pose_result.confidences.get(kp1_name, 0) >= min_conf and
            pose_result.confidences.get(kp2_name, 0) >= min_conf):
            pt1 = tuple(int(v) for v in pose_result.keypoints_2d[kp1_name])
            pt2 = tuple(int(v) for v in pose_result.keypoints_2d[kp2_name])
            avg_c = (pose_result.confidences.get(kp1_name, 0) +
                     pose_result.confidences.get(kp2_name, 0)) / 2
            line_color = _conf_to_color(avg_c) if show_conf_colors else (255, 255, 0)
            thickness = max(2, int(avg_c * 4))
            cv2.line(vis, pt1, pt2, line_color, thickness)

    # Keypoints — confidence 히트맵
    for name, (x, y) in pose_result.keypoints_2d.items():
        conf = pose_result.confidences.get(name, 0)
        if conf >= min_conf:
            color = _conf_to_color(conf) if show_conf_colors else KEYPOINT_COLORS.get(name, (255, 255, 255))
            radius = max(4, int(conf * 10))
            cv2.circle(vis, (int(x), int(y)), radius, color, -1)
            cv2.circle(vis, (int(x), int(y)), radius + 2, (255, 255, 255), 1)
            # 짧은 이름 표시
            short_name = name.replace("left_", "L").replace("right_", "R")
            cv2.putText(vis, f"{short_name}:{conf:.2f}",
                       (int(x) + 8, int(y) - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # 관절 각도 표시
    if show_angles and pose_result.joint_angles:
        for angle_name, angle_val in pose_result.joint_angles.items():
            # 각도를 관절 위치에 표시
            joint_name = angle_name.replace("_angle", "")
            if joint_name in pose_result.keypoints_2d:
                jx, jy = pose_result.keypoints_2d[joint_name]
                # 각도 배경 박스
                text = f"{angle_val:.0f}°"
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                bx, by = int(jx) - tw - 10, int(jy) - 5
                cv2.rectangle(vis, (bx - 2, by - th - 2), (bx + tw + 2, by + 4),
                              (0, 0, 0), -1)
                # 무릎 각도: 180=펴짐(파랑), 90=굽힘(주황)
                angle_ratio = max(0, min(1, (angle_val - 90) / 90))
                angle_color = (0, int(200 * angle_ratio), int(255 * (1 - angle_ratio)))
                cv2.putText(vis, text, (bx, by),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, angle_color, 2)

    # 상단 정보
    cv2.putText(vis, f"{model_name} | {pose_result.inference_time_ms:.1f}ms",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    avg_conf = pose_result.get_lower_limb_confidence()
    has_ll = pose_result.has_lower_limb()
    status = "OK" if has_ll else "FAIL"
    status_color = (0, 255, 0) if has_ll else (0, 0, 255)

    cv2.putText(vis, f"Lower Limb: {status} (conf={avg_conf:.2f})",
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)

    # keypoint 개수 표시
    detected_kps = [k for k, c in pose_result.confidences.items() if c >= min_conf]
    cv2.putText(vis, f"Keypoints: {len(detected_kps)}/{len(LOWER_LIMB_KEYPOINTS)}",
               (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return vis
