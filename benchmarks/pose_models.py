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

하체 전용 인식 관련 참고:
  - Top-down 방식 (RTMPose, MediaPipe): 사람 검출 → 크롭 → 포즈 추정
    → 상체 안 보이면 사람 검출 실패 가능!
  - Bottom-up/Single-shot 방식 (YOLOv8-Pose): keypoint 직접 검출
    → 부분 인식에 상대적으로 강함
  - 이 벤치마크로 실제로 어떤 모델이 하체만 보일 때 동작하는지 확인
"""

import numpy as np
import time
import cv2
from abc import ABC, abstractmethod


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
    Ultralytics YOLOv8 Pose
    - Detection + Pose를 한번에 (single-shot)
    - 17 COCO keypoints (heel/toe 없음)
    - 부분 인식에 상대적으로 강함 (YOLO가 partial object 감지 가능)
    - TensorRT 변환 지원
    """

    KEYPOINT_MAP = {
        11: "left_hip", 12: "right_hip",
        13: "left_knee", 14: "right_knee",
        15: "left_ankle", 16: "right_ankle",
    }

    def __init__(self, model_size="n", use_tensorrt=False):
        name = f"YOLOv8{model_size}-Pose"
        if use_tensorrt:
            name += " (TRT)"
        super().__init__(name)
        self.model_size = model_size
        self.use_tensorrt = use_tensorrt
        self.model = None

    def load(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("ultralytics 미설치. pip install ultralytics")

        model_name = f"yolov8{self.model_size}-pose.pt"
        print(f"  [{self.name}] 모델 로드 중: {model_name}")
        self.model = YOLO(model_name)

        if self.use_tensorrt:
            print(f"  [{self.name}] TensorRT 엔진 변환 중 (최초 1회, 수분 소요)...")
            engine_path = self.model.export(format="engine", half=True)
            self.model = YOLO(engine_path)

        # 워밍업 (첫 추론은 느림)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model(dummy, verbose=False)

        self.is_loaded = True
        print(f"  [{self.name}] 로드 완료")

    def predict(self, rgb_image: np.ndarray) -> PoseResult:
        result = PoseResult()

        results = self.model(rgb_image, verbose=False)

        if not results or len(results) == 0:
            return result

        r = results[0]
        if r.keypoints is None or len(r.keypoints) == 0:
            return result

        boxes = r.boxes
        if boxes is None or len(boxes) == 0:
            return result

        # 가장 큰 bbox (= 가장 가까운 사람) 선택
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

        result.detected = True
        for coco_idx, our_name in self.KEYPOINT_MAP.items():
            if coco_idx < len(xy):
                x, y = float(xy[coco_idx, 0]), float(xy[coco_idx, 1])
                # (0,0)은 미감지 keypoint
                if x > 0 or y > 0:
                    result.keypoints_2d[our_name] = (x, y)
                    result.confidences[our_name] = float(conf[coco_idx])
                else:
                    result.confidences[our_name] = 0.0

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
                 use_tensorrt=False):
        name = f"RTMPose ({mode})"
        if use_tensorrt:
            name += " (TRT)"
        super().__init__(name)
        self.mode = mode
        self.backend = backend
        self.device = device
        self.use_tensorrt = use_tensorrt
        self.body = None

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
        if self.use_tensorrt:
            backend = "onnxruntime"
            # TensorRT EP를 사용하도록 환경 설정
            import os
            os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"
            os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
            os.environ["ORT_TENSORRT_CACHE_PATH"] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "trt_cache"
            )
            os.makedirs(os.environ["ORT_TENSORRT_CACHE_PATH"], exist_ok=True)
            print(f"  [{self.name}] TensorRT EP 활성화 (FP16, 최초 빌드 시 수분 소요)")

        print(f"  [{self.name}] 로드 중... (최초 실행 시 모델 자동 다운로드)")
        self.body = Body(
            mode=self.mode,
            to_openpose=False,
            backend=backend,
            device=self.device,
        )

        # TensorRT EP 설정 (rtmlib 내부 ONNX session에 적용)
        if self.use_tensorrt:
            self._apply_tensorrt_provider()

        # 워밍업 (TensorRT 첫 실행 시 엔진 빌드됨)
        print(f"  [{self.name}] 워밍업 중...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            self.body(dummy)

        self.is_loaded = True
        print(f"  [{self.name}] 로드 완료 (backend={backend}, device={self.device}, TRT={self.use_tensorrt})")

    def _apply_tensorrt_provider(self):
        """rtmlib 내부 ONNX session에 TensorRT provider 적용"""
        try:
            import onnxruntime as ort
            trt_providers = [
                ('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': os.environ.get("ORT_TENSORRT_CACHE_PATH", "./trt_cache"),
                }),
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                }),
            ]
            # rtmlib Body 내부의 det_model과 pose_model 세션 교체
            if hasattr(self.body, 'det_model') and hasattr(self.body.det_model, 'session'):
                det_model_path = self.body.det_model.model_path if hasattr(self.body.det_model, 'model_path') else None
                if det_model_path:
                    self.body.det_model.session = ort.InferenceSession(
                        det_model_path, providers=trt_providers)
            if hasattr(self.body, 'pose_model') and hasattr(self.body.pose_model, 'session'):
                pose_model_path = self.body.pose_model.model_path if hasattr(self.body.pose_model, 'model_path') else None
                if pose_model_path:
                    self.body.pose_model.session = ort.InferenceSession(
                        pose_model_path, providers=trt_providers)
            print(f"  [{self.name}] TensorRT provider 적용 완료")
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
                s = float(scrs[coco_idx])
                if x > 0 or y > 0:
                    result.keypoints_2d[our_name] = (x, y)
                    result.confidences[our_name] = s
                else:
                    result.confidences[our_name] = 0.0

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
    # 정확한 매핑은 모델 출력을 확인하면서 조정 필요
    FOOT_MAP = {
        17: "left_toe",    # left_big_toe
        19: "left_heel",   # left_heel
        20: "right_toe",   # right_big_toe
        22: "right_heel",  # right_heel
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
        if self.use_tensorrt:
            backend = "onnxruntime"
            import os
            os.environ["ORT_TENSORRT_FP16_ENABLE"] = "1"
            os.environ["ORT_TENSORRT_ENGINE_CACHE_ENABLE"] = "1"
            os.environ["ORT_TENSORRT_CACHE_PATH"] = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "trt_cache"
            )
            os.makedirs(os.environ["ORT_TENSORRT_CACHE_PATH"], exist_ok=True)
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
            self._apply_tensorrt_provider()

        # 워밍업 (TensorRT 첫 실행 시 엔진 빌드됨)
        print(f"  [{self.name}] 워밍업 중...")
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for _ in range(5):
            self.wholebody(dummy)

        self.is_loaded = True
        print(f"  [{self.name}] 로드 완료 (TRT={self.use_tensorrt})")

    def _apply_tensorrt_provider(self):
        """rtmlib 내부 ONNX session에 TensorRT provider 적용"""
        try:
            import onnxruntime as ort
            import os
            trt_providers = [
                ('TensorrtExecutionProvider', {
                    'device_id': 0,
                    'trt_max_workspace_size': 2 * 1024 * 1024 * 1024,
                    'trt_fp16_enable': True,
                    'trt_engine_cache_enable': True,
                    'trt_engine_cache_path': os.environ.get("ORT_TENSORRT_CACHE_PATH", "./trt_cache"),
                }),
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                }),
            ]
            if hasattr(self.wholebody, 'det_model') and hasattr(self.wholebody.det_model, 'session'):
                det_path = getattr(self.wholebody.det_model, 'model_path', None)
                if det_path:
                    self.wholebody.det_model.session = ort.InferenceSession(
                        det_path, providers=trt_providers)
            if hasattr(self.wholebody, 'pose_model') and hasattr(self.wholebody.pose_model, 'session'):
                pose_path = getattr(self.wholebody.pose_model, 'model_path', None)
                if pose_path:
                    self.wholebody.pose_model.session = ort.InferenceSession(
                        pose_path, providers=trt_providers)
            print(f"  [{self.name}] TensorRT provider 적용 완료")
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

        # Foot keypoints
        for wb_idx, our_name in self.FOOT_MAP.items():
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


def draw_pose(image, pose_result: PoseResult, model_name="", min_conf=0.3):
    """이미지에 하체 pose 그리기"""
    vis = image.copy()

    # Skeleton 연결선
    for kp1_name, kp2_name in SKELETON_CONNECTIONS:
        if (kp1_name in pose_result.keypoints_2d and
            kp2_name in pose_result.keypoints_2d and
            pose_result.confidences.get(kp1_name, 0) >= min_conf and
            pose_result.confidences.get(kp2_name, 0) >= min_conf):
            pt1 = tuple(int(v) for v in pose_result.keypoints_2d[kp1_name])
            pt2 = tuple(int(v) for v in pose_result.keypoints_2d[kp2_name])
            cv2.line(vis, pt1, pt2, (255, 255, 0), 2)

    # Keypoints
    for name, (x, y) in pose_result.keypoints_2d.items():
        conf = pose_result.confidences.get(name, 0)
        if conf >= min_conf:
            color = KEYPOINT_COLORS.get(name, (255, 255, 255))
            radius = max(3, int(conf * 8))
            cv2.circle(vis, (int(x), int(y)), radius, color, -1)
            cv2.putText(vis, f"{name}:{conf:.2f}",
                       (int(x) + 5, int(y) - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

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
