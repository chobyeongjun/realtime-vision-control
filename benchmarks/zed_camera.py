"""
ZED X Mini 카메라 유틸리티
- RGB 이미지 + Depth Map 캡처
- Body Tracking 없이 경량 모드로 동작
- 벤치마크용 프레임 제공
"""

import numpy as np
import os
import time

try:
    import pyzed.sl as sl
    HAS_ZED = True
except ImportError:
    HAS_ZED = False
    print("[WARNING] pyzed 미설치 - 웹캠 폴백 모드로 동작")

import cv2


class ZEDCamera:
    """ZED X Mini RGB + Depth 캡처 (Body Tracking OFF)"""

    def __init__(self, resolution="HD720", fps=30):
        if not HAS_ZED:
            raise RuntimeError("ZED SDK가 설치되지 않았습니다")

        self.zed = sl.Camera()
        self.init_params = sl.InitParameters()

        # 해상도 설정
        res_map = {
            "SVGA": sl.RESOLUTION.SVGA,         # 960x600 (ZED X Mini 최적)
            "HD1080": sl.RESOLUTION.HD1080,      # 1920x1080
            "HD1200": sl.RESOLUTION.HD1200,      # 1920x1200
            "HD720": sl.RESOLUTION.HD720,        # 1280x720 (ZED X Mini 미지원)
            "HD2K": sl.RESOLUTION.HD2K,
            "VGA": sl.RESOLUTION.VGA,
        }
        self.init_params.camera_resolution = res_map.get(resolution, sl.RESOLUTION.SVGA)
        self.init_params.camera_fps = fps
        self.init_params.depth_mode = sl.DEPTH_MODE.NEURAL  # 경량 depth
        self.init_params.coordinate_units = sl.UNIT.METER
        self.init_params.depth_minimum_distance = 0.3
        self.init_params.depth_maximum_distance = 5.0

        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.point_cloud = sl.Mat()

        self.runtime_params = sl.RuntimeParameters()
        self.is_open = False

    def open(self):
        err = self.zed.open(self.init_params)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"ZED 열기 실패: {err}")
        self.is_open = True

        info = self.zed.get_camera_information()
        print(f"[ZED] 카메라 열림: {info.camera_model}")
        print(f"  해상도: {info.camera_configuration.resolution.width}x"
              f"{info.camera_configuration.resolution.height}")
        print(f"  FPS: {info.camera_configuration.fps}")
        return True

    def grab(self):
        """프레임 캡처. 성공 시 True"""
        if not self.is_open:
            return False
        return self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS

    def get_rgb(self):
        """RGB 이미지 (numpy BGR)"""
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        return self.image.get_data()[:, :, :3].copy()  # BGRA → BGR

    def get_depth(self):
        """Depth map (numpy float32, meters)"""
        self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
        return self.depth.get_data().copy()

    def get_point_cloud(self):
        """3D Point Cloud (numpy float32, Nx4: x,y,z,rgba)"""
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA)
        return self.point_cloud.get_data().copy()

    def pixel_to_3d(self, x, y, depth_map=None):
        """2D 픽셀 좌표 → 3D 좌표 (ZED depth 사용)"""
        if depth_map is None:
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            depth_map = self.depth.get_data()

        x, y = int(x), int(y)
        h, w = depth_map.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            z = depth_map[y, x]
            if np.isfinite(z) and z > 0:
                # ZED 카메라 intrinsics로 역투영
                calib = self.zed.get_camera_information().camera_configuration.calibration_parameters
                fx = calib.left_cam.fx
                fy = calib.left_cam.fy
                cx = calib.left_cam.cx
                cy = calib.left_cam.cy
                x3d = (x - cx) * z / fx
                y3d = (y - cy) * z / fy
                return np.array([x3d, y3d, z], dtype=np.float32)
        return None

    def close(self):
        if self.is_open:
            self.zed.close()
            self.is_open = False

    def __del__(self):
        self.close()


class WebcamFallback:
    """ZED 없을 때 일반 웹캠으로 폴백 (depth 없음)"""

    def __init__(self, camera_id=0, width=1280, height=720):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.frame = None
        self.is_open = False

    def open(self):
        self.is_open = self.cap.isOpened()
        if self.is_open:
            print(f"[Webcam] 카메라 열림: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
                  f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        return self.is_open

    def grab(self):
        ret, self.frame = self.cap.read()
        return ret

    def get_rgb(self):
        return self.frame.copy() if self.frame is not None else None

    def get_depth(self):
        return None  # 웹캠은 depth 없음

    def pixel_to_3d(self, x, y, depth_map=None):
        return None

    def close(self):
        self.cap.release()
        self.is_open = False


class VideoFileSource:
    """동영상 파일에서 프레임 읽기 (카메라 없이 벤치마크용)"""

    def __init__(self, video_path, loop=True, **kwargs):
        self.video_path = video_path
        self.loop = loop
        self.cap = None
        self.frame = None
        self.is_open = False

    def open(self):
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"동영상 파일 열기 실패: {self.video_path}")
        self.is_open = True
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"[Video] 파일 열림: {self.video_path}")
        print(f"  해상도: {w}x{h}, FPS: {fps:.1f}, 총 프레임: {total}")
        return True

    def grab(self):
        ret, self.frame = self.cap.read()
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, self.frame = self.cap.read()
        return ret

    def get_rgb(self):
        return self.frame.copy() if self.frame is not None else None

    def get_depth(self):
        return None

    def pixel_to_3d(self, x, y, depth_map=None):
        return None

    def close(self):
        if self.cap:
            self.cap.release()
        self.is_open = False


class SVO2FileSource:
    """ZED SVO2 파일에서 RGB + Depth 재생 (녹화 영상으로 3D 벤치마크)"""

    def __init__(self, svo_path, loop=True, **kwargs):
        if not HAS_ZED:
            raise RuntimeError("SVO2 재생에는 ZED SDK가 필요합니다")
        self.svo_path = svo_path
        self.loop = loop
        self.zed = sl.Camera()
        self.image = sl.Mat()
        self.depth = sl.Mat()
        self.runtime_params = sl.RuntimeParameters()
        self.is_open = False
        self._total_frames = 0
        self._calib = None

    def open(self):
        init = sl.InitParameters()
        init.set_from_svo_file(self.svo_path)
        init.svo_real_time_mode = False  # 최대 속도로 재생
        init.depth_mode = sl.DEPTH_MODE.NEURAL
        init.coordinate_units = sl.UNIT.METER

        err = self.zed.open(init)
        if err != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"SVO2 열기 실패: {err}")

        self.is_open = True
        info = self.zed.get_camera_information()
        w = info.camera_configuration.resolution.width
        h = info.camera_configuration.resolution.height
        fps = info.camera_configuration.fps
        self._total_frames = self.zed.get_svo_number_of_frames()
        self._calib = info.camera_configuration.calibration_parameters

        print(f"[SVO2] 파일 열림: {self.svo_path}")
        print(f"  해상도: {w}x{h}, FPS: {fps}, 총 프레임: {self._total_frames}")
        print(f"  Depth: 활성 (3D 벤치마크 가능)")
        return True

    def grab(self):
        if not self.is_open:
            return False
        err = self.zed.grab(self.runtime_params)
        if err == sl.ERROR_CODE.SUCCESS:
            return True
        elif err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
            if self.loop:
                self.zed.set_svo_position(0)
                return self.zed.grab(self.runtime_params) == sl.ERROR_CODE.SUCCESS
            return False
        return False

    def get_rgb(self):
        self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
        return self.image.get_data()[:, :, :3].copy()

    def get_depth(self):
        self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
        return self.depth.get_data().copy()

    def pixel_to_3d(self, x, y, depth_map=None):
        if depth_map is None:
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            depth_map = self.depth.get_data()

        x, y = int(x), int(y)
        h, w = depth_map.shape[:2]
        if 0 <= x < w and 0 <= y < h:
            z = depth_map[y, x]
            if np.isfinite(z) and z > 0:
                fx = self._calib.left_cam.fx
                fy = self._calib.left_cam.fy
                cx = self._calib.left_cam.cx
                cy = self._calib.left_cam.cy
                x3d = (x - cx) * z / fx
                y3d = (y - cy) * z / fy
                return np.array([x3d, y3d, z], dtype=np.float32)
        return None

    def close(self):
        if self.is_open:
            self.zed.close()
            self.is_open = False

    def __del__(self):
        self.close()


def create_camera(use_zed=True, video_path=None, **kwargs):
    """카메라 팩토리 함수

    SVO2/SVO 파일은 자동으로 ZED SDK로 열어 depth 포함 재생.
    일반 동영상(mp4, avi 등)은 OpenCV로 열어 RGB만 재생.
    """
    if video_path:
        ext = os.path.splitext(video_path)[1].lower()
        if ext in ('.svo2', '.svo') and HAS_ZED:
            return SVO2FileSource(video_path, **kwargs)
        return VideoFileSource(video_path, **kwargs)
    elif use_zed and HAS_ZED:
        return ZEDCamera(**kwargs)
    else:
        return WebcamFallback(**kwargs)
