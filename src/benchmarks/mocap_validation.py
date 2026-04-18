#!/usr/bin/env python3
"""
Mocap 비교 검증 프레임워크
==========================
Pose Estimation 결과를 Motion Capture Ground Truth와 비교합니다.
Mocap 데이터 수집 후 사용할 구조를 미리 준비합니다.

지원 형식:
    - CSV (범용 export)
    - C3D (Vicon, OptiTrack) [추후]
    - BVH (모션 캡처 교환 형식) [추후]

사용법:
    python3 mocap_validation.py --mocap mocap_data/trial01.csv \
                                 --benchmark results/benchmark_*.json
"""

import numpy as np
import csv
import os
from dataclasses import dataclass, field


# ============================================================================
# 데이터 구조
# ============================================================================
@dataclass
class MocapFrame:
    """단일 Mocap 프레임"""
    timestamp: float           # seconds
    keypoints_3d: dict = field(default_factory=dict)  # name -> (x, y, z) meters
    joint_angles: dict = field(default_factory=dict)   # name -> angle degrees
    metadata: dict = field(default_factory=dict)


@dataclass
class ComparisonResult:
    """비교 결과"""
    # Per-keypoint 3D 오차 (mm)
    mpjpe: dict = field(default_factory=dict)      # name -> mean error (mm)
    mpjpe_overall: float = 0.0                      # 전체 평균 MPJPE
    mpjpe_lower_limb: float = 0.0                   # 하체만 MPJPE

    # Per-joint 각도 오차 (degrees)
    joint_angle_mae: dict = field(default_factory=dict)  # name -> MAE (deg)

    # PCK (Percentage of Correct Keypoints)
    pck_50mm: float = 0.0    # 50mm 이내 정확도
    pck_100mm: float = 0.0   # 100mm 이내 정확도

    # 프레임 수
    num_frames: int = 0
    num_matched: int = 0


# ============================================================================
# Mocap 데이터 로더
# ============================================================================
LOWER_LIMB_KEYS = [
    "left_hip", "left_knee", "left_ankle",
    "right_hip", "right_knee", "right_ankle",
    "left_heel", "left_toe", "right_heel", "right_toe",
]


class MocapDataset:
    """Mocap Ground Truth 데이터셋"""

    def __init__(self):
        self.frames = []
        self.source_format = ""
        self.capture_fps = 0

    def load_csv(self, filepath):
        """
        CSV 형식 로드.

        기대 형식:
            timestamp,left_hip_x,left_hip_y,left_hip_z,left_knee_x,...
            0.000,0.123,0.456,0.789,0.234,...

        keypoint 이름은 헤더의 _x, _y, _z 접미사에서 추출합니다.
        관절 각도 열은 _angle 접미사로 인식합니다.
        """
        self.source_format = "csv"
        self.frames = []

        with open(filepath, 'r') as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames

            # keypoint 이름 추출
            kp_names = set()
            angle_names = set()
            for h in headers:
                if h.endswith('_x'):
                    kp_names.add(h[:-2])
                elif h.endswith('_angle'):
                    angle_names.add(h)

            for row in reader:
                frame = MocapFrame(
                    timestamp=float(row.get('timestamp', 0)),
                )

                # 3D keypoints
                for kp_name in kp_names:
                    x_key = f"{kp_name}_x"
                    y_key = f"{kp_name}_y"
                    z_key = f"{kp_name}_z"
                    if x_key in row and y_key in row and z_key in row:
                        try:
                            x = float(row[x_key])
                            y = float(row[y_key])
                            z = float(row[z_key])
                            if np.isfinite(x) and np.isfinite(y) and np.isfinite(z):
                                frame.keypoints_3d[kp_name] = (x, y, z)
                        except (ValueError, TypeError):
                            pass

                # 관절 각도
                for angle_col in angle_names:
                    try:
                        frame.joint_angles[angle_col] = float(row[angle_col])
                    except (ValueError, TypeError):
                        pass

                self.frames.append(frame)

        if self.frames:
            dt_list = []
            for i in range(1, len(self.frames)):
                dt = self.frames[i].timestamp - self.frames[i-1].timestamp
                if dt > 0:
                    dt_list.append(dt)
            if dt_list:
                self.capture_fps = 1.0 / np.mean(dt_list)

        print(f"  Mocap 로드: {len(self.frames)} frames, {self.capture_fps:.1f} Hz")
        return self

    def load_c3d(self, filepath):
        """C3D 형식 (Vicon, OptiTrack). 추후 구현."""
        raise NotImplementedError(
            "C3D 로더 미구현. c3d 패키지 필요: pip install c3d\n"
            "현재는 Mocap 소프트웨어에서 CSV로 export 후 load_csv() 사용하세요.\n"
            "Vicon Nexus: Pipeline > Export > Export ASCII\n"
            "OptiTrack Motive: Edit > Export to CSV"
        )

    def load_bvh(self, filepath):
        """BVH 형식. 추후 구현."""
        raise NotImplementedError(
            "BVH 로더 미구현.\n"
            "현재는 CSV export 사용하세요."
        )

    def get_frame_at_time(self, timestamp):
        """가장 가까운 시간의 프레임 반환"""
        if not self.frames:
            return None
        closest = min(self.frames, key=lambda f: abs(f.timestamp - timestamp))
        return closest


# ============================================================================
# 시간 동기화
# ============================================================================
class TemporalAlignment:
    """카메라-Mocap 시간 동기화"""

    def __init__(self, offset=0.0):
        self.offset = offset  # camera_time + offset = mocap_time

    def set_manual_offset(self, offset):
        """수동 시간 오프셋 설정"""
        self.offset = offset

    def compute_cross_correlation(self, camera_signal, mocap_signal,
                                   camera_timestamps, mocap_timestamps):
        """
        교차 상관을 이용한 자동 시간 동기화.

        두 시계열 신호(예: 무릎 각도)의 cross-correlation으로 최적 offset 탐색.

        Args:
            camera_signal: list of float (예: left_knee_flexion)
            mocap_signal: list of float
            camera_timestamps: list of float
            mocap_timestamps: list of float

        Returns:
            optimal offset (seconds)
        """
        if len(camera_signal) < 10 or len(mocap_signal) < 10:
            return 0.0

        # 공통 시간 범위로 리샘플링 (100Hz)
        dt = 0.01
        t_start = max(camera_timestamps[0], mocap_timestamps[0])
        t_end = min(camera_timestamps[-1], mocap_timestamps[-1])

        if t_end <= t_start:
            return 0.0

        common_t = np.arange(t_start, t_end, dt)
        cam_interp = np.interp(common_t, camera_timestamps, camera_signal)
        moc_interp = np.interp(common_t, mocap_timestamps, mocap_signal)

        # Normalize
        cam_interp = (cam_interp - np.mean(cam_interp)) / (np.std(cam_interp) + 1e-8)
        moc_interp = (moc_interp - np.mean(moc_interp)) / (np.std(moc_interp) + 1e-8)

        # Cross-correlation
        corr = np.correlate(cam_interp, moc_interp, mode='full')
        lag = np.argmax(corr) - len(moc_interp) + 1
        self.offset = lag * dt

        return self.offset

    def camera_to_mocap_time(self, camera_time):
        """카메라 시간 → Mocap 시간"""
        return camera_time + self.offset


# ============================================================================
# 비교 메트릭 계산
# ============================================================================
def compute_mocap_comparison(model_keypoints_3d_per_frame,
                              model_timestamps,
                              mocap_dataset,
                              alignment=None,
                              max_time_diff=0.05):
    """
    모델 출력과 Mocap Ground Truth 비교.

    Args:
        model_keypoints_3d_per_frame: list of dict (name -> (x,y,z))
        model_timestamps: list of float
        mocap_dataset: MocapDataset
        alignment: TemporalAlignment (None이면 offset=0)
        max_time_diff: 매칭 허용 최대 시간 차 (sec)

    Returns:
        ComparisonResult
    """
    if alignment is None:
        alignment = TemporalAlignment(0.0)

    result = ComparisonResult()
    per_kp_errors = {}    # name -> list of errors (mm)
    per_angle_errors = {} # name -> list of errors (deg)
    pck_50_count = 0
    pck_100_count = 0
    total_kp_comparisons = 0

    for i, (model_kps, cam_t) in enumerate(zip(model_keypoints_3d_per_frame, model_timestamps)):
        mocap_t = alignment.camera_to_mocap_time(cam_t)
        mocap_frame = mocap_dataset.get_frame_at_time(mocap_t)

        if mocap_frame is None:
            continue

        time_diff = abs(mocap_frame.timestamp - mocap_t)
        if time_diff > max_time_diff:
            continue

        result.num_matched += 1

        # Per-keypoint 3D 오차
        for kp_name in model_kps:
            if kp_name in mocap_frame.keypoints_3d:
                model_pt = np.array(model_kps[kp_name])
                mocap_pt = np.array(mocap_frame.keypoints_3d[kp_name])
                error_mm = np.linalg.norm(model_pt - mocap_pt) * 1000  # m -> mm

                if kp_name not in per_kp_errors:
                    per_kp_errors[kp_name] = []
                per_kp_errors[kp_name].append(error_mm)

                total_kp_comparisons += 1
                if error_mm < 50:
                    pck_50_count += 1
                if error_mm < 100:
                    pck_100_count += 1

    result.num_frames = len(model_keypoints_3d_per_frame)

    # MPJPE
    if per_kp_errors:
        all_errors = []
        ll_errors = []
        for kp_name, errors in per_kp_errors.items():
            mean_err = float(np.mean(errors))
            result.mpjpe[kp_name] = mean_err
            all_errors.extend(errors)
            if kp_name in LOWER_LIMB_KEYS:
                ll_errors.extend(errors)

        result.mpjpe_overall = float(np.mean(all_errors)) if all_errors else 0.0
        result.mpjpe_lower_limb = float(np.mean(ll_errors)) if ll_errors else 0.0

    # PCK
    if total_kp_comparisons > 0:
        result.pck_50mm = pck_50_count / total_kp_comparisons * 100
        result.pck_100mm = pck_100_count / total_kp_comparisons * 100

    return result


def print_comparison_report(result):
    """비교 결과 터미널 출력"""
    print()
    print("=" * 60)
    print("  Mocap vs Model Comparison")
    print("=" * 60)
    print(f"  총 프레임: {result.num_frames}")
    print(f"  매칭 프레임: {result.num_matched}")
    print()
    print(f"  MPJPE (전체): {result.mpjpe_overall:.1f} mm")
    print(f"  MPJPE (하체): {result.mpjpe_lower_limb:.1f} mm")
    print(f"  PCK@50mm: {result.pck_50mm:.1f}%")
    print(f"  PCK@100mm: {result.pck_100mm:.1f}%")

    if result.mpjpe:
        print()
        print(f"  Per-keypoint MPJPE (mm):")
        for kp, err in sorted(result.mpjpe.items(), key=lambda x: -x[1]):
            is_ll = " *" if kp in LOWER_LIMB_KEYS else ""
            print(f"    {kp:<25} {err:>7.1f}{is_ll}")

    if result.joint_angle_mae:
        print()
        print(f"  Joint Angle MAE (degrees):")
        for ja, mae in sorted(result.joint_angle_mae.items()):
            print(f"    {ja:<30} {mae:>6.1f}")


# ============================================================================
# Main
# ============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Mocap 비교 검증")
    parser.add_argument("--mocap", type=str, help="Mocap CSV 파일 경로")
    parser.add_argument("--demo", action="store_true", help="합성 데이터로 데모 실행")
    args = parser.parse_args()

    if args.demo:
        print("  합성 데이터로 데모 실행...")

        # 합성 Mocap 데이터
        dataset = MocapDataset()
        for i in range(100):
            t = i * 0.01
            frame = MocapFrame(
                timestamp=t,
                keypoints_3d={
                    "left_hip": (0.1, 0.9 + np.sin(t) * 0.01, 1.5),
                    "left_knee": (0.1, 0.5 + np.sin(t * 2) * 0.05, 1.5),
                    "left_ankle": (0.1, 0.1, 1.5),
                    "right_hip": (-0.1, 0.9 + np.sin(t) * 0.01, 1.5),
                    "right_knee": (-0.1, 0.5 + np.sin(t * 2) * 0.05, 1.5),
                    "right_ankle": (-0.1, 0.1, 1.5),
                },
            )
            dataset.frames.append(frame)

        # 합성 모델 출력 (약간의 노이즈 추가)
        model_kps = []
        model_ts = []
        for frame in dataset.frames:
            noisy = {}
            for k, v in frame.keypoints_3d.items():
                noise = np.random.randn(3) * 0.02  # 20mm 노이즈
                noisy[k] = tuple(np.array(v) + noise)
            model_kps.append(noisy)
            model_ts.append(frame.timestamp)

        result = compute_mocap_comparison(model_kps, model_ts, dataset)
        print_comparison_report(result)

    elif args.mocap:
        dataset = MocapDataset()
        dataset.load_csv(args.mocap)
        print(f"  로드 완료: {len(dataset.frames)} frames")
    else:
        print("  --mocap 또는 --demo 인자를 지정하세요.")
        print("  자세한 사용법: python3 mocap_validation.py --help")
