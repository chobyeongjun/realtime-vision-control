"""
관절 각도 계산 모듈
==================
pose_processor_node.cpp의 JointAngleCalculator 로직을 Python으로 미러링.
3D 및 2D keypoint에서 하체 관절 각도를 계산합니다.

참조: pose_processor_node.cpp:108-118 (computeAngle)
      gait_phase_detector.hpp:298-335 (hip/knee/ankle angle)
"""

import numpy as np


# ============================================================================
# 기본 각도 계산
# ============================================================================
def compute_angle_3d(parent, joint, child):
    """
    3개의 3D 포인트로 관절 각도 계산.
    pose_processor_node.cpp:108-118의 JointAngleCalculator::computeAngle과 동일.

    Args:
        parent: (x, y, z) 부모 관절 (예: hip)
        joint:  (x, y, z) 현재 관절 (예: knee)
        child:  (x, y, z) 자식 관절 (예: ankle)

    Returns:
        angle in radians
    """
    p = np.array(parent, dtype=np.float64)
    j = np.array(joint, dtype=np.float64)
    c = np.array(child, dtype=np.float64)

    v1 = p - j
    v2 = c - j

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0

    v1 = v1 / n1
    v2 = v2 / n2

    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.arccos(cos_angle))


def compute_angle_2d(parent, joint, child):
    """
    3개의 2D 포인트로 관절 각도 계산 (depth 없을 때 근사).

    Args:
        parent: (x, y) pixel 좌표
        joint:  (x, y) pixel 좌표
        child:  (x, y) pixel 좌표

    Returns:
        angle in radians
    """
    p = np.array(parent[:2], dtype=np.float64)
    j = np.array(joint[:2], dtype=np.float64)
    c = np.array(child[:2], dtype=np.float64)

    v1 = p - j
    v2 = c - j

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 < 1e-9 or n2 < 1e-9:
        return 0.0

    v1 = v1 / n1
    v2 = v2 / n2

    cos_angle = np.clip(np.dot(v1, v2), -1.0, 1.0)
    return float(np.arccos(cos_angle))


# ============================================================================
# 하체 관절 각도 정의
# ============================================================================
# (angle_name, parent_keypoint, joint_keypoint, child_keypoint)
LOWER_LIMB_ANGLE_DEFS = [
    ("left_knee_flexion",  "left_hip",   "left_knee",  "left_ankle"),
    ("right_knee_flexion", "right_hip",  "right_knee", "right_ankle"),
    ("left_hip_flexion",   "left_knee",  "left_hip",   "right_hip"),
    ("right_hip_flexion",  "right_knee", "right_hip",  "left_hip"),
    # ankle 각도는 heel/toe가 있는 모델에서만 계산 가능
    ("left_ankle_dorsiflexion",  "left_knee",  "left_ankle", "left_toe"),
    ("right_ankle_dorsiflexion", "right_knee", "right_ankle", "right_toe"),
]


def compute_lower_limb_angles(pose_result, use_3d=True):
    """
    PoseResult에서 하체 관절 각도를 일괄 계산.

    Args:
        pose_result: PoseResult 객체 (keypoints_3d, keypoints_2d, confidences)
        use_3d: True면 keypoints_3d 사용, False면 keypoints_2d 사용

    Returns:
        dict: angle_name -> angle_degrees
    """
    keypoints = pose_result.keypoints_3d if use_3d else pose_result.keypoints_2d
    compute_fn = compute_angle_3d if use_3d else compute_angle_2d

    angles = {}
    for angle_name, parent_name, joint_name, child_name in LOWER_LIMB_ANGLE_DEFS:
        if (parent_name in keypoints and
            joint_name in keypoints and
            child_name in keypoints):
            # confidence 체크
            min_conf = min(
                pose_result.confidences.get(parent_name, 0),
                pose_result.confidences.get(joint_name, 0),
                pose_result.confidences.get(child_name, 0),
            )
            if min_conf < 0.3:
                continue

            angle_rad = compute_fn(
                keypoints[parent_name],
                keypoints[joint_name],
                keypoints[child_name],
            )

            # knee flexion: 180 - angle (완전 신전 = 0도)
            # gait_phase_detector.hpp:313-323 참조
            if "knee_flexion" in angle_name:
                angle_deg = (np.pi - angle_rad) * 180.0 / np.pi
            # ankle dorsiflexion: angle - 90도 (neutral = 0도)
            # gait_phase_detector.hpp:325-335 참조
            elif "ankle_dorsiflexion" in angle_name:
                angle_deg = (angle_rad - np.pi / 2.0) * 180.0 / np.pi
            # hip flexion: raw angle
            else:
                angle_deg = angle_rad * 180.0 / np.pi

            angles[angle_name] = angle_deg

    return angles
