"""
3D 정확도 메트릭 (Self-Consistency)
====================================
Mocap Ground Truth 없이 사용 가능한 자기 일관성 메트릭.
뼈 길이 안정성, 좌우 대칭성, 해부학적 타당성을 평가합니다.
"""

import numpy as np


# ============================================================================
# 뼈 연결 정의 (3D 길이 측정용)
# ============================================================================
BONE_DEFINITIONS = {
    "left_thigh":  ("left_hip",   "left_knee"),
    "right_thigh": ("right_hip",  "right_knee"),
    "left_shank":  ("left_knee",  "left_ankle"),
    "right_shank": ("right_knee", "right_ankle"),
    "left_foot_heel":  ("left_ankle",  "left_heel"),
    "right_foot_heel": ("right_ankle", "right_heel"),
    "left_foot_toe":   ("left_ankle",  "left_toe"),
    "right_foot_toe":  ("right_ankle", "right_toe"),
    "pelvis_width":    ("left_hip",    "right_hip"),
}

# 정상 성인 뼈 길이 범위 (meters)
ANATOMICAL_RANGES = {
    "left_thigh":  (0.35, 0.55),
    "right_thigh": (0.35, 0.55),
    "left_shank":  (0.30, 0.50),
    "right_shank": (0.30, 0.50),
    "left_foot_heel":  (0.03, 0.15),
    "right_foot_heel": (0.03, 0.15),
    "left_foot_toe":   (0.05, 0.30),
    "right_foot_toe":  (0.05, 0.30),
    "pelvis_width":    (0.20, 0.40),
}

# 좌우 대칭 쌍
SYMMETRY_PAIRS = [
    ("left_thigh", "right_thigh"),
    ("left_shank", "right_shank"),
    ("left_foot_heel", "right_foot_heel"),
    ("left_foot_toe", "right_foot_toe"),
]


def compute_bone_lengths(keypoints_3d):
    """
    3D keypoints에서 뼈 길이를 계산.

    Args:
        keypoints_3d: dict, name -> (x, y, z)

    Returns:
        dict: bone_name -> length_meters
    """
    lengths = {}
    for bone_name, (kp1, kp2) in BONE_DEFINITIONS.items():
        if kp1 in keypoints_3d and kp2 in keypoints_3d:
            p1 = np.array(keypoints_3d[kp1])
            p2 = np.array(keypoints_3d[kp2])
            dist = np.linalg.norm(p2 - p1)
            if np.isfinite(dist) and dist > 0.001:
                lengths[bone_name] = float(dist)
    return lengths


def compute_bone_length_stability(bone_lengths_per_frame):
    """
    프레임 간 뼈 길이 안정성 평가 (CV: Coefficient of Variation).
    낮을수록 안정적 = 더 정확한 3D 추정.

    Args:
        bone_lengths_per_frame: list of dict (compute_bone_lengths 결과)

    Returns:
        dict with keys:
            mean: bone_name -> mean_length
            std:  bone_name -> std_length
            cv:   bone_name -> coefficient_of_variation (std/mean)
    """
    if not bone_lengths_per_frame:
        return {"mean": {}, "std": {}, "cv": {}}

    # 각 뼈별로 길이 모으기
    bone_values = {}
    for frame_lengths in bone_lengths_per_frame:
        for bone_name, length in frame_lengths.items():
            if bone_name not in bone_values:
                bone_values[bone_name] = []
            bone_values[bone_name].append(length)

    result = {"mean": {}, "std": {}, "cv": {}}
    for bone_name, values in bone_values.items():
        if len(values) < 5:
            continue
        arr = np.array(values)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        cv = std / mean if mean > 0.001 else float('inf')
        result["mean"][bone_name] = mean
        result["std"][bone_name] = std
        result["cv"][bone_name] = cv

    return result


def compute_symmetry_score(bone_lengths):
    """
    좌우 대칭성 점수 (단일 프레임).
    정상인에서 좌/우 대퇴골, 경골 길이는 ~5% 이내.

    Args:
        bone_lengths: dict, bone_name -> length

    Returns:
        dict: pair_name -> asymmetry_ratio (|L-R| / avg, 0=완전대칭)
    """
    scores = {}
    for left_name, right_name in SYMMETRY_PAIRS:
        if left_name in bone_lengths and right_name in bone_lengths:
            l_len = bone_lengths[left_name]
            r_len = bone_lengths[right_name]
            avg = (l_len + r_len) / 2.0
            if avg > 0.001:
                asymmetry = abs(l_len - r_len) / avg
                pair_key = left_name.replace("left_", "")
                scores[pair_key] = float(asymmetry)
    return scores


def compute_depth_validity_rate(keypoints_3d, keypoints_2d):
    """
    2D keypoint 중 3D 변환에 성공한 비율.

    Args:
        keypoints_3d: dict
        keypoints_2d: dict

    Returns:
        float: 0~1
    """
    if not keypoints_2d:
        return 0.0
    valid_3d = sum(1 for k in keypoints_2d if k in keypoints_3d)
    return valid_3d / len(keypoints_2d)


def compute_anatomical_plausibility(bone_lengths):
    """
    뼈 길이가 해부학적 범위 내인지 확인.

    Args:
        bone_lengths: dict

    Returns:
        dict: bone_name -> {"in_range": bool, "value": float, "range": (min, max)}
    """
    results = {}
    for bone_name, (min_len, max_len) in ANATOMICAL_RANGES.items():
        if bone_name in bone_lengths:
            val = bone_lengths[bone_name]
            results[bone_name] = {
                "in_range": min_len <= val <= max_len,
                "value": val,
                "range": (min_len, max_len),
            }
    return results


def aggregate_symmetry_scores(symmetry_per_frame):
    """
    여러 프레임의 symmetry score를 집계.

    Args:
        symmetry_per_frame: list of dict (compute_symmetry_score 결과)

    Returns:
        dict: pair_name -> {"mean": float, "std": float}
    """
    if not symmetry_per_frame:
        return {}

    pair_values = {}
    for frame_scores in symmetry_per_frame:
        for pair_name, score in frame_scores.items():
            if pair_name not in pair_values:
                pair_values[pair_name] = []
            pair_values[pair_name].append(score)

    result = {}
    for pair_name, values in pair_values.items():
        arr = np.array(values)
        result[pair_name] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
        }
    return result
