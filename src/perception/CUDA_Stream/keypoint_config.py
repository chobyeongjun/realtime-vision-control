"""Keypoint schema + skeleton definitions.

Supports two built-in presets:

  * ``coco17``  — standard COCO 17 keypoints (Ultralytics YOLOv8/26 pose default)
  * ``lowlimb6`` — 6-point lower-body subset used by the H-Walker
                  custom-trained model (hip/knee/ankle × left/right).
                  This is the PRIMARY schema for the rehab workflow.

Add new presets here (not in pipeline.py) so shm_publisher, postprocess,
and the control-loop contract all stay in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


@dataclass(frozen=True)
class KeypointSchema:
    """Complete specification of a pose-head output schema."""

    name: str
    keypoints: Tuple[str, ...]            # index → joint name
    segments: Tuple[Tuple[str, str], ...]  # bone (parent, child) pairs
    symmetric_pairs: Tuple[Tuple[str, str], ...]  # for symmetry checks
    # joint-angle triplets (parent, vertex, child): used by the control
    # loop's impedance model. NB: hip flexion is defined SYMMETRICALLY —
    # both left and right hip use the contralateral knee as the reference.
    # This intentionally differs from mainline `joint_angles.py:89-90`
    # which has a known asymmetry bug (skiro-learnings / P0-1).
    angles: Tuple[Tuple[str, str, str, str], ...]

    @property
    def num_keypoints(self) -> int:
        return len(self.keypoints)

    def index(self, name: str) -> int:
        return self.keypoints.index(name)


COCO17 = KeypointSchema(
    name="coco17",
    keypoints=(
        "nose",
        "left_eye", "right_eye",
        "left_ear", "right_ear",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ),
    segments=(
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
        ("left_shoulder", "left_hip"),
        ("right_shoulder", "right_hip"),
        ("left_hip", "left_knee"),
        ("right_hip", "right_knee"),
        ("left_knee", "left_ankle"),
        ("right_knee", "right_ankle"),
        ("left_shoulder", "left_elbow"),
        ("right_shoulder", "right_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_elbow", "right_wrist"),
    ),
    symmetric_pairs=(
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    ),
    angles=(
        # P0-1 FIX: hip flexion uses contralateral KNEE (not hip) as reference —
        # both sides symmetric. Mainline's left/right_hip cross-reference
        # gives asymmetric angles for identical poses.
        ("left_hip_flexion",  "left_knee",  "left_hip",  "right_knee"),
        ("right_hip_flexion", "right_knee", "right_hip", "left_knee"),
        ("left_knee_flexion",  "left_hip",  "left_knee",  "left_ankle"),
        ("right_knee_flexion", "right_hip", "right_knee", "right_ankle"),
    ),
)


LOWLIMB6 = KeypointSchema(
    name="lowlimb6",
    keypoints=(
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    ),
    segments=(
        ("left_hip", "right_hip"),      # pelvis width
        ("left_hip", "left_knee"),      # left thigh
        ("right_hip", "right_knee"),    # right thigh
        ("left_knee", "left_ankle"),    # left shank
        ("right_knee", "right_ankle"),  # right shank
    ),
    symmetric_pairs=(
        ("left_hip", "right_hip"),
        ("left_knee", "right_knee"),
        ("left_ankle", "right_ankle"),
    ),
    angles=(
        # hip flexion uses contralateral KNEE (symmetric formulation — P0-1 fix)
        ("left_hip_flexion",  "left_knee",  "left_hip",  "right_knee"),
        ("right_hip_flexion", "right_knee", "right_hip", "left_knee"),
        ("left_knee_flexion",  "left_hip",  "left_knee",  "left_ankle"),
        ("right_knee_flexion", "right_hip", "right_knee", "right_ankle"),
    ),
)


PRESETS: Dict[str, KeypointSchema] = {
    COCO17.name: COCO17,
    LOWLIMB6.name: LOWLIMB6,
}


def get_schema(name: str) -> KeypointSchema:
    if name not in PRESETS:
        raise ValueError(f"unknown keypoint schema {name!r}; choose from {list(PRESETS)}")
    return PRESETS[name]
