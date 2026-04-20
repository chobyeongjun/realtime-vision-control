"""Schema + symmetric joint-angle sanity tests.

These run on CPU (torch CPU tensors) — no CUDA required.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from perception.CUDA_Stream.joint_angles_v2 import (
    check_symmetry, compute_angles,
)
from perception.CUDA_Stream.keypoint_config import (
    COCO17, LOWLIMB6, PRESETS, get_schema,
)


def test_presets_registered():
    assert "coco17" in PRESETS
    assert "lowlimb6" in PRESETS
    assert PRESETS["coco17"] is COCO17
    assert PRESETS["lowlimb6"] is LOWLIMB6


def test_get_schema_unknown_raises():
    with pytest.raises(ValueError):
        get_schema("nope")


def test_coco17_has_17_keypoints():
    assert COCO17.num_keypoints == 17


def test_lowlimb6_has_6_keypoints():
    assert LOWLIMB6.num_keypoints == 6
    names = set(LOWLIMB6.keypoints)
    assert names == {
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "left_ankle", "right_ankle",
    }


def test_symmetric_hip_flexion_matches_contralateral_knee():
    """P0-1 fix: both schemas must use contralateral KNEE (not hip)."""
    for schema in (COCO17, LOWLIMB6):
        by_name = {a[0]: a for a in schema.angles}
        l = by_name["left_hip_flexion"]
        r = by_name["right_hip_flexion"]
        # (name, a, b, c) — c is the reference joint
        assert l[3] == "right_knee", f"{schema.name}: left hip flexion must reference right_knee"
        assert r[3] == "left_knee", f"{schema.name}: right hip flexion must reference left_knee"


def test_symmetry_check_under_mirrored_pose():
    """Synthetic mirrored pose must give left/right angles within 1°."""
    for schema in (COCO17, LOWLIMB6):
        report = check_symmetry(schema, tol=1.0)
        assert report.ok(tol=1.0), (
            f"{schema.name}: max asymmetry {report.max_asymmetry_deg:.2f}°"
        )


def test_bone_segments_are_valid_indices():
    for schema in (COCO17, LOWLIMB6):
        for parent, child in schema.segments:
            assert parent in schema.keypoints
            assert child in schema.keypoints


def test_compute_angles_on_identity_pose_returns_finite():
    """Keypoints stacked on top of each other are degenerate — compute_angles
    must return NaN rather than raise."""
    pts = torch.zeros((LOWLIMB6.num_keypoints, 3))
    angles = compute_angles(pts, LOWLIMB6)
    # all angles NaN because vector norms are zero
    import math
    for v in angles.values():
        assert math.isnan(v)
