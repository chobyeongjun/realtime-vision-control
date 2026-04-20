"""Constraint tests — CPU torch, no CUDA needed."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from perception.CUDA_Stream.constraints import (
    BoneLengthConstraint, ConstraintStack, JointVelocityBound,
)
from perception.CUDA_Stream.keypoint_config import LOWLIMB6


DEVICE = torch.device("cpu")


def _rest_pose():
    """Symmetric standing pose used as calibration baseline."""
    pts = torch.zeros((6, 3), device=DEVICE)
    pts[LOWLIMB6.index("left_hip")] = torch.tensor([+0.10, 0.0, 0.0])
    pts[LOWLIMB6.index("right_hip")] = torch.tensor([-0.10, 0.0, 0.0])
    pts[LOWLIMB6.index("left_knee")] = torch.tensor([+0.10, -0.40, 0.0])
    pts[LOWLIMB6.index("right_knee")] = torch.tensor([-0.10, -0.40, 0.0])
    pts[LOWLIMB6.index("left_ankle")] = torch.tensor([+0.10, -0.80, 0.0])
    pts[LOWLIMB6.index("right_ankle")] = torch.tensor([-0.10, -0.80, 0.0])
    return pts


def test_bone_length_not_armed_by_default():
    c = BoneLengthConstraint(LOWLIMB6, calibration_frames=3, max_calibration_std_m=1.0, device=DEVICE)
    assert not c.armed
    _, dec = c.apply(_rest_pose())
    assert dec.accept and dec.reason == "not_armed"


def test_bone_length_arms_after_calibration():
    c = BoneLengthConstraint(LOWLIMB6, calibration_frames=3, max_calibration_std_m=1.0, device=DEVICE)
    for _ in range(3):
        c.observe(_rest_pose())
    assert c.armed
    _, dec = c.apply(_rest_pose())
    assert dec.accept and dec.reason == "ok"


def test_bone_length_rejects_collapsed_thigh():
    c = BoneLengthConstraint(
        LOWLIMB6, tolerance=0.2, calibration_frames=3,
        max_calibration_std_m=1.0, device=DEVICE,
    )
    for _ in range(3):
        c.observe(_rest_pose())
    bad = _rest_pose().clone()
    # collapse left knee upward so left thigh length drops ~75%
    bad[LOWLIMB6.index("left_knee")] = torch.tensor([+0.10, -0.10, 0.0])
    _, dec = c.apply(bad)
    assert not dec.accept
    assert dec.reason == "bone_length_out_of_range"


def test_bone_length_rejects_noisy_calibration():
    """Mainline P0-3 parity: std > 10mm during calibration must reject ref."""
    c = BoneLengthConstraint(
        LOWLIMB6, tolerance=0.2, calibration_frames=4,
        max_calibration_std_m=0.010, device=DEVICE,
    )
    # Feed poses with ~50mm random jitter per frame — way above 10mm
    torch.manual_seed(0)
    for i in range(4):
        p = _rest_pose().clone()
        p[LOWLIMB6.index("left_knee"), 1] -= 0.05 * i
        c.observe(p)
    # Noisy → not armed
    assert not c.armed
    # Subsequent quiet frames should eventually arm
    for _ in range(5):
        c.observe(_rest_pose())
    assert c.armed


def test_bone_length_clamp_mode_shrinks_error():
    c = BoneLengthConstraint(
        LOWLIMB6, tolerance=0.2, calibration_frames=3, clamp=True,
        max_calibration_std_m=1.0, device=DEVICE,
    )
    for _ in range(3):
        c.observe(_rest_pose())
    bad = _rest_pose().clone()
    bad[LOWLIMB6.index("left_knee")] = torch.tensor([+0.10, -0.10, 0.0])
    corrected, dec = c.apply(bad)
    # clamp mode accepts but adjusts child joint
    assert dec.accept
    assert dec.reason == "bone_length_clamped"
    corr_len = float((corrected[LOWLIMB6.index("left_knee")] - corrected[LOWLIMB6.index("left_hip")]).norm())
    assert 0.3 < corr_len < 0.5  # near the 0.40 reference


def test_joint_velocity_first_frame_always_accepts():
    v = JointVelocityBound(max_velocity_mps=1.0, device=DEVICE)
    _, dec = v.apply(_rest_pose(), ts_s=0.0)
    assert dec.accept and dec.reason == "first_frame"


def test_joint_velocity_rejects_teleport():
    v = JointVelocityBound(max_velocity_mps=2.0, device=DEVICE)
    v.apply(_rest_pose(), ts_s=0.0)
    bad = _rest_pose().clone()
    # 1 m jump on right ankle within 10ms → 100 m/s
    bad[LOWLIMB6.index("right_ankle")] += torch.tensor([1.0, 0.0, 0.0])
    _, dec = v.apply(bad, ts_s=0.01)
    assert not dec.accept
    assert dec.reason == "joint_velocity_exceeded"
    # next frame is compared to the ORIGINAL, not the rejected one
    _, dec2 = v.apply(_rest_pose(), ts_s=0.02)
    assert dec2.accept


def test_stack_default_is_empty_and_always_accepts():
    stack = ConstraintStack()
    _, dec = stack.apply(_rest_pose(), ts_s=0.0)
    assert dec.accept


def test_stack_combined_bone_and_velocity():
    stack = ConstraintStack(
        bone_length=BoneLengthConstraint(
            LOWLIMB6, calibration_frames=3,
            max_calibration_std_m=1.0, device=DEVICE,
        ),
        joint_velocity=JointVelocityBound(max_velocity_mps=5.0, device=DEVICE),
    )
    for i in range(3):
        stack.observe(_rest_pose())
    for i in range(3):
        _, dec = stack.apply(_rest_pose(), ts_s=i * 0.01)
        assert dec.accept
    # teleport
    bad = _rest_pose().clone()
    bad[LOWLIMB6.index("right_ankle")] += torch.tensor([2.0, 0.0, 0.0])
    _, dec = stack.apply(bad, ts_s=0.04)
    assert not dec.accept
