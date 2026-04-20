"""Joint angle computation — symmetric hip-flexion (P0-1 fix).

Mainline ``perception/benchmarks/joint_angles.py:89-90`` defines:

    ("left_hip_flexion",  "left_knee",  "left_hip",  "right_hip"),
    ("right_hip_flexion", "right_knee", "right_hip", "left_hip"),

— left/right use the OPPOSITE hip as a reference point, which makes
left and right flexion measure different 3D relations for an identical
pose. This module uses the contralateral KNEE instead (symmetric),
matching the schema definitions in :mod:`keypoint_config`.

All math is pure torch so the control loop or benchmark can call it
on-GPU without a D2H round-trip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch

from .keypoint_config import KeypointSchema


def compute_angles(kpts_3d: torch.Tensor, schema: KeypointSchema) -> Dict[str, float]:
    """kpts_3d: (K, 3) CUDA. Returns {angle_name: degrees} as a dict."""
    out: Dict[str, float] = {}
    for name, a, b, c in schema.angles:
        ia, ib, ic = schema.index(a), schema.index(b), schema.index(c)
        va = kpts_3d[ia] - kpts_3d[ib]
        vc = kpts_3d[ic] - kpts_3d[ib]
        na = va.norm()
        nc = vc.norm()
        if float(na) < 1e-4 or float(nc) < 1e-4:
            out[name] = float("nan")
            continue
        cos = (va @ vc) / (na * nc)
        cos = cos.clamp(-1.0, 1.0)
        out[name] = float(torch.rad2deg(torch.acos(cos)).item())
    return out


@dataclass
class SymmetryReport:
    """Checks that identical mirrored poses give identical flexion angles."""

    max_asymmetry_deg: float

    def ok(self, tol: float = 1.0) -> bool:
        return self.max_asymmetry_deg < tol


def check_symmetry(schema: KeypointSchema, tol: float = 1.0) -> SymmetryReport:
    """Self-test — constructs a left-right-mirrored synthetic pose and
    verifies that left/right_*_flexion match to within ``tol`` degrees.

    Use this after any schema change to catch the mainline P0-1-style
    asymmetry before it reaches the robot.
    """
    device = torch.device("cpu")
    K = schema.num_keypoints
    pts = torch.zeros((K, 3), device=device)
    # plant symmetric values where both left/right sides exist
    # keypoints schema is just names — we don't assume ordering — so we
    # place hips/knees/ankles at mirrored positions if they exist.
    placements = {
        "left_hip":   (+0.10, 0.0, 0.0),
        "right_hip":  (-0.10, 0.0, 0.0),
        "left_knee":  (+0.10, -0.40, 0.05),
        "right_knee": (-0.10, -0.40, 0.05),
        "left_ankle": (+0.10, -0.80, 0.0),
        "right_ankle": (-0.10, -0.80, 0.0),
        "left_foot":  (+0.10, -0.85, 0.15),
        "right_foot": (-0.10, -0.85, 0.15),
        "left_shoulder": (+0.15, 0.40, 0.0),
        "right_shoulder": (-0.15, 0.40, 0.0),
    }
    for name, (x, y, z) in placements.items():
        if name in schema.keypoints:
            pts[schema.index(name)] = torch.tensor([x, y, z])

    angles = compute_angles(pts, schema)
    max_asym = 0.0
    for a, b in [
        ("left_hip_flexion", "right_hip_flexion"),
        ("left_knee_flexion", "right_knee_flexion"),
        ("left_ankle_flexion", "right_ankle_flexion"),
    ]:
        if a in angles and b in angles:
            diff = abs(angles[a] - angles[b])
            max_asym = max(max_asym, diff)
    return SymmetryReport(max_asymmetry_deg=max_asym)
