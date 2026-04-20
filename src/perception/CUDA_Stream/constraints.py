"""Optional post-hoc 3D constraints — default OFF.

We ship two constraints that the mainline C++ path provides, but keep
them **opt-in** because prior experiments showed they can form positive
feedback loops with the detector (skiro-learnings:
"SegmentConstraint OFF + smoothing OFF → 인식 정상화",
"yolo26s-lower6-v2 left_thigh 9px vs right 31px — SegmentConstraint
피드백 루프 의심").

Call order matters:
  1. calibrate(...)  — during N frames of quiet standing, collect bone
                       lengths. Constraints stay disarmed.
  2. enable()        — once calibration statistics are stable, flip the
                       switch. Hard-gate invalid frames instead of
                       modifying keypoints whenever possible.

All ops are torch on CUDA — no .cpu() in hot path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import torch

from .keypoint_config import KeypointSchema

LOGGER = logging.getLogger(__name__)


@dataclass
class ConstraintDecision:
    """What the constraint stack decided about this frame."""

    accept: bool
    reason: str = ""
    max_bone_drift: float = 0.0   # m
    max_joint_velocity: float = 0.0  # m/s

    def to_flags(self) -> dict:
        return {
            "accept": self.accept,
            "reason": self.reason,
            "max_bone_drift_m": self.max_bone_drift,
            "max_joint_velocity_mps": self.max_joint_velocity,
        }


class BoneLengthConstraint:
    """Reject or clamp frames whose bone lengths diverge too far.

    Hard-gate mode (default): if any bone drifts > ``tolerance`` × ref_len,
    the whole frame is rejected (return accept=False). The pipeline then
    falls back to the previous good frame — no keypoint is rewritten, so
    no feedback loop is possible.

    Clamp mode (opt-in): the offending child joint is pulled along the
    bone axis toward the reference length. Matches mainline behavior.
    """

    def __init__(
        self,
        schema: KeypointSchema,
        tolerance: float = 0.25,          # ±25% default — looser than C++ to reduce false rejects
        calibration_frames: int = 30,
        max_calibration_std_m: float = 0.010,  # skiro-learnings: ref rejected
                                                 # when per-segment std > 10 mm
                                                 # (mainline P0-3 parity)
        device: Optional[torch.device] = None,
        clamp: bool = False,               # default: hard-gate only
    ) -> None:
        self.schema = schema
        self.tolerance = tolerance
        self.calibration_frames = calibration_frames
        self.max_calibration_std_m = max_calibration_std_m
        self.device = device or torch.device("cuda:0")
        self.clamp = clamp
        self._armed = False
        self._calib_buffer: list[torch.Tensor] = []
        self._ref_lengths: Optional[torch.Tensor] = None  # (S,)
        self._last_calib_std: Optional[torch.Tensor] = None
        # pre-compute parent/child index pairs (on GPU)
        idx = [(schema.index(p), schema.index(c)) for p, c in schema.segments]
        self._parent_idx = torch.tensor([p for p, _ in idx], device=self.device)
        self._child_idx = torch.tensor([c for _, c in idx], device=self.device)

    @property
    def armed(self) -> bool:
        return self._armed

    def reset(self) -> None:
        self._armed = False
        self._calib_buffer.clear()
        self._ref_lengths = None

    def observe(self, kpts_3d: torch.Tensor) -> None:
        """Collect bone lengths during calibration. No-op once armed.

        Parity with mainline P0-3: the reference is accepted only when
        every segment has std ≤ ``max_calibration_std_m`` (default 10 mm).
        If calibration_frames arrive but std is too noisy (subject moved,
        detector was unstable), we RESET the buffer and keep collecting —
        no bad ref ever arms the constraint.
        """
        if self._armed:
            return
        lens = self._bone_lengths(kpts_3d)
        if torch.isfinite(lens).all() and (lens > 1e-3).all():
            self._calib_buffer.append(lens)
        if len(self._calib_buffer) >= self.calibration_frames:
            stacked = torch.stack(self._calib_buffer, dim=0)
            stds = stacked.std(dim=0)
            self._last_calib_std = stds
            max_std = float(stds.max().item())
            if max_std > self.max_calibration_std_m:
                LOGGER.warning(
                    "BoneLengthConstraint calibration rejected: "
                    "max std %.1f mm > %.1f mm — keep subject still and retry",
                    max_std * 1000.0,
                    self.max_calibration_std_m * 1000.0,
                )
                # drop half the buffer so we re-gather but don't start over
                self._calib_buffer = self._calib_buffer[self.calibration_frames // 2:]
                return
            self._ref_lengths = torch.median(stacked, dim=0).values
            self._armed = True
            LOGGER.info(
                "BoneLengthConstraint armed: ref=%s std(mm)=%s",
                [round(v, 3) for v in self._ref_lengths.detach().cpu().tolist()],
                [round(v * 1000.0, 1) for v in stds.detach().cpu().tolist()],
            )

    def apply(
        self, kpts_3d: torch.Tensor
    ) -> tuple[torch.Tensor, ConstraintDecision]:
        """Return (possibly-corrected kpts_3d, decision)."""
        if not self._armed or self._ref_lengths is None:
            return kpts_3d, ConstraintDecision(accept=True, reason="not_armed")

        lens = self._bone_lengths(kpts_3d)
        ratio = lens / (self._ref_lengths + 1e-6)
        drift = (lens - self._ref_lengths).abs()
        max_drift = float(drift.max().item())

        ok = ((ratio > 1.0 - self.tolerance) & (ratio < 1.0 + self.tolerance))
        if bool(ok.all().item()):
            return kpts_3d, ConstraintDecision(
                accept=True, reason="ok", max_bone_drift=max_drift
            )

        if not self.clamp:
            # hard-gate: reject the whole frame; caller falls back to prev
            return kpts_3d, ConstraintDecision(
                accept=False, reason="bone_length_out_of_range",
                max_bone_drift=max_drift,
            )

        # clamp mode — pull child along bone axis toward reference length
        corrected = kpts_3d.clone()
        for i, (p_idx, c_idx) in enumerate(zip(self._parent_idx.tolist(), self._child_idx.tolist())):
            if bool(ok[i].item()):
                continue
            pr = corrected[p_idx]
            ch = corrected[c_idx]
            vec = ch - pr
            cur = float(vec.norm().item())
            if cur < 1e-6:
                continue
            target = float(self._ref_lengths[i].item())
            corrected[c_idx] = pr + vec * (target / cur)
        return corrected, ConstraintDecision(
            accept=True, reason="bone_length_clamped", max_bone_drift=max_drift
        )

    def _bone_lengths(self, kpts_3d: torch.Tensor) -> torch.Tensor:
        parents = kpts_3d.index_select(0, self._parent_idx)
        children = kpts_3d.index_select(0, self._child_idx)
        return torch.linalg.norm(children - parents, dim=1)  # (S,)


class JointVelocityBound:
    """Reject frames with impossible inter-frame joint velocities.

    P0-4: hard gate on Euclidean velocity (m/s) of each joint between
    consecutive accepted frames. Default max = 5 m/s — aggressive human
    walking peaks at ~4 m/s foot velocity, so this catches teleportation
    artifacts (detector flipping left/right, etc.).
    """

    def __init__(
        self,
        max_velocity_mps: float = 5.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.max_velocity_mps = max_velocity_mps
        self.device = device or torch.device("cuda:0")
        self._prev_kpts: Optional[torch.Tensor] = None
        self._prev_ts: Optional[float] = None

    def reset(self) -> None:
        self._prev_kpts = None
        self._prev_ts = None

    def apply(
        self, kpts_3d: torch.Tensor, ts_s: float
    ) -> tuple[torch.Tensor, ConstraintDecision]:
        if self._prev_kpts is None or self._prev_ts is None:
            self._prev_kpts = kpts_3d.clone()
            self._prev_ts = ts_s
            return kpts_3d, ConstraintDecision(accept=True, reason="first_frame")

        dt = max(ts_s - self._prev_ts, 1e-3)
        vel = torch.linalg.norm(kpts_3d - self._prev_kpts, dim=1) / dt  # (K,)
        vmax = float(vel.max().item())
        if vmax > self.max_velocity_mps:
            # reject; do NOT update prev so the next frame is still
            # compared against the last accepted one.
            return kpts_3d, ConstraintDecision(
                accept=False, reason="joint_velocity_exceeded",
                max_joint_velocity=vmax,
            )

        self._prev_kpts = kpts_3d.clone()
        self._prev_ts = ts_s
        return kpts_3d, ConstraintDecision(
            accept=True, reason="ok", max_joint_velocity=vmax
        )


@dataclass
class ConstraintStack:
    """Compose enabled constraints. Default: everything OFF."""

    bone_length: Optional[BoneLengthConstraint] = None
    joint_velocity: Optional[JointVelocityBound] = None

    def observe(self, kpts_3d: torch.Tensor) -> None:
        if self.bone_length is not None:
            self.bone_length.observe(kpts_3d)

    def apply(
        self, kpts_3d: torch.Tensor, ts_s: float
    ) -> tuple[torch.Tensor, ConstraintDecision]:
        current = kpts_3d
        flags = ConstraintDecision(accept=True, reason="ok")

        if self.bone_length is not None:
            current, dec = self.bone_length.apply(current)
            if not dec.accept:
                return current, dec
            flags.max_bone_drift = dec.max_bone_drift

        if self.joint_velocity is not None:
            current, dec = self.joint_velocity.apply(current, ts_s)
            flags.max_joint_velocity = dec.max_joint_velocity
            if not dec.accept:
                return current, dec

        return current, flags

    def reset(self) -> None:
        if self.bone_length is not None:
            self.bone_length.reset()
        if self.joint_velocity is not None:
            self.joint_velocity.reset()
