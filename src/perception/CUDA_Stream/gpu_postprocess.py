"""GPU postprocess — YOLOv8/26-pose decode → 3D (depth sample) → optional filter.

Schema-aware: supports ``coco17`` (17 kpts, last-dim 56) and the H-Walker
custom ``lowlimb6`` (6 kpts, last-dim 23) via :mod:`keypoint_config`.

IMPORTANT defaults (from skiro-learnings / user feedback):

  * ``use_filter=False`` — OneEuro was observed to *suppress* detection on
    the custom lower-limb model; initial low-confidence keypoints get
    zero'd and the filter remembers that zero, dragging subsequent real
    detections toward 0. **Leave off by default.**
  * Bone-length constraint lives in :mod:`constraints` — **do NOT enable**
    during calibration / initial warm-up (positive feedback loop observed
    with yolo26s-lower6-v2: left-thigh collapsed to 9px vs 31px right).

Graph-capture safe: the ``int(...argmax.item())`` D2H in this module means
**do NOT** include the postprocess step in ``cuda_graph.GraphedStep``;
graph capture targets only preproc + infer.

Output last-dim layout for YOLOv8/26 pose head:
    raw: (1, N, 5 + K*3), where K = num_keypoints.
    last_dim = 5 + K*3  →  coco17 ⇒ 56,  lowlimb6 ⇒ 23.
    Legacy detect+cls layout (last_dim = 6 + K*3) is also auto-detected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .gpu_preprocess import LetterboxParams
from .keypoint_config import COCO17, KeypointSchema, get_schema


@dataclass
class PoseResult:
    kpts_2d_px: torch.Tensor   # (K, 2) original-image pixels, CUDA
    kpts_3d_m: torch.Tensor    # (K, 3) meters in camera frame, CUDA
    kpt_conf: torch.Tensor     # (K,) float, CUDA
    box_conf: float
    valid: bool                # False when no person was detected above threshold
    depth_invalid_ratio: float  # 0..1 — fraction of keypoints whose depth was bad


class OneEuroFilter1D:
    """Vectorized OneEuro filter (last-axis samples, flexible shape).

    NB: invalid samples (caller passes NaN) are skipped — this prevents
    the "zero sink" failure mode we hit with the lower-limb model.
    """

    def __init__(
        self,
        shape: Tuple[int, ...],
        min_cutoff: float = 2.5,   # higher than old 1.0 → less lag
        beta: float = 0.3,         # stronger adaptation for fast walking
        d_cutoff: float = 1.0,
        device: Optional[torch.device] = None,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.device = device or torch.device("cuda:0")
        self.x_prev: Optional[torch.Tensor] = None
        self.dx_prev = torch.zeros(shape, device=self.device)
        self.t_prev: Optional[float] = None

    def reset(self) -> None:
        self.x_prev = None
        self.dx_prev.zero_()
        self.t_prev = None

    @staticmethod
    def _alpha(cutoff, dt: float):
        tau = 1.0 / (2.0 * 3.141592653589793 * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x: torch.Tensor, t_s: float) -> torch.Tensor:
        if self.x_prev is None or self.t_prev is None:
            self.x_prev = x.clone()
            self.t_prev = t_s
            return x
        dt = max(t_s - self.t_prev, 1e-3)
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(self.d_cutoff, dt)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev
        cutoff = self.min_cutoff + self.beta * dx_hat.abs()
        alpha = self._alpha(cutoff, dt)
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t_s
        return x_hat


class GpuPostprocessor:
    """Decode pose head output, sample depth, optionally smooth."""

    def __init__(
        self,
        schema: KeypointSchema = COCO17,
        conf_threshold: float = 0.35,
        kpt_conf_threshold: float = 0.30,
        depth_patch: int = 3,           # sample a 3×3 patch and take nanmedian
        occluded_count: Optional[int] = None,  # None = max(2, K//3) — relative to schema
        device: Optional[torch.device] = None,
        use_filter: bool = False,        # ⚠ skiro-learnings: keep OFF by default
        filter_params: Optional[dict] = None,
    ) -> None:
        self.schema = schema
        self.K = schema.num_keypoints
        self.conf_threshold = conf_threshold
        self.kpt_conf_threshold = kpt_conf_threshold
        self.depth_patch = max(1, int(depth_patch))
        # For COCO17 the default of 2 (12%) is strict enough, but for
        # lowlimb6 requiring ≥2 occlusions is 33% of all joints — too
        # harsh for a gait rehab workflow where a single leg occlusion
        # is common. Use max(2, K // 3) so:
        #   K=6  → 2 (same absolute bar, but better calibrated)
        #   K=17 → 5 (~30% — less chatty than 2/17 = 12%)
        self.occluded_count = (
            occluded_count if occluded_count is not None else max(2, self.K // 3)
        )
        self.device = device or torch.device("cuda:0")
        self.use_filter = use_filter
        self._filter: Optional[OneEuroFilter1D] = None
        if use_filter:
            fp = filter_params or {}
            self._filter = OneEuroFilter1D(
                (self.K, 3), device=self.device, **fp
            )

    @classmethod
    def from_schema_name(cls, name: str, **kwargs) -> "GpuPostprocessor":
        return cls(schema=get_schema(name), **kwargs)

    # ------------------------------------------------------------------
    def __call__(
        self,
        raw_output: torch.Tensor,
        depth_hw: Optional[torch.Tensor],
        lb_params: LetterboxParams,
        calibration: dict,
        stream: torch.cuda.Stream,
        ts_s: float,
    ) -> PoseResult:
        if raw_output.ndim != 3 or raw_output.shape[0] != 1:
            raise ValueError(
                f"raw_output must be (1,N,L), got {tuple(raw_output.shape)}"
            )
        last = raw_output.shape[2]
        expected_pose = 5 + self.K * 3
        expected_legacy = 6 + self.K * 3
        if last == expected_pose:
            kpt_offset = 5
        elif last == expected_legacy:
            kpt_offset = 6
        else:
            raise ValueError(
                f"raw_output last dim {last} doesn't match schema {self.schema.name} "
                f"(expected {expected_pose} pose or {expected_legacy} legacy)"
            )

        K = self.K
        with torch.cuda.stream(stream):
            det = raw_output[0]  # (N, last)
            conf = det[:, 4]
            # Combine the first two D2H syncs: one .cpu() call is ~1µs,
            # two .item() calls each force their own sync. We need both
            # the argmax index and its value — do it once and reuse.
            best = torch.argmax(conf)
            best_tuple = torch.stack([best.float(), conf[best]]).cpu()
            best_idx = int(best_tuple[0].item())
            box_conf = float(best_tuple[1].item())

            if box_conf < self.conf_threshold:
                zeros2 = torch.zeros((K, 2), device=self.device)
                zeros3 = torch.zeros((K, 3), device=self.device)
                zerosc = torch.zeros((K,), device=self.device)
                return PoseResult(
                    kpts_2d_px=zeros2, kpts_3d_m=zeros3, kpt_conf=zerosc,
                    box_conf=box_conf, valid=False, depth_invalid_ratio=1.0,
                )

            kpts = det[best_idx, kpt_offset : kpt_offset + K * 3].view(K, 3)
            xy_letter = kpts[:, :2]
            kp_conf = kpts[:, 2]

            xy_src = xy_letter.clone()
            xy_src[:, 0] = (xy_src[:, 0] - lb_params.pad_x) / lb_params.scale
            xy_src[:, 1] = (xy_src[:, 1] - lb_params.pad_y) / lb_params.scale
            xy_src[:, 0].clamp_(0, lb_params.src_w - 1)
            xy_src[:, 1].clamp_(0, lb_params.src_h - 1)

            if depth_hw is not None:
                kpts_3d, depth_invalid = self._lift_to_3d(xy_src, depth_hw, calibration, kp_conf)
            else:
                kpts_3d = torch.zeros((K, 3), device=self.device)
                kpts_3d[:, :2] = xy_src
                depth_invalid = 1.0

            # Occlusion handling — occlusion threshold scales with schema.
            # We count low-confidence keypoints on-device then do ONE
            # .item() here (not two separate argmax/item calls) to
            # minimize D2H sync in the hot path.
            num_low_conf = int((kp_conf < self.kpt_conf_threshold).sum().item())
            occluded = num_low_conf >= self.occluded_count

            if self._filter is not None and not occluded:
                kpts_3d = self._filter(kpts_3d, ts_s)

            return PoseResult(
                kpts_2d_px=xy_src,
                kpts_3d_m=kpts_3d,
                kpt_conf=kp_conf,
                box_conf=box_conf,
                valid=not occluded,
                depth_invalid_ratio=depth_invalid,
            )

    # ------------------------------------------------------------------
    def _lift_to_3d(
        self,
        xy: torch.Tensor,
        depth_hw: torch.Tensor,
        calib: dict,
        kp_conf: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        H, W = depth_hw.shape
        K = xy.shape[0]
        r = self.depth_patch // 2
        u = xy[:, 0].clamp(r, W - r - 1).long()
        v = xy[:, 1].clamp(r, H - r - 1).long()

        patches = []
        for dv in range(-r, r + 1):
            for du in range(-r, r + 1):
                patches.append(depth_hw[v + dv, u + du])
        patches = torch.stack(patches, dim=0)  # (patch², K)

        valid = torch.isfinite(patches) & (patches > 0.0)
        patches = torch.where(valid, patches, torch.full_like(patches, float("nan")))
        z = torch.nanmedian(patches, dim=0).values  # (K,)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)

        # per-joint invalid ratio — useful for watchdog
        invalid_ratio = float((~valid).float().mean().item())

        z = torch.where(kp_conf >= self.kpt_conf_threshold, z, torch.zeros_like(z))

        fx, fy = calib["fx"], calib["fy"]
        cx, cy = calib["cx"], calib["cy"]
        x3 = (xy[:, 0] - cx) * z / fx
        y3 = (xy[:, 1] - cy) * z / fy
        xyz_cam = torch.stack([x3, y3, z], dim=1)  # (K, 3) camera frame

        # If the bridge provided an IMU-based rotation, project into a
        # gravity-aligned world frame. This matches mainline Method B
        # (ZEDIMUWorldFrame) so sagittal view stays vertical even when
        # the camera is tilted ~32° on the walker mount.
        R = calib.get("R_world_from_cam")
        if R is not None:
            # R: (3, 3) on same device; xyz_cam @ R.T == (R @ xyz.T).T
            xyz_world = xyz_cam @ R.t()
            return xyz_world, invalid_ratio
        return xyz_cam, invalid_ratio
