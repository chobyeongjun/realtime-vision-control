"""GPU-dependent tests — auto-skipped when CUDA is absent (CI / laptop)."""

from __future__ import annotations

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():  # pragma: no cover — CI guard
    pytest.skip("CUDA not available", allow_module_level=True)

from perception.CUDA_Stream.gpu_postprocess import (
    GpuPostprocessor,
    OneEuroFilter1D,
)
from perception.CUDA_Stream.gpu_preprocess import GpuPreprocessor
from perception.CUDA_Stream.stream_manager import StreamManager


DEVICE = torch.device("cuda:0")


def test_stream_manager_builds_four_streams():
    sm = StreamManager(device=DEVICE, pinned_bytes=1024)
    assert set(sm.streams.keys()) == {"capture", "preproc", "infer", "post"}
    # stream_ptr is an int (raw cuda handle)
    for name in sm.streams:
        ptr = sm.stream_ptr(name)
        assert isinstance(ptr, int)


def test_preprocessor_letterbox_roundtrip():
    pre = GpuPreprocessor(imgsz=640, dtype=torch.float32, device=DEVICE)
    H, W = 1200, 1920
    rgb = torch.randint(0, 256, (H, W, 3), dtype=torch.uint8, device=DEVICE)
    stream = torch.cuda.Stream(device=DEVICE)
    out, params = pre(rgb, stream=stream)
    stream.synchronize()
    assert out.shape == (1, 3, 640, 640)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # undo letterbox should map corners back within ±1 px
    xy = torch.tensor([[params.pad_x, params.pad_y]], dtype=torch.float32,
                      device=DEVICE)
    xy_src = pre.undo_letterbox(xy, params)
    assert abs(xy_src[0, 0].item()) < 1.5
    assert abs(xy_src[0, 1].item()) < 1.5


def test_oneeuro_converges_on_steady_signal():
    f = OneEuroFilter1D((4,), min_cutoff=1.0, beta=0.0, device=DEVICE)
    value = torch.ones(4, device=DEVICE) * 3.0
    for i in range(30):
        out = f(value, t_s=i * 0.01)
    assert torch.allclose(out, value, atol=1e-3)


def test_postprocessor_empty_when_low_conf():
    post = GpuPostprocessor(conf_threshold=0.5, device=DEVICE, use_filter=False)
    # fake detection with box_conf=0.1 (below threshold)
    raw = torch.zeros((1, 1, 57), device=DEVICE)
    raw[0, 0, 4] = 0.1
    from perception.CUDA_Stream.gpu_preprocess import LetterboxParams
    lb = LetterboxParams(scale=0.5, pad_x=20, pad_y=0, src_h=1200, src_w=1920, dst=640)
    stream = torch.cuda.Stream(device=DEVICE)
    result = post(raw, None, lb, {"fx": 600, "fy": 600, "cx": 320, "cy": 240}, stream, 0.0)
    stream.synchronize()
    assert not result.valid
    assert result.kpts_2d_px.abs().sum().item() == 0.0


def test_postprocessor_valid_decodes_coco17_pose_layout():
    """Pose head: last dim = 56, keypoints start at offset 5."""
    post = GpuPostprocessor(conf_threshold=0.1, device=DEVICE, use_filter=False)
    raw = torch.zeros((1, 2, 56), device=DEVICE)
    raw[0, 0, 4] = 0.9  # box conf
    raw[0, 1, 4] = 0.2
    # 17 keypoints at (320, 240), conf=0.8 — offset 5 for pose head
    for k in range(17):
        raw[0, 0, 5 + k * 3 + 0] = 320.0
        raw[0, 0, 5 + k * 3 + 1] = 240.0
        raw[0, 0, 5 + k * 3 + 2] = 0.8
    from perception.CUDA_Stream.gpu_preprocess import LetterboxParams
    lb = LetterboxParams(scale=1.0, pad_x=0, pad_y=0, src_h=480, src_w=640, dst=640)
    stream = torch.cuda.Stream(device=DEVICE)
    result = post(raw, None, lb, {"fx": 600, "fy": 600, "cx": 320, "cy": 240}, stream, 0.0)
    stream.synchronize()
    assert result.valid
    assert result.kpts_2d_px.shape == (17, 2)
    assert math.isclose(result.kpts_2d_px[0, 0].item(), 320.0, abs_tol=0.5)
    assert math.isclose(result.kpts_2d_px[0, 1].item(), 240.0, abs_tol=0.5)


def test_postprocessor_legacy_57_ch_layout():
    """Legacy detect-head layout (last dim = 57) — keypoints at offset 6."""
    post = GpuPostprocessor(conf_threshold=0.1, device=DEVICE, use_filter=False)
    raw = torch.zeros((1, 1, 57), device=DEVICE)
    raw[0, 0, 4] = 0.9
    for k in range(17):
        raw[0, 0, 6 + k * 3 + 0] = 100.0
        raw[0, 0, 6 + k * 3 + 1] = 200.0
        raw[0, 0, 6 + k * 3 + 2] = 0.6
    from perception.CUDA_Stream.gpu_preprocess import LetterboxParams
    lb = LetterboxParams(scale=1.0, pad_x=0, pad_y=0, src_h=480, src_w=640, dst=640)
    stream = torch.cuda.Stream(device=DEVICE)
    result = post(raw, None, lb, {"fx": 600, "fy": 600, "cx": 320, "cy": 240}, stream, 0.0)
    stream.synchronize()
    assert result.valid
    assert math.isclose(result.kpts_2d_px[0, 0].item(), 100.0, abs_tol=0.5)


def test_postprocessor_mixed_invalid_depth():
    """NaN/0/negative depth values should produce zero'd 3D keypoints."""
    post = GpuPostprocessor(
        conf_threshold=0.1, kpt_conf_threshold=0.1, device=DEVICE, use_filter=False
    )
    raw = torch.zeros((1, 1, 56), device=DEVICE)
    raw[0, 0, 4] = 0.9
    # keypoints at spread positions, all high-conf
    coords = [(100, 100), (200, 200), (300, 300), (400, 400)] + [(150, 150)] * 13
    for k, (x, y) in enumerate(coords):
        raw[0, 0, 5 + k * 3 + 0] = float(x)
        raw[0, 0, 5 + k * 3 + 1] = float(y)
        raw[0, 0, 5 + k * 3 + 2] = 0.9

    depth = torch.full((480, 640), 2.5, device=DEVICE)  # valid 2.5 m
    # poison specific points
    depth[99:102, 99:102] = float("nan")  # kpt 0 → invalid
    depth[199:202, 199:202] = 0.0  # kpt 1 → invalid
    depth[299:302, 299:302] = -1.0  # kpt 2 → invalid
    # kpt 3 stays at 2.5

    from perception.CUDA_Stream.gpu_preprocess import LetterboxParams
    lb = LetterboxParams(scale=1.0, pad_x=0, pad_y=0, src_h=480, src_w=640, dst=640)
    stream = torch.cuda.Stream(device=DEVICE)
    result = post(raw, depth, lb, {"fx": 600, "fy": 600, "cx": 320, "cy": 240}, stream, 0.0)
    stream.synchronize()
    # kpt 0,1,2 invalid → z = 0
    assert result.kpts_3d_m[0, 2].item() == 0.0
    assert result.kpts_3d_m[1, 2].item() == 0.0
    assert result.kpts_3d_m[2, 2].item() == 0.0
    # kpt 3 valid → z ≈ 2.5
    assert math.isclose(result.kpts_3d_m[3, 2].item(), 2.5, abs_tol=0.1)


def test_oneeuro_reset_clears_state():
    from perception.CUDA_Stream.gpu_postprocess import OneEuroFilter1D

    f = OneEuroFilter1D((4,), device=DEVICE)
    # feed some data
    for i in range(5):
        out1 = f(torch.ones(4, device=DEVICE) * (1.0 + i), t_s=i * 0.01)
    f.reset()
    # after reset the very first call should return the input unchanged
    x = torch.ones(4, device=DEVICE) * 42.0
    out = f(x, t_s=0.0)
    assert torch.allclose(out, x)
