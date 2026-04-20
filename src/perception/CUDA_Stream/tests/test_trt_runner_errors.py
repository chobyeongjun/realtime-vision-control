"""TRTRunner failure path tests.

These exercise argument-validation paths that run before any TensorRT
kernel is touched, so they work even when a real engine file is absent.
"""

from __future__ import annotations

import pytest

pytest.importorskip("tensorrt")
pytest.importorskip("torch")

from perception.CUDA_Stream.trt_runner import TRTRunner


def test_missing_engine_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        TRTRunner(tmp_path / "nope.engine")


def test_corrupt_engine_raises_runtime(tmp_path):
    bogus = tmp_path / "bad.engine"
    bogus.write_bytes(b"not a real tensorrt engine")
    with pytest.raises((RuntimeError, Exception)):
        TRTRunner(bogus)
