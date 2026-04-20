"""Watchdog unit tests — CPU-only.

We fake ``torch.cuda.Stream`` with ``unittest.mock.Mock`` so these tests
run anywhere (CI, laptop, dev machine).
"""

from __future__ import annotations

import os
import tempfile
import time
from unittest.mock import MagicMock

import numpy as np
import pytest

# Importing the watchdog pulls torch in; fine for CPU-only.
torch = pytest.importorskip("torch")

from perception.CUDA_Stream.watchdog import StreamWatchdog


class FakeStream:
    def __init__(self, idle: bool = True) -> None:
        self._idle = idle

    def set_idle(self, v: bool) -> None:
        self._idle = v

    def query(self) -> bool:
        return self._idle


@pytest.fixture
def estop_path(tmp_path):
    p = tmp_path / "estop_sentinel"
    yield str(p)
    if os.path.exists(p):
        os.unlink(p)


def test_publish_timeout_fires_fallback(estop_path):
    stream = FakeStream(idle=True)
    cb = MagicMock()
    wd = StreamWatchdog(
        streams={"infer": stream},
        publish_timeout_ms=10.0,
        stream_timeout_ms=1000.0,
        startup_grace_s=0.0,  # disable the production 5s grace for unit tests
        fallback_cb=cb,
        estop_file=estop_path,
        poll_hz=200.0,
    )
    wd.start()
    try:
        time.sleep(0.05)  # > publish_timeout
    finally:
        wd.stop()
    assert cb.call_count >= 1, "fallback should have fired at least once"
    assert os.path.exists(estop_path), "estop sentinel file should exist"


def test_fallback_only_re_fires_on_new_unhealthy_transition(estop_path):
    stream = FakeStream(idle=True)
    cb = MagicMock()
    wd = StreamWatchdog(
        streams={"infer": stream},
        publish_timeout_ms=5.0,
        stream_timeout_ms=1000.0,
        startup_grace_s=0.0,
        fallback_cb=cb,
        estop_file=estop_path,
        poll_hz=500.0,
    )
    wd.start()
    try:
        time.sleep(0.05)
        calls_after_first_unhealthy = cb.call_count
        # simulate healthy period — publishes resume
        wd.note_publish()
        time.sleep(0.02)
        # now go unhealthy again
        time.sleep(0.05)
    finally:
        wd.stop()
    assert calls_after_first_unhealthy >= 1
    # after healthy → unhealthy round trip, there should be at least one
    # additional fallback firing
    assert cb.call_count > calls_after_first_unhealthy


def test_force_publisher_invalidates_on_fault(estop_path):
    publisher = MagicMock()
    publisher.K = 6  # lowlimb6 schema
    stream = FakeStream(idle=True)
    wd = StreamWatchdog(
        streams={"infer": stream},
        publish_timeout_ms=5.0,
        stream_timeout_ms=1000.0,
        startup_grace_s=0.0,
        publisher=publisher,
        fallback_cb=None,
        estop_file=estop_path,
        poll_hz=500.0,
    )
    wd.start()
    try:
        time.sleep(0.05)
    finally:
        wd.stop()
    assert publisher.publish.called
    kwargs = publisher.publish.call_args.kwargs
    assert kwargs["valid"] is False
    # watchdog must read publisher.K dynamically (K=6 here)
    np.testing.assert_array_equal(kwargs["kpts_3d_m"], np.zeros((6, 3), np.float32))
    np.testing.assert_array_equal(kwargs["kpt_conf"], np.zeros((6,), np.float32))
    np.testing.assert_array_equal(kwargs["kpts_2d_px"], np.zeros((6, 2), np.float32))


def test_stream_busy_timeout(estop_path):
    stream = FakeStream(idle=False)  # never idle
    cb = MagicMock()
    wd = StreamWatchdog(
        streams={"infer": stream},
        publish_timeout_ms=10_000.0,  # rule out publish path
        stream_timeout_ms=5.0,
        startup_grace_s=10_000.0,  # prevent publish path entirely
        fallback_cb=cb,
        estop_file=estop_path,
        poll_hz=500.0,
    )
    wd.note_publish()  # prevent publish path
    wd.start()
    try:
        time.sleep(0.05)
    finally:
        wd.stop()
    assert cb.call_count >= 1
    reason = cb.call_args.args[0]
    assert "infer" in reason and "busy" in reason
