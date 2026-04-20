"""Tests for shm_publisher — CPU-only, no GPU required."""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np
import pytest

from perception.CUDA_Stream.shm_publisher import (
    DEFAULT_NAME,
    FORBIDDEN_NAMES,
    ShmPublisher,
    ShmReader,
)


def _rand_keypoints(seed: int = 0, K: int = 17):
    rng = np.random.default_rng(seed)
    return (
        rng.standard_normal((K, 3)).astype(np.float32),
        rng.random((K,)).astype(np.float32),
        rng.random((K, 2)).astype(np.float32) * 600,
    )


@pytest.fixture
def shm_name():
    name = f"test_hwalker_pose_cuda_{os.getpid()}"
    yield name
    # cleanup in case a test left it behind
    path = f"/dev/shm/{name}"
    if os.path.exists(path):
        try:
            os.unlink(path)
        except OSError:
            pass


def test_forbidden_mainline_name():
    for bad in FORBIDDEN_NAMES:
        with pytest.raises(ValueError):
            ShmPublisher(num_keypoints=17, name=bad)


@pytest.mark.parametrize("K", [6, 17])
def test_roundtrip_single_writer_reader(shm_name, K):
    pub = ShmPublisher(num_keypoints=K, name=shm_name, create=True)
    try:
        kpts3, kconf, kpts2 = _rand_keypoints(K=K)
        pub.publish(
            frame_id=42,
            ts_ns=1_700_000_000_000_000_000,
            kpts_3d_m=kpts3,
            kpt_conf=kconf,
            kpts_2d_px=kpts2,
            box_conf=0.91,
            valid=True,
            depth_invalid_ratio=0.11,
        )
        reader = ShmReader(name=shm_name)
        try:
            assert reader.K == K
            got = reader.read()
            assert got is not None
            fid, ts, g3, gc, g2, bc, valid, dir_ = got
            assert fid == 42
            assert ts == 1_700_000_000_000_000_000
            assert valid is True
            assert g3.shape == (K, 3)
            assert gc.shape == (K,)
            assert g2.shape == (K, 2)
            np.testing.assert_allclose(g3, kpts3, atol=1e-6)
            np.testing.assert_allclose(gc, kconf, atol=1e-6)
            np.testing.assert_allclose(g2, kpts2, atol=1e-6)
            assert abs(bc - 0.91) < 1e-4
            assert abs(dir_ - 0.11) < 1e-4
        finally:
            reader.close()
    finally:
        pub.close()


def test_invalid_shapes_rejected(shm_name):
    pub = ShmPublisher(num_keypoints=6, name=shm_name, create=True)
    try:
        with pytest.raises(ValueError):
            pub.publish(
                frame_id=1, ts_ns=0,
                kpts_3d_m=np.zeros((5, 3), dtype=np.float32),
                kpt_conf=np.zeros((6,), dtype=np.float32),
                kpts_2d_px=np.zeros((6, 2), dtype=np.float32),
                box_conf=0.0, valid=False,
            )
        with pytest.raises(ValueError):
            pub.publish(
                frame_id=1, ts_ns=0,
                kpts_3d_m=np.zeros((6, 3), dtype=np.float64),
                kpt_conf=np.zeros((6,), dtype=np.float32),
                kpts_2d_px=np.zeros((6, 2), dtype=np.float32),
                box_conf=0.0, valid=False,
            )
    finally:
        pub.close()


def _reader_proc(shm_name: str, n_reads: int, out_queue: mp.Queue) -> None:
    reader = ShmReader(name=shm_name)
    seen_partial = 0
    last_fid = -1
    for _ in range(n_reads):
        got = reader.read(max_retries=64)
        if got is None:
            seen_partial += 1
            continue
        fid = got[0]
        if fid < last_fid:
            seen_partial += 1
        last_fid = fid
        time.sleep(0.0005)
    reader.close()
    out_queue.put(seen_partial)


def test_seqlock_no_torn_reads_concurrent(shm_name):
    pub = ShmPublisher(num_keypoints=6, name=shm_name, create=True)
    try:
        kpts3, kconf, kpts2 = _rand_keypoints(K=6)
        out = mp.Queue()
        reader = mp.Process(target=_reader_proc, args=(shm_name, 500, out))
        reader.start()
        for i in range(2000):
            pub.publish(
                frame_id=i, ts_ns=i * 1_000_000,
                kpts_3d_m=kpts3 + i * 0.01,
                kpt_conf=kconf, kpts_2d_px=kpts2,
                box_conf=0.5 + (i % 10) * 0.01, valid=True,
            )
        reader.join(timeout=5)
        assert not reader.is_alive()
        seen_partial = out.get_nowait()
        assert seen_partial < 25, f"too many partial reads: {seen_partial}"
    finally:
        pub.close()
