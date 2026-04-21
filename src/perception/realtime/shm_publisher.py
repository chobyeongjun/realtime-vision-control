"""
shm_publisher.py
================
Python → C++ POSIX shared memory 기반 pose 데이터 퍼블리셔.

메모리 레이아웃은 pose_shm.h 와 완전히 일치해야 한다.
  ctypes.sizeof(PoseShmCtype) == 36  (pose_shm.h SHM_SIZE)

pyzed / OpenCV 와 충돌 없음 — mmap + ctypes만 사용.

사용 예시:
    from shm_publisher import ShmPublisher, FlexionAngles
    pub = ShmPublisher()
    pub.open()
    pub.write_pose(FlexionAngles(
        left_knee_deg=155.0, right_knee_deg=158.0,
        left_hip_deg=20.0,   right_hip_deg=18.0,
        gait_phase=0.0,
        timestamp_us=time.monotonic() * 1e6,
        valid=True, method='A',
    ))
    pub.close()
"""

import ctypes
import mmap
import os
import time
from dataclasses import dataclass
from typing import Literal

# ─── 상수 ────────────────────────────────────────────────────────────────────
SHM_NAME: str = "/hwalker_pose"   # pose_shm.h SHM_NAME 과 동일
SHM_SIZE: int = 36                # pose_shm.h SHM_SIZE 와 동일 (bytes)
_SHM_PATH: str = f"/dev/shm{SHM_NAME}"  # Linux; macOS는 shm_open syscall 필요


# ─── ctypes 구조체 ────────────────────────────────────────────────────────────
# #pragma pack(1) + 명시적 _pad 와 동일한 레이아웃
class _PoseShmCtype(ctypes.Structure):
    """
    pose_shm.h PoseShm 과 메모리 레이아웃 100% 동일.

    offset  0 : timestamp_us   (c_double, 8 bytes)
    offset  8 : left_knee_deg  (c_float,  4 bytes)
    offset 12 : right_knee_deg (c_float,  4 bytes)
    offset 16 : left_hip_deg   (c_float,  4 bytes)
    offset 20 : right_hip_deg  (c_float,  4 bytes)
    offset 24 : gait_phase     (c_float,  4 bytes)
    offset 28 : valid          (c_uint8,  1 byte)
    offset 29 : seq            (c_uint8,  1 byte)
    offset 30 : method         (c_uint8,  1 byte)
    offset 31 : _pad[5]        (c_uint8,  5 bytes)
    total      36 bytes
    """
    _pack_ = 1  # 컴파일러 자동 패딩 금지
    _fields_ = [
        ("timestamp_us",   ctypes.c_double),        # [μs] time.monotonic() * 1e6
        ("left_knee_deg",  ctypes.c_float),         # [deg] 완전신전=180, 굴곡시 감소
        ("right_knee_deg", ctypes.c_float),         # [deg]
        ("left_hip_deg",   ctypes.c_float),         # [deg] sagittal flexion
        ("right_hip_deg",  ctypes.c_float),         # [deg]
        ("gait_phase",     ctypes.c_float),         # [0.0~1.0] Teensy IMU에서 채움
        ("valid",          ctypes.c_uint8),         # 1=유효, 0=무효
        ("seq",            ctypes.c_uint8),         # 0~255 순환 카운터
        ("method",         ctypes.c_uint8),         # ord('A')=65 or ord('B')=66
        ("_pad",           ctypes.c_uint8 * 5),     # 항상 0, 예약
    ]

# 크기 검증 — import 시점에 즉시 확인
assert ctypes.sizeof(_PoseShmCtype) == SHM_SIZE, (
    f"PoseShmCtype 크기 오류: {ctypes.sizeof(_PoseShmCtype)} != {SHM_SIZE}. "
    "pose_shm.h 와 레이아웃 불일치."
)


# ─── 공개 데이터 클래스 ───────────────────────────────────────────────────────
@dataclass
class FlexionAngles:
    """
    YOLO/ZED 파이프라인에서 ShmPublisher로 넘기는 단일 프레임 데이터.

    timestamp_us : CLOCK_MONOTONIC 기준 [μs] (time.monotonic() * 1e6)
    *_deg        : 관절 굴곡각 [deg] — 0.0 이면 미감지
    gait_phase   : [0.0~1.0] Teensy serial에서 수신 (미수신 시 0.0)
    valid        : 4개 이상 관절 감지 시 True
    method       : 'A' = ZED depth 기반, 'B' = MediaPipe 2D 기반
    """
    left_knee_deg:  float = 0.0
    right_knee_deg: float = 0.0
    left_hip_deg:   float = 0.0
    right_hip_deg:  float = 0.0
    gait_phase:     float = 0.0       # [0.0~1.0] Teensy IMU
    timestamp_us:   float = 0.0       # [μs]
    valid:          bool  = False
    method:         Literal['A', 'B'] = 'A'


# ─── 퍼블리셔 ─────────────────────────────────────────────────────────────────
class ShmPublisher:
    """
    POSIX shared memory 에 PoseShm 구조체를 직접 씁니다.

    - mmap + ctypes overlay 방식: 복사 없이 shm에 직접 필드 할당 (~1μs)
    - multiprocessing.shared_memory를 사용하지 않는 이유:
        Python 3.8+ 전용이며, 내부적으로 추가 metadata 블록을 prepend해
        C++ side에서 raw offset 계산이 복잡해진다.
    - pyzed / OpenCV와 충돌 없음 (순수 POSIX syscall + ctypes)

    Linux  : /dev/shm/<name> 파일로 생성
    macOS  : shm_open(3) — /dev/shm 대신 Mach 커널 shm 사용
    """

    def __init__(self, name: str = SHM_NAME, size: int = SHM_SIZE) -> None:
        self._name  = name
        self._size  = size
        self._fd: int | None       = None
        self._mm: mmap.mmap | None = None
        self._shm: _PoseShmCtype | None = None
        self._seq: int = 0  # 0~255 순환

    def open(self) -> None:
        """
        POSIX shm 생성(없으면) 또는 열기.
        이미 존재하면 재사용 (다중 프로세스 restart 허용).
        """
        import posix_ipc  # type: ignore[import-untyped]
        try:
            mem = posix_ipc.SharedMemory(
                self._name,
                flags=posix_ipc.O_CREAT,
                mode=0o666,
                size=self._size,
            )
        except Exception as exc:
            raise RuntimeError(f"shm_open({self._name}) 실패: {exc}") from exc

        self._fd = mem.fd
        # fd를 mmap으로 매핑
        self._mm = mmap.mmap(self._fd, self._size)
        # ctypes를 mmap 버퍼 위에 overlay — 추가 복사 없음
        self._shm = _PoseShmCtype.from_buffer(self._mm)
        # 초기화: 모든 필드 0, valid=0
        ctypes.memset(ctypes.addressof(self._shm), 0, self._size)

    def write_pose(self, flexion: FlexionAngles) -> None:
        """
        FlexionAngles → shm에 seqlock 패턴으로 기록.

        Seqlock 규약 (Linux kernel 동일):
          - 쓰기 시작 전: seq를 홀수로 (writer in progress)
          - 모든 필드 write
          - 쓰기 종료: seq를 짝수로 (consistent)
          - Reader는 seq 짝수일 때만 읽고, 읽은 전후 seq 비교

        호출 비용: ~1μs (mmap 캐시 히트 기준)
        """
        if self._shm is None:
            raise RuntimeError("open() 먼저 호출하세요.")

        shm = self._shm

        # Step 1: writer 진입 표시 (seq 홀수) — reader는 이 값 보면 재시도
        self._seq = (self._seq + 1) & 0xFF
        if self._seq % 2 == 0:
            # 짝수면 한번 더 올려 홀수로 (writer start)
            self._seq = (self._seq + 1) & 0xFF
        shm.seq = self._seq

        # Step 2: 데이터 write (이 사이에 C++이 seq=odd 읽으면 retry)
        shm.timestamp_us   = flexion.timestamp_us
        shm.left_knee_deg  = flexion.left_knee_deg
        shm.right_knee_deg = flexion.right_knee_deg
        shm.left_hip_deg   = flexion.left_hip_deg
        shm.right_hip_deg  = flexion.right_hip_deg
        shm.gait_phase     = flexion.gait_phase
        shm.valid          = 1 if flexion.valid else 0
        shm.method         = ord(flexion.method)
        ctypes.memset(ctypes.addressof(shm._pad), 0, 5)

        # Step 3: writer 종료 표시 (seq 짝수) — 일관 상태 완료
        self._seq = (self._seq + 1) & 0xFF
        shm.seq = self._seq

    def close(self) -> None:
        """mmap 해제 및 shm 닫기. (shm 이름은 제거하지 않음 — 재시작 허용) """
        self._shm = None
        if self._mm is not None:
            self._mm.close()
            self._mm = None
        if self._fd is not None:
            os.close(self._fd)
            self._fd = None

    def unlink(self) -> None:
        """shm 이름을 /dev/shm에서 완전 제거. 마지막 프로세스 종료 시 호출. """
        try:
            import posix_ipc
            posix_ipc.unlink_shared_memory(self._name)
        except Exception:
            pass

    def __enter__(self) -> "ShmPublisher":
        self.open()
        return self

    def __exit__(self, *_) -> None:
        self.close()
