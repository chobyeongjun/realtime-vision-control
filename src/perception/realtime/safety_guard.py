"""
safety_guard.py
===============
Depth 인식 실패 시 안전 레벨 관리.

레벨 정의:
    LEVEL_0 (정상)   : 6개 keypoint 모두 유효, knee flexion 계산 정상
    LEVEL_1 (경고)   : depth 일부 손실 (4~5개 유효), Kp 50% 감소
    LEVEL_2 (비상)   : 연속 3프레임 이상 invalid → pretension(5N) fallback
    LEVEL_3 (정지)   : 연속 10프레임 이상 → Teensy E_STOP

전환 규칙:
    LEVEL_0 → LEVEL_1 : depth_valid_count < 6 OR bone_length 이상
    LEVEL_1 → LEVEL_2 : consecutive_invalid >= WARN_FRAMES (3)
    LEVEL_2 → LEVEL_3 : consecutive_invalid >= ESTOP_FRAMES (10)
    LEVEL_X → LEVEL_0 : 연속 RECOVER_FRAMES (5) 정상 프레임

주요 출력:
    SafetyState.level          : 현재 레벨 (0~3)
    SafetyState.kp_scale       : Kp 배율 (1.0 / 0.5 / 0.0 / 0.0)
    SafetyState.send_estop     : True이면 Teensy에 E_STOP 전송 필요
    SafetyState.valid_for_ctrl : True이면 제어 가능
"""

from __future__ import annotations

import time
import sys
import os
from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

_bench_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'benchmarks')
if _bench_dir not in sys.path:
    sys.path.insert(0, _bench_dir)

from joint_3d import JointState3D, BONE_RANGES


# ─── 레벨 정의 ───────────────────────────────────────────────────────────────

class SafetyLevel(IntEnum):
    NORMAL   = 0   # 정상
    DEGRADED = 1   # 경고: 부분 depth 손실
    FALLBACK = 2   # 비상: pretension only
    ESTOP    = 3   # 정지: 모터 비활성화


# ─── 파라미터 ─────────────────────────────────────────────────────────────────

WARN_FRAMES    = 3    # 연속 invalid → FALLBACK
ESTOP_FRAMES   = 10   # 연속 invalid → ESTOP
RECOVER_FRAMES = 5    # 연속 정상 → LEVEL_0 복귀
BONE_TOL       = 0.20 # 뼈 길이 허용 오차 ±20%

# 레벨별 Kp 배율
KP_SCALE = {
    SafetyLevel.NORMAL:   1.0,
    SafetyLevel.DEGRADED: 0.5,
    SafetyLevel.FALLBACK: 0.0,
    SafetyLevel.ESTOP:    0.0,
}


# ─── 출력 구조체 ──────────────────────────────────────────────────────────────

@dataclass
class SafetyState:
    level:          SafetyLevel = SafetyLevel.NORMAL
    kp_scale:       float = 1.0
    valid_for_ctrl: bool  = True   # True → 임피던스 제어 가능
    send_estop:     bool  = False  # True → Teensy E_STOP 전송 필요
    reason:         str   = ''
    consecutive_invalid: int = 0
    consecutive_valid:   int = 0


# ─── Safety Guard ─────────────────────────────────────────────────────────────

class DepthSafetyGuard:
    """
    매 프레임 JointState3D를 입력받아 SafetyState를 반환.

    사용법:
        guard = DepthSafetyGuard()

        # 메인 루프에서:
        safety = guard.update(joint_state)
        if safety.send_estop:
            serial.send_estop()
        if safety.valid_for_ctrl:
            kp_effective = KP_KNEE * safety.kp_scale
            ... 임피던스 계산 ...
        else:
            # pretension fallback (Teensy timeout으로 자동 처리됨)
            pass
    """

    def __init__(self):
        self._consec_invalid: int = 0
        self._consec_valid:   int = 0
        self._level: SafetyLevel  = SafetyLevel.NORMAL
        self._estop_sent: bool    = False
        self._calib_bone_len: dict = {}   # {bone_name: float} 초기 교정값

    def set_calibrated_bone_lengths(self, bone_lengths: dict):
        """
        서 있는 상태에서 측정한 뼈 길이 교정값 설정.
        이후 프레임에서 ±BONE_TOL 초과 시 DEGRADED 처리.
        """
        self._calib_bone_len = dict(bone_lengths)

    def update(self, state: JointState3D) -> SafetyState:
        """
        JointState3D → SafetyState.

        프레임마다 호출. 상태 머신을 갱신하고 현재 SafetyState 반환.
        """
        out = SafetyState()

        # ── 이 프레임이 "정상"인지 판단 ──────────────────────────────────────
        frame_ok, reason = self._is_frame_ok(state)

        if frame_ok:
            self._consec_invalid = 0
            self._consec_valid  += 1
        else:
            self._consec_invalid += 1
            self._consec_valid   = 0

        # ── 레벨 전환 ─────────────────────────────────────────────────────────
        if self._consec_invalid >= ESTOP_FRAMES:
            self._level = SafetyLevel.ESTOP
        elif self._consec_invalid >= WARN_FRAMES:
            self._level = SafetyLevel.FALLBACK
        elif not frame_ok:
            # 1~2프레임 문제 → DEGRADED (Kp 절반)
            self._level = SafetyLevel.DEGRADED
        else:
            # 연속 정상 → 복귀
            if self._consec_valid >= RECOVER_FRAMES:
                self._level = SafetyLevel.NORMAL
            elif self._level == SafetyLevel.DEGRADED:
                pass  # DEGRADED 유지 중 (아직 RECOVER_FRAMES 미달)
            # FALLBACK/ESTOP에서는 RECOVER_FRAMES 충족 후에만 복귀
            elif self._level in (SafetyLevel.FALLBACK, SafetyLevel.ESTOP):
                if self._consec_valid >= RECOVER_FRAMES:
                    self._level = SafetyLevel.NORMAL
                    self._estop_sent = False

        # ── E_STOP 전송 (한 번만) ─────────────────────────────────────────────
        send_estop = False
        if self._level == SafetyLevel.ESTOP and not self._estop_sent:
            send_estop = True
            self._estop_sent = True
            _log_safety('CRITICAL', f'E_STOP 발동: {self._consec_invalid}프레임 연속 invalid')

        # ── 결과 구성 ──────────────────────────────────────────────────────────
        out.level               = self._level
        out.kp_scale            = KP_SCALE[self._level]
        out.valid_for_ctrl      = (self._level in (SafetyLevel.NORMAL, SafetyLevel.DEGRADED))
        out.send_estop          = send_estop
        out.reason              = reason if not frame_ok else 'OK'
        out.consecutive_invalid = self._consec_invalid
        out.consecutive_valid   = self._consec_valid
        return out

    # ── 내부 헬퍼 ─────────────────────────────────────────────────────────────

    def _is_frame_ok(self, state: JointState3D) -> tuple[bool, str]:
        """
        이 프레임이 제어에 사용 가능한지 판단.

        Returns: (ok: bool, reason: str)
        """
        # 1. Knee flexion 계산 가능한 keypoint 있어야
        if not state.valid:
            return False, f'keypoint 부족 (depth_valid={state.depth_valid_count})'

        # 2. Knee flexion 자체가 None이면 안 됨
        if state.left_knee_flexion is None and state.right_knee_flexion is None:
            return False, 'knee_flexion 계산 불가'

        # 3. Depth 유효 개수 (최소 4개)
        if state.depth_valid_count < 4:
            return False, f'depth_valid={state.depth_valid_count} < 4'

        # 4. 뼈 길이 범위 체크
        for bone, length in state.bone_lengths.items():
            if bone in BONE_RANGES:
                lo, hi = BONE_RANGES[bone]
                if not (lo * 0.8 <= length <= hi * 1.2):
                    return False, f'{bone} 길이 이상: {length:.3f}m'

        # 5. 교정값 대비 ±BONE_TOL 이상 이탈
        for bone, ref in self._calib_bone_len.items():
            if bone in state.bone_lengths and ref > 0.01:
                ratio = abs(state.bone_lengths[bone] - ref) / ref
                if ratio > BONE_TOL:
                    return False, f'{bone} 교정 이탈: {ratio*100:.0f}%'

        # 6. Knee flexion 물리적 범위 체크 (-10° ~ 100°)
        for flex in (state.left_knee_flexion, state.right_knee_flexion):
            if flex is not None and not (-10.0 <= flex <= 100.0):
                return False, f'knee_flexion 범위 이탈: {flex:.1f}°'

        return True, 'OK'


def _log_safety(level: str, msg: str):
    ts = time.strftime('%H:%M:%S')
    print(f'[{ts}][SAFETY/{level}] {msg}', file=sys.stderr)
