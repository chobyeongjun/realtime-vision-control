"""
bone_constraint.py
==================
3D bone length constraint — 움직임 중 keypoint outlier 제거.

원리:
  정자세 캘리브레이션 중 뼈 길이 샘플 30프레임 수집 → 중앙값을 ref로 저장.
  보행 중 매 프레임: 뼈 길이가 ref ±tolerance 벗어나면 child joint를
  부모→자식 방향으로 ref 길이 위치에 투영 보정.

설계 원칙 (과거 실패로부터):
  - 2D keypoint는 절대 건드리지 않음 (depth 깨짐 — 2026-04-15 기록)
  - 3D 좌표에만 적용
  - **Static ref** — 캘리브 시 확정 후 고정. 보행 중 절대 변경 안 함 (피드백 루프 방지)
  - Std > 10mm면 ref 무효 처리 (캘리브 재시도 권장)
  - Child joint만 이동 (parent 기준 유지)
  - Camera frame / world frame 무관하게 작동 (뼈 길이는 회전 불변)

사용 패턴:
    bc = BoneLengthConstraint(tolerance=0.20, std_threshold=0.010)
    # 캘리브 중 (정자세 30프레임)
    for frame in calib_frames:
        bc.add_sample(raw_3d)
    ok = bc.finalize()
    if not ok:
        print("캘리브 재시도 권장 — std 초과")
    # 보행 중 매 프레임
    corrected_3d, n_hit = bc.apply(raw_3d)
"""

from __future__ import annotations

import numpy as np
from typing import Dict, Tuple


# 뼈 세그먼트 정의: (parent, child). child가 보정 대상.
SEGMENTS_3D = [
    ('left_hip',   'left_knee'),
    ('left_knee',  'left_ankle'),
    ('right_hip',  'right_knee'),
    ('right_knee', 'right_ankle'),
]


class BoneLengthConstraint:
    """3D 뼈 길이 제약 (정자세 ref 기반).

    Attributes:
        tolerance: ref 길이 대비 허용 편차 (기본 ±20%)
        std_threshold: 캘리브 std 이하여야 ref 채택 (기본 10mm)
        min_samples: 최소 샘플 수 (기본 20)
    """

    def __init__(self,
                 tolerance: float = 0.20,
                 std_threshold: float = 0.010,
                 min_samples: int = 20):
        self.tolerance = tolerance
        self.std_threshold = std_threshold
        self.min_samples = min_samples
        self._samples: Dict[Tuple[str, str], list] = {}
        self._ref:     Dict[Tuple[str, str], float] = {}
        self._stats:   Dict[Tuple[str, str], dict]  = {}
        self._ready = False
        self._finalize_tried = False   # 한 번만 시도 — 실패해도 로그 도배 방지
        self._hit_counter = {seg: 0 for seg in SEGMENTS_3D}
        self._frame_counter = 0

    # ── 캘리브 phase ────────────────────────────────────────────────────────

    def add_sample(self, raw_3d: dict) -> None:
        """캘리브 프레임마다 호출 — 각 segment 뼈 길이 샘플 1개 누적.

        finalize 시도한 뒤엔 no-op (메모리 축적 방지).
        재시도 원하면 reset() 먼저 호출.
        """
        if self._finalize_tried:
            return
        for seg in SEGMENTS_3D:
            parent, child = seg
            if parent in raw_3d and child in raw_3d:
                p = np.asarray(raw_3d[parent])
                c = np.asarray(raw_3d[child])
                length = float(np.linalg.norm(c - p))
                # sanity: 10cm 이하, 80cm 이상은 확실히 오류. 샘플에서 제외.
                if 0.10 < length < 0.80:
                    self._samples.setdefault(seg, []).append(length)

    def sample_count(self) -> int:
        """현재 누적 샘플 수 (segment 중 최솟값)."""
        if not self._samples:
            return 0
        return min(len(v) for v in self._samples.values())

    def finalize(self) -> bool:
        """캘리브 종료 시 호출. median을 ref로 저장하고 std 검증.

        **한 번만 수행** — 실패해도 재호출 시 이전 결과 반환 (로그 도배 방지).
        재시도하려면 reset() 먼저.

        Returns:
            True: std 검증 통과, ref 신뢰 가능 → apply() 활성화
            False: std 초과 / 샘플 부족 → apply() 무효 (캘리브 재시도 권장)
        """
        if self._finalize_tried:
            return self._ready   # 이미 시도함. 로그 없이 조용히 결과 반환.
        self._finalize_tried = True

        if not self._samples:
            print('[BoneConstraint] 샘플 없음 — finalize 실패')
            self._ready = False
            return False

        all_ok = True
        for seg in SEGMENTS_3D:
            lengths = self._samples.get(seg, [])
            if len(lengths) < self.min_samples:
                print(f'[BoneConstraint] {seg[0]}→{seg[1]}: '
                      f'샘플 부족 ({len(lengths)} < {self.min_samples})')
                all_ok = False
                continue
            arr    = np.array(lengths)
            median = float(np.median(arr))
            std    = float(arr.std())

            self._stats[seg] = {
                'mean':   float(arr.mean()),
                'std':    std,
                'median': median,
                'min':    float(arr.min()),
                'max':    float(arr.max()),
                'n':      len(arr),
            }

            if std > self.std_threshold:
                print(f'[BoneConstraint] {seg[0]}→{seg[1]}: '
                      f'std {std*1000:.1f}mm > {self.std_threshold*1000:.0f}mm '
                      f'(캘리브 부정확 — 움직이지 말 것)')
                all_ok = False
                continue

            self._ref[seg] = median

        if all_ok and len(self._ref) >= 4:
            self._ready = True
            print('[BoneConstraint] ref 확정 (static):')
            for seg in SEGMENTS_3D:
                if seg in self._ref:
                    s = self._stats[seg]
                    print(f"  {seg[0]:11} → {seg[1]:11} "
                          f"{self._ref[seg]*100:5.1f}cm  "
                          f"(std {s['std']*1000:4.1f}mm, N={s['n']})")
        else:
            self._ready = False
            print('[BoneConstraint] ref 채택 안 됨 — apply() 무효, 캘리브 재시도 권장')

        return self._ready

    # ── 보행 phase ──────────────────────────────────────────────────────────

    def apply(self, raw_3d: dict) -> Tuple[dict, int]:
        """매 프레임 호출 — outlier 뼈 길이 감지 시 child joint 투영 보정.

        Args:
            raw_3d: {joint_name: (X, Y, Z)} (camera or world frame 무관)

        Returns:
            (corrected_3d, hit_count)
              corrected_3d: 보정된 dict (원본 유지, 새 dict 반환)
              hit_count:    이번 프레임에서 constraint 적용된 seg 개수 (0~4)
        """
        self._frame_counter += 1
        if not self._ready:
            return raw_3d, 0

        out = dict(raw_3d)  # shallow copy
        hit = 0

        for seg in SEGMENTS_3D:
            if seg not in self._ref:
                continue
            parent, child = seg
            if parent not in out or child not in out:
                continue

            p = np.asarray(out[parent], dtype=np.float64)
            c = np.asarray(out[child],  dtype=np.float64)
            vec = c - p
            cur_len = float(np.linalg.norm(vec))
            if cur_len < 1e-6:
                continue

            ref = self._ref[seg]
            rel_err = abs(cur_len - ref) / ref

            if rel_err > self.tolerance:
                # outlier — child를 ref 위치로 투영 (parent→child 방향 유지)
                new_c = p + vec / cur_len * ref
                out[child] = new_c.astype(np.float32)
                hit += 1
                self._hit_counter[seg] += 1

        return out, hit

    # ── 상태/진단 ──────────────────────────────────────────────────────────

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def tried_finalize(self) -> bool:
        """finalize 한 번 시도했는지. True면 더 이상 sample 수집/finalize 안 함."""
        return self._finalize_tried

    @property
    def ref_dict(self) -> dict:
        """진단용 — 확정된 ref 길이 (m)."""
        return {f'{p}->{c}': v for (p, c), v in self._ref.items()}

    def hit_summary(self) -> dict:
        """누적 프레임 중 각 segment별 constraint 발동 횟수."""
        if self._frame_counter == 0:
            return {}
        return {
            f'{seg[0]}->{seg[1]}': {
                'hits':  n,
                'ratio': n / self._frame_counter,
            }
            for seg, n in self._hit_counter.items()
        }

    def reset(self) -> None:
        """처음부터 다시 — 캘리브 재시도용."""
        self._samples.clear()
        self._ref.clear()
        self._stats.clear()
        self._ready = False
        self._finalize_tried = False
        self._hit_counter = {seg: 0 for seg in SEGMENTS_3D}
        self._frame_counter = 0
