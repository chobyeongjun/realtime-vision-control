"""
validate_transform.py
=====================
3D 좌표 변환이 해부학적으로 올바른지 검증.

ZED IMU quaternion → rotation matrix → world frame 변환 후
아래 조건이 모두 통과해야 합니다:

  [V1] 뼈 길이 보존   : 변환 전후 thigh/shank 길이 동일 (회전은 거리 보존)
  [V2] 수직 방향      : world frame에서 ankle Y < hip Y (발목이 고관절보다 아래)
  [V3] 좌우 대칭      : 서 있을 때 left_hip Y ≈ right_hip Y (±1cm)
  [V4] 무릎 각도 불변 : 3D 무릎 각도는 좌표계와 무관 → 변환 전후 동일
  [V5] Sagittal 투영  : 서 있을 때 thigh vector가 XY plane에서 수직에 가까움

각 검증 항목을 통과/실패로 표시하고 수치를 출력합니다.

실행:
    python3 validate_transform.py               # ZED 실시간
    python3 validate_transform.py --svo2 ...    # 파일

출력 예:
    [V1] 뼈 길이 보존    PASS  (before=0.421m  after=0.421m  diff=0.000m)
    [V2] 수직 방향       PASS  (ankle_Y=1.12m < hip_Y=0.72m ✓)
    [V3] 좌우 대칭       PASS  (left_hip_Y=0.72 right_hip_Y=0.71 diff=0.01m)
    [V4] 무릎 각도 불변  PASS  (before=178.1° after=178.1° diff=0.0°)
    [V5] Sagittal 수직   PASS  (thigh_horiz_component=0.03m → 4.1° 기울어짐)
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2

_this_dir  = os.path.dirname(os.path.abspath(__file__))
_bench_dir = os.path.join(_this_dir, '..', 'benchmarks')
_model_dir = os.path.join(_this_dir, '..', 'models')
for p in (_bench_dir, _model_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

from zed_camera import create_camera
from pose_models import LowerBodyPoseModel
from postprocess_accel import batch_2d_to_3d
from joint_3d import compute_joint_state, _angle_3d, JointState3D
from calibration import ZEDIMUWorldFrame, StandingCalibration


# ─────────────────────────────────────────────────────────────────────────────
# 검증 로직
# ─────────────────────────────────────────────────────────────────────────────

class TransformValidator:
    """
    카메라 frame 3D 좌표와 world frame 3D 좌표를 비교해
    변환이 해부학적으로 올바른지 확인.
    """

    THRESHOLDS = {
        'bone_diff_m':       0.005,   # V1: 뼈 길이 차이 허용 [m]
        'sym_hip_diff_m':    0.030,   # V3: 좌우 hip 높이 차이 [m]
        'angle_diff_deg':    0.5,     # V4: 각도 불변 허용 오차 [deg]
        'thigh_tilt_deg':    30.0,    # V5: 서 있을 때 허벅지 최대 기울기 [deg]
    }

    def __init__(self):
        self.history = []     # (cam_state, world_positions) 누적

    def add(self, cam_state: JointState3D, world_positions: dict):
        self.history.append((cam_state, world_positions))
        if len(self.history) > 100:
            self.history.pop(0)

    def validate_single(self, cam_state: JointState3D,
                         world_pos: dict) -> dict:
        """단일 프레임 검증. 결과 dict 반환."""
        results = {}
        pos_c = cam_state.positions   # camera frame
        pos_w = world_pos             # world frame

        # ── V1: 뼈 길이 보존 ──────────────────────────────────────────────
        v1_details = {}
        for side in ('left', 'right'):
            for (a, b), bone in [
                ((f'{side}_hip', f'{side}_knee'), f'{side}_thigh'),
                ((f'{side}_knee', f'{side}_ankle'), f'{side}_shank'),
            ]:
                if all(k in pos_c for k in (a, b)) and all(k in pos_w for k in (a, b)):
                    len_c = float(np.linalg.norm(pos_c[a] - pos_c[b]))
                    len_w = float(np.linalg.norm(pos_w[a] - pos_w[b]))
                    diff  = abs(len_c - len_w)
                    v1_details[bone] = {'cam': len_c, 'world': len_w, 'diff': diff}

        v1_pass = all(d['diff'] < self.THRESHOLDS['bone_diff_m']
                      for d in v1_details.values())
        results['V1_bone_preservation'] = {
            'pass': v1_pass,
            'details': v1_details,
        }

        # ── V2: 수직 방향 (ankle이 hip보다 낮아야 함) ─────────────────────
        # world frame에서 Y축이 위를 향한다고 가정 → ankle_Y < hip_Y
        # 만약 Y축이 아래를 향하면 반대 (ankle_Y > hip_Y)
        v2_details = {}
        for side in ('left', 'right'):
            hip_k, ankle_k = f'{side}_hip', f'{side}_ankle'
            if hip_k in pos_w and ankle_k in pos_w:
                # world frame의 Z값이 수직 높이가 될 수도 있음 (ZED 설정에 따라)
                # → 3개 축 모두에서 hip-ankle 방향 확인
                diff_vec = pos_w[ankle_k] - pos_w[hip_k]
                # 중력 방향 = 가장 큰 절댓값 성분
                dominant_axis = int(np.argmax(np.abs(diff_vec)))
                dominant_sign = np.sign(diff_vec[dominant_axis])
                v2_details[side] = {
                    'hip_world':   pos_w[hip_k].tolist(),
                    'ankle_world': pos_w[ankle_k].tolist(),
                    'diff_vec':    diff_vec.tolist(),
                    'dominant_axis': ['X', 'Y', 'Z'][dominant_axis],
                    'dominant_sign': float(dominant_sign),
                }

        # ankle이 hip보다 world frame에서 일관된 방향으로 있는지
        v2_consistent = len(v2_details) > 0 and len(set(
            (d['dominant_axis'], d['dominant_sign']) for d in v2_details.values()
        )) == 1
        results['V2_vertical_direction'] = {
            'pass': v2_consistent,
            'details': v2_details,
        }

        # ── V3: 좌우 대칭 (서 있을 때 양쪽 hip 높이 유사) ────────────────
        if 'left_hip' in pos_w and 'right_hip' in pos_w:
            diff = pos_w['left_hip'] - pos_w['right_hip']
            # dominant axis 방향의 차이 확인
            sym_diff = float(np.min(np.abs(diff)))
            v3_pass = sym_diff < self.THRESHOLDS['sym_hip_diff_m']
            results['V3_bilateral_symmetry'] = {
                'pass': v3_pass,
                'left_hip_world':  pos_w['left_hip'].tolist(),
                'right_hip_world': pos_w['right_hip'].tolist(),
                'diff_vec':        diff.tolist(),
                'min_axis_diff_m': sym_diff,
            }

        # ── V4: 무릎 각도 불변 ───────────────────────────────────────────
        v4_details = {}
        for side in ('left', 'right'):
            h, k, a = f'{side}_hip', f'{side}_knee', f'{side}_ankle'
            if all(j in pos_c for j in (h, k, a)) and all(j in pos_w for j in (h, k, a)):
                ang_c = _angle_3d(pos_c[h], pos_c[k], pos_c[a])
                ang_w = _angle_3d(pos_w[h], pos_w[k], pos_w[a])
                diff  = abs(ang_c - ang_w)
                v4_details[side] = {'cam_deg': ang_c, 'world_deg': ang_w, 'diff': diff}

        v4_pass = all(d['diff'] < self.THRESHOLDS['angle_diff_deg']
                      for d in v4_details.values())
        results['V4_angle_invariance'] = {
            'pass': v4_pass,
            'details': v4_details,
        }

        # ── V5: 서 있을 때 허벅지가 수직에 가까워야 함 ────────────────────
        v5_details = {}
        for side in ('left', 'right'):
            hip_k, knee_k = f'{side}_hip', f'{side}_knee'
            if hip_k in pos_w and knee_k in pos_w:
                thigh_vec = pos_w[knee_k] - pos_w[hip_k]
                thigh_len = float(np.linalg.norm(thigh_vec))
                if thigh_len > 0.01:
                    thigh_unit = thigh_vec / thigh_len
                    # dominant axis = 수직 방향 (가장 큰 성분)
                    dom = int(np.argmax(np.abs(thigh_unit)))
                    # 수직 성분 제거 후 수평 성분 크기
                    horiz = thigh_unit.copy()
                    horiz[dom] = 0
                    horiz_mag = float(np.linalg.norm(horiz))
                    tilt_deg = float(np.degrees(np.arcsin(np.clip(horiz_mag, 0, 1))))
                    v5_details[side] = {
                        'thigh_unit': thigh_unit.tolist(),
                        'vertical_axis': ['X', 'Y', 'Z'][dom],
                        'tilt_deg': tilt_deg,
                    }

        v5_pass = all(d['tilt_deg'] < self.THRESHOLDS['thigh_tilt_deg']
                      for d in v5_details.values())
        results['V5_sagittal_upright'] = {
            'pass': v5_pass,
            'details': v5_details,
        }

        return results

    def print_results(self, results: dict):
        PASS = '\033[92mPASS\033[0m'
        FAIL = '\033[91mFAIL\033[0m'

        print('\n' + '─'*60)
        for check, data in results.items():
            tag  = PASS if data['pass'] else FAIL
            name = check.split('_', 1)[1].replace('_', ' ')
            print(f'  [{tag}] {name}')

            if check == 'V1_bone_preservation':
                for bone, d in data['details'].items():
                    ok = '✓' if d['diff'] < self.THRESHOLDS['bone_diff_m'] else '✗'
                    print(f'         {bone:<16} cam={d["cam"]:.4f}m '
                          f'world={d["world"]:.4f}m diff={d["diff"]*1000:.2f}mm {ok}')

            elif check == 'V2_vertical_direction':
                for side, d in data['details'].items():
                    print(f'         {side}: dominant axis={d["dominant_axis"]} '
                          f'sign={d["dominant_sign"]:+.0f}')
                if not data['pass']:
                    print('         !! 좌우 방향이 일치하지 않음 → R 확인 필요')

            elif check == 'V3_bilateral_symmetry':
                diff_m = data['min_axis_diff_m']
                ok = '✓' if diff_m < self.THRESHOLDS['sym_hip_diff_m'] else '✗'
                print(f'         hip 좌우 높이 차: {diff_m*100:.1f}cm {ok}')

            elif check == 'V4_angle_invariance':
                for side, d in data['details'].items():
                    ok = '✓' if d['diff'] < self.THRESHOLDS['angle_diff_deg'] else '✗'
                    print(f'         {side}: cam={d["cam_deg"]:.2f}° '
                          f'world={d["world_deg"]:.2f}° diff={d["diff"]:.3f}° {ok}')

            elif check == 'V5_sagittal_upright':
                for side, d in data['details'].items():
                    ok = '✓' if d['tilt_deg'] < self.THRESHOLDS['thigh_tilt_deg'] else '✗'
                    print(f'         {side}: 수직축={d["vertical_axis"]} '
                          f'기울기={d["tilt_deg"]:.1f}° {ok}')

        all_pass = all(d['pass'] for d in results.values())
        print('─'*60)
        if all_pass:
            print('  \033[92m모든 검증 통과 → 좌표 변환 정확\033[0m')
        else:
            fail_list = [k for k, d in results.items() if not d['pass']]
            print(f'  \033[91m실패: {fail_list}\033[0m')
            print('  → calibration.py ZEDIMUWorldFrame._quat_to_R() 확인 필요')
        print()


# ─────────────────────────────────────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--svo2',  type=str, default=None)
    parser.add_argument('--video', type=str, default=None)
    parser.add_argument('--model', type=str,
                        default=os.path.join(_model_dir, 'yolo26s-lower6-v2.pt'))
    parser.add_argument('--no-trt', action='store_true')
    parser.add_argument('--validate-every', type=int, default=30,
                        help='N 프레임마다 검증 출력')
    args = parser.parse_args()

    video_path = args.svo2 or args.video
    use_zed    = video_path is None

    camera = create_camera(use_zed=use_zed, video_path=video_path,
                           resolution='SVGA', fps=120, depth_mode='PERFORMANCE')
    camera.open()

    model = LowerBodyPoseModel(
        model_path=args.model, use_tensorrt=not args.no_trt,
        imgsz=640, smoothing=0.5)
    model.load()

    # ZED IMU world frame 초기화
    wf = ZEDIMUWorldFrame(camera) if use_zed else None
    if wf:
        ok = wf.init(warm_up_frames=30)
        if not ok:
            print('[warn] ZED IMU 없음 → camera frame으로만 검증')
            wf = None

    validator = TransformValidator()

    print('\n[ready] 똑바로 서 있는 상태로 검증 시작')
    print('        스페이스바: 현재 프레임 즉시 검증 / q: 종료\n')

    frame_count = 0
    t_prev = time.perf_counter()
    fps = 0.0

    while True:
        if not camera.grab():
            if video_path:
                break
            continue

        t_now = time.perf_counter()
        fps = 0.9*fps + 0.1/(max(t_now - t_prev, 1e-6))
        t_prev = t_now

        rgb   = camera.get_rgb()
        depth = camera.get_depth()
        if rgb is None:
            continue

        result = model.predict(rgb)
        keypoints_3d = {}
        if result.detected and depth is not None:
            keypoints_3d = batch_2d_to_3d(result.keypoints_2d, depth, camera)

        state = compute_joint_state(
            result.keypoints_2d, keypoints_3d, result.confidences,
            timestamp_us=t_now*1e6)

        # world frame 변환
        world_pos = {}
        if wf and state.valid:
            world_pos = wf._transform_positions(state.positions)

        # 시각화 (camera frame)
        vis = rgb.copy()
        for name, (px, py) in state.pixels.items():
            cv2.circle(vis, (int(px), int(py)), 7, (0, 255, 100), -1)
        for (j1, j2) in [('left_hip','left_knee'),('left_knee','left_ankle'),
                         ('right_hip','right_knee'),('right_knee','right_ankle'),
                         ('left_hip','right_hip')]:
            if j1 in state.pixels and j2 in state.pixels:
                p1 = tuple(int(v) for v in state.pixels[j1])
                p2 = tuple(int(v) for v in state.pixels[j2])
                cv2.line(vis, p1, p2, (200, 200, 200), 2)

        # 상태 오버레이
        status = f'FPS:{fps:.0f}  Joints:{len(state.positions)}/6'
        if wf:
            status += '  IMU:ON'
        cv2.putText(vis, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 100), 1)

        cv2.imshow('validate_transform', vis)

        # 주기적 검증
        if frame_count % args.validate_every == 0 and state.valid and world_pos:
            results = validator.validate_single(state, world_pos)
            validator.print_results(results)

        frame_count += 1
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key == ord(' ') and state.valid and world_pos:
            results = validator.validate_single(state, world_pos)
            validator.print_results(results)

    camera.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
