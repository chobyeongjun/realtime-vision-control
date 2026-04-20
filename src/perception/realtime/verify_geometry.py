"""
verify_geometry.py
==================
Step 0 검증 스크립트: ZED 3D 좌표계와 관절 위치가 물리적으로 올바른지 확인.

실행:
    cd src/hw_perception/realtime
    python3 verify_geometry.py
    python3 verify_geometry.py --video ../benchmarks/recordings/walk.mp4
    python3 verify_geometry.py --svo2  ../benchmarks/recordings/walk.svo2

확인 항목:
    1. 카메라 좌표계 방향 (X/Y/Z가 실제로 어느 방향인지)
    2. 뼈 길이 (thigh 0.33~0.52m, shank 0.30~0.48m)
    3. 무릎 각도 (서 있을 때 ~170~180°)
    4. depth 분포 (hip < knee < ankle 이어야 함 — 카메라가 위에 있으므로)
    5. 좌우 대칭성

종료: q 또는 ESC
"""

import sys
import os
import time
import argparse
import numpy as np
import cv2

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
_this_dir  = os.path.dirname(os.path.abspath(__file__))
_bench_dir = os.path.join(_this_dir, '..', 'benchmarks')
_model_dir = os.path.join(_this_dir, '..', 'models')
for p in [_bench_dir, _model_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from zed_camera import create_camera, PipelinedCamera
from pose_models import LowerBodyPoseModel
from postprocess_accel import batch_2d_to_3d
from joint_3d import compute_joint_state, validate_bone_lengths, JOINT_NAMES

try:
    from trt_pose_engine import TRTPoseEngine
    HAS_DIRECT_TRT = True
except ImportError:
    HAS_DIRECT_TRT = False

# ── 시각화 색상 ────────────────────────────────────────────────────────────────
COLORS = {
    'left_hip':    (255, 100,  50),
    'right_hip':   ( 50, 100, 255),
    'left_knee':   (255, 200,  50),
    'right_knee':  ( 50, 200, 255),
    'left_ankle':  (255,  50, 200),
    'right_ankle': ( 50, 255, 200),
}
SKELETON = [
    ('left_hip',  'right_hip'),
    ('left_hip',  'left_knee'),
    ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'),
    ('right_knee','right_ankle'),
]


# ── Sagittal Plane 시각화 ─────────────────────────────────────────────────────

def draw_sagittal(state, fps: float, width=400, height=500) -> np.ndarray:
    """3D 좌표를 sagittal plane (Y-Z 측면뷰)으로 시각화.

    ZED 카메라 좌표계:
      Y = 아래 방향 (+), Z = depth 방향 (+)
    Sagittal view:
      가로 = Z (앞뒤), 세로 = -Y (위아래, 뒤집어서 위가 +)
    """
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)  # 어두운 배경

    # 제목
    cv2.putText(canvas, 'Sagittal Plane', (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, f'FPS: {fps:.1f}', (width - 100, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)

    if not state.valid or not state.positions:
        cv2.putText(canvas, 'No Detection', (width//2 - 60, height//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1, cv2.LINE_AA)
        return canvas

    # 3D 좌표 → sagittal 좌표 변환
    # Z(depth) → 가로, Y(vertical) → 세로(반전)
    positions = state.positions
    zs = [p[2] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    if not zs:
        return canvas

    z_min, z_max = min(zs), max(zs)
    y_min, y_max = min(ys), max(ys)

    # 스케일 + 마진
    margin = 60
    z_range = max(z_max - z_min, 0.01)
    y_range = max(y_max - y_min, 0.01)
    scale = min((width - 2 * margin) / z_range, (height - 2 * margin) / y_range) * 0.8
    z_center = (z_min + z_max) / 2
    y_center = (y_min + y_max) / 2

    def to_screen(y3d, z3d):
        sx = int(margin + (z3d - z_center) * scale + (width - 2 * margin) / 2)
        sy = int(margin + (y3d - y_center) * scale + (height - 2 * margin) / 2)
        return (sx, sy)

    # 좌우 다리 그리기
    sides = [
        ('left', (100, 200, 255), (50, 150, 255)),   # 왼쪽: 파란 계열
        ('right', (255, 150, 100), (255, 100, 50)),   # 오른쪽: 빨간 계열
    ]

    for side, line_color, joint_color in sides:
        hip_k = f'{side}_hip'
        knee_k = f'{side}_knee'
        ankle_k = f'{side}_ankle'

        pts = {}
        for name in [hip_k, knee_k, ankle_k]:
            if name in positions:
                y3d, z3d = positions[name][1], positions[name][2]
                pts[name] = to_screen(y3d, z3d)

        # 뼈 선
        if hip_k in pts and knee_k in pts:
            cv2.line(canvas, pts[hip_k], pts[knee_k], line_color, 3)
        if knee_k in pts and ankle_k in pts:
            cv2.line(canvas, pts[knee_k], pts[ankle_k], line_color, 3)

        # 관절 점
        for name, pt in pts.items():
            cv2.circle(canvas, pt, 7, joint_color, -1)
            cv2.circle(canvas, pt, 7, (255, 255, 255), 1)

    # hip 연결선 (waist)
    if 'left_hip' in positions and 'right_hip' in positions:
        lh = to_screen(positions['left_hip'][1], positions['left_hip'][2])
        rh = to_screen(positions['right_hip'][1], positions['right_hip'][2])
        cv2.line(canvas, lh, rh, (150, 150, 150), 2)

    # 각도 표시
    info_y = height - 140
    angle_data = [
        ('L Knee', state.left_knee_angle),
        ('R Knee', state.right_knee_angle),
        ('L Thigh', state.left_thigh_inclination),
        ('R Thigh', state.right_thigh_inclination),
    ]
    for label, val in angle_data:
        if val is not None:
            color = (100, 255, 100) if 'L' in label else (100, 200, 255)
            cv2.putText(canvas, f'{label}: {val:.1f} deg', (15, info_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            info_y += 20

    # 뼈 길이 표시
    info_y += 10
    for bone, length in sorted(state.bone_lengths.items()):
        color = (200, 200, 200)
        cv2.putText(canvas, f'{bone}: {length:.3f}m', (15, info_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)
        info_y += 16

    # 축 라벨
    cv2.putText(canvas, 'Z (depth) ->', (width - 120, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)
    cv2.putText(canvas, 'Y (vert)', (5, height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (100, 100, 100), 1, cv2.LINE_AA)

    return canvas


# ── 오버레이 렌더링 ────────────────────────────────────────────────────────────

def draw_overlay(img: np.ndarray, state, valid_bones: dict, fps: float) -> np.ndarray:
    vis = img.copy()
    h, w = vis.shape[:2]

    # 스켈레톤 선
    for j1, j2 in SKELETON:
        if j1 in state.pixels and j2 in state.pixels:
            p1 = tuple(int(v) for v in state.pixels[j1])
            p2 = tuple(int(v) for v in state.pixels[j2])
            cv2.line(vis, p1, p2, (180, 180, 180), 2)

    # 관절 점 + depth
    for name, (px, py) in state.pixels.items():
        pt = (int(px), int(py))
        color = COLORS.get(name, (200, 200, 200))
        cv2.circle(vis, pt, 8, color, -1)
        cv2.circle(vis, pt, 8, (255, 255, 255), 1)

        if name in state.positions:
            z = state.positions[name][2]
            cv2.putText(vis, f'{z:.2f}m', (pt[0]+10, pt[1]-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    # ── 정보 패널 (왼쪽 상단) ──
    lines = []
    lines.append(f'FPS: {fps:.1f}')
    lines.append(f'Joints: {len(state.positions)}/6')
    lines.append('')

    # 뼈 길이
    lines.append('-- Bone Lengths --')
    for bone, length in sorted(state.bone_lengths.items()):
        ok = valid_bones.get(bone, True)
        tag = 'OK' if ok else 'WARN'
        lines.append(f'  {bone:<16} {length:.3f}m [{tag}]')
    lines.append('')

    # 관절 각도
    lines.append('-- Joint Angles --')
    if state.left_knee_angle is not None:
        lines.append(f'  L knee  {state.left_knee_angle:6.1f} deg')
    if state.right_knee_angle is not None:
        lines.append(f'  R knee  {state.right_knee_angle:6.1f} deg')
    if state.left_thigh_inclination is not None:
        lines.append(f'  L thigh {state.left_thigh_inclination:6.1f} deg')
    if state.right_thigh_inclination is not None:
        lines.append(f'  R thigh {state.right_thigh_inclination:6.1f} deg')
    lines.append('')

    # depth 순서 확인 (hip < knee < ankle 이어야 함)
    lines.append('-- Depth Order (hip<knee<ankle?) --')
    for side in ('left', 'right'):
        hip_k, knee_k, ankle_k = f'{side}_hip', f'{side}_knee', f'{side}_ankle'
        if all(k in state.positions for k in (hip_k, knee_k, ankle_k)):
            zh = state.positions[hip_k][2]
            zk = state.positions[knee_k][2]
            za = state.positions[ankle_k][2]
            ok = zh < zk < za
            tag = 'OK' if ok else 'FAIL'
            lines.append(f'  {side:<5} hip={zh:.2f} knee={zk:.2f} ankle={za:.2f} [{tag}]')

    # 패널 배경
    pad = 8
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.42, 1
    panel_w = 280
    panel_h = len(lines) * 16 + pad * 2
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, vis, 0.4, 0, vis)

    for i, line in enumerate(lines):
        y = pad + i * 16 + 12
        color = (200, 255, 200) if line.startswith('--') else (220, 220, 220)
        cv2.putText(vis, line, (pad, y), font, fs, color, th, cv2.LINE_AA)

    # 좌표계 설명 (우하단)
    cx_info = [
        'ZED Camera Frame:',
        '  X = image right (+)',
        '  Y = image down  (+)',
        '  Z = depth away  (+)',
        'Walker mount: Z ~ vertical dist',
    ]
    for i, line in enumerate(cx_info):
        y = h - (len(cx_info) - i) * 16 - 4
        cv2.putText(vis, line, (w - 270, y), font, 0.38,
                    (180, 220, 255), 1, cv2.LINE_AA)

    return vis


def print_coordinate_summary(state):
    """콘솔에 좌표 요약 출력."""
    print('\n' + '='*55)
    print('3D 좌표 (카메라 좌표계, 단위: m)')
    print(f'  {"Joint":<14} {"X":>7} {"Y":>7} {"Z(depth)":>9} {"Conf":>6}')
    print('-'*55)
    for name in JOINT_NAMES:
        if name in state.positions:
            x, y, z = state.positions[name]
            c = state.confs.get(name, 0)
            print(f'  {name:<14} {x:7.3f} {y:7.3f} {z:9.3f} {c:6.2f}')
    print()
    print('뼈 길이:')
    for bone, length in sorted(state.bone_lengths.items()):
        lo, hi = {'thigh': (0.33, 0.52), 'shank': (0.30, 0.48)}.get(
            bone.split('_')[1], (0, 1))
        tag = 'OK' if lo <= length <= hi else '!! OUT OF RANGE !!'
        print(f'  {bone:<16} {length:.3f} m  {tag}')
    print()
    print('관절 각도:')
    for label, val in [
        ('L knee', state.left_knee_angle),
        ('R knee', state.right_knee_angle),
        ('L thigh', state.left_thigh_inclination),
        ('R thigh', state.right_thigh_inclination),
    ]:
        if val is not None:
            print(f'  {label}  {val:.1f} deg')
    print('='*55)


# ── 메인 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='ZED 3D 좌표계 검증')
    parser.add_argument('--video', type=str, default=None,
                        help='MP4 파일 경로 (없으면 ZED 실시간)')
    parser.add_argument('--svo2', type=str, default=None,
                        help='SVO2 파일 경로')
    # TRT 엔진이 있으면 직접 로드, 없으면 .pt로 폴백
    _default_engine = os.path.join(_model_dir, 'yolo26s-lower6-v2.engine')
    _default_model = _default_engine if os.path.exists(_default_engine) \
        else os.path.join(_model_dir, 'yolo26s-lower6-v2.pt')
    parser.add_argument('--model', type=str,
                        default=_default_model,
                        help='lower6 모델 경로 (.engine 또는 .pt)')
    parser.add_argument('--no-trt', action='store_true',
                        help='TRT 없이 PyTorch로 실행')
    parser.add_argument('--fps', type=int, default=120,
                        help='카메라 FPS (기본 120, ZED X Mini SVGA 최대)')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='모델 입력 해상도 (기본 640)')
    parser.add_argument('--direct-trt', action='store_true',
                        help='Ultralytics 우회, TRT 직접 호출 (최고 속도)')
    parser.add_argument('--no-display', action='store_true',
                        help='OpenCV 화면 표시 비활성화 (속도 최적화)')
    parser.add_argument('--print-interval', type=int, default=60,
                        help='콘솔 출력 간격 (프레임 수)')
    args = parser.parse_args()

    # ── 카메라 초기화 ──
    video_path = args.svo2 or args.video
    use_zed    = (video_path is None)

    print(f'[init] 카메라: {"ZED 실시간" if use_zed else video_path}')
    raw_camera = create_camera(
        use_zed=use_zed,
        video_path=video_path,
        resolution='SVGA',
        fps=args.fps,
        depth_mode='PERFORMANCE',
    )
    # PipelinedCamera: grab과 predict 병렬화, segfault 없음
    camera = PipelinedCamera(raw_camera)
    camera.open()

    # ── 모델 초기화 ──
    if args.direct_trt and HAS_DIRECT_TRT:
        # Ultralytics 완전 우회 — TRT 직접 호출
        # TRTPoseEngine.load()가 .direct.engine → .engine → ONNX 빌드 자동 처리
        engine_path = os.path.join(_model_dir, f'yolo26s-lower6-v2-{args.imgsz}.engine')
        print(f'[init] 모델: DirectTRT  imgsz={args.imgsz}')
        model = TRTPoseEngine(engine_path, imgsz=args.imgsz)
        model.load()
    else:
        use_trt = not args.no_trt
        print(f'[init] 모델: {args.model}  TRT={use_trt}  imgsz={args.imgsz}')
        model = LowerBodyPoseModel(
            model_path=args.model,
            use_tensorrt=use_trt,
            imgsz=args.imgsz,
            smoothing=0.0,
            segment_constraint=True,
        )
        model.load()

    # ── 워밍업 ──
    print('[init] 워밍업 20프레임...')
    warmup_done = 0
    while warmup_done < 20:
        rgb = camera.get()
        if rgb is not None:
            model.predict(rgb)
            camera.release()
            warmup_done += 1

    print('[ready] 실행 중. q/ESC로 종료')
    print('        서 있는 상태에서 뼈 길이와 각도를 확인하세요.')
    print()

    frame_count   = 0
    t_prev        = time.perf_counter()
    fps_display   = 0.0
    last_state    = None
    prev_kpts_2d  = {}
    prev_confs    = {}
    prev_3d       = {}   # 3D EMA smoothing
    SMOOTH_ALPHA  = 0.8  # 현재 80% + 이전 20% (딜레이 최소)

    # ── 프로파일링 ──
    profile_accum = {'fetch': 0, 'predict': 0, '2d_to_3d': 0,
                     'imu': 0, 'joint_state': 0, 'overlay': 0, 'imshow': 0}
    profile_count = 0
    PROFILE_INTERVAL = 120  # 120프레임마다 출력

    while True:
        _t0 = time.perf_counter()
        # PipelinedCamera: RGB만 가져옴 (grab은 캡처 스레드에서 완료)
        rgb = camera.get()
        _t1 = time.perf_counter()
        if rgb is None:
            continue

        t_now = time.perf_counter()
        fps_display = 0.9 * fps_display + 0.1 * (1.0 / max(t_now - t_prev, 1e-6))
        t_prev = t_now
        _t2 = _t1

        # ── 추론 (GPU 독점 — grab 안 돌아감) ──
        is_bgra = (rgb.ndim == 3 and rgb.shape[2] == 4)
        if is_bgra and hasattr(model, 'predict_bgra'):
            result = model.predict_bgra(rgb)
        elif is_bgra:
            result = model.predict(cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR))
        else:
            result = model.predict(rgb)

        # ── detection 실패 시 마지막 성공 결과 유지 ──
        if not result.detected and prev_kpts_2d:
            result.keypoints_2d = dict(prev_kpts_2d)
            result.confidences = dict(prev_confs)
            result.detected = True
        if result.detected:
            prev_kpts_2d = dict(result.keypoints_2d)
            prev_confs = dict(result.confidences)
        _t3 = time.perf_counter()

        # ── predict 끝난 후 depth + IMU (GPU 독점) ──
        depth = camera.get_depth()
        keypoints_3d = {}
        if result.detected and depth is not None:
            raw_3d = batch_2d_to_3d(result.keypoints_2d, depth, camera.camera)
            # 3D EMA smoothing
            for name, pos in raw_3d.items():
                if name in prev_3d:
                    px, py, pz = prev_3d[name]
                    cx, cy, cz = pos
                    raw_3d[name] = (
                        SMOOTH_ALPHA * cx + (1 - SMOOTH_ALPHA) * px,
                        SMOOTH_ALPHA * cy + (1 - SMOOTH_ALPHA) * py,
                        SMOOTH_ALPHA * cz + (1 - SMOOTH_ALPHA) * pz,
                    )
                prev_3d[name] = raw_3d[name]
            keypoints_3d = raw_3d
        _t4 = time.perf_counter()

        world_up = camera.get_gravity_vector()
        _t5 = time.perf_counter()

        # ── 다음 grab 시작 허용 ──
        camera.release()

        # ── JointState3D 계산 ──
        state = compute_joint_state(
            keypoints_2d=result.keypoints_2d,
            keypoints_3d=keypoints_3d,
            confidences=result.confidences,
            timestamp_us=t_now * 1e6,
            world_up_vec=world_up,
        )
        last_state = state
        _t6 = time.perf_counter()

        # ── 뼈 길이 검증 ──
        valid_bones = validate_bone_lengths(state) if state.valid else {}

        # ── 시각화 ──
        if not args.no_display:
            display_img = cv2.cvtColor(rgb, cv2.COLOR_BGRA2BGR) if is_bgra else rgb
            vis = draw_overlay(display_img, state, valid_bones, fps_display)
            sagittal = draw_sagittal(state, fps_display)
            _t7 = time.perf_counter()
            cv2.imshow('verify_geometry', vis)
            cv2.imshow('sagittal', sagittal)
            _t8 = time.perf_counter()
        else:
            _t7 = time.perf_counter()
            _t8 = _t7

        # ── 프로파일 누적 ──
        profile_accum['fetch']      += _t1 - _t0   # async에서는 ~0ms (이미 준비됨)
        profile_accum['predict']    += _t3 - _t2
        profile_accum['2d_to_3d']   += _t4 - _t3
        profile_accum['imu']        += _t5 - _t4
        profile_accum['joint_state']+= _t6 - _t5
        profile_accum['overlay']    += _t7 - _t6
        profile_accum['imshow']     += _t8 - _t7
        profile_count += 1

        if profile_count >= PROFILE_INTERVAL:
            total = sum(profile_accum.values())
            print(f'\n[PROFILE] {profile_count}프레임 평균 (총 {total/profile_count*1000:.1f}ms/frame):')
            for k, v in profile_accum.items():
                avg_ms = v / profile_count * 1000
                pct = v / total * 100 if total > 0 else 0
                print(f'  {k:<12} {avg_ms:6.1f}ms  ({pct:4.1f}%)')
            print()
            profile_accum = {k: 0 for k in profile_accum}
            profile_count = 0

        # ── 콘솔 출력 ──
        if frame_count % args.print_interval == 0 and state.valid:
            print_coordinate_summary(state)

        frame_count += 1

        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
        # 스페이스바: 현재 프레임 강제 출력
        elif key == ord(' ') and last_state is not None:
            print_coordinate_summary(last_state)

    # ── 정리 ──
    camera.close()
    cv2.destroyAllWindows()

    if last_state and last_state.valid:
        print('\n[최종 상태]')
        print_coordinate_summary(last_state)


if __name__ == '__main__':
    main()
