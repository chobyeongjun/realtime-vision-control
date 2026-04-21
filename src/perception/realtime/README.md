# hw_perception/realtime — 실시간 하지 자세 추정 파이프라인

ZED X Mini + YOLO26s-lower6 기반의 실시간 하지 6관절 3D 추정.
제어기(`hw_control`)에 무릎/고관절 굴곡 각도를 공급한다.

---

## 파일 구조

```
realtime/
├── joint_3d.py            # 핵심 3D 관절 추정 (재사용 가능한 라이브러리)
├── calibration.py         # Method A/B 좌표 기준점 설정 + 속도 비교
├── verify_geometry.py     # Step 0: 뼈 길이·각도 수치 확인
├── validate_transform.py  # Step 1: world frame 변환 정확도 검증 (V1~V5)
└── README.md
```

---

## 전제 조건

Jetson Orin NX 기준:

```bash
source venv/bin/activate        # system-site-packages 포함 venv
pip install ultralytics rtmlib  # 이미 설치되어 있으면 생략
```

모델 파일 위치 (이미 있음):
```
src/hw_perception/models/
├── yolo26s-lower6.pt
└── yolo26s-lower6-v2.pt    ← 권장
```

---

## 실행 순서

### Step 0 — 3D 좌표 및 뼈 길이 확인

카메라를 Walker에 장착한 상태에서 사람이 카메라 아래에 서서 확인한다.

```bash
cd src/hw_perception/realtime

# ZED 실시간
python3 verify_geometry.py

# SVO2 파일로 오프라인 테스트
python3 verify_geometry.py --svo2 ../benchmarks/recordings/walk.svo2

# TRT 없이 빠른 확인
python3 verify_geometry.py --no-trt
```

**확인 항목:**

| 항목 | 정상 범위 | 의미 |
|------|-----------|------|
| thigh 길이 | 0.33 ~ 0.52 m | 대퇴골 |
| shank 길이 | 0.30 ~ 0.48 m | 경골 |
| 무릎 각도 (서 있을 때) | 170 ~ 180° | 완전 신전 |
| depth 순서 | hip < knee < ankle | 카메라가 위에 있음 |

**키 조작:**
- `스페이스바` : 현재 프레임 즉시 콘솔 출력
- `q` / `ESC` : 종료

---

### Step 1 — World Frame 변환 검증

ZED IMU로 얻은 rotation matrix가 해부학적으로 올바른지 5가지 조건으로 검증한다.

```bash
python3 validate_transform.py

# 파일 사용
python3 validate_transform.py --svo2 ../benchmarks/recordings/walk.svo2
```

**검증 항목 (V1~V5):**

| 항목 | 조건 | 실패 시 의미 |
|------|------|-------------|
| V1 뼈 길이 보존 | diff < 0.5 mm | R 행렬 계산 버그 |
| V2 수직 방향 | ankle 방향 일관 | 중력 정렬 실패 |
| V3 좌우 대칭 | 양 hip 높이 diff < 3 cm | IMU 중력 오정렬 |
| V4 각도 불변 | diff < 0.5° | R 수치 오류 |
| V5 허벅지 수직 | 기울기 < 30° | world frame 방향 오류 |

**모든 항목이 PASS여야** 다음 단계(제어기 연결)로 진행한다.

---

### Step 2 — 캘리브레이션 방식 선택

```bash
# Method A vs B 속도 비교
python3 -c "from calibration import run_latency_benchmark; run_latency_benchmark()"
```

| 방식 | 설명 | 장점 | 단점 |
|------|------|------|------|
| **A** StandingCalibration | 피험자가 서 있는 5초 동안 neutral 측정 | 단순, IMU 불필요 | 피험자마다 매번 필요 |
| **B** ZEDIMUWorldFrame | ZED 내장 IMU → world frame 자동 정렬 | 카메라 위치 변경에 강건 | pyzed 필수 |

**권장:** `validate_transform.py` V1~V5 모두 PASS이면 **B 사용**.
실패하면 **A 사용**.

---

## 코드 사용법 (다른 모듈에서 import)

### 기본 파이프라인

```python
from realtime.joint_3d import compute_joint_state, JointState3D
from realtime.calibration import ZEDIMUWorldFrame, FlexionAngles

# --- 초기화 ---
wf = ZEDIMUWorldFrame(camera)
wf.init(warm_up_frames=30)          # IMU warm-up

# --- 매 프레임 ---
result      = model.predict(rgb)    # YOLO26s-lower6
raw_3d      = batch_2d_to_3d(result.keypoints_2d, depth, camera)
state       = compute_joint_state(result.keypoints_2d, raw_3d, result.confidences)
flexion     = wf.to_flexion(state)  # → FlexionAngles

# --- 제어기로 전달 ---
# flexion.left_knee   [deg]  0° = 완전 신전, 60° = peak swing
# flexion.right_knee  [deg]
# flexion.left_hip    [deg]
# flexion.right_hip   [deg]
# flexion.valid       bool
```

### Method A (ZED IMU 없을 때)

```python
from realtime.calibration import StandingCalibration

cal = StandingCalibration(n_frames=30)

# --- 캘리브레이션 (서 있는 상태에서 ~1초) ---
while not cal.done:
    state = compute_joint_state(...)
    cal.update(state)               # 30프레임 모이면 자동 확정

# 캘리브레이션 결과 저장 (피험자별 재사용)
cal.save('calib_subject01.json')

# --- 보행 중 ---
flexion = cal.to_flexion(state)     # → FlexionAngles
```

---

## 출력 형식 (`FlexionAngles`)

```python
@dataclass
class FlexionAngles:
    left_knee:  float | None   # 0° ~ 70°, 굴곡할수록 증가
    right_knee: float | None
    left_hip:   float | None   # 0° ~ 35°, 굴곡할수록 증가
    right_hip:  float | None
    method:     str            # 'A' or 'B'
    latency_us: float          # 이 계산에 걸린 시간 [μs]
    valid:      bool           # False면 관절 미감지
```

Winter 2009 reference와 동일한 convention:
- 완전 신전 = 0°
- 굴곡할수록 양수

---

## 알려진 제약

| 항목 | 내용 |
|------|------|
| 카메라 FPS | 50Hz (SVGA 기준) |
| 추론 지연 | ~16 ms (YOLO26s FP16 TRT) |
| 전체 파이프라인 지연 | ~30 ms |
| 유효 관절 최소 수 | 4개 이상 (6개 중) |
| depth 유효 거리 | 0.1 ~ 3.0 m |
| confidence 임계값 | 0.5 (joint_3d.py `CONF_THRESHOLD`) |

---

## 문제 해결

**`LowerBodyPoseModel model_path 미지정` 오류**
→ `--model` 옵션으로 `.pt` 파일 경로를 명시하거나 기본 경로 확인:
```
src/hw_perception/models/yolo26s-lower6-v2.pt
```

**뼈 길이 OUT OF RANGE**
→ confidence가 낮아 keypoint가 엉뚱한 곳에 찍힘.
`CONF_THRESHOLD` 값을 높이거나 조명/거리 조정.

**V3 bilateral symmetry FAIL**
→ 카메라가 정중앙이 아닌 한쪽으로 치우쳐 있거나
ZED IMU가 아직 안정화 안 됨. `wf.init(warm_up_frames=50)`으로 늘리기.

**depth 값이 모두 NaN**
→ ZED depth mode가 OFF. `depth_mode='PERFORMANCE'`로 설정 확인.
