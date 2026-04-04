# 하체 전용 Pose Estimation 모델 학습 인수인계서

> 작성일: 2026-03-23
> 프로젝트: RealTime_Pose_Estimation
> 목적: 새 채팅 세션에서 이 문서를 기반으로 학습 계획 수립 및 실행

---

## 1. 프로젝트 배경

### 1.1 현재 시스템
- **목적**: 워커(이동식 보행기) 사용자의 실시간 하지 포즈 추정 → 로봇 제어
- **추론 장비**: NVIDIA Jetson Orin NX 16GB (JetPack 6.x)
- **카메라**: ZED X Mini (SVGA 960×600 @ 120fps, NEURAL depth)
- **현재 모델**: YOLO11-pose (COCO 17 키포인트, 발목까지만)

### 1.2 문제점
- COCO 17 키포인트는 **발목(ankle)까지만** 제공
- 보행 분석에 필요한 **Toe(발가락)와 Heel(뒤꿈치)** 키포인트가 없음
- 전신 모델(133 키포인트)은 불필요한 상체/얼굴/손 키포인트로 리소스 낭비

### 1.3 해결 방향
- **하체 전용 경량 모델**을 별도 학습
- 기존 전신 YOLO로 사람 검출 → 하체 crop → 하체 전용 모델로 정밀 추론 (2-Stage)
- 또는 하체 전용 모델 단독 사용 (1-Stage)

---

## 2. 학습 목표

### 2.1 하체 전용 키포인트 정의 (10개)

```
인덱스   이름              좌우
──────────────────────────────
  0     left_hip          좌
  1     right_hip         우
  2     left_knee         좌
  3     right_knee        우
  4     left_ankle        좌
  5     right_ankle       우
  6     left_big_toe      좌
  7     right_big_toe     우
  8     left_heel         좌
  9     right_heel        우
```

- `flip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]` (좌우 대칭)
- small_toe는 보행 분석에 불필요하므로 제외
- hip은 crop 기준점 + 고관절 각도 계산에 필요

### 2.2 모델 사양
- **베이스 모델**: YOLO11n-pose 또는 YOLO11s-pose
- **입력 크기**: 320×320 또는 416×416 (하체 crop이므로 작아도 됨)
- **키포인트**: 10개 (kpt_shape: [10, 3])
- **출력**: 하체 bbox + 10 키포인트 좌표 + confidence

---

## 3. 학습 데이터

### 3.1 COCO-WholeBody 데이터셋 (주 데이터소스)

- **출처**: https://github.com/jin-s13/COCO-WholeBody
- **이미지**: COCO 2017 Train/Val (기존 COCO 이미지 그대로 사용)
  - Train: http://images.cocodataset.org/zips/train2017.zip (~18GB)
  - Val: http://images.cocodataset.org/zips/val2017.zip (~1GB)
- **어노테이션**: COCO-WholeBody V1.0 (Google Drive에서 다운로드)
- **포함 키포인트**: 133개 (17 body + 6 foot + 68 face + 42 hand)
- **발 키포인트 6개**:
  - idx 17: left_big_toe
  - idx 18: left_small_toe
  - idx 19: left_heel
  - idx 20: right_big_toe
  - idx 21: right_small_toe
  - idx 22: right_heel
- **foot_valid 필드**: 발이 명확히 보이는 경우만 True
- **주의**: 20%+ 발 어노테이션이 불완전 (1-2개 키포인트 누락)

### 3.2 CMU Foot Keypoint Dataset (보조/참고)

- **출처**: https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/
- **규모**: 14K 학습용 + 545 검증용 (COCO 이미지 기반)
- **키포인트 6개**: big_toe, small_toe, heel (좌/우)

### 3.3 YOLO 포맷 변환 참고

- https://github.com/Eva20150932/coco-foot-and-leg (YOLO 포맷 변환 예시)

---

## 4. 데이터 전처리 파이프라인

### 4.1 COCO-WholeBody → 하체 전용 변환 순서

```
1) COCO-WholeBody JSON 로드
2) 각 person annotation에서:
   a) foot_valid == True인 것만 필터링
   b) body keypoints에서 hip(11,12), knee(13,14), ankle(15,16) 추출
   c) foot keypoints에서 big_toe(0,1), heel(4,5) 추출 (small_toe 제외)
   d) 모든 키포인트가 visible(v>0)인 것만 사용
3) 하체 bounding box 계산:
   - x_min, x_max: 모든 하체 키포인트의 x 범위 + 여유(20%)
   - y_min: hip_y - 여유(15%)
   - y_max: toe/heel의 y + 여유(10%)
4) 이미지에서 하체 영역 crop
5) 키포인트 좌표를 crop 기준으로 재계산 (normalize to 0~1)
6) YOLO 포맷으로 저장:
   - labels: class x_center y_center width height kx1 ky1 kv1 kx2 ky2 kv2 ... (10개)
```

### 4.2 변환 스크립트 (작성 필요)

```python
# convert_coco_wholebody_to_lower_body.py
#
# 입력: COCO-WholeBody annotation JSON + COCO 2017 이미지
# 출력: YOLO 포맷 하체 전용 데이터셋
#
# 주요 기능:
#   - foot_valid 필터링
#   - 하체 crop + 키포인트 좌표 변환
#   - 불완전한 발 어노테이션 처리 (v=0으로 마킹)
#   - train/val 분리 유지
#   - 데이터 증강 옵션 (flip, scale, brightness)
```

---

## 5. 학습 환경

### 5.1 학습 서버 사양
- **GPU**: NVIDIA RTX 5090 × 2장
- **OS**: Linux
- **용도**: 모델 학습 전용 (추론은 Jetson에서)

### 5.2 필수 소프트웨어 설치

```bash
# Python 환경
python3 -m venv lower_body_train
source lower_body_train/bin/activate

# 핵심 패키지
pip install ultralytics           # YOLO11
pip install pycocotools           # COCO 어노테이션 파싱
pip install opencv-python-headless
pip install numpy pandas matplotlib
pip install tqdm pillow

# Multi-GPU 학습용
# ultralytics 내장 DDP 지원 (별도 설치 불필요)
```

### 5.3 데이터 디렉토리 구조

```
lower_body_training/
├── convert_coco_wholebody_to_lower_body.py   # 변환 스크립트
├── lower_body_pose.yaml                       # YOLO 학습 설정
├── train_lower_body.py                        # 학습 스크립트
├── export_for_jetson.py                       # Jetson용 모델 변환
│
├── data/
│   ├── coco/                   # COCO 원본
│   │   ├── train2017/          # 이미지
│   │   ├── val2017/
│   │   └── annotations/       # COCO-WholeBody JSON
│   │
│   └── lower_body/             # 변환된 하체 전용 데이터
│       ├── train/
│       │   ├── images/
│       │   └── labels/
│       └── val/
│           ├── images/
│           └── labels/
│
└── runs/                       # 학습 결과
    └── pose/
        └── lower_body_v1/
            └── weights/
                ├── best.pt     # → Jetson으로 전송
                └── last.pt
```

---

## 6. 학습 설정

### 6.1 YOLO 데이터 설정 파일

```yaml
# lower_body_pose.yaml
path: ./data/lower_body
train: train/images
val: val/images

kpt_shape: [10, 3]   # 10 키포인트, 각각 (x, y, visibility)

names:
  0: lower_body

flip_idx: [1, 0, 3, 2, 5, 4, 7, 6, 9, 8]
```

### 6.2 학습 하이퍼파라미터

```python
from ultralytics import YOLO

model = YOLO("yolo11s-pose.pt")  # small 모델 (nano도 가능)

model.train(
    data="lower_body_pose.yaml",
    epochs=200,
    imgsz=416,            # 하체 crop이므로 크지 않아도 됨
    batch=128,            # RTX 5090 x2이면 큰 배치 가능
    device=[0, 1],        # 듀얼 GPU
    workers=16,
    patience=30,          # early stopping
    cos_lr=True,          # cosine lr scheduler

    # 데이터 증강
    flipud=0.0,           # 상하 반전 비활성화 (하체 방향 고정)
    fliplr=0.5,           # 좌우 반전
    mosaic=1.0,           # 모자이크 증강
    scale=0.5,            # 스케일 변화

    # 저장
    project="runs/pose",
    name="lower_body_v1",
)
```

### 6.3 예상 학습 시간
- RTX 5090 x2, batch 128, 200 epochs
- COCO-WholeBody 유효 데이터 ~60K 이미지 기준
- **예상: 2~4시간**

---

## 7. 모델 내보내기 (Jetson 배포용)

### 7.1 ONNX 변환

```python
from ultralytics import YOLO

model = YOLO("runs/pose/lower_body_v1/weights/best.pt")

# ONNX 내보내기
model.export(
    format="onnx",
    imgsz=416,
    simplify=True,
    opset=17,
)
# → best.onnx 생성
```

### 7.2 TensorRT 변환 (Jetson에서 실행)

```python
# Jetson Orin NX에서 실행
from ultralytics import YOLO

model = YOLO("best.onnx")
model.export(
    format="engine",
    imgsz=416,
    half=True,        # FP16
    device=0,
)
# → best.engine 생성 (Jetson에서 최적 추론)
```

### 7.3 Jetson 전송

```bash
# 학습 서버 → Jetson
scp runs/pose/lower_body_v1/weights/best.pt user@jetson:/path/to/models/
scp runs/pose/lower_body_v1/weights/best.onnx user@jetson:/path/to/models/
# TensorRT 엔진은 Jetson에서 직접 빌드해야 함 (아키텍처 다름)
```

---

## 8. Jetson 통합 계획

### 8.1 추론 파이프라인 (2-Stage)

```
[Frame from ZED X Mini]
         │
    ┌────▼────┐
    │ Stage 1  │  기존 YOLO11-pose (전신)
    │ 전신검출  │  → 사람 bbox + hip/ankle 위치
    └────┬────┘
         │ hip~ankle 영역 crop (+여유)
    ┌────▼────┐
    │ Stage 2  │  하체 전용 모델 (새로 학습)
    │ 하체정밀  │  → hip, knee, ankle, toe, heel
    └────┬────┘
         │
    ┌────▼────┐
    │ 후처리   │  관절각도 계산 (joint_angles.py 확장)
    │ + 3D    │  + ZED depth로 3D 좌표 생성
    └────┬────┘
         │
    ┌────▼────┐
    │ 로봇    │  제어 신호 출력
    │ 제어    │
    └─────────┘
```

### 8.2 기존 코드 수정 필요 사항

| 파일 | 수정 내용 |
|------|----------|
| `pose_models.py` | `LowerBodyPoseModel` 클래스 추가 (10 키포인트) |
| `joint_angles.py` | toe/heel 기반 발목 각도 계산 추가 |
| `metrics_3d.py` | toe/heel 뼈 길이 메트릭 추가 |
| `run_benchmark.py` | 하체 모델 벤치마크 옵션 추가 |
| `zed_camera.py` | 변경 없음 |

### 8.3 새로운 키포인트 매핑

```python
# 기존 COCO 17 (발목까지)
COCO_LOWER = {
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16,
}

# 하체 전용 모델 (새 모델)
LOWER_BODY_KPT = {
    'left_hip': 0, 'right_hip': 1,
    'left_knee': 2, 'right_knee': 3,
    'left_ankle': 4, 'right_ankle': 5,
    'left_big_toe': 6, 'right_big_toe': 7,
    'left_heel': 8, 'right_heel': 9,
}
```

---

## 9. 성능 목표

| 메트릭 | 목표 |
|--------|------|
| 하체 모델 추론 시간 (Jetson, TRT FP16) | < 5ms |
| 전체 파이프라인 (Stage1 + Stage2) | < 15ms (~66fps) |
| Toe/Heel 키포인트 정확도 (PCK@0.05) | > 70% |
| 검출률 (하체 보일 때) | > 95% |

---

## 10. 작업 순서 체크리스트

```
Phase 1: 환경 준비 (학습 서버)
  □ Python venv 생성 + 패키지 설치
  □ COCO 2017 이미지 다운로드 (train + val)
  □ COCO-WholeBody 어노테이션 다운로드

Phase 2: 데이터 전처리
  □ convert_coco_wholebody_to_lower_body.py 작성
  □ COCO-WholeBody → 하체 전용 YOLO 포맷 변환
  □ 변환 결과 검증 (시각화로 키포인트 확인)
  □ 데이터 통계 확인 (유효 샘플 수, 키포인트 분포)

Phase 3: 모델 학습
  □ lower_body_pose.yaml 작성
  □ YOLO11-pose 학습 실행 (RTX 5090 x2)
  □ 학습 곡선 모니터링 (loss, mAP)
  □ best.pt 검증 (val 데이터셋)

Phase 4: 모델 내보내기 + 전송
  □ ONNX 변환
  □ Jetson으로 전송
  □ Jetson에서 TensorRT 엔진 빌드
  □ 추론 속도 확인

Phase 5: Jetson 통합
  □ pose_models.py에 LowerBodyPoseModel 추가
  □ joint_angles.py에 toe/heel 각도 계산 추가
  □ 2-Stage 파이프라인 구현
  □ 실시간 데모 테스트
  □ 벤치마크 실행
```

---

## 11. 참고 링크

- COCO-WholeBody: https://github.com/jin-s13/COCO-WholeBody
- CMU Foot Keypoints: https://cmu-perceptual-computing-lab.github.io/foot_keypoint_dataset/
- COCO Foot-and-Leg (YOLO): https://github.com/Eva20150932/coco-foot-and-leg
- Ultralytics YOLO Pose: https://docs.ultralytics.com/tasks/pose/
- Ultralytics Multi-GPU: https://docs.ultralytics.com/guides/distributed-training/

---

## 12. 새 채팅 시작 시 프롬프트 예시

```
이 인수인계서를 읽고 하체 전용 Pose Estimation 모델 학습을 진행해줘.

환경:
- 학습 서버: Linux, RTX 5090 x2
- 배포 대상: Jetson Orin NX 16GB

HANDOVER_LowerBody_Training.md 파일을 참고해서
Phase 1부터 순서대로 진행해줘.
```
