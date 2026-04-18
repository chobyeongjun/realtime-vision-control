# PPT 생성 프롬프트

> 이 문서를 다른 AI 채팅에 붙여넣으면 python-pptx로 PPT를 자동 생성할 수 있습니다.

---

## 프롬프트 시작

아래 데이터와 구조를 기반으로 `python-pptx`를 사용하여 PPT 파일을 생성하는 Python 스크립트를 작성해주세요.

### 요구사항
- 파일명: `hwalker_pose_benchmark.pptx`
- 디자인: 다크 배경 (#1a1a2e) + 흰색 텍스트, 포인트 컬러 파란색 (#4fc3f7)
- 차트: matplotlib로 생성 → 이미지로 삽입
- 테이블: python-pptx 네이티브 테이블
- 언어: 한국어
- 총 10 슬라이드

---

### 슬라이드 1: 표지

```
제목: H-Walker 실시간 하체 포즈 추정
부제: 모델 비교 분석 결과 보고
날짜: 2026-03-24
환경: Jetson Orin NX 16GB + ZED X Mini (SVGA@120fps, Global Shutter)
핵심 목표: E2E Latency < 50ms | 하체 Keypoint 정확도 | 3D 포즈
```

---

### 슬라이드 2: 비교 대상 모델

테이블 데이터:

| 모델 | Keypoints 수 | Foot KP | 가속 방식 | 특징 |
|------|-------------|---------|----------|------|
| MediaPipe Pose | 33 | O | CPU (TFLite) | Google, 경량 |
| YOLO-Pose (v8/11/26) | 17 (COCO) | X | TensorRT FP16 | Ultralytics, GPU 최적화 |
| RTMPose | 17 (COCO) | X | ONNX Runtime | OpenMMLab, Top-down |
| RTMPose Wholebody | 133 | O | ONNX Runtime | 전신+발+손 |
| MoveNet | 17 | X | TFLite | Google, 초경량 |
| ZED Body Tracking | 38 | O | ZED SDK 내장 | 3D 직접 제공 |

---

### 슬라이드 3: 벤치마크 결과 요약 (핵심)

상단 텍스트: "12개 모델 성공, 3개 에러 — E2E <50ms 달성은 YOLO 계열 6개뿐"

테이블 데이터 (이 데이터를 테이블로 만들어주세요):

```python
benchmark_data = [
    # (모델명, FPS, Infer_ms, E2E_ms, P95_E2E, under50_pct, detect_pct, lower_pct, conf, foot)
    ("YOLOv8n-Pose",     27.7, 27.4, 35.7, 38.2,  100.0, 100.0, 100.0, 0.97, False),
    ("YOLOv8s-Pose",     25.1, 32.4, 39.5, 43.4,  100.0, 100.0, 100.0, 0.99, False),
    ("YOLO11n-Pose",     23.8, 32.7, 41.6, 44.7,  100.0, 100.0, 100.0, 0.99, False),
    ("YOLO11s-Pose",     24.5, 33.1, 40.4, 43.2,   99.7, 100.0, 100.0, 0.99, False),
    ("YOLO26n-Pose",     22.4, 35.3, 44.2, 46.5,  100.0, 100.0, 100.0, 0.97, False),
    ("YOLO26s-Pose",     22.3, 36.9, 44.4, 47.5,   99.4, 100.0, 100.0, 0.99, False),
    ("MediaPipe (c=0)",  15.5, 46.6, 63.6, 71.1,    0.0,  94.4,  94.4, 0.77, True),
    ("MediaPipe (c=1)",  12.5, 62.4, 79.3, 125.2,   1.6,  52.9,  51.9, 0.59, True),
    ("RTMPose (lw)",      6.6, 132.8, 150.5, 177.7,  0.0, 100.0, 100.0, 0.60, False),
    ("RTMPose (bal)",     1.4, 709.9, 726.9, 772.3,  0.0, 100.0, 100.0, 0.61, False),
    ("RTMPose WB (lw)",   4.1, 228.2, 245.6, 292.4,  0.0, 100.0, 100.0, 0.68, True),
    ("RTMPose WB (bal)",  1.1, 882.2, 899.9, 1046.7, 0.0, 100.0, 100.0, 0.73, True),
]
```

컬럼: 모델 | FPS | Infer(ms) | E2E(ms) | P95 E2E | <50ms% | 인식률% | Conf
- <50ms% 100인 행은 초록색 배경, 0인 행은 빨간색 배경으로 강조

---

### 슬라이드 4: E2E Latency 비교 차트

matplotlib **가로 막대 차트** (horizontal bar chart):
- Y축: 모델명 (E2E 오름차순 정렬)
- X축: ms
- 색상: <50ms 모델은 파란색, >50ms는 회색
- 50ms 위치에 빨간 세로 점선 + "목표: 50ms" 라벨
- 차트 제목: "E2E Latency 비교 (낮을수록 좋음)"

데이터:
```python
latency_chart = {
    "YOLOv8n-Pose": 35.7,
    "YOLOv8s-Pose": 39.5,
    "YOLO11s-Pose": 40.4,
    "YOLO11n-Pose": 41.6,
    "YOLO26n-Pose": 44.2,
    "YOLO26s-Pose": 44.4,
    "MediaPipe (c=0)": 63.6,
    "MediaPipe (c=1)": 79.3,
    "RTMPose (lw)": 150.5,
    "RTMPose WB (lw)": 245.6,
    "RTMPose (bal)": 726.9,
    "RTMPose WB (bal)": 899.9,
}
```

---

### 슬라이드 5: Latency 분해 차트

matplotlib **스택 바 차트** (상위 6개 모델만):
- X축: 모델명
- Y축: ms
- 3개 스택: Grab(파란), Inference(주황), PostProc(초록)
- 50ms 빨간 점선

데이터:
```python
latency_breakdown = {
    # (모델, grab_ms, infer_ms, postproc_ms)
    "YOLOv8n": (7.5, 27.4, 0.8),
    "YOLOv8s": (6.4, 32.4, 0.7),
    "YOLO11n": (8.2, 32.7, 0.7),
    "YOLO11s": (6.6, 33.1, 0.7),
    "YOLO26n": (8.2, 35.3, 0.7),
    "YOLO26s": (6.8, 36.9, 0.7),
}
```

---

### 슬라이드 6: YOLO 세대별 아키텍처 비교

테이블:

| 구분 | YOLOv8 (2023) | YOLO11 (2024) | YOLO26 (2026) |
|------|--------------|--------------|--------------|
| Backbone 블록 | C2f | C3k2 + C2PSA | C3k2, DFL 제거 |
| NMS | 필요 (후처리) | 필요 (후처리) | NMS-Free (제거) |
| Keypoint Loss | 기본 regression | 기본 regression | RLE (불확실성 모델링) |
| Optimizer | SGD/AdamW | SGD/AdamW | MuSGD |
| Params (n) | 3.3M | 2.9M | 2.9M |
| FLOPs (n) | 9.2B | 7.6B | 7.5B |
| COCO mAP (n) | 50.4 | 50.0 | 57.2 (+7.2) |
| CPU ONNX (n) | 80ms | 52ms | 40ms (-50%) |

하단 주석: "YOLO26은 정확도 최고(+7 mAP)이지만, Jetson GPU에서는 NMS-Free 이점이 적어 YOLOv8이 더 빠름"

---

### 슬라이드 7: TensorRT FP16 가속 효과

테이블 (YOLO26s 기준):

| 지표 | Non-TRT | TRT FP16 | 개선폭 |
|------|---------|----------|--------|
| Inference | 36.9ms | 22.7ms | -38.6% |
| E2E | 44.4ms | 38.7ms | -12.9% |
| FPS | 22.3 | 25.8 | +15.7% |
| P95 E2E | 47.5ms | 45.4ms | <50ms 달성 |

추가 텍스트:
- "Camera Grab (11.5ms)이 전체 E2E의 29.7% — Inference 다음 최대 병목"
- "TRT 적용 시 모든 YOLO 모델 <50ms 안정적 달성 가능"

---

### 슬라이드 8: 3D 안정성 & 관절 각도

2단 레이아웃:

왼쪽 - 3D Bone Length CV 테이블:
```python
bone_stability = {
    # (bone, CV, status, mean_m)
    "left_thigh":   (0.051, "WARN", 0.32),
    "right_thigh":  (0.035, "OK",   0.32),
    "left_shank":   (0.040, "OK",   0.33),
    "right_shank":  (0.107, "BAD",  0.32),
    "pelvis_width": (0.017, "OK",   0.18),
}
# OK < 0.05, WARN 0.05~0.08, BAD > 0.08
```

오른쪽 - 관절 각도 범위 (서있는 자세):
```python
joint_angles = {
    "left_knee_flexion":  (13.0, 9.0),   # mean, std
    "right_knee_flexion": (16.5, 6.2),
    "left_hip_flexion":   (91.7, 1.1),
    "right_hip_flexion":  (84.7, 2.1),
}
```

하단: "right_shank CV=0.107 (BAD) — ZED Depth 노이즈 영향, 필터링 필요"

---

### 슬라이드 9: 에러 모델 & 제외 사유

테이블:

| 모델 | 상태 | 에러 원인 | 해결 방법 |
|------|------|----------|----------|
| MoveNet Lightning | ERROR | NumPy 2.x 비호환 (tflite_runtime) | pip install "numpy<2" |
| MoveNet Thunder | ERROR | 동일 | 동일 |
| ZED Body Tracking | ERROR | Positional Tracking 미활성화 | enablePositionalTracking() 추가 |
| MediaPipe (c=1) | 제외 | 인식률 52.9% | 실사용 불가 |
| RTMPose 전체 | 참고 | CUDA EP 미사용 → CPU fallback | onnxruntime-gpu 재설치 |

---

### 슬라이드 10: 최종 추천 & 다음 단계

상단 - 추천 테이블 (강조):

| 시나리오 | 추천 모델 | E2E | FPS | 근거 |
|----------|----------|-----|-----|------|
| ★ 최고 속도 | YOLOv8n-Pose | 35.7ms | 27.7 | 가장 낮은 latency |
| ★ 속도+정확도 균형 | YOLOv8s-Pose | 39.5ms | 25.1 | Conf 0.99, 100% 인식 |
| ★ COCO 정확도 최고 | YOLO26s + TRT | 38.7ms | 25.8 | mAP 63.0 최고 |
| 발 KP 필요 시 | MediaPipe (c=0) | 63.6ms | 15.5 | 유일한 실용 옵션 |

하단 - 다음 단계:
1. YOLOv8s-Pose + TRT FP16 벤치마크 실행
2. MoveNet / ZED BT 에러 수정 후 재측정
3. 선정 모델로 보행 시나리오 3D 파이프라인 테스트
4. H-Walker 실제 탑재 및 실시간 동작 검증

---

## 스타일 가이드

```python
# 색상 팔레트
COLORS = {
    "bg_dark":     "#1a1a2e",   # 슬라이드 배경
    "text_white":  "#ffffff",   # 기본 텍스트
    "accent_blue": "#4fc3f7",   # 포인트 (제목, 강조)
    "accent_green":"#66bb6a",   # 달성/OK
    "accent_red":  "#ef5350",   # 미달/FAIL/BAD
    "accent_orange":"#ffa726",  # 경고/WARN
    "table_header":"#0d47a1",   # 테이블 헤더
    "table_row1":  "#1e1e3f",   # 테이블 홀수 행
    "table_row2":  "#16213e",   # 테이블 짝수 행
    "chart_pass":  "#4fc3f7",   # 차트 - 50ms 이하
    "chart_fail":  "#78909c",   # 차트 - 50ms 초과
}

# 폰트
FONT_TITLE = "맑은 고딕"  # 또는 "Noto Sans KR"
FONT_BODY = "맑은 고딕"
FONT_SIZE_TITLE = 28
FONT_SIZE_SUBTITLE = 18
FONT_SIZE_BODY = 14
FONT_SIZE_TABLE = 11

# 슬라이드 크기: 와이드스크린 16:9
```

## 프롬프트 끝
