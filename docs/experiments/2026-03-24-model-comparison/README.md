# 2026-03-24 Pose Model Vendor Comparison

## 목적
6개 pose estimation library/모델을 Jetson Orin NX에서 동일 조건으로 측정 → H-Walker용 모델 선택 근거.

## 측정 환경
- Jetson Orin NX 16GB, MAXN
- ZED X Mini SVGA 960×600 @120fps
- TensorRT FP16 (지원 모델만)
- 각 모델 15초 측정

## 결과 요약
| 모델 | FPS | E2E (ms) | p95 | <50ms | 인식률 | Confidence |
|---|---|---|---|---|---|---|
| YOLOv8n-Pose | 27.7 | 35.7 | 38.2 | ✅ | 100% | 0.97 |
| YOLOv8s-Pose | 25.1 | 39.5 | 43.4 | ✅ | 100% | 0.99 |
| YOLO11s-Pose | 24.5 | 40.4 | 43.2 | ✅ | 100% | 0.99 |
| **YOLO26s-Pose** | **22.3** | **44.4** | **47.5** | ✅ | **100%** | **0.99** |
| MediaPipe (c=0) | 15.5 | 63.6 | 71.1 | ❌ | 94.4% | 0.77 |
| RTMPose (light) | 6.6 | 150.5 | 177.7 | ❌ | 100% | 0.60 |

## 결정
**YOLO26s 채택** — 가장 높은 confidence + budget 안. Fine-tuning 후 18ms로 가속됨 (2026-03-27).

## 데이터
```
data/                          # 6 timestamp × results.json (각 측정 raw)
├── 20260324_142547/results.json
├── 20260324_143326/results.json
├── ...
videos/                        # 데모 영상 (LFS)
├── YOLO26n.mp4 / YOLO26n_TRT.mp4
├── YOLO26s.mp4 / YOLO26s_TRT.mp4
├── RTMPose_WB_lightweight.mp4 / *_TRT.mp4
├── RTMPose_WB_balanced.mp4 / *_TRT.mp4
```

## Related
- `docs/lessons/model_selection_01.md` — 선택 근거 상세
- `docs/evolution/perception-evolution.md` §2 — Library 비교 종합
