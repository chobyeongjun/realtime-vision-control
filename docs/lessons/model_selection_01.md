# YOLO vs RTMPose — 왜 YOLO를 선택했는가

## 결론 (한 줄 요약)
> Jetson Orin NX에서 TensorRT FP16 기준, YOLO 계열만 E2E < 50ms 목표를 달성했다. RTMPose는 이론상 빠르지만 Jetson에서 CUDA EP가 활성화되지 않아 CPU fallback으로 동작했다.

---

## 배경

H-Walker 제어를 위해 E2E Latency < 50ms 요구사항 존재.  
후보 모델: YOLOv8/11/26-Pose, RTMPose (17kpt), RTMPose Wholebody (133kpt), MediaPipe, MoveNet, ZED Body Tracking.

---

## 실험 / 관찰

Jetson Orin NX 16GB + ZED X Mini (SVGA@120fps)에서 각 15초 측정.

| 모델 | E2E(ms) | <50ms 달성 | 비고 |
|------|---------|-----------|------|
| YOLOv8n-Pose | 35.7 | ✅ 100% | TRT FP16 |
| YOLOv8s-Pose | 39.5 | ✅ 100% | TRT FP16 |
| YOLO26s-Pose | 44.4 | ✅ 99.4% | TRT FP16 |
| RTMPose (lightweight) | 150.5 | ❌ 0% | CPU fallback |
| RTMPose Wholebody | 245.6 | ❌ 0% | CPU fallback |
| MediaPipe (c=0) | 63.6 | ❌ 0% | TFLite CPU |

---

## 교훈

**RTMPose의 실패 원인**: `onnxruntime-gpu` 설치 시 `CUDAExecutionProvider` 경고 발생 → CPU fallback으로 동작. Jetson aarch64에서 CUDA EP를 제대로 활성화하려면 Jetson 전용 onnxruntime-gpu 빌드가 필요하며, 일반 PyPI 버전으로는 TRT EP/CUDA EP가 동작하지 않는다.

**YOLO의 강점**: TensorRT FP16 엔진을 Ultralytics가 직접 내보내므로, Jetson TensorRT와 완전히 통합된다. 별도 EP 설정 없이 GPU 추론이 보장된다.

---

## 왜 이 방법인가 (설계 근거)

**YOLO 대신 RTMPose를 쓰면 좋은 이유 (이론)**  
- Top-down 방식 → 사람 crop 후 pose 추정 → 가려짐에 강함  
- SimCC 헤드로 heatmap 대신 분류 기반 keypoint 예측 → 빠름  
- COCO 기준 정확도가 YOLO보다 높음

**그런데 Jetson에서 YOLO를 선택한 이유 (현실)**  
- Jetson에서 RTMPose CUDA 가속 활성화 난이도 높음 (Jetson 전용 빌드 필요)  
- YOLO는 TRT FP16 워크플로우가 Ultralytics에 내장 → 설정 없이 바로 GPU 가속  
- E2E 35~44ms로 목표 달성, Confidence 0.97~0.99로 안정적  

**YOLO26 대신 YOLOv8을 쓰는 이유**  
- YOLO26은 NMS-Free + RLE loss로 논문 COCO mAP는 높지만 Jetson GPU에서 NMS-Free 이점이 적음  
- 실측: YOLOv8n(35.7ms) < YOLO26n(44.2ms) — 오히려 구 버전이 빠름  

---

## References

- [RTMPose: Real-Time Multi-Person Pose Estimation (2023)](https://arxiv.org/abs/2303.07399)
- [YOLOv8 Technical Report - Ultralytics](https://docs.ultralytics.com)
- [YOLO26 / YOLO12 Release Notes](https://docs.ultralytics.com/models/)
- [TensorRT Execution Provider - ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)
- [Jetson onnxruntime-gpu 전용 빌드](https://eaidynamic.github.io/onnxruntime-jetson/)
