# 논문 레퍼런스

실험 결정과 설계에 참고한 논문/문서 모음.

---

## Pose Estimation 모델

| 논문 | 저자 | 연도 | 핵심 기여 | 링크 |
|------|------|------|----------|------|
| RTMPose | Jiang et al. | 2023 | SimCC 헤드, 실시간 Top-down pose | [arXiv:2303.07399](https://arxiv.org/abs/2303.07399) |
| YOLOv8 | Ultralytics | 2023 | C2f backbone, Pose head 통합 | [Ultralytics Docs](https://docs.ultralytics.com) |
| MediaPipe Pose | Bazarevsky et al. | 2020 | BlazePose, 33kpt, 실시간 CPU | [arXiv:2006.10204](https://arxiv.org/abs/2006.10204) |
| MoveNet | Google | 2021 | CenterNet 기반 초경량 pose | [TF Hub](https://tfhub.dev/google/movenet) |

---

## TensorRT / 추론 가속

| 문서 | 설명 | 링크 |
|------|------|------|
| TensorRT Developer Guide | FP16/INT8 quantization, engine build | [NVIDIA Docs](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/) |
| ONNX Runtime TRT EP | Jetson에서 TensorRT Execution Provider | [ORT Docs](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html) |
| Jetson onnxruntime-gpu | Jetson aarch64 전용 빌드 | [eaidynamic.github.io](https://eaidynamic.github.io/onnxruntime-jetson/) |

---

## 신호 처리 / 필터링

| 논문 | 설명 | 링크 |
|------|------|------|
| One Euro Filter | 실시간 keypoint 떨림 감소 | [casiez.net](https://gery.casiez.net/1euro/) |

---

## 데이터셋

| 데이터셋 | 설명 | 링크 |
|---------|------|------|
| COCO 2017 Keypoints | 17kpt, 118K 학습 이미지 | [cocodataset.org](https://cocodataset.org/#keypoints-2020) |
| COCO-WholeBody | 133kpt (body+foot+face+hand) | [GitHub](https://github.com/jin-s13/COCO-WholeBody) |
