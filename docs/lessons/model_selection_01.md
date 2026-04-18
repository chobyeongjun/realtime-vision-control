---
title: 왜 YOLO26s를 선택했나 (실측 근거)
date: 2026-03-24
tags: [lesson, model-selection]
---

# 모델 선택: YOLO26s-Pose

## 결론
YOLO26s-Pose 선택. E2E < 50ms 달성 + Conf 0.99.

## 근거 (실측)
- 12개 모델 비교 결과 YOLO 계열만 E2E < 50ms 달성
- RTMPose: CUDA EP 미작동으로 150ms 이상 → 탈락
- MediaPipe: 워커 환경(하체만 노출) 인식률 94.4% → 탈락
- YOLO26s: 44.4ms, P95 47.5ms, Conf 0.99 → **채택**

## 실험 참조
- [[../../research-vault/realtime-vision-control/experiments/2026-03-24-model-comparison]]
