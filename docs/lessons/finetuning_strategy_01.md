---
title: 왜 6kpt Fine-Tuning인가 (속도 2.4배 근거)
date: 2026-03-27
tags: [lesson, finetuning]
---

# Fine-Tuning 전략: 하체 6kpt

## 결론
전신 17kpt → 하체 6kpt 집중으로 추론 44ms → 18ms (2.4배).

## 근거
- heel/toe는 IMU 담당 → 비전 불필요
- 필요 keypoint: hip/knee/ankle × 좌우 = 6개
- capacity 집중 효과: mAP50 88.5%, Conf 0.99

## 실험 참조
- [[../../research-vault/realtime-vision-control/experiments/2026-03-27-finetuning-v1]]
