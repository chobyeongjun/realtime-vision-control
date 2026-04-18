# 6kpt Fine-Tuning — 왜 전신 17kpt 대신 하체 전용 모델인가

## 결론 (한 줄 요약)
> 전신 17kpt 모델을 하체 6kpt 전용으로 Fine-Tuning하면 추론 속도 2.4배 향상(44ms→18ms), Confidence 향상(0.97→0.99), 인식률 동일(100%)을 달성했다.

---

## 배경

- H-Walker는 hip / knee / ankle 3쌍(6개) keypoint만 필요
- 기존 YOLO26s-pose는 COCO 17kpt(전신) 모델 → capacity가 불필요한 상체/얼굴에 낭비
- heel/toe는 IMU 센서로 커버 → 비전 모델이 담당할 필요 없음

---

## 실험 / 관찰

RTX 5090에서 COCO 2017 데이터로 Fine-Tuning (17kpt→6kpt 변환 후 학습).

| 항목 | YOLO26s 17kpt (원본) | YOLO26s 6kpt (Fine-Tuned) |
|------|---------------------|--------------------------|
| 추론 속도 (Jetson TRT FP16) | 44ms | **18ms** |
| 인식률 | 100% | **100%** |
| Confidence | 0.97 | **0.99** |
| Best epoch | — | 302 / 352 (EarlyStopping) |
| Pose mAP50 | — | 88.5% |
| 학습 시간 | — | 25.8h (RTX 5090, AutoBatch) |

---

## 교훈

**capacity 집중 효과**: 출력 채널 수가 줄면(17→6 keypoint) Pose Head의 파라미터가 줄고, 같은 Backbone/Neck 용량이 더 적은 타깃에 집중 → 정확도와 속도 동시 향상.

**데이터 변환 전략**: COCO 17kpt에서 hip(11,12) / knee(13,14) / ankle(15,16)만 추출 → 별도 어노테이션 없이 기존 데이터 재활용 가능. 약 82K 학습 샘플 확보.

**EarlyStopping 효과**: patience=50으로 설정 시 불필요한 학습 자동 종료. 500 epoch 설정했으나 302에서 수렴.

---

## 왜 이 방법인가 (설계 근거)

**대안 1: 전신 모델 그대로 사용**  
→ 불필요한 상체 keypoint 계산 → 추론 낭비. 44ms로 목표(50ms) 대비 여유가 없음.

**대안 2: RTMPose Wholebody (133kpt)**  
→ heel/toe 포함이지만 E2E 245ms로 목표 5배 초과. Jetson 비적합.

**대안 3: 2-Stage (전신 검출 → 하체 crop → 하체 전용 모델)**  
→ 구현 복잡도 증가, Stage1 latency 추가. 현재 1-Stage 18ms로 충분하므로 불필요.

**선택한 방법: 1-Stage Fine-Tuning**  
→ 구현 단순, 추론 파이프라인 변경 없음, 속도 2.4배 향상으로 목표 달성.

---

## References

- [Ultralytics YOLO Pose Training](https://docs.ultralytics.com/tasks/pose/)
- [COCO Dataset — Keypoint Annotations](https://cocodataset.org/#keypoints-2020)
- [Transfer Learning for Pose Estimation (Ultralytics 공식 가이드)](https://docs.ultralytics.com/guides/model-training-tips/)
- [One-Euro Filter — 실시간 필터링](https://gery.casiez.net/1euro/)
