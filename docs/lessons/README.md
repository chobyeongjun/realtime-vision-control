# Lessons — 작성 가이드

여기에는 **디버깅 내역이 아닌** 실험과 설계에서 얻은 교훈을 기록합니다.

---

## 파일 명명 규칙

```
<주제>_<번호>.md
예: model_selection_01.md / pipeline_design_01.md / hardware_constraint_01.md
```

## 각 파일의 구조 (템플릿)

```markdown
# [교훈 제목]

## 결론 (한 줄 요약)
> 핵심 메시지를 한 줄로.

## 배경
왜 이 문제를 마주쳤는가.

## 실험 / 관찰
무엇을 해봤고 무엇을 측정했는가.

## 교훈
실험 결과가 말해주는 것.

## 왜 이 방법인가 (설계 근거)
대안들과 비교해서 이 선택을 한 이유.

## References
- [논문/문서 제목](URL)
- [논문/문서 제목](URL)
```

---

## 현재 파일 목록

| 파일 | 주제 |
|------|------|
| `model_selection_01.md` | YOLO vs RTMPose — 왜 YOLO를 선택했는가 |
| `finetuning_strategy_01.md` | 6kpt Fine-Tuning — 왜 전신 모델 대신 하체 전용인가 |
