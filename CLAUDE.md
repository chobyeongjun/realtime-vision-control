# realtime-vision-control

## 프로젝트
실시간 포즈 추정 + 제어 (ZED X Mini + YOLO26s → impedance/ILC/MPC)

## 핵심 제약
- E2E latency < 50ms (p95)
- ZED: HD1080/HD1200/SVGA만 사용
- 제어 주기: outer 10-30Hz

## 규칙 (전체 규칙 → ~/research-vault/20_Meta/claude-rules.md)
- 하드웨어 수치는 공식 스펙 확인 필수, 추측 금지
- 로봇: Exosuit (외골격/exoskeleton 금지)
- 작업 완료 시 커밋 + 푸시 자동 수행

## Vault 연동
- 실험 기록: ~/research-vault/realtime-vision-control/experiments/
- 미팅 노트: ~/research-vault/realtime-vision-control/meetings/
- 논문 자료: ~/research-vault/realtime-vision-control/papers/
- Wiki: ~/research-vault/10_Wiki/

## 세션 시작 시
1. 관련 Wiki 노트 확인 (admittance-control, ak60-motor 등)
2. 최근 실험 확인: ~/research-vault/realtime-vision-control/experiments/

## 실험 저장 규칙 (자동화)

실험 코드 실행이 완료되면 반드시:

```bash
# 1. /tmp/exp_config.json 먼저 작성
cat > /tmp/exp_config.json << 'JSON'
{
  "name": "experiment_name_snake_case",
  "hypothesis": "이 실험이 검증하려는 것 1문장",
  "method": "사용한 알고리즘/방법",
  "parameters": {"camera": "ZED X Mini", "control": "impedance"},
  "key_metrics": {"latency_ms": 0.0, "accuracy_pct": 0},
  "status": "success",
  "limitations": "발견된 한계점",
  "next_experiment": "다음 실험 방향"
}
JSON

# 2. 세션 종료 시 Stop hook이 자동으로 save_experiment.py 실행
# → ~/research-vault/realtime-vision-control/experiments/ 에 저장
```

**수동 실행:** `python3 ~/.scripts/save_experiment.py --config /tmp/exp_config.json`

### 그래프 규칙
- PNG (DPI 150) + PDF (벡터) 두 형식으로 저장
- 축 레이블 단위 필수: `Time [s]`, `Latency [ms]`, `Error [px]`
- proposed=`#2196F3`, baseline=`#F44336`
