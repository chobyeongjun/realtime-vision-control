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
