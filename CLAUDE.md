# Claude Code 작업 규칙

> 이 파일은 Claude Code가 작업할 때 반드시 따라야 할 규칙입니다.
> 모든 세션에서 이 파일을 먼저 읽고 규칙을 따르세요.

---

## 1. 코드 변경 후 반드시 알려줄 것

모든 코드 수정/변경 후 사용자에게 **Jetson에서 실행할 명령어**를 알려주세요:

```bash
# 1) 최신 코드 가져오기
cd ~/RealTime_Pose_Estimation
git pull origin <현재-브랜치-이름>

# 2) 변경된 파일 목록 확인
git log --oneline -3

# 3) 실행 (변경된 스크립트에 따라)
source venv/bin/activate
python3 benchmarks/<실행할-스크립트>.py
```

**매번 빠짐없이** git pull 명령어부터 실행 명령어까지 전체 과정을 안내하세요.

---

## 2. 에러 발생 시 로깅 규칙

에러가 발생하고 해결되면 반드시:

1. **CHANGELOG.md**에 기록:
   - 에러 증상 (어떤 명령/스크립트에서 발생)
   - 에러 메시지 (핵심 부분)
   - 원인 분석
   - 해결 방법 (수정한 파일, 변경 내용)
   - 관련 커밋 해시

2. **커밋 메시지**에 `Fix:` 접두사 사용

3. 심각한 에러는 **TROUBLESHOOTING.md**에도 추가

---

## 3. 결과 분석 후 메모 규칙

벤치마크/테스트 결과를 보여준 후 반드시:

1. **CHANGELOG.md**에 결과 요약 기록:
   - 어떤 테스트를 실행했는지
   - 주요 수치 (FPS, Latency, 인식률 등)
   - 이전 대비 변화가 있다면 비교
   - 다음 단계 권장사항

2. **HANDOVER.md** 업데이트:
   - 현재 프로젝트 상태
   - 최근 변경사항 반영
   - 알려진 이슈 업데이트

---

## 4. 인수인계 규칙

코드 변경이 있을 때마다 **HANDOVER.md**를 최신 상태로 유지하세요:

- 새 파일이 추가되면 파일 설명 추가
- 기존 파일이 수정되면 변경 내용 기록
- 설정/환경이 바뀌면 설치 가이드 업데이트
- 갑자기 세션이 끊겨도 다음 세션에서 바로 이어갈 수 있도록

---

## 5. 브랜치 규칙

- **개발 브랜치**: `claude/analyze-project-results-FjIrj`
- **소스 브랜치** (Jetson 코드 원본): `claude/fix-python-dependencies-DsY3U`
- 항상 지정된 개발 브랜치에서 작업
- main/master에 직접 push 금지

---

## 6. 프로젝트 컨텍스트

- **목적**: H-Walker (보행 보조 로봇)용 실시간 하체 포즈 추정
- **환경**: Jetson Orin NX 16GB + ZED X Mini (Global Shutter, SVGA@120fps)
- **비교 모델 5종**: MediaPipe, YOLO26, RTMPose, RTMPose Wholebody, ZED Body Tracking
- **핵심 요구사항**: E2E Latency < 50ms, 하체 keypoint 정확도, 3D 포즈
- **스코어 기반 평가 X** → **정확한 지표 기반 비교 분석**으로 최고 모델 선정
