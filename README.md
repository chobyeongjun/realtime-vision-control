# H-Walker Realtime Vision Control

실시간 하체 포즈 추정 시스템 — NVIDIA Jetson Orin NX + ZED X Mini 기반, H-Walker(보행 보조 로봇) 제어용

## 환경

| 항목 | 사양 |
|------|------|
| 추론 보드 | Jetson Orin NX 16GB (JetPack 6.x) |
| 카메라 | ZED X Mini (SVGA 960×600 @ 120fps, Global Shutter) |
| 학습 서버 | RTX 5090 (학습 전용) |
| 핵심 목표 | E2E Latency < 50ms, 하체 6kpt 정확도, 3D 포즈 |

## 빠른 시작 (Jetson)

```bash
cd ~/RealTime_Pose_Estimation
git pull origin main
source venv/bin/activate

# 모델 검증
python3 src/benchmarks/verify_models.py

# 벤치마크 실행 (전체 모델, 각 15초)
python3 src/benchmarks/run_benchmark.py

# 하체 6kpt 모델 실시간 데모
python3 src/benchmarks/run_cropped_demo.py
```

## 주요 모델

| 모델 | E2E(ms) | FPS | 비고 |
|------|---------|-----|------|
| YOLOv8n-Pose (17kpt) | 35.7 | 27.7 | 최고 속도 |
| YOLOv8s-Pose (17kpt) | 39.5 | 25.1 | 속도+정확도 균형 |
| **하체 6kpt Fine-Tuned** | **~18** | **~55** | ★ 권장 (H-Walker용) |
| MediaPipe (c=0) | 63.6 | 15.5 | 발 KP 필요 시 |

## 디렉토리 구조

```
├── src/
│   ├── benchmarks/      # 추론 + 벤치마크 코드 (pose_models.py, zed_camera.py 등)
│   └── training/        # 하체 Fine-Tuning 스크립트 (RTX 5090에서 실행)
├── models/              # 학습된 모델 파일 (yolo26s-lower6.pt, .onnx)
├── docs/
│   ├── hardware/        # INSTALL_GUIDE.md — Jetson 설치 가이드
│   ├── meetings/        # PPT_PROMPT.md — 미팅/발표 자료
│   ├── paper/           # 논문, 포스터, 그래프
│   └── reference/       # TROUBLESHOOTING.md, PROJECT_NOTES.md, 기타 참고 문서
├── CHANGELOG.md         # 전체 변경 이력 + 벤치마크 결과
├── CLAUDE.md            # Claude Code 작업 규칙
└── requirements.txt     # Python 패키지 목록
```

## 문서 안내

- **처음 설치**: `docs/hardware/INSTALL_GUIDE.md`
- **에러 발생 시**: `docs/reference/TROUBLESHOOTING.md`
- **벤치마크 결과 / 변경 이력**: `CHANGELOG.md`
- **명령어 모음 / 하드웨어 스펙**: `docs/reference/PROJECT_NOTES.md`
