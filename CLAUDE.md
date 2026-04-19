# realtime-vision-control
## 프로젝트 개요
ZED X Mini + YOLO/MediaPipe/RTMPose 실시간 포즈 추정 → 케이블 드리븐 워커 제어.
## 구조
- `src/perception/benchmarks/` — 모델 벤치마크 스크립트
- `src/control/impedance|ilc|mpc_ilc/` — 제어 알고리즘
- `docs/hardware/zed_x_mini_spec.md`
- `docs/paper/main.tex` — LaTeX 논문
## 핵심 제약
- E2E 레이턴시: **< 50ms**
- ZED: HD1080/HD1200만 지원 (HD720 불가)
- 워커 환경 = 하체만 노출 → YOLOv8-Pose 사용 (Top-down 단독 금지)
- Depth: `np.isfinite(z) and z > 0` 가드 필수
