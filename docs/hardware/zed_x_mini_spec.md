# ZED X Mini — Hardware Specification
## 카메라 사양
| 항목 | 값 |
|------|----|
| 모델 | ZED X Mini |
| 지원 해상도 | SVGA / HD1080 / HD1200 |
| ⚠ HD720 | **미지원** (폴백 → SVGA) |
| FPS | 15/30/60 해상도별 상이 |
| Depth 범위 | 0.2m ~ 20m |
| SDK | ZED SDK 5.2.1 |
| JetPack | 6.x (CUDA 12.6) |
## 알려진 제약
- 반사면/가림 시 Depth = NaN/0 → `np.isfinite(z) and z > 0` 필수
- 데몬 충돌 시: `src/perception/benchmarks/reset_zed.sh`
- torch: `setup_jetson.sh` 경유 필수 (직접 pip install 금지)
- venv: `python3 -m venv env --system-site-packages`
