# Troubleshooting

## 🔴 하지 말아야 할 것들 (past incidents — vault skiro-learnings)

- **GDM(X server) 끄지 말 것** — Argus/EGL 연쇄 실패 → ZED segfault + 드라이버 락업 → 리부트 강제. ZED X Mini는 **GMSL2/CSI** → Argus 프레임워크 → EGL 컨텍스트 필요 → EGL은 X server 공급
- **jetson_clocks 생략 금지** — GPU 306MHz(max 33%)로 떨어져 predict 40→13ms, GR3D util 11%→85%. 부팅 후 매번 실행 (persistent 아님)
- **OneEuro filter `use_filter=True` 금지 (yolo26s-lower6-v2)** — 2026-04-17 실측: joints 0/6 영구 기각. 초기 low-conf keypoint가 0으로 zero out → 필터가 0 기억 → 이후 실제 detection도 0으로 suppress
- **SegmentConstraint on 2D 금지** — 피드백 루프 고착 (2026-04-15). 본 모듈은 **3D only** + **static ref** + **std ≤ 10mm 검증 gate** 3중 차단. `calibrate()` → `armed=True` 순서만
- **NEURAL depth 모드 금지** — YOLO와 GPU SM 경합해 predict 30ms x2.4 spike, FPS 74→29 (60% 손실). NEURAL_LIGHT도 30% 손실 기각. **PERFORMANCE 고정** (런타임 `ValueError` 차단)
- **cv2.imshow sagittal display 금지** — 실험 중엔 `--no-display`. FPS 74 → 42 반토막 (X11 + waitKey + GPU 경합)
- **ZED `copy=False` 금지** — PipelinedCamera race로 state.valid=False 캘리 0% 고착. `deep_copy=True` 강제 (SVGA 2.3MB copy=0.5ms 감수)
- **22→15핀 CSI 어댑터 금지** — Waveshare Orin NX 22핀 보드만 GMSL2 라우팅 정상. 어댑터는 물리 연결만 되고 I2C bus 9에 카메라 안 잡힘 (vault zed-x-mini-jetson-setup.md)
- **imgsz 480 금지** — 사용자 영구 거부, 640 유지
- **cv2.cuda 사용 금지** — JetPack OpenCV CUDA 미빌드. torchvision GPU 사용

## Mainline과의 공존 (중요)

Mainline `control` 브랜치에는 이미 P0 트랙 적용 완료 (커밋 `ed4d933f`). 본 모듈은 **완전 격리**됨:

| 구분 | Mainline | CUDA_Stream |
|------|----------|-------------|
| SHM 경로 | `/hwalker_pose` (36B) | `/hwalker_pose_cuda` (≈300-500B, 동적 K) |
| Bone constraint | `realtime/bone_constraint.py` (CPU numpy) | `CUDA_Stream/constraints.py` (GPU torch) |
| Velocity gate | 20°/frame (각도 기반) | m/s (3D 위치 기반) |
| 엔진 | `yolo26s-lower6-v2-640.direct.engine` | 같은 엔진 사용 가능 (schema=lowlimb6) |

제어 루프(`hw_control`)는 mainline `/hwalker_pose` 소비 중. 본 모듈의 `/hwalker_pose_cuda`는 **별도 consumer** 붙일 때만 사용. 두 경로 동시 운용 시 mainline 영향 0.


## Segfault right after first inference
- Likely `set_tensor_address` pointer invalidation. Our TRTRunner re-binds
  in `bind_input_address`. Confirm `GpuPreprocessor.out.data_ptr()` hasn't
  changed between warmup and the first live frame.
- If you mutate `pre.out` shape mid-run, re-bind.

## "cudaErrorInvalidResourceHandle" inside execute_async_v3
- `StreamManager` was destroyed but TRT context is still alive. Shut down
  TRTRunner before StreamManager.
- Or stream was created on a different device. Check `StreamManager.device`.

## Hang for ~5 seconds then fallback triggered
- Watchdog detected a stream stuck > 50ms. Likely ZED didn't produce a frame.
  Check `bridge.latest()` returning `None` repeatedly — the ZED thread may
  have died. `tegrastats` often shows 0% GPU during this window.

## Depth invalid ratio > 30%
- See skiro-learnings: ZED `copy=False` race. We default to
  `get_data(deep_copy=True)` — don't flip this.
- Reset ZED: `bridge.stop(); bridge = ZEDGpuBridge(...); bridge.open()`.

## CUDA Graph capture failed
- Shapes not fixed. Either set `imgsz` explicitly (done by default) or
  keep graph disabled. Correctness is unaffected.

## SHM file left behind after crash
```bash
rm /dev/shm/hwalker_pose_cuda  # NEVER rm /dev/shm/hwalker_pose (mainline)
```

## Throttling (thermal)
`sudo tegrastats --interval 500` — look for `CPU@`, `GPU@`, `SOC@` freq drop.
40W MAXN_SUPER without a proper heatsink will throttle in ~1-2 minutes on
ZED streaming + YOLO workload.

## "RuntimeError: CUDA error: no kernel image is available"
- Wrong PyTorch wheel for your JetPack. Use the NVIDIA-provided wheels for
  JetPack 6.x (Orin NX 16GB = arm64 + sm_87).

## Mainline collision alarm
If `preflight` says `/dev/shm/hwalker_pose_cuda` collision, another
instance of this module is running. Kill it before starting a new one.
If `/dev/shm/hwalker_pose` is missing, that's mainline's fault — do NOT
create it from this module.
