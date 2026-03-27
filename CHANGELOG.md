# 변경 이력 (Changelog)

> 모든 수정사항, 에러 해결, 결과 분석을 시간순으로 기록합니다.
> 새 항목은 맨 위에 추가합니다.

---

## [2026-03-27] 1차 Fine-Tuning 완료 + Jetson 배포 + 2차 파이프라인 구축

**브랜치**: `claude/analyze-project-results-FjIrj`

### 1차 학습 결과 (RTX 5090, COCO 데이터)
- **Best epoch**: 302/352 (EarlyStopping patience=50으로 자동 종료)
- **Pose mAP50**: 88.5%
- **Pose mAP50-95**: 77.7%
- **학습 시간**: 25.8시간 (RTX 5090, batch=24 AutoBatch)

### Jetson 실시간 테스트 결과
| 항목 | 기존 YOLO26s (17kpt) | 새 하체 모델 (6kpt) | 개선 |
|------|---------------------|-------------------|------|
| 추론 속도 | 44ms | **18ms** | **2.4배 빠름** |
| 인식률 | 100% | **100%** | 동일 |
| Confidence | 0.97 | **0.99** | 향상 |
| 키포인트 | 17개 (전신) | **6개 (하체)** | 집중도↑ |

### 해결한 에러들

#### Fix 1: `total_mem` → `total_memory` (PyTorch 2.11 호환)
- **증상**: `AttributeError: 'torch._C._CudaDeviceProperties' object has no attribute 'total_mem'`
- **원인**: PyTorch 2.11에서 속성명 변경
- **해결**: `train_lower_body.py:93` — `total_mem` → `total_memory`
- **커밋**: `d88fed3`

#### Fix 2: `runs/pose/runs/pose/` 이중 경로
- **증상**: 학습 결과가 `runs/pose/runs/pose/lower_body_v1/`에 저장됨
- **원인**: `--project=runs/pose` 설정 + Ultralytics가 내부적으로 `pose/` 추가
- **해결**: `--project` 기본값을 `runs`로 변경
- **커밋**: `36ca5fe`

#### Fix 3: TRT 엔진 `task=detect` 인식 오류
- **증상**: `WARNING: Unable to automatically guess model task, assuming 'task=detect'` → not detected
- **원인**: TRT 엔진 파일에 task 메타데이터 없음
- **해결**: `YOLO(engine_path, task="pose")` 명시적 task 지정
- **커밋**: `37e5018`

#### Fix 4: `SegmentLengthConstraint.apply()` 메서드 없음
- **증상**: `AttributeError: 'SegmentLengthConstraint' object has no attribute 'apply'`
- **원인**: 실제 메서드명은 `update()`
- **해결**: `.apply()` → `.update()` 변경
- **커밋**: `7bc1b58`

#### Fix 5: `SegmentLengthConstraint.update()` 반환값 대입 오류
- **증상**: `TypeError: argument of type 'bool' is not iterable`
- **원인**: `update()`는 in-place 수정 + bool 반환인데 결과를 대입함
- **해결**: `result.keypoints_2d = self._seg_constraint.update(...)` → `self._seg_constraint.update(...)`
- **커밋**: `28a25e8`

#### Fix 6: imgsz 416 → 640 변경
- **증상**: 카메라 crop 640×600인데 416으로 축소 → 유용한 픽셀 손실
- **원인**: 하체만 찍는 카메라에서 416은 불필요한 다운스케일
- **해결**: train/export/inference 모두 imgsz=640으로 통일
- **커밋**: `585a91c`

### 추가된 파일 (2차 Fine-Tuning용)
| 파일 | 목적 |
|------|------|
| `training/auto_label_walker.py` | 워커 ZED 영상 자동 라벨링 |
| `training/walker_data.yaml` | 워커 데이터 학습 설정 |
| `models/yolo26s-lower6.pt` | 1차 학습 모델 (pretrained 6kpt) |
| `models/yolo26s-lower6.onnx` | 1차 ONNX |

### 학습 기본값 (RTX 5090 최적화)
```
epochs=500, patience=50, batch=-1 (AutoBatch), workers=16, imgsz=640
```

---

## [2026-03-25] 하체 전용 Fine-Tuning 프레임워크 구축

**브랜치**: `claude/analyze-project-results-FjIrj`

### 배경
- 기존 모델(YOLO26s-pose 등)은 전신 17kpt로 학습 → 발목(ankle)까지만 인식
- heel/toe는 IMU 센서가 커버 → 비전 모델은 hip/knee/ankle 6kpt에 집중
- 하체에 capacity를 집중하여 정확도 향상 + 떨림 감소 목표

### 추가된 파일 (training/ 디렉토리)
| 파일 | 목적 |
|------|------|
| `training/__init__.py` | 패키지 초기화 |
| `training/convert_coco_to_lower_body.py` | COCO 17kpt → 하체 6kpt YOLO 포맷 변환 |
| `training/validate_dataset.py` | 변환된 데이터셋 품질 검증 + 시각화 |
| `training/lower_body_pose.yaml` | YOLO 학습용 데이터셋 설정 (kpt_shape: [6, 3]) |
| `training/train_lower_body.py` | YOLO26s-pose Fine-Tuning (Multi-GPU, dry-run) |
| `training/export_for_jetson.py` | 학습 모델 ONNX/TRT 내보내기 |

### 수정된 파일
- `benchmarks/pose_models.py`: `LowerBodyPoseModel` 클래스 + MODEL_REGISTRY 추가
  - 원본 `YOLOv8Pose`와 완전 분리
  - 1-Stage / 2-Stage 파이프라인 지원
  - Registry: `"lower_body"`, `"lower_body_2stage"`

### 핵심 설계
- **6 keypoints**: left/right × hip/knee/ankle (heel/toe는 IMU 담당)
- **YOLO26s-pose pretrained → Fine-Tuning**: Backbone/Neck 가중치 재사용, Pose Head만 재초기화
- **표준 COCO 데이터셋 ~150K+ annotations** 활용 (COCO-WholeBody 불필요)
- **원본 모델 보존**: yolo26s-pose.pt 절대 수정 안 함, 커스텀은 yolo26s-lower6 명명

### 다음 단계
1. 학습 서버에서 COCO 데이터 다운로드 + 변환
2. dry-run (5 epoch) → 본 학습 (200 epoch)
3. ONNX 내보내기 → Jetson 전송 → TRT FP16 빌드
4. 벤치마크 비교 (기존 17kpt vs 새 6kpt)

---

## [2026-03-24] 벤치마크 실행 결과 분석 (12개 모델 성공, 3개 에러)

**브랜치**: `claude/analyze-project-results-FjIrj`

### 벤치마크 환경
- Jetson Orin NX 16GB + ZED X Mini (SVGA@120fps, Global Shutter)
- 측정 시간: 각 모델 15초
- Depth: ON

### 성공 모델 결과 요약 (E2E Latency 순)

| 모델 | FPS | Infer(ms) | E2E(ms) | P95 E2E | <50ms% | 인식률 | Conf | Foot |
|------|-----|-----------|---------|---------|--------|--------|------|------|
| YOLOv8n-Pose | 27.7 | 27.4 | 35.7 | 38.2 | 100% | 100% | 0.97 | No |
| YOLOv8s-Pose | 25.1 | 32.4 | 39.5 | 43.4 | 100% | 100% | 0.99 | No |
| YOLO11s-Pose | 24.5 | 33.1 | 40.4 | 43.2 | 99.7% | 100% | 0.99 | No |
| YOLO11n-Pose | 23.8 | 32.7 | 41.6 | 44.7 | 100% | 100% | 0.99 | No |
| YOLO26n-Pose | 22.4 | 35.3 | 44.2 | 46.5 | 100% | 100% | 0.97 | No |
| YOLO26s-Pose | 22.3 | 36.9 | 44.4 | 47.5 | 99.4% | 100% | 0.99 | No |
| MediaPipe (c=0) | 15.5 | 46.6 | 63.6 | 71.1 | 0% | 94.4% | 0.77 | Yes |
| MediaPipe (c=1) | 12.5 | 62.4 | 79.3 | 125.2 | 1.6% | 52.9% | 0.59 | Yes |
| RTMPose (lightweight) | 6.6 | 132.8 | 150.5 | 177.7 | 0% | 100% | 0.60 | No |
| RTMPose Wholebody (lw) | 4.1 | 228.2 | 245.6 | 292.4 | 0% | 100% | 0.68 | Yes |
| RTMPose (balanced) | 1.4 | 709.9 | 726.9 | 772.3 | 0% | 100% | 0.61 | No |
| RTMPose Wholebody (bal) | 1.1 | 882.2 | 899.9 | 1046.7 | 0% | 100% | 0.73 | Yes |

### 에러 모델 (3개)

| 모델 | 에러 원인 |
|------|----------|
| MoveNet (lightning) | NumPy 2.x 비호환 — tflite_runtime이 NumPy 1.x로 컴파일됨 |
| MoveNet (thunder) | 동일 |
| ZED BT (FAST) | Positional Tracking 미활성화 상태에서 Body Tracking 호출 |

### 핵심 분석

1. **E2E <50ms 달성**: YOLO 계열 6개 모델만 달성 (99.4~100%)
2. **최고 속도**: YOLOv8n-Pose (27.7 FPS, E2E 35.7ms)
3. **최고 균형**: YOLOv8s-Pose (25.1 FPS, E2E 39.5ms, Conf 0.99)
4. **RTMPose**: ONNX Runtime CPU fallback으로 추정 — CUDAExecutionProvider 경고 발생
5. **MediaPipe c=1**: 인식률 52.9%로 사용 불가 수준
6. **발 키포인트**: Foot 지원 모델 중 실용적인 것은 MediaPipe c=0뿐 (E2E 63.6ms)

### 3D 안정성 공통 이슈
- 모든 모델에서 `right_shank` CV가 높음 (0.05~0.17) — Depth 노이즈 영향
- Wholebody foot_heel CV 0.33~0.54로 매우 불안정

### 다음 단계
- [ ] MoveNet NumPy 에러 수정 (numpy<2 재설치 확인)
- [ ] ZED BT positional tracking 활성화 후 재측정
- [ ] RTMPose CUDA EP 미사용 원인 조사
- [ ] YOLOv8s-Pose 기반 3D 파이프라인 심화 테스트

---

## [2026-03-24] 모델 비교 프레임워크 구축 + YOLO26 + 영상 녹화 + Global Shutter

**브랜치**: `claude/analyze-project-results-FjIrj`
**커밋**: `b6aa67a`

### 변경 내용

#### 1. 소스 코드 병합
- `claude/fix-python-dependencies-DsY3U` 브랜치의 전체 코드를 현재 브랜치로 병합
- 포함 항목: YOLO26 지원, INT8/TRT 비교, 3D 파이프라인, C++ 후처리, 영상 녹화

#### 2. 새 파일 추가
| 파일 | 설명 |
|------|------|
| `benchmarks/run_comparison_video.py` | **5종 모델 그리드 비교 영상 생성기** (2x3 레이아웃, 프레임별 CSV) |
| `CLAUDE.md` | Claude Code 작업 규칙 (git pull 안내, 에러 로깅, 메모 규칙) |
| `CHANGELOG.md` | 변경 이력 로그 (이 파일) |

#### 3. 수정된 파일
| 파일 | 변경 내용 |
|------|----------|
| `benchmarks/zed_camera.py` | ZED X Mini Global Shutter 문서화, `configure_for_benchmark()` 추가, `get_camera_info()` 추가 |
| `benchmarks/run_record_demo.py` | 관절 각도 오버레이 추가, 프레임별 상세 CSV 출력 추가 |
| `benchmarks/run_benchmark.py` | 카메라 메타데이터(셔터 타입, calibration)를 결과 JSON에 포함 |
| `benchmarks/run_comparison_video.py` | YOLO26으로 기본 모델 변경 (`yolo_version="v26"`) |

#### 4. Global Shutter 검토 결과
- **결론**: 별도 코드 설정 불필요 (하드웨어 레벨 자동 적용)
- ZED X Mini는 Global Shutter 센서 → 모든 픽셀 동시 촬영 → motion blur 없음
- 추가한 것: 카메라 정보에 셔터 타입 자동 감지/기록, 노출 시간 제어 옵션

### 비교 대상 모델 5종 (확정)
| 모델 | 파일명 | Keypoints | Foot | 비고 |
|------|--------|-----------|------|------|
| MediaPipe Pose | - (내장) | 33 | O | 경량, CPU 가능 |
| **YOLO26-Pose** | `yolo26n-pose.pt` | 17 (COCO) | X | **최신 2026.01**, NMS-free, RLE |
| RTMPose | rtmlib | 17 (COCO) | X | SOTA 속도/정확도 |
| RTMPose Wholebody | rtmlib | 133 | O | Foot keypoint 직접 제공 |
| ZED Body Tracking | ZED SDK | 38 | O | 3D 직접 제공 |

### 다음 단계
- [ ] Jetson에서 `git pull` 후 벤치마크 실행
- [ ] 결과 데이터로 상세 비교 분석
- [ ] 최적 모델 1개 선정 및 최적화 전략 수립

---

## [2026-03-18 이전] 초기 벤치마크 프레임워크 (소스 브랜치)

**브랜치**: `claude/fix-python-dependencies-DsY3U`

총 17건의 이슈 해결 (상세 내용은 HANDOVER.md 참조):
- Phase 1: 프로젝트 초기 셋업
- Phase 2: 모델 추가 및 카메라 최적화
- Phase 3: Jetson 환경 설치 문제 해결
- Phase 4: TensorRT 초기화 및 벤치마크 수정
- Phase 5: INT8 양자화 지원
- Phase 6: 영상 녹화 기능
- Phase 7: 3D 파이프라인 구축
- Phase 8: 성능 최적화 (zero-copy, GPU 전처리)
