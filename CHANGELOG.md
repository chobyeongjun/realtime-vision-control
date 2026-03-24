# 변경 이력 (Changelog)

> 모든 수정사항, 에러 해결, 결과 분석을 시간순으로 기록합니다.
> 새 항목은 맨 위에 추가합니다.

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
