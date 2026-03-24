# 변경 이력 (Changelog)

> 모든 수정사항, 에러 해결, 결과 분석을 시간순으로 기록합니다.
> 새 항목은 맨 위에 추가합니다.

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
