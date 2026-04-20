# Consumer Contract — mainline 제어 루프가 지켜야 할 계약

본 문서는 `hw_treadmill/`, `hw_overground/` 등 mainline 제어 루프가
`/hwalker_pose_cuda` 경로를 소비할 때 **반드시** 구현해야 하는 최소 로직이다.

이 모듈은 격리 규칙 #1(Mainline 수정 금지)을 엄격히 지키므로, 아래 코드를
이 모듈이 직접 mainline에 삽입하지 않는다. 팀에서 별도 PR로 적용한다.

## 발행 채널 (CUDA_Stream 측이 제공)

### 1) SHM — `/dev/shm/hwalker_pose_cuda`

| offset | size | 타입 | 의미 |
|-------:|-----:|:---:|-----|
| 0  | 4   | uint32 | seqlock counter (짝수일 때 stable) |
| 4  | 4   | uint32 | frame_id |
| 8  | 8   | uint64 | ts_ns |
| 16 | 4   | float  | box_conf |
| 20 | 4   | uint32 | **valid_flag (0 = 무효, safe-stop)** |
| 24 | 204 | float×51 | kpts_3d_m (17×3, meter) |
| 228| 68  | float×17 | kpt_conf |
| 296| 136 | float×34 | kpts_2d_px (17×2) |

seqlock 프로토콜은 `shm_publisher.ShmReader.read()` 참고.

### 2) Estop sentinel — `/dev/shm/hwalker_pose_cuda_estop`

워치독이 stream hang, publish timeout, depth-invalid spike 중 하나를
감지하면 이 파일을 작성한다. 파일 **존재 자체**가 safe-stop 신호이며
내용은 사람용 reason. 복구 시 워치독이 파일을 unlink 한다.

## Reader 참고 구현 (mainline PR에 추가)

```python
from multiprocessing import shared_memory
from pathlib import Path
import struct

ESTOP = Path("/dev/shm/hwalker_pose_cuda_estop")
SHM_NAME = "hwalker_pose_cuda"

def perception_tick(motor, shm_reader, max_age_ms=50):
    # 1) estop sentinel → hard 0N
    if ESTOP.exists():
        motor.command_force(0.0)  # AK60 70N 이내
        return "estop_sentinel"

    # 2) SHM read
    data = shm_reader.read(max_retries=16)
    if data is None:
        motor.command_force(0.0)
        return "shm_read_failed"

    frame_id, ts_ns, kpts_3d, kpt_conf, kpts_2d, box_conf, valid = data
    if not valid:
        motor.command_force(0.0)
        return "valid_false"

    # 3) staleness check — 제어 주기 대비 프레임이 너무 오래됐으면 무효
    age_ms = (time.time_ns() - ts_ns) / 1e6
    if age_ms > max_age_ms:
        motor.command_force(0.0)
        return "stale"

    # 4) 정상 경로 — keypoint 기반 impedance 계산 후 force 명령
    target_force = impedance_model(kpts_3d, kpt_conf)
    motor.command_force(min(target_force, 70.0))  # hard ceiling
    return "ok"
```

## 테스트 가이드

mainline PR 적용 시 추가할 테스트:
1. `test_estop_sentinel_zeros_force` — 파일 존재 시 `command_force(0.0)` 호출
2. `test_valid_false_zeros_force` — valid=0 일 때 동일
3. `test_stale_data_zeros_force` — `age_ms > max_age_ms` 일 때 동일
4. `test_ak60_force_ceiling` — 계산된 target_force가 70N 초과해도 70N로 clamp

## 미구현 시 위험

본 contract 없이 CUDA_Stream을 활성화하면, 워치독이 이상을 감지해도
제어 루프가 **이전 `valid=1` 데이터**를 계속 사용한다 →
최악의 경우 AK60이 70N 한도까지 잘못된 힘 명령 수행.
본 모듈의 격리 원칙 #3("실패해도 OK") 은 consumer contract가
충족된 **다음에야** 성립한다.
