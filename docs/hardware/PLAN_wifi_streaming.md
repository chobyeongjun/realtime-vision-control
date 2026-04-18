# Wi-Fi 실시간 스트리밍 계획
> Jetson → 노트북으로 포즈 추정 영상 실시간 전송

## 구조
```
Jetson (120fps 추론) ──TCP 소켓 (30fps JPEG)──→ 노트북 Python GUI
```
- 추론은 120fps 그대로 유지
- 스트리밍은 별도 스레드 30fps → 추론 성능 영향 없음
- JPEG q=70 기준 프레임당 ~30KB → 초당 ~1MB → Wi-Fi 여유

## 구현 파일 3개

### 1. `benchmarks/stream_server.py` (Jetson측, 신규)
- 별도 스레드 TCP 서버 (메인 루프 블로킹 없음)
- 최신 프레임만 JPEG 압축 후 30fps 간격 전송
- 프로토콜: `[4바이트 big-endian uint32 JPEG크기] + [JPEG 데이터]`
- 클라이언트 여러 대 동시 접속 지원
- 파라미터: port(9000), jpeg_quality(70), stream_fps(30)

### 2. `benchmarks/stream_client.py` (노트북측, 신규)
- tkinter GUI 또는 OpenCV 창으로 수신 프레임 표시
- 사용법: `python3 stream_client.py <jetson_ip> 9000`
- 추가 의존성 없음 (socket, struct, tkinter 모두 표준 라이브러리)

### 3. `benchmarks/run_cropped_demo.py` (수정)
- `--stream` 플래그 추가
- `--stream-port 9000`, `--stream-fps 30`, `--stream-quality 70` 옵션
- 모델 로드 후 StreamServer 시작
- 시각화 프레임 생성 후 `streamer.update_frame(vis)` 한 줄 추가
- 종료 시 `streamer.stop()` 호출

## 사용법
```bash
# Jetson에서
python3 run_cropped_demo.py --stream

# 노트북에서
python3 stream_client.py 192.168.x.x 9000
```

## 예상 지연
| 구간 | 지연 |
|------|------|
| JPEG 인코딩 (640x600) | ~1ms |
| TCP 전송 (Wi-Fi) | ~2-5ms |
| 디코딩 + GUI 렌더링 | ~2ms |
| **총 지연** | **~5-10ms** |

## BLE와 병행 가능
- 스트리밍: 영상 확인용 (Wi-Fi)
- BLE: 관절 각도/좌표 데이터 전송 (~170 bytes/프레임)
- 두 채널 독립 동작
