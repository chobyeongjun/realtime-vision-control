#!/usr/bin/env python3
"""
YOLO26s-Pose 하체 전용 Fine-Tuning 스크립트
============================================
YOLO26s-pose pretrained (COCO 17kpt) → 하체 6kpt 모델로 파인튜닝.

⚠️  원본 yolo26s-pose.pt는 절대 수정하지 않습니다.
    학습 결과는 runs/pose/lower_body_*/ 에 별도 저장됩니다.

사용법:
    # 사전 테스트 (5 epoch, loss 감소 확인)
    python train_lower_body.py --dry-run

    # 본 학습 (RTX 5090 x2)
    python train_lower_body.py --device 0,1 --batch 128

    # 학습 재개
    python train_lower_body.py --resume runs/pose/lower_body_yolo26s_v1/weights/last.pt
"""

import argparse
import os
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="YOLO26s-Pose 하체 전용 Fine-Tuning"
    )

    # 모델 설정
    parser.add_argument(
        "--model", default="yolo26s-pose.pt",
        help="Pretrained 모델 (기본: yolo26s-pose.pt, 자동 다운로드)"
    )
    parser.add_argument(
        "--data", default=None,
        help="데이터셋 YAML (기본: training/lower_body_pose.yaml)"
    )

    # 학습 설정 (RTX 5090 32GB 최대 활용 기본값)
    parser.add_argument("--imgsz", type=int, default=640, help="입력 해상도 (기본: 640, 카메라 crop 640x600에 맞춤)")
    parser.add_argument("--epochs", type=int, default=500, help="최대 에포크 (기본: 500, patience로 자동 조기종료)")
    parser.add_argument("--batch", type=int, default=-1, help="배치 크기 (기본: -1 = AutoBatch, VRAM 한계까지 자동 최대)")
    parser.add_argument("--device", default="0", help="GPU 장치 (기본: 0)")
    parser.add_argument("--workers", type=int, default=16, help="데이터 로더 워커 (기본: 16, GPU 풀 가동)")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience (기본: 50, 50번 연속 개선 없으면 종료)")

    # 학습률
    parser.add_argument("--lr0", type=float, default=0.01, help="초기 학습률 (기본: 0.01)")
    parser.add_argument("--lrf", type=float, default=0.01, help="최종 학습률 비율 (기본: 0.01)")

    # 출력 설정
    parser.add_argument("--project", default="runs", help="학습 결과 프로젝트 (기본: runs)")
    parser.add_argument("--name", default="lower_body_yolo26s_v1", help="실험 이름")

    # 안전 기능
    parser.add_argument("--dry-run", action="store_true", help="5 epoch 사전 테스트 (loss 감소 확인)")
    parser.add_argument("--resume", default=None, help="학습 재개 (last.pt 경로)")

    return parser.parse_args()


def validate_environment(args):
    """학습 환경 사전 검증"""
    print("=" * 60)
    print("  환경 검증")
    print("=" * 60)

    # Python 버전
    print(f"  Python: {sys.version.split()[0]}")

    # ultralytics 확인
    try:
        import ultralytics
        print(f"  Ultralytics: {ultralytics.__version__}")
    except ImportError:
        print("  ERROR: ultralytics 미설치!")
        print("  pip install ultralytics")
        sys.exit(1)

    # PyTorch + CUDA
    try:
        import torch
        print(f"  PyTorch: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"  GPU count: {gpu_count}")
            for i in range(gpu_count):
                name = torch.cuda.get_device_name(i)
                mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"    GPU {i}: {name} ({mem:.1f} GB)")
        else:
            print("  WARNING: CUDA 없음! CPU 학습은 매우 느립니다.")
    except ImportError:
        print("  ERROR: PyTorch 미설치!")
        sys.exit(1)

    # 데이터셋 확인
    data_yaml = args.data
    if data_yaml and Path(data_yaml).exists():
        print(f"  데이터셋 YAML: {data_yaml} ✓")
    else:
        print(f"  데이터셋 YAML: {data_yaml}")
        print(f"  WARNING: YAML 파일이 존재하지 않습니다. 경로를 확인하세요.")

    # device 파싱 확인
    device_str = args.device
    if "," in device_str:
        devices = [int(d.strip()) for d in device_str.split(",")]
        print(f"  학습 장치: GPU {devices} (Multi-GPU DDP)")
    else:
        print(f"  학습 장치: GPU {device_str}")

    print()


def find_data_yaml(args) -> str:
    """데이터셋 YAML 파일 찾기"""
    if args.data:
        return args.data

    # 기본 경로들 시도
    candidates = [
        Path(__file__).parent / "lower_body_pose.yaml",
        Path("training/lower_body_pose.yaml"),
        Path("lower_body_pose.yaml"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    print("ERROR: 데이터셋 YAML을 찾을 수 없습니다.")
    print("  --data 옵션으로 직접 지정하세요.")
    sys.exit(1)


def parse_device(device_str: str):
    """device 문자열을 ultralytics 형식으로 변환"""
    if "," in device_str:
        return [int(d.strip()) for d in device_str.split(",")]
    return int(device_str)


def train(args):
    """메인 학습 실행"""
    from ultralytics import YOLO

    data_yaml = find_data_yaml(args)
    device = parse_device(args.device)

    # dry-run 모드
    epochs = 5 if args.dry_run else args.epochs
    name = f"{args.name}_dryrun" if args.dry_run else args.name

    if args.dry_run:
        print("=" * 60)
        print("  🧪 DRY-RUN 모드 (5 epoch 사전 테스트)")
        print("  - kpt_shape [6,3] 적용 확인")
        print("  - loss 감소 추이 확인")
        print("  - 문제 없으면 --dry-run 빼고 본 학습 실행")
        print("=" * 60)

    # 모델 로드
    if args.resume:
        print(f"\n  학습 재개: {args.resume}")
        model = YOLO(args.resume)
    else:
        print(f"\n  Pretrained 모델 로드: {args.model}")
        print(f"  ⚠️  원본 {args.model}은 수정되지 않습니다.")
        model = YOLO(args.model)

    print(f"  데이터셋: {data_yaml}")
    print(f"  입력 크기: {args.imgsz}×{args.imgsz}")
    print(f"  에포크: {epochs}")
    print(f"  배치: {args.batch}")
    print(f"  장치: {device}")
    print()

    # 학습 실행
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        patience=args.patience if not args.dry_run else epochs,  # dry-run은 early stop 안 함
        cos_lr=True,
        lr0=args.lr0,
        lrf=args.lrf,

        # 데이터 증강 (워커 전면 상반 카메라 시점에 맞춤)
        flipud=0.0,            # 상하 반전 OFF (사람은 항상 직립)
        fliplr=0.5,            # 좌우 반전 ON
        mosaic=1.0,            # 모자이크 증강
        scale=0.5,             # 스케일 변화 ±50%
        degrees=5.0,           # 약간의 회전 (워커 기울어짐 대비)
        translate=0.1,         # 위치 변화 (카메라 흔들림 대비)
        perspective=0.0005,    # 약간의 원근 변화 (상반 시점 보정)
        hsv_h=0.015,           # 색조 변화
        hsv_s=0.7,             # 채도 변화
        hsv_v=0.4,             # 밝기 변화

        # 저장
        project=args.project,
        name=name,
        exist_ok=True,
        save=True,
        save_period=10,        # 10 epoch마다 체크포인트
        plots=True,            # 학습 곡선 저장

        # 학습 재개
        resume=args.resume is not None,
    )

    # 결과 요약
    print("\n" + "=" * 60)
    print("  학습 완료!")
    print("=" * 60)

    weights_dir = Path(args.project) / name / "weights"
    if weights_dir.exists():
        best_pt = weights_dir / "best.pt"
        last_pt = weights_dir / "last.pt"
        print(f"  Best weights: {best_pt}")
        print(f"  Last weights: {last_pt}")

        if best_pt.exists():
            size_mb = best_pt.stat().st_size / 1e6
            print(f"  Best size: {size_mb:.1f} MB")

    if args.dry_run:
        print("\n  📋 DRY-RUN 결과 확인 사항:")
        print("  1. loss가 에포크마다 감소했는가?")
        print("  2. WARNING: shape mismatch 메시지가 Pose Head에만 나왔는가?")
        print("  3. kpt_shape: [6, 3]이 올바르게 적용되었는가?")
        print(f"\n  문제 없으면 본 학습:")
        print(f"    python train_lower_body.py --device {args.device} --batch {args.batch}")
    else:
        print(f"\n  다음 단계:")
        print(f"    1. 학습 곡선 확인: {Path(args.project) / name}/")
        print(f"    2. 모델 내보내기:")
        print(f"       python export_for_jetson.py --weights {weights_dir / 'best.pt'}")

    return results


def main():
    args = parse_args()
    validate_environment(args)
    train(args)


if __name__ == "__main__":
    main()
