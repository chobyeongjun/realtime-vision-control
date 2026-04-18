#!/usr/bin/env python3
"""
학습된 하체 모델 내보내기 (Jetson 배포용)
=========================================
학습된 best.pt → ONNX / TensorRT 엔진으로 변환.

⚠️  TensorRT 엔진은 반드시 Jetson에서 빌드해야 합니다!
    학습 서버(x86)에서 빌드한 엔진은 Jetson(aarch64)에서 작동하지 않습니다.

워크플로우:
    1. [학습 서버] ONNX 내보내기
       python export_for_jetson.py --weights best.pt --format onnx

    2. [학습 서버] models/ 에 복사 + GitHub push
       cp best.pt models/yolo26s-lower6.pt
       cp best.onnx models/yolo26s-lower6.onnx
       git add models/ && git commit && git push

    3. [Jetson] git pull로 모델 받기
       git pull origin claude/analyze-project-results-FjIrj

    4. [Jetson] TensorRT FP16 엔진 빌드
       python training/export_for_jetson.py --weights models/yolo26s-lower6.pt --format engine --half
"""

import argparse
import platform
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="하체 모델 Jetson 배포용 내보내기"
    )
    parser.add_argument(
        "--weights", required=True,
        help="학습된 모델 경로 (best.pt 또는 best.onnx)"
    )
    parser.add_argument(
        "--format", default="onnx", choices=["onnx", "engine"],
        help="내보내기 포맷 (기본: onnx)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="입력 해상도 (학습 시 사용한 값과 동일해야 함, 기본: 640)"
    )
    parser.add_argument(
        "--half", action="store_true",
        help="FP16 정밀도 (TensorRT 권장)"
    )
    parser.add_argument(
        "--simplify", action="store_true", default=True,
        help="ONNX 단순화 (기본: True)"
    )
    parser.add_argument(
        "--opset", type=int, default=17,
        help="ONNX opset 버전 (기본: 17)"
    )
    parser.add_argument(
        "--validate", action="store_true",
        help="내보내기 후 val 데이터셋으로 정확도 검증"
    )
    parser.add_argument(
        "--data", default=None,
        help="검증용 데이터셋 YAML (--validate 시 필요)"
    )
    parser.add_argument(
        "--output-name", default=None,
        help="출력 파일명 (기본: 자동 생성, 예: yolo26s-lower6-416)"
    )
    return parser.parse_args()


def check_platform(export_format: str):
    """플랫폼 체크 (TRT는 Jetson에서만 빌드 권장)"""
    arch = platform.machine()

    if export_format == "engine":
        if arch == "x86_64":
            print("  ⚠️  WARNING: x86_64에서 TensorRT 빌드 중!")
            print("  ⚠️  이 엔진은 Jetson (aarch64)에서 작동하지 않습니다.")
            print("  ⚠️  ONNX로 내보낸 후 Jetson에서 TRT 빌드하세요.")
            print()
            response = input("  계속하시겠습니까? (y/N): ")
            if response.lower() != "y":
                print("  취소됨. ONNX로 내보내려면:")
                print("    python export_for_jetson.py --weights <path> --format onnx")
                sys.exit(0)
        elif arch == "aarch64":
            print(f"  ✅ Jetson 플랫폼 감지 (aarch64)")

    print(f"  Platform: {arch}")


def export_model(args):
    """모델 내보내기"""
    from ultralytics import YOLO

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"  ERROR: 모델 파일 없음: {weights_path}")
        sys.exit(1)

    print(f"\n  모델 로드: {weights_path}")
    model = YOLO(str(weights_path))

    # 출력 이름 결정
    if args.output_name:
        output_stem = args.output_name
    else:
        output_stem = f"yolo26s-lower6-{args.imgsz}"  # 예: yolo26s-lower6-640

    if args.format == "onnx":
        print(f"  ONNX 내보내기 (opset={args.opset}, simplify={args.simplify})")
        exported_path = model.export(
            format="onnx",
            imgsz=args.imgsz,
            simplify=args.simplify,
            opset=args.opset,
        )
        print(f"  ✅ ONNX 저장: {exported_path}")

        # 파일 크기
        onnx_path = Path(exported_path)
        if onnx_path.exists():
            size_mb = onnx_path.stat().st_size / 1e6
            print(f"  파일 크기: {size_mb:.1f} MB")

        print(f"\n  다음 단계 (GitHub 경유):")
        print(f"    mkdir -p models")
        print(f"    cp {args.weights} models/yolo26s-lower6.pt")
        print(f"    cp {exported_path} models/yolo26s-lower6.onnx")
        print(f"    git add models/ && git commit -m 'Add trained lower body model' && git push")
        print(f"    # Jetson에서: git pull → TRT 빌드")
        print(f"    python training/export_for_jetson.py --weights models/yolo26s-lower6.pt --format engine --half")

    elif args.format == "engine":
        half = args.half
        print(f"  TensorRT 내보내기 (FP16={half})")

        exported_path = model.export(
            format="engine",
            imgsz=args.imgsz,
            half=half,
            device=0,
        )
        print(f"  ✅ TensorRT 엔진 저장: {exported_path}")

        engine_path = Path(exported_path)
        if engine_path.exists():
            size_mb = engine_path.stat().st_size / 1e6
            print(f"  파일 크기: {size_mb:.1f} MB")

    return exported_path


def validate_model(model_path: str, data_yaml: str, imgsz: int):
    """내보낸 모델 정확도 검증"""
    from ultralytics import YOLO

    print(f"\n  검증 실행: {model_path}")
    print(f"  데이터셋: {data_yaml}")

    model = YOLO(model_path)
    results = model.val(
        data=data_yaml,
        imgsz=imgsz,
        batch=1,
        device=0,
    )

    print(f"\n  === 검증 결과 ===")
    print(f"  mAP50:     {results.box.map50:.4f}")
    print(f"  mAP50-95:  {results.box.map:.4f}")

    if hasattr(results, 'pose'):
        print(f"  Pose mAP50:    {results.pose.map50:.4f}")
        print(f"  Pose mAP50-95: {results.pose.map:.4f}")

    return results


def main():
    args = parse_args()

    print("=" * 60)
    print("  하체 모델 내보내기 (Jetson 배포용)")
    print("=" * 60)

    check_platform(args.format)

    # 내보내기
    exported_path = export_model(args)

    # 검증
    if args.validate:
        data_yaml = args.data
        if data_yaml is None:
            # 기본 경로 시도
            candidates = [
                Path(__file__).parent / "lower_body_pose.yaml",
                Path("training/lower_body_pose.yaml"),
            ]
            for c in candidates:
                if c.exists():
                    data_yaml = str(c)
                    break

        if data_yaml:
            validate_model(exported_path, data_yaml, args.imgsz)
        else:
            print("  WARNING: 검증용 YAML 없음. --data 옵션 지정 필요.")

    print(f"\n  완료! 내보낸 모델: {exported_path}")


if __name__ == "__main__":
    main()
