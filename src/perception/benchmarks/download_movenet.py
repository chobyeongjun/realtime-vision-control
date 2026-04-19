#!/usr/bin/env python3
"""
MoveNet 모델 다운로드
=====================
TFLite 모델을 TensorFlow Hub에서 다운로드합니다.

사용법:
    python3 download_movenet.py              # Lightning + Thunder 모두 다운로드
    python3 download_movenet.py --lightning   # Lightning만
    python3 download_movenet.py --thunder     # Thunder만
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import shutil
import tempfile

MODELS = {
    "lightning": {
        "url": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite",
        "filename": "movenet_lightning.tflite",
        "input_size": 192,
        "description": "MoveNet Lightning - 초고속 (192x192, ~3ms on Jetson)",
    },
    "thunder": {
        "url": "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite",
        "filename": "movenet_thunder.tflite",
        "input_size": 256,
        "description": "MoveNet Thunder - 정확 (256x256, ~8ms on Jetson)",
    },
}

# float16 버전 (대안)
MODELS_FP16 = {
    "lightning": {
        "url": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite",
        "filename": "movenet_lightning_fp16.tflite",
    },
    "thunder": {
        "url": "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite",
        "filename": "movenet_thunder_fp16.tflite",
    },
}


def download_file(url, dest_path, description=""):
    """URL에서 파일 다운로드"""
    if os.path.exists(dest_path):
        print(f"  이미 존재: {dest_path}")
        return True

    print(f"  다운로드 중: {description}")
    print(f"    URL: {url}")
    print(f"    저장: {dest_path}")

    try:
        # TF Hub URL은 리다이렉트될 수 있음
        req = urllib.request.Request(url, headers={
            'User-Agent': 'Mozilla/5.0'
        })
        with urllib.request.urlopen(req, timeout=60) as response:
            content_type = response.headers.get('Content-Type', '')
            data = response.read()

            # zip 파일인 경우 압축 해제
            if 'zip' in content_type or data[:4] == b'PK\x03\x04':
                with tempfile.NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                try:
                    with zipfile.ZipFile(tmp_path, 'r') as zf:
                        # tflite 파일 찾기
                        tflite_files = [f for f in zf.namelist() if f.endswith('.tflite')]
                        if tflite_files:
                            with zf.open(tflite_files[0]) as src:
                                with open(dest_path, 'wb') as dst:
                                    dst.write(src.read())
                        else:
                            print(f"    경고: zip 내 tflite 파일 없음")
                            return False
                finally:
                    os.unlink(tmp_path)
            else:
                with open(dest_path, 'wb') as f:
                    f.write(data)

        size_mb = os.path.getsize(dest_path) / (1024 * 1024)
        print(f"  ✓ 완료 ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"  ✗ 다운로드 실패: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def download_from_kaggle(variant, model_dir):
    """Kaggle에서 MoveNet 모델 다운로드 (TF Hub 실패 시 대안)"""
    filename = MODELS[variant]["filename"]
    dest_path = os.path.join(model_dir, filename)

    if os.path.exists(dest_path):
        print(f"  이미 존재: {dest_path}")
        return True

    kaggle_urls = {
        "lightning": "https://www.kaggle.com/models/google/movenet/tfLite/singlepose-lightning-tflite-int8/1",
        "thunder": "https://www.kaggle.com/models/google/movenet/tfLite/singlepose-thunder-tflite-int8/1",
    }

    print(f"  Kaggle에서 다운로드 시도...")
    print(f"    수동 다운로드: {kaggle_urls.get(variant, 'N/A')}")
    print(f"    다운로드 후 {dest_path} 에 저장하세요")
    return False


def main():
    parser = argparse.ArgumentParser(description="MoveNet 모델 다운로드")
    parser.add_argument("--lightning", action="store_true", help="Lightning만 다운로드")
    parser.add_argument("--thunder", action="store_true", help="Thunder만 다운로드")
    parser.add_argument("--fp16", action="store_true", help="float16 버전도 다운로드")
    parser.add_argument("--model-dir", default=None, help="모델 저장 디렉토리")
    args = parser.parse_args()

    model_dir = args.model_dir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models")
    os.makedirs(model_dir, exist_ok=True)

    # 다운로드할 모델 결정
    if args.lightning:
        variants = ["lightning"]
    elif args.thunder:
        variants = ["thunder"]
    else:
        variants = ["lightning", "thunder"]

    print("=" * 50)
    print("  MoveNet 모델 다운로드")
    print("=" * 50)
    print(f"  저장 디렉토리: {model_dir}")
    print()

    success = 0
    for variant in variants:
        info = MODELS[variant]
        dest_path = os.path.join(model_dir, info["filename"])

        print(f"\n--- {info['description']} ---")
        print(f"  입력 크기: {info['input_size']}x{info['input_size']}")

        if download_file(info["url"], dest_path, info["description"]):
            success += 1
        else:
            # TF Hub 실패 시 Kaggle 안내
            download_from_kaggle(variant, model_dir)

    # float16 버전
    if args.fp16:
        print("\n--- float16 버전 ---")
        for variant in variants:
            info = MODELS_FP16[variant]
            dest_path = os.path.join(model_dir, info["filename"])
            download_file(info["url"], dest_path, f"{variant} (float16)")

    # 결과 요약
    print(f"\n{'=' * 50}")
    print(f"  다운로드 완료: {success}/{len(variants)} 모델")
    print(f"{'=' * 50}")

    # 파일 목록
    print(f"\n  모델 파일:")
    for f in sorted(os.listdir(model_dir)):
        if f.endswith(('.tflite', '.onnx')):
            size = os.path.getsize(os.path.join(model_dir, f))
            print(f"    {f:<40} {size / 1024:.0f} KB")

    print(f"\n  검증: python3 verify_models.py")
    print(f"  벤치마크: python3 run_benchmark.py --models movenet --visualize")


if __name__ == "__main__":
    main()
