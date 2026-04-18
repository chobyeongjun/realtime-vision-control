#!/usr/bin/env python3
"""
변환된 하체 데이터셋 품질 검증 + 시각화
=======================================
convert_coco_to_lower_body.py로 변환된 데이터셋의 품질을 검증합니다.

사용법:
    python validate_dataset.py \
        --dataset-dir ./data/lower_body \
        --num-samples 20 \
        --output-dir ./data/lower_body/validation_samples
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np


# 하체 6 키포인트 정보
LOWER_BODY_NAMES = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# 스켈레톤 연결
SKELETON = [
    (0, 2),  # left_hip → left_knee
    (2, 4),  # left_knee → left_ankle
    (1, 3),  # right_hip → right_knee
    (3, 5),  # right_knee → right_ankle
    (0, 1),  # left_hip → right_hip
]

# 색상: 좌측=초록, 우측=파랑
COLORS_LEFT = (0, 255, 0)
COLORS_RIGHT = (255, 0, 0)
KPT_COLORS = [
    COLORS_LEFT, COLORS_RIGHT,   # hip
    COLORS_LEFT, COLORS_RIGHT,   # knee
    COLORS_LEFT, COLORS_RIGHT,   # ankle
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="하체 데이터셋 품질 검증 + 시각화"
    )
    parser.add_argument(
        "--dataset-dir", required=True,
        help="변환된 데이터셋 경로 (train/, val/ 포함)"
    )
    parser.add_argument(
        "--split", default="val",
        help="검증할 split (기본: val)"
    )
    parser.add_argument(
        "--num-samples", type=int, default=20,
        help="시각화할 샘플 수 (기본: 20)"
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="시각화 결과 저장 경로 (기본: dataset-dir/validation_samples)"
    )
    return parser.parse_args()


# ============================================================================
# 라벨 파싱
# ============================================================================

def parse_yolo_label(label_path: str) -> list:
    """
    YOLO 라벨 파일 파싱.

    Returns:
        list of dict: [{bbox: (cx,cy,w,h), keypoints: [(x,y,v),...]}]
    """
    annotations = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5 + 6 * 3:  # class + bbox(4) + 6kpt*3
                continue

            cls = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])

            keypoints = []
            for i in range(6):
                idx = 5 + i * 3
                kx = float(parts[idx])
                ky = float(parts[idx + 1])
                kv = int(parts[idx + 2])
                keypoints.append((kx, ky, kv))

            annotations.append({
                "class": cls,
                "bbox": (cx, cy, w, h),
                "keypoints": keypoints,
            })

    return annotations


# ============================================================================
# 통계 계산
# ============================================================================

def compute_dataset_statistics(dataset_dir: str, split: str) -> dict:
    """데이터셋 전체 통계 계산"""
    label_dir = Path(dataset_dir) / split / "labels"
    image_dir = Path(dataset_dir) / split / "images"

    if not label_dir.exists():
        return {"error": f"Label directory not found: {label_dir}"}

    stats = {
        "total_images": 0,
        "total_labels": 0,
        "total_annotations": 0,
        "annotations_per_image": [],
        "bbox_widths": [],
        "bbox_heights": [],
        "bbox_aspect_ratios": [],
        "per_kpt_visible": [0] * 6,
        "per_kpt_total": 0,
        "degenerate_count": 0,
        "degenerate_files": [],
    }

    # 이미지 수
    if image_dir.exists():
        stats["total_images"] = len(list(image_dir.glob("*.jpg")) +
                                     list(image_dir.glob("*.png")))

    # 라벨 파일 순회
    label_files = sorted(label_dir.glob("*.txt"))
    stats["total_labels"] = len(label_files)

    for lf in label_files:
        annotations = parse_yolo_label(str(lf))
        stats["annotations_per_image"].append(len(annotations))

        for ann in annotations:
            stats["total_annotations"] += 1
            cx, cy, w, h = ann["bbox"]

            stats["bbox_widths"].append(w)
            stats["bbox_heights"].append(h)
            if h > 0:
                stats["bbox_aspect_ratios"].append(w / h)

            # degenerate 체크
            is_degenerate = False
            if w < 0.01 or h < 0.01:
                is_degenerate = True
            if h > 0 and (w / h > 5 or w / h < 0.2):
                is_degenerate = True

            # all keypoints at (0,0)
            all_zero = all(kx == 0 and ky == 0 for kx, ky, kv in ann["keypoints"])
            if all_zero:
                is_degenerate = True

            if is_degenerate:
                stats["degenerate_count"] += 1
                stats["degenerate_files"].append(str(lf.name))

            # per-keypoint visibility
            stats["per_kpt_total"] += 1
            for i, (kx, ky, kv) in enumerate(ann["keypoints"]):
                if kv > 0:
                    stats["per_kpt_visible"][i] += 1

    return stats


def print_statistics(stats: dict, split: str):
    """통계 출력"""
    if "error" in stats:
        print(f"  ERROR: {stats['error']}")
        return

    print(f"\n{'='*60}")
    print(f"  데이터셋 통계: {split}")
    print(f"{'='*60}")
    print(f"  이미지 수:       {stats['total_images']}")
    print(f"  라벨 파일 수:    {stats['total_labels']}")
    print(f"  총 annotations:  {stats['total_annotations']}")

    if stats["annotations_per_image"]:
        ann_arr = np.array(stats["annotations_per_image"])
        print(f"  이미지당 ann:    mean={ann_arr.mean():.1f}, "
              f"max={ann_arr.max()}, min={ann_arr.min()}")

    if stats["bbox_widths"]:
        bw = np.array(stats["bbox_widths"])
        bh = np.array(stats["bbox_heights"])
        print(f"\n  BBox 크기 (normalized):")
        print(f"    Width:  mean={bw.mean():.3f}, std={bw.std():.3f}, "
              f"min={bw.min():.3f}, max={bw.max():.3f}")
        print(f"    Height: mean={bh.mean():.3f}, std={bh.std():.3f}, "
              f"min={bh.min():.3f}, max={bh.max():.3f}")

    if stats["bbox_aspect_ratios"]:
        ar = np.array(stats["bbox_aspect_ratios"])
        print(f"    Aspect: mean={ar.mean():.2f}, std={ar.std():.2f}")

    print(f"\n  키포인트별 Visibility Rate:")
    total = max(stats["per_kpt_total"], 1)
    for i, name in enumerate(LOWER_BODY_NAMES):
        rate = stats["per_kpt_visible"][i] / total * 100
        bar = "█" * int(rate / 2) + "░" * (50 - int(rate / 2))
        print(f"    {name:15s}: {bar} {rate:5.1f}%")

    print(f"\n  Degenerate labels: {stats['degenerate_count']}")
    if stats["degenerate_files"]:
        for f in stats["degenerate_files"][:5]:
            print(f"    - {f}")
        if len(stats["degenerate_files"]) > 5:
            print(f"    ... and {len(stats['degenerate_files']) - 5} more")

    # 품질 판정
    print(f"\n  === 품질 판정 ===")
    issues = []
    if stats["total_annotations"] < 1000:
        issues.append("⚠ annotation 수가 1000 미만 (학습에 부족)")
    if stats["degenerate_count"] > stats["total_annotations"] * 0.05:
        issues.append("⚠ degenerate label이 5% 초과")
    for i, name in enumerate(LOWER_BODY_NAMES):
        rate = stats["per_kpt_visible"][i] / total * 100
        if rate < 70:
            issues.append(f"⚠ {name} visibility {rate:.0f}% (70% 미만)")

    if issues:
        for issue in issues:
            print(f"    {issue}")
    else:
        print(f"    ✅ 모든 품질 기준 통과!")


# ============================================================================
# 시각화
# ============================================================================

def visualize_sample(image_path: str, label_path: str, output_path: str):
    """하나의 샘플을 시각화하여 저장"""
    img = cv2.imread(image_path)
    if img is None:
        print(f"  WARNING: Cannot read image: {image_path}")
        return False

    h, w = img.shape[:2]
    annotations = parse_yolo_label(label_path)

    for ann in annotations:
        cx, cy, bw, bh = ann["bbox"]

        # bbox 그리기 (denormalize)
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

        # 키포인트 그리기
        kpt_pixels = []
        for i, (kx, ky, kv) in enumerate(ann["keypoints"]):
            px = int(kx * w)
            py = int(ky * h)
            kpt_pixels.append((px, py, kv))

            if kv > 0 and px > 0 and py > 0:
                color = KPT_COLORS[i]
                radius = 6 if kv == 2 else 4
                thickness = -1 if kv == 2 else 2  # filled=visible, hollow=occluded
                cv2.circle(img, (px, py), radius, color, thickness)
                # 이름 표시
                short = LOWER_BODY_NAMES[i].replace("left_", "L").replace("right_", "R")
                cv2.putText(img, short, (px + 8, py - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 스켈레톤 그리기
        for k1, k2 in SKELETON:
            px1, py1, v1 = kpt_pixels[k1]
            px2, py2, v2 = kpt_pixels[k2]
            if v1 > 0 and v2 > 0 and px1 > 0 and py1 > 0 and px2 > 0 and py2 > 0:
                color = KPT_COLORS[k1]
                cv2.line(img, (px1, py1), (px2, py2), color, 2)

    cv2.imwrite(output_path, img)
    return True


def visualize_samples(dataset_dir: str, split: str, num_samples: int,
                      output_dir: str):
    """랜덤 샘플 시각화"""
    image_dir = Path(dataset_dir) / split / "images"
    label_dir = Path(dataset_dir) / split / "labels"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 라벨이 있는 이미지 찾기
    label_files = sorted(label_dir.glob("*.txt"))
    if not label_files:
        print(f"  WARNING: No label files found in {label_dir}")
        return

    # 랜덤 샘플링
    sample_files = random.sample(label_files, min(num_samples, len(label_files)))

    print(f"\n  시각화: {len(sample_files)} 샘플 → {out_dir}")

    success = 0
    for lf in sample_files:
        stem = lf.stem
        # 이미지 찾기 (jpg 또는 png)
        img_path = None
        for ext in [".jpg", ".png", ".jpeg"]:
            candidate = image_dir / f"{stem}{ext}"
            if candidate.exists():
                img_path = str(candidate)
                break

        if img_path is None:
            continue

        out_path = str(out_dir / f"{stem}_vis.jpg")
        if visualize_sample(img_path, str(lf), out_path):
            success += 1

    print(f"  완료: {success}/{len(sample_files)} 샘플 저장됨")


# ============================================================================
# 메인
# ============================================================================

def main():
    args = parse_args()

    output_dir = args.output_dir or str(Path(args.dataset_dir) / "validation_samples")

    print("=" * 60)
    print("  하체 데이터셋 품질 검증")
    print("=" * 60)

    # 통계 계산
    stats = compute_dataset_statistics(args.dataset_dir, args.split)
    print_statistics(stats, args.split)

    # 시각화
    visualize_samples(args.dataset_dir, args.split, args.num_samples, output_dir)

    # 메타데이터와 통계 저장
    stats_path = Path(args.dataset_dir) / f"validation_stats_{args.split}.json"
    # numpy array를 list로 변환
    serializable_stats = {}
    for k, v in stats.items():
        if isinstance(v, np.ndarray):
            serializable_stats[k] = v.tolist()
        elif isinstance(v, list) and v and isinstance(v[0], (np.floating, np.integer)):
            serializable_stats[k] = [float(x) for x in v]
        else:
            serializable_stats[k] = v

    with open(stats_path, "w") as f:
        json.dump(serializable_stats, f, indent=2, default=str)
    print(f"\n  통계 저장: {stats_path}")


if __name__ == "__main__":
    main()
