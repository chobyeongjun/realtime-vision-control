#!/usr/bin/env python3
"""
COCO 17-Keypoint → 하체 전용 6-Keypoint YOLO 포맷 변환
=======================================================
표준 COCO person_keypoints 어노테이션에서 하체 6개 키포인트만 추출하여
YOLO pose 학습용 포맷으로 변환합니다.

사용법:
    python convert_coco_to_lower_body.py \
        --coco-dir ./data/coco \
        --output-dir ./data/lower_body \
        --min-visible-kpts 4 \
        --bbox-padding 0.2

입력 구조:
    data/coco/
    ├── train2017/           # 이미지
    ├── val2017/
    └── annotations/
        ├── person_keypoints_train2017.json
        └── person_keypoints_val2017.json

출력 구조:
    data/lower_body/
    ├── train/
    │   ├── images/  → symlink to coco images
    │   └── labels/  → .txt YOLO labels
    └── val/
        ├── images/
        └── labels/

하체 6 Keypoints:
    0: left_hip      (COCO idx 11)
    1: right_hip     (COCO idx 12)
    2: left_knee     (COCO idx 13)
    3: right_knee    (COCO idx 14)
    4: left_ankle    (COCO idx 15)
    5: right_ankle   (COCO idx 16)
"""

import argparse
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

# ============================================================================
# COCO → 하체 키포인트 매핑
# ============================================================================
# COCO 17 keypoints 중 하체 6개의 인덱스
COCO_TO_LOWER = {
    11: 0,   # left_hip
    12: 1,   # right_hip
    13: 2,   # left_knee
    14: 3,   # right_knee
    15: 4,   # left_ankle
    16: 5,   # right_ankle
}

LOWER_BODY_NAMES = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]

# 좌우 대칭 매핑 (fliplr 증강용)
FLIP_IDX = [1, 0, 3, 2, 5, 4]

NUM_KEYPOINTS = 6


def parse_args():
    parser = argparse.ArgumentParser(
        description="COCO 17kpt → 하체 6kpt YOLO 포맷 변환"
    )
    parser.add_argument(
        "--coco-dir", required=True,
        help="COCO 데이터셋 루트 (train2017/, val2017/, annotations/ 포함)"
    )
    parser.add_argument(
        "--output-dir", default="./data/lower_body",
        help="변환된 데이터 출력 경로 (기본: ./data/lower_body)"
    )
    parser.add_argument(
        "--min-visible-kpts", type=int, default=4,
        help="최소 visible 키포인트 수 (기본: 4, 6개 중 4개 이상 보여야 사용)"
    )
    parser.add_argument(
        "--bbox-padding", type=float, default=0.2,
        help="하체 bbox 패딩 비율 (기본: 0.2 = 20%%)"
    )
    parser.add_argument(
        "--min-bbox-size", type=float, default=0.02,
        help="최소 bbox 크기 (이미지 대비 비율, 기본: 0.02 = 2%%)"
    )
    parser.add_argument(
        "--splits", nargs="+", default=["train", "val"],
        help="변환할 split (기본: train val)"
    )
    parser.add_argument(
        "--copy-images", action="store_true",
        help="이미지를 symlink 대신 복사 (Windows 호환)"
    )
    return parser.parse_args()


# ============================================================================
# 핵심 변환 함수
# ============================================================================

def load_coco_annotations(annotation_path: str) -> dict:
    """COCO 어노테이션 JSON 로드"""
    print(f"  Loading: {annotation_path}")
    with open(annotation_path, "r") as f:
        data = json.load(f)
    print(f"  Images: {len(data['images'])}, Annotations: {len(data['annotations'])}")
    return data


def extract_lower_body_keypoints(ann: dict) -> list:
    """
    COCO person annotation에서 하체 6개 키포인트 추출.

    Args:
        ann: COCO annotation dict (keypoints 필드 필수)

    Returns:
        list of (x, y, v) tuples for 6 lower body keypoints
        v: 0=not labeled, 1=labeled but not visible, 2=labeled and visible
    """
    kps = ann.get("keypoints", [])
    if len(kps) < 51:  # 17 * 3 = 51
        return None

    lower_kps = []
    for coco_idx in sorted(COCO_TO_LOWER.keys()):
        x = kps[coco_idx * 3]
        y = kps[coco_idx * 3 + 1]
        v = kps[coco_idx * 3 + 2]
        lower_kps.append((x, y, v))

    return lower_kps


def compute_lower_body_bbox(keypoints: list, img_w: int, img_h: int,
                             padding: float = 0.2) -> tuple:
    """
    Visible 키포인트로부터 하체 bbox 계산.

    Args:
        keypoints: list of (x, y, v), 6개
        img_w, img_h: 이미지 크기
        padding: bbox 패딩 비율

    Returns:
        (cx, cy, w, h) normalized to [0, 1], or None if invalid
    """
    # visible 키포인트만 사용
    visible_pts = [(x, y) for x, y, v in keypoints if v > 0 and x > 0 and y > 0]
    if len(visible_pts) < 2:
        return None

    xs = [p[0] for p in visible_pts]
    ys = [p[1] for p in visible_pts]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # bbox 크기
    bw = x_max - x_min
    bh = y_max - y_min

    # 패딩 적용 (상단 hip쪽은 여유 더 크게, 하단 ankle은 보통)
    pad_x = bw * padding
    pad_y_top = bh * (padding + 0.05)    # hip 위로 25% 여유
    pad_y_bottom = bh * (padding + 0.0)  # ankle 아래로 20% 여유

    x_min = max(0, x_min - pad_x)
    x_max = min(img_w, x_max + pad_x)
    y_min = max(0, y_min - pad_y_top)
    y_max = min(img_h, y_max + pad_y_bottom)

    # YOLO 포맷: center_x, center_y, width, height (normalized)
    cx = ((x_min + x_max) / 2.0) / img_w
    cy = ((y_min + y_max) / 2.0) / img_h
    w = (x_max - x_min) / img_w
    h = (y_max - y_min) / img_h

    # clamp to [0, 1]
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w = max(0.0, min(1.0, w))
    h = max(0.0, min(1.0, h))

    return (cx, cy, w, h)


def extract_lower_body_annotation(ann: dict, img_w: int, img_h: int,
                                   min_visible: int = 4,
                                   bbox_padding: float = 0.2,
                                   min_bbox_size: float = 0.02) -> str:
    """
    하나의 COCO person annotation을 YOLO 하체 라벨 라인으로 변환.

    Args:
        ann: COCO annotation
        img_w, img_h: 이미지 크기
        min_visible: 최소 visible 키포인트 수
        bbox_padding: bbox 패딩 비율
        min_bbox_size: 최소 bbox 크기 비율

    Returns:
        YOLO label line string, or None if filtered out
    """
    # 너무 작은 annotation 스킵
    if ann.get("area", 0) < 100:
        return None

    # iscrowd 스킵
    if ann.get("iscrowd", 0):
        return None

    # 하체 키포인트 추출
    lower_kps = extract_lower_body_keypoints(ann)
    if lower_kps is None:
        return None

    # visible 키포인트 수 체크
    n_visible = sum(1 for _, _, v in lower_kps if v > 0)
    if n_visible < min_visible:
        return None

    # 하체 bbox 계산
    bbox = compute_lower_body_bbox(lower_kps, img_w, img_h, bbox_padding)
    if bbox is None:
        return None

    cx, cy, w, h = bbox

    # 너무 작은 bbox 필터
    if w < min_bbox_size or h < min_bbox_size:
        return None

    # YOLO 라벨 라인 구성: class cx cy w h kx1 ky1 kv1 ... kx6 ky6 kv6
    parts = [f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"]

    for x, y, v in lower_kps:
        if v > 0 and x > 0 and y > 0:
            # 좌표를 이미지 크기로 normalize
            nx = x / img_w
            ny = y / img_h
            # YOLO pose format: v=0 (not visible), v=1 (occluded), v=2 (visible)
            # Ultralytics는 v>0이면 loss 계산에 포함
            nv = 2 if v == 2 else 1
            parts.append(f"{nx:.6f} {ny:.6f} {nv}")
        else:
            # 라벨 안 된 키포인트: 0 0 0
            parts.append("0.000000 0.000000 0")

    return " ".join(parts)


# ============================================================================
# Split 변환
# ============================================================================

def get_split_paths(coco_dir: str, split: str) -> tuple:
    """split별 이미지 디렉토리와 어노테이션 파일 경로"""
    coco_dir = Path(coco_dir)

    if split == "train":
        img_dir = coco_dir / "train2017"
        ann_file = coco_dir / "annotations" / "person_keypoints_train2017.json"
    elif split == "val":
        img_dir = coco_dir / "val2017"
        ann_file = coco_dir / "annotations" / "person_keypoints_val2017.json"
    else:
        raise ValueError(f"Unknown split: {split}")

    if not img_dir.exists():
        print(f"  WARNING: Image directory not found: {img_dir}")
        return None, None
    if not ann_file.exists():
        print(f"  WARNING: Annotation file not found: {ann_file}")
        return None, None

    return str(img_dir), str(ann_file)


def convert_split(coco_dir: str, output_dir: str, split: str,
                  min_visible: int = 4, bbox_padding: float = 0.2,
                  min_bbox_size: float = 0.02, copy_images: bool = False) -> dict:
    """
    하나의 split (train 또는 val) 변환.

    Returns:
        statistics dict
    """
    print(f"\n{'='*60}")
    print(f"  Converting: {split}")
    print(f"{'='*60}")

    img_dir, ann_file = get_split_paths(coco_dir, split)
    if img_dir is None or ann_file is None:
        return {"error": f"Missing files for {split}"}

    # 어노테이션 로드
    coco_data = load_coco_annotations(ann_file)

    # 이미지 ID → 이미지 정보 매핑
    img_info = {}
    for img in coco_data["images"]:
        img_info[img["id"]] = img

    # 출력 디렉토리 생성
    out_img_dir = Path(output_dir) / split / "images"
    out_lbl_dir = Path(output_dir) / split / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    # 통계
    stats = {
        "total_annotations": 0,
        "converted": 0,
        "skipped_area": 0,
        "skipped_crowd": 0,
        "skipped_no_kpts": 0,
        "skipped_few_visible": 0,
        "skipped_small_bbox": 0,
        "skipped_no_bbox": 0,
        "images_with_labels": 0,
        "images_total": len(img_info),
        "per_kpt_visible": [0] * NUM_KEYPOINTS,
    }

    # 이미지별 라벨 그룹핑
    labels_per_image = defaultdict(list)

    for ann in coco_data["annotations"]:
        stats["total_annotations"] += 1

        img_id = ann["image_id"]
        if img_id not in img_info:
            continue

        img = img_info[img_id]
        img_w, img_h = img["width"], img["height"]

        label_line = extract_lower_body_annotation(
            ann, img_w, img_h,
            min_visible=min_visible,
            bbox_padding=bbox_padding,
            min_bbox_size=min_bbox_size,
        )

        if label_line is None:
            # 스킵 이유 분류
            if ann.get("iscrowd", 0):
                stats["skipped_crowd"] += 1
            elif ann.get("area", 0) < 100:
                stats["skipped_area"] += 1
            else:
                # 키포인트 부족 등
                lower_kps = extract_lower_body_keypoints(ann)
                if lower_kps is None:
                    stats["skipped_no_kpts"] += 1
                else:
                    n_vis = sum(1 for _, _, v in lower_kps if v > 0)
                    if n_vis < min_visible:
                        stats["skipped_few_visible"] += 1
                    else:
                        stats["skipped_small_bbox"] += 1
            continue

        stats["converted"] += 1
        labels_per_image[img_id].append(label_line)

        # per-keypoint visibility 통계
        lower_kps = extract_lower_body_keypoints(ann)
        if lower_kps:
            for i, (_, _, v) in enumerate(lower_kps):
                if v > 0:
                    stats["per_kpt_visible"][i] += 1

    # 라벨 파일 작성 + 이미지 링크
    linked_images = set()
    for img_id, lines in labels_per_image.items():
        img = img_info[img_id]
        filename = img["file_name"]
        stem = Path(filename).stem

        # 라벨 파일 작성
        label_path = out_lbl_dir / f"{stem}.txt"
        with open(label_path, "w") as f:
            f.write("\n".join(lines) + "\n")

        # 이미지 링크 (symlink 또는 copy)
        src_img = Path(img_dir) / filename
        dst_img = out_img_dir / filename

        if not dst_img.exists() and src_img.exists():
            if copy_images:
                import shutil
                shutil.copy2(str(src_img), str(dst_img))
            else:
                try:
                    os.symlink(os.path.abspath(str(src_img)), str(dst_img))
                except OSError:
                    # symlink 실패 시 copy 시도
                    import shutil
                    shutil.copy2(str(src_img), str(dst_img))

        linked_images.add(img_id)

    stats["images_with_labels"] = len(linked_images)

    # 통계 출력
    print(f"\n  --- {split} 변환 결과 ---")
    print(f"  총 annotations:      {stats['total_annotations']}")
    print(f"  변환 성공:            {stats['converted']}")
    print(f"  라벨 있는 이미지:     {stats['images_with_labels']} / {stats['images_total']}")
    print(f"  스킵 - crowd:         {stats['skipped_crowd']}")
    print(f"  스킵 - 작은 area:     {stats['skipped_area']}")
    print(f"  스킵 - kpt 부족:      {stats['skipped_few_visible']}")
    print(f"  스킵 - bbox 작음:     {stats['skipped_small_bbox']}")

    if stats["converted"] > 0:
        print(f"\n  키포인트별 visibility rate:")
        for i, name in enumerate(LOWER_BODY_NAMES):
            rate = stats["per_kpt_visible"][i] / stats["converted"] * 100
            print(f"    {name:15s}: {rate:5.1f}% ({stats['per_kpt_visible'][i]}/{stats['converted']})")

    return stats


# ============================================================================
# 메인
# ============================================================================

def main():
    args = parse_args()

    print("=" * 60)
    print("  COCO → 하체 6kpt YOLO 포맷 변환")
    print("=" * 60)
    print(f"  COCO 디렉토리: {args.coco_dir}")
    print(f"  출력 디렉토리:  {args.output_dir}")
    print(f"  최소 visible:   {args.min_visible_kpts}/6")
    print(f"  bbox 패딩:      {args.bbox_padding:.0%}")
    print(f"  최소 bbox 크기: {args.min_bbox_size:.0%}")
    print(f"  이미지 모드:    {'복사' if args.copy_images else 'symlink'}")

    # COCO 디렉토리 존재 확인
    coco_dir = Path(args.coco_dir)
    if not coco_dir.exists():
        print(f"\n  ERROR: COCO 디렉토리가 없습니다: {coco_dir}")
        print(f"  다음 구조가 필요합니다:")
        print(f"    {coco_dir}/train2017/")
        print(f"    {coco_dir}/val2017/")
        print(f"    {coco_dir}/annotations/person_keypoints_train2017.json")
        print(f"    {coco_dir}/annotations/person_keypoints_val2017.json")
        sys.exit(1)

    all_stats = {}
    for split in args.splits:
        stats = convert_split(
            coco_dir=args.coco_dir,
            output_dir=args.output_dir,
            split=split,
            min_visible=args.min_visible_kpts,
            bbox_padding=args.bbox_padding,
            min_bbox_size=args.min_bbox_size,
            copy_images=args.copy_images,
        )
        all_stats[split] = stats

    # 전체 요약
    print(f"\n{'='*60}")
    print(f"  변환 완료!")
    print(f"{'='*60}")

    total_converted = 0
    total_images = 0
    for split, stats in all_stats.items():
        if "error" in stats:
            print(f"  {split}: ERROR - {stats['error']}")
        else:
            total_converted += stats["converted"]
            total_images += stats["images_with_labels"]
            print(f"  {split}: {stats['converted']} labels, "
                  f"{stats['images_with_labels']} images")

    print(f"\n  총 라벨: {total_converted}")
    print(f"  총 이미지: {total_images}")
    print(f"\n  출력 경로: {args.output_dir}")

    # 메타데이터 저장
    meta_path = Path(args.output_dir) / "conversion_meta.json"
    meta = {
        "source": "COCO person_keypoints",
        "keypoints": LOWER_BODY_NAMES,
        "num_keypoints": NUM_KEYPOINTS,
        "flip_idx": FLIP_IDX,
        "coco_mapping": {str(k): v for k, v in COCO_TO_LOWER.items()},
        "min_visible_kpts": args.min_visible_kpts,
        "bbox_padding": args.bbox_padding,
        "stats": all_stats,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  메타데이터: {meta_path}")


if __name__ == "__main__":
    main()
