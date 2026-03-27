#!/usr/bin/env python3
"""
워커 ZED 영상 자동 라벨링 (2차 Fine-Tuning용)
=============================================
1차 학습된 하체 6kpt 모델을 사용하여 워커 보행 영상에서
자동으로 keypoint 라벨을 생성합니다.

워크플로우:
    1. [Jetson] 워커 보행 영상 촬영
       python benchmarks/record_zed.py --output recordings/walk_01.mp4 --duration 300

    2. [Jetson or RTX] 자동 라벨링
       python training/auto_label_walker.py \
           --video recordings/walk_01.mp4 \
           --model models/yolo26s-lower6.pt \
           --output-dir data/walker

    3. [RTX 5090] 2차 Fine-Tuning
       python training/train_lower_body.py \
           --model models/yolo26s-lower6.pt \
           --data training/walker_data.yaml \
           --name lower_body_v2_walker

키포인트: 6개 (hip/knee/ankle × left/right)
"""

import argparse
import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


# 하체 6 키포인트 이름
LOWER_BODY_NAMES = [
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="워커 ZED 영상 자동 라벨링 (2차 Fine-Tuning용)"
    )
    parser.add_argument(
        "--video", required=True, nargs="+",
        help="입력 영상 경로 (여러 개 가능: --video walk1.mp4 walk2.mp4)"
    )
    parser.add_argument(
        "--model", default="models/yolo26s-lower6.pt",
        help="1차 학습된 모델 (기본: models/yolo26s-lower6.pt)"
    )
    parser.add_argument(
        "--output-dir", default="data/walker",
        help="출력 경로 (기본: data/walker)"
    )
    parser.add_argument(
        "--frame-interval", type=int, default=4,
        help="프레임 추출 간격 (기본: 4 = 120fps 영상에서 30fps로 추출)"
    )
    parser.add_argument(
        "--min-conf", type=float, default=0.5,
        help="최소 confidence (기본: 0.5, 이하면 라벨 생성 안 함)"
    )
    parser.add_argument(
        "--min-kpts", type=int, default=4,
        help="최소 visible 키포인트 수 (기본: 4/6)"
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="모델 입력 해상도 (기본: 640)"
    )
    parser.add_argument(
        "--visualize", type=int, default=20,
        help="시각화 샘플 수 (기본: 20)"
    )
    parser.add_argument(
        "--split-ratio", type=float, default=0.9,
        help="train/val 분할 비율 (기본: 0.9 = 90%% train)"
    )
    return parser.parse_args()


def auto_label_video(video_path, model, output_dir, frame_interval=4,
                     min_conf=0.5, min_kpts=4, imgsz=640):
    """
    하나의 영상에서 자동 라벨링.

    Returns:
        dict: statistics
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: 영상 열기 실패: {video_path}")
        return {"error": True}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    video_name = Path(video_path).stem
    img_dir = Path(output_dir) / "all" / "images"
    lbl_dir = Path(output_dir) / "all" / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    print(f"  영상: {video_path}")
    print(f"  해상도: {w}x{h}, FPS: {fps:.0f}, 총 프레임: {total_frames}")
    print(f"  추출 간격: {frame_interval} (매 {frame_interval}프레임마다)")

    stats = {
        "total_frames": total_frames,
        "extracted": 0,
        "labeled": 0,
        "skipped_low_conf": 0,
        "skipped_few_kpts": 0,
    }

    frame_idx = 0
    pbar = tqdm(total=total_frames // frame_interval, desc=f"  {video_name}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval != 0:
            frame_idx += 1
            continue

        stats["extracted"] += 1

        # 모델 추론
        results = model(frame, verbose=False, imgsz=imgsz, conf=0.25)

        if results and len(results) > 0 and results[0].keypoints is not None:
            kps_data = results[0].keypoints.data
            if len(kps_data) > 0:
                # 가장 큰 (가장 가까운) 사람 선택
                best_idx = 0
                if len(kps_data) > 1:
                    # bbox 면적이 가장 큰 사람
                    if results[0].boxes is not None and len(results[0].boxes) > 0:
                        areas = (results[0].boxes.xyxy[:, 2] - results[0].boxes.xyxy[:, 0]) * \
                                (results[0].boxes.xyxy[:, 3] - results[0].boxes.xyxy[:, 1])
                        best_idx = int(areas.argmax())

                person_kps = kps_data[best_idx].cpu().numpy()

                # confidence 체크
                kpt_list = []
                visible_count = 0
                for i in range(min(6, len(person_kps))):
                    x, y = float(person_kps[i][0]), float(person_kps[i][1])
                    conf = float(person_kps[i][2]) if len(person_kps[i]) > 2 else 0.0

                    if conf >= min_conf and x > 0 and y > 0:
                        kpt_list.append((x / w, y / h, 2))  # normalized + visible
                        visible_count += 1
                    else:
                        kpt_list.append((0.0, 0.0, 0))

                if visible_count < min_kpts:
                    stats["skipped_few_kpts"] += 1
                    frame_idx += 1
                    pbar.update(1)
                    continue

                # bbox 계산 (visible keypoints로부터)
                vis_pts = [(kx * w, ky * h) for kx, ky, kv in kpt_list if kv > 0]
                if len(vis_pts) < 2:
                    frame_idx += 1
                    pbar.update(1)
                    continue

                xs = [p[0] for p in vis_pts]
                ys = [p[1] for p in vis_pts]
                bx_min, bx_max = min(xs), max(xs)
                by_min, by_max = min(ys), max(ys)
                bw = bx_max - bx_min
                bh = by_max - by_min

                # 패딩
                pad = 0.2
                bx_min = max(0, bx_min - bw * pad)
                bx_max = min(w, bx_max + bw * pad)
                by_min = max(0, by_min - bh * (pad + 0.05))
                by_max = min(h, by_max + bh * pad)

                cx = ((bx_min + bx_max) / 2) / w
                cy = ((by_min + by_max) / 2) / h
                nw = (bx_max - bx_min) / w
                nh = (by_max - by_min) / h

                # YOLO 라벨 라인
                parts = [f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"]
                for kx, ky, kv in kpt_list:
                    parts.append(f"{kx:.6f} {ky:.6f} {kv}")
                label_line = " ".join(parts)

                # 이미지 저장
                img_name = f"{video_name}_{frame_idx:06d}.jpg"
                img_path = img_dir / img_name
                cv2.imwrite(str(img_path), frame)

                # 라벨 저장
                lbl_name = f"{video_name}_{frame_idx:06d}.txt"
                lbl_path = lbl_dir / lbl_name
                with open(lbl_path, "w") as f:
                    f.write(label_line + "\n")

                stats["labeled"] += 1
            else:
                stats["skipped_low_conf"] += 1
        else:
            stats["skipped_low_conf"] += 1

        frame_idx += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    return stats


def split_train_val(output_dir, split_ratio=0.9):
    """all/ 폴더를 train/val로 분할"""
    import random

    all_img_dir = Path(output_dir) / "all" / "images"
    all_lbl_dir = Path(output_dir) / "all" / "labels"

    if not all_img_dir.exists():
        print("  ERROR: all/images/ 없음")
        return

    images = sorted(all_img_dir.glob("*.jpg"))
    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)
    train_imgs = images[:split_idx]
    val_imgs = images[split_idx:]

    for split_name, img_list in [("train", train_imgs), ("val", val_imgs)]:
        split_img_dir = Path(output_dir) / split_name / "images"
        split_lbl_dir = Path(output_dir) / split_name / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in img_list:
            stem = img_path.stem
            lbl_path = all_lbl_dir / f"{stem}.txt"

            # symlink
            dst_img = split_img_dir / img_path.name
            dst_lbl = split_lbl_dir / lbl_path.name

            if not dst_img.exists():
                try:
                    os.symlink(os.path.abspath(str(img_path)), str(dst_img))
                except OSError:
                    import shutil
                    shutil.copy2(str(img_path), str(dst_img))

            if lbl_path.exists() and not dst_lbl.exists():
                try:
                    os.symlink(os.path.abspath(str(lbl_path)), str(dst_lbl))
                except OSError:
                    import shutil
                    shutil.copy2(str(lbl_path), str(dst_lbl))

    print(f"  train: {len(train_imgs)} images")
    print(f"  val:   {len(val_imgs)} images")


def visualize_samples(output_dir, num_samples=20):
    """자동 라벨 시각화"""
    img_dir = Path(output_dir) / "all" / "images"
    lbl_dir = Path(output_dir) / "all" / "labels"
    vis_dir = Path(output_dir) / "visualization"
    vis_dir.mkdir(parents=True, exist_ok=True)

    import random
    images = sorted(img_dir.glob("*.jpg"))
    if not images:
        return

    samples = random.sample(images, min(num_samples, len(images)))

    SKELETON = [(0, 2), (2, 4), (1, 3), (3, 5), (0, 1)]
    COLORS_L = (0, 255, 0)
    COLORS_R = (255, 0, 0)
    KPT_COLORS = [COLORS_L, COLORS_R, COLORS_L, COLORS_R, COLORS_L, COLORS_R]

    for img_path in samples:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5 + 6 * 3:
                    continue

                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                x1 = int((cx - bw / 2) * w)
                y1 = int((cy - bh / 2) * h)
                x2 = int((cx + bw / 2) * w)
                y2 = int((cy + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 255), 2)

                kpts = []
                for i in range(6):
                    idx = 5 + i * 3
                    kx = float(parts[idx]) * w
                    ky = float(parts[idx + 1]) * h
                    kv = int(parts[idx + 2])
                    kpts.append((int(kx), int(ky), kv))

                    if kv > 0 and kx > 0 and ky > 0:
                        cv2.circle(img, (int(kx), int(ky)), 6, KPT_COLORS[i], -1)
                        name = LOWER_BODY_NAMES[i].replace("left_", "L").replace("right_", "R")
                        cv2.putText(img, name, (int(kx) + 8, int(ky) - 5),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, KPT_COLORS[i], 1)

                for k1, k2 in SKELETON:
                    if kpts[k1][2] > 0 and kpts[k2][2] > 0:
                        cv2.line(img, (kpts[k1][0], kpts[k1][1]),
                                (kpts[k2][0], kpts[k2][1]), KPT_COLORS[k1], 2)

        cv2.imwrite(str(vis_dir / f"{img_path.stem}_vis.jpg"), img)

    print(f"  시각화: {len(samples)}장 → {vis_dir}")


def main():
    args = parse_args()

    print("=" * 60)
    print("  워커 영상 자동 라벨링 (2차 Fine-Tuning용)")
    print("=" * 60)

    # 모델 로드
    from ultralytics import YOLO
    print(f"\n  모델 로드: {args.model}")
    model = YOLO(args.model, task="pose")

    # 각 영상 처리
    all_stats = {}
    for video_path in args.video:
        stats = auto_label_video(
            video_path, model, args.output_dir,
            frame_interval=args.frame_interval,
            min_conf=args.min_conf,
            min_kpts=args.min_kpts,
            imgsz=args.imgsz,
        )
        all_stats[video_path] = stats

    # 통계 출력
    total_labeled = sum(s.get("labeled", 0) for s in all_stats.values())
    total_extracted = sum(s.get("extracted", 0) for s in all_stats.values())
    print(f"\n  === 자동 라벨링 완료 ===")
    print(f"  총 추출 프레임: {total_extracted}")
    print(f"  라벨 생성:      {total_labeled}")
    print(f"  성공률:         {total_labeled / max(total_extracted, 1) * 100:.1f}%")

    # train/val 분할
    print(f"\n  train/val 분할 (비율: {args.split_ratio})")
    split_train_val(args.output_dir, args.split_ratio)

    # 시각화
    if args.visualize > 0:
        visualize_samples(args.output_dir, args.visualize)

    # 메타데이터 저장
    meta = {
        "source": "walker_auto_label",
        "model_used": args.model,
        "min_conf": args.min_conf,
        "min_kpts": args.min_kpts,
        "frame_interval": args.frame_interval,
        "stats": {k: v for k, v in all_stats.items()},
        "total_labeled": total_labeled,
    }
    meta_path = Path(args.output_dir) / "auto_label_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"\n  다음 단계:")
    print(f"    1. 시각화 확인: {args.output_dir}/visualization/")
    print(f"    2. 2차 Fine-Tuning:")
    print(f"       python training/train_lower_body.py \\")
    print(f"           --model models/yolo26s-lower6.pt \\")
    print(f"           --data training/walker_data.yaml \\")
    print(f"           --name lower_body_v2_walker")


if __name__ == "__main__":
    main()
