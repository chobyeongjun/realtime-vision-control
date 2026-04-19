#!/usr/bin/env python3
"""
RTMPose Wholebody Foot Keypoint 인덱스 검증
============================================
COCO-Wholebody 133 keypoints에서 foot 관련 인덱스가 올바른지 시각적으로 확인합니다.

사용법:
    python3 verify_foot_indices.py                     # 웹캠으로 실시간 확인
    python3 verify_foot_indices.py --image test.jpg    # 이미지 파일로 확인
    python3 verify_foot_indices.py --save              # 결과 이미지 저장

예상 COCO-Wholebody foot keypoint 순서 (index 17-22):
    17: left_big_toe
    18: left_small_toe
    19: left_heel
    20: right_big_toe
    21: right_small_toe
    22: right_heel

현재 FOOT_MAP (pose_models.py):
    17 -> left_toe   (left_big_toe)
    19 -> left_heel
    20 -> right_toe  (right_big_toe)
    22 -> right_heel
"""

import sys
import os
import argparse
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


FOOT_LABELS = {
    17: "17:L_big_toe",
    18: "18:L_small_toe",
    19: "19:L_heel",
    20: "20:R_big_toe",
    21: "21:R_small_toe",
    22: "22:R_heel",
}

FOOT_COLORS = {
    17: (0, 255, 0),    # green - left
    18: (0, 220, 0),    # green variant - left small toe
    19: (0, 150, 0),    # dark green - left heel
    20: (255, 0, 0),    # blue (BGR) - right
    21: (220, 0, 0),    # blue variant - right small toe
    22: (150, 0, 0),    # dark blue - right heel
}

BODY_LABELS = {
    11: "11:L_hip",
    12: "12:R_hip",
    13: "13:L_knee",
    14: "14:R_knee",
    15: "15:L_ankle",
    16: "16:R_ankle",
}


def verify_with_image(image, mode="balanced", backend="onnxruntime", device="cuda", save_path=None):
    """단일 이미지에서 foot keypoint 검증"""
    try:
        from rtmlib import Wholebody
    except ImportError:
        print("rtmlib 미설치. pip install rtmlib onnxruntime-gpu")
        return

    print(f"모델 로드 중... (mode={mode})")
    wb = Wholebody(mode=mode, to_openpose=False, backend=backend, device=device)

    # 워밍업
    dummy = np.zeros((480, 640, 3), dtype=np.uint8)
    wb(dummy)

    keypoints, scores = wb(image)
    if keypoints is None or len(keypoints) == 0:
        print("사람 감지 실패! 전신이 보이는 이미지를 사용하세요.")
        return

    kps = keypoints[0]
    scrs = scores[0]
    num_kps = len(kps)

    print(f"\n총 keypoint 수: {num_kps}")
    print(f"{'Index':>5} {'X':>8} {'Y':>8} {'Score':>7}  Label")
    print("-" * 50)

    # body lower limb (11-16)
    for idx, label in BODY_LABELS.items():
        if idx < num_kps:
            x, y = kps[idx]
            s = scrs[idx]
            print(f"{idx:>5} {x:>8.1f} {y:>8.1f} {s:>7.3f}  {label}")

    print()

    # foot keypoints (17-22)
    for idx, label in FOOT_LABELS.items():
        if idx < num_kps:
            x, y = kps[idx]
            s = scrs[idx]
            print(f"{idx:>5} {x:>8.1f} {y:>8.1f} {s:>7.3f}  {label}")

    # 시각화
    vis = image.copy()

    # body keypoints (ankle 기준)
    for idx, label in BODY_LABELS.items():
        if idx < num_kps:
            x, y = int(kps[idx][0]), int(kps[idx][1])
            if x > 0 or y > 0:
                cv2.circle(vis, (x, y), 6, (255, 255, 0), -1)
                cv2.putText(vis, label, (x + 8, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # foot keypoints (크게 표시)
    for idx, label in FOOT_LABELS.items():
        if idx < num_kps:
            x, y = int(kps[idx][0]), int(kps[idx][1])
            s = float(scrs[idx])
            if x > 0 or y > 0:
                color = FOOT_COLORS[idx]
                cv2.circle(vis, (x, y), 10, color, -1)
                cv2.circle(vis, (x, y), 12, (255, 255, 255), 2)
                cv2.putText(vis, f"{label} ({s:.2f})", (x + 15, y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 범례
    cv2.putText(vis, "GREEN = LEFT foot | BLUE = RIGHT foot",
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(vis, "YELLOW = body (hip/knee/ankle)",
               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    if save_path:
        cv2.imwrite(save_path, vis)
        print(f"\n결과 저장: {save_path}")

    return vis


def main():
    parser = argparse.ArgumentParser(description="RTMPose Wholebody Foot Keypoint 검증")
    parser.add_argument("--image", type=str, help="검증할 이미지 파일 경로")
    parser.add_argument("--mode", default="balanced", choices=["lightweight", "balanced", "performance"])
    parser.add_argument("--backend", default="onnxruntime")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save", action="store_true", help="결과 이미지 저장")
    args = parser.parse_args()

    save_path = None
    if args.save:
        os.makedirs("results", exist_ok=True)
        save_path = "results/foot_keypoint_verification.jpg"

    if args.image:
        image = cv2.imread(args.image)
        if image is None:
            print(f"이미지 로드 실패: {args.image}")
            return
        vis = verify_with_image(image, args.mode, args.backend, args.device, save_path)
        if vis is not None:
            cv2.imshow("Foot Keypoint Verification", vis)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        # 웹캠 실시간
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("웹캠 열기 실패")
            return

        try:
            from rtmlib import Wholebody
        except ImportError:
            print("rtmlib 미설치")
            return

        print(f"모델 로드 중...")
        wb = Wholebody(mode=args.mode, to_openpose=False,
                       backend=args.backend, device=args.device)
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        wb(dummy)
        print("준비 완료. 'q'로 종료, 's'로 현재 프레임 저장")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            keypoints, scores = wb(frame)
            vis = frame.copy()

            if keypoints is not None and len(keypoints) > 0:
                kps = keypoints[0]
                scrs = scores[0]
                num_kps = len(kps)

                for idx, label in BODY_LABELS.items():
                    if idx < num_kps:
                        x, y = int(kps[idx][0]), int(kps[idx][1])
                        if x > 0 or y > 0:
                            cv2.circle(vis, (x, y), 5, (255, 255, 0), -1)
                            cv2.putText(vis, label, (x + 8, y - 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

                for idx, label in FOOT_LABELS.items():
                    if idx < num_kps:
                        x, y = int(kps[idx][0]), int(kps[idx][1])
                        s = float(scrs[idx])
                        if x > 0 or y > 0:
                            color = FOOT_COLORS[idx]
                            cv2.circle(vis, (x, y), 8, color, -1)
                            cv2.circle(vis, (x, y), 10, (255, 255, 255), 2)
                            cv2.putText(vis, f"{label} ({s:.2f})", (x + 12, y + 5),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.putText(vis, "GREEN=LEFT foot | BLUE=RIGHT foot | YELLOW=body",
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

            cv2.imshow("Foot Keypoint Verification", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                save_p = "results/foot_keypoint_verification.jpg"
                os.makedirs("results", exist_ok=True)
                cv2.imwrite(save_p, vis)
                print(f"저장: {save_p}")

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
