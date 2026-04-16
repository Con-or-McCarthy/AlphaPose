#!/usr/bin/env python3
"""Export AlphaPose joints and build approximate person masks.

This script reads AlphaPose JSON output (default COCO-like format) and writes:
1) Per-frame joint tables (JSON + CSV).
2) Binary masks per person/frame and an optional merged frame mask.

Mask generation is pose-derived (skeleton tubes + joint disks + optional convex hull),
so it is an approximation rather than true semantic segmentation.
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


COCO17_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]

COCO17_LIMBS = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export AlphaPose joints and pose-derived masks.")
    parser.add_argument("--input_json", required=True, help="Path to AlphaPose result json.")
    parser.add_argument(
        "--image_dir",
        default="",
        help="Directory containing source frames. Used to infer frame size for masks.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for tables and masks.",
    )
    parser.add_argument(
        "--min_conf",
        type=float,
        default=0.2,
        help="Minimum keypoint confidence to include in tables/masks.",
    )
    parser.add_argument(
        "--line_thickness",
        type=int,
        default=14,
        help="Skeleton limb thickness for mask rendering.",
    )
    parser.add_argument(
        "--joint_radius",
        type=int,
        default=8,
        help="Joint disk radius for mask rendering.",
    )
    parser.add_argument(
        "--no_hull",
        action="store_true",
        help="Disable convex hull fill over confident joints.",
    )
    return parser.parse_args()


def load_results(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Expected a list of detections in input json.")
    return data


def group_by_image(detections: Sequence[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for det in detections:
        image_id = str(det.get("image_id", ""))
        if not image_id:
            continue
        grouped[image_id].append(det)
    return grouped


def decode_keypoints(raw_keypoints: Sequence[float]) -> np.ndarray:
    arr = np.array(raw_keypoints, dtype=np.float32)
    if arr.size % 3 != 0:
        raise ValueError("keypoints length must be divisible by 3.")
    return arr.reshape(-1, 3)


def joint_name(idx: int, num_joints: int) -> str:
    if num_joints == 17 and idx < len(COCO17_NAMES):
        return COCO17_NAMES[idx]
    return f"joint_{idx:02d}"


def infer_limb_pairs(num_joints: int) -> List[Tuple[int, int]]:
    if num_joints == 17:
        return COCO17_LIMBS
    return []


def ensure_frame_shape(image_dir: Path, image_name: str, detections: Sequence[dict]) -> Tuple[int, int]:
    if image_dir:
        image_path = image_dir / image_name
        if image_path.exists():
            img = cv2.imread(str(image_path))
            if img is not None:
                h, w = img.shape[:2]
                return h, w

    max_x = 0.0
    max_y = 0.0
    for det in detections:
        kp = decode_keypoints(det.get("keypoints", []))
        if kp.size == 0:
            continue
        max_x = max(max_x, float(np.max(kp[:, 0])))
        max_y = max(max_y, float(np.max(kp[:, 1])))

    width = max(1, int(np.ceil(max_x)) + 2)
    height = max(1, int(np.ceil(max_y)) + 2)
    return height, width


def render_person_mask(
    kpts: np.ndarray,
    limbs: Sequence[Tuple[int, int]],
    shape_hw: Tuple[int, int],
    min_conf: float,
    line_thickness: int,
    joint_radius: int,
    use_hull: bool,
) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    valid_points: List[Tuple[int, int]] = []

    for i, (x, y, c) in enumerate(kpts):
        if c < min_conf:
            continue
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        if xi < 0 or yi < 0 or xi >= w or yi >= h:
            continue
        valid_points.append((xi, yi))
        cv2.circle(mask, (xi, yi), joint_radius, 255, thickness=-1)

    for a, b in limbs:
        if a >= len(kpts) or b >= len(kpts):
            continue
        xa, ya, ca = kpts[a]
        xb, yb, cb = kpts[b]
        if ca < min_conf or cb < min_conf:
            continue
        p1 = (int(round(float(xa))), int(round(float(ya))))
        p2 = (int(round(float(xb))), int(round(float(yb))))
        cv2.line(mask, p1, p2, 255, thickness=line_thickness)

    if use_hull and len(valid_points) >= 3:
        pts = np.array(valid_points, dtype=np.int32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        cv2.fillConvexPoly(mask, hull, 255)

    return mask


def write_frame_table_json(frame_path: Path, rows: List[dict]) -> None:
    payload = {"num_rows": len(rows), "joints": rows}
    with frame_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_frame_table_csv(frame_path: Path, rows: List[dict]) -> None:
    fieldnames = ["image_id", "person_id", "joint_index", "joint_name", "x", "y", "confidence"]
    with frame_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def sanitize_stem(image_id: str) -> str:
    return Path(image_id).stem


def main() -> None:
    args = parse_args()
    input_json = Path(args.input_json)
    image_dir = Path(args.image_dir) if args.image_dir else Path("")
    out_root = Path(args.output_dir)
    out_tables = out_root / "joint_tables"
    out_masks_people = out_root / "masks" / "per_person"
    out_masks_frame = out_root / "masks" / "per_frame"
    out_tables.mkdir(parents=True, exist_ok=True)
    out_masks_people.mkdir(parents=True, exist_ok=True)
    out_masks_frame.mkdir(parents=True, exist_ok=True)

    detections = load_results(input_json)
    grouped = group_by_image(detections)
    if not grouped:
        print("No valid detections found.")
        return

    total_people = 0
    total_frames = 0

    for image_id in sorted(grouped.keys()):
        frame_dets = grouped[image_id]
        if not frame_dets:
            continue
        total_frames += 1

        frame_shape = ensure_frame_shape(image_dir, image_id, frame_dets)
        frame_mask = np.zeros(frame_shape, dtype=np.uint8)
        frame_rows: List[dict] = []

        for local_person_idx, det in enumerate(frame_dets):
            key_raw = det.get("keypoints", [])
            kpts = decode_keypoints(key_raw)
            num_joints = int(kpts.shape[0])
            limbs = infer_limb_pairs(num_joints)

            person_id = det.get("idx", local_person_idx)
            person_mask = render_person_mask(
                kpts=kpts,
                limbs=limbs,
                shape_hw=frame_shape,
                min_conf=args.min_conf,
                line_thickness=args.line_thickness,
                joint_radius=args.joint_radius,
                use_hull=not args.no_hull,
            )
            frame_mask = np.maximum(frame_mask, person_mask)

            stem = sanitize_stem(image_id)
            person_mask_path = out_masks_people / f"{stem}_person-{person_id}.png"
            cv2.imwrite(str(person_mask_path), person_mask)

            for j, (x, y, conf) in enumerate(kpts):
                frame_rows.append(
                    {
                        "image_id": image_id,
                        "person_id": person_id,
                        "joint_index": j,
                        "joint_name": joint_name(j, num_joints),
                        "x": float(x),
                        "y": float(y),
                        "confidence": float(conf),
                    }
                )
            total_people += 1

        stem = sanitize_stem(image_id)
        write_frame_table_json(out_tables / f"{stem}.json", frame_rows)
        write_frame_table_csv(out_tables / f"{stem}.csv", frame_rows)
        cv2.imwrite(str(out_masks_frame / f"{stem}.png"), frame_mask)

    print(f"Done. Frames: {total_frames}, persons: {total_people}")
    print(f"Joint tables: {out_tables}")
    print(f"Person masks: {out_masks_people}")
    print(f"Frame masks: {out_masks_frame}")


if __name__ == "__main__":
    main()
