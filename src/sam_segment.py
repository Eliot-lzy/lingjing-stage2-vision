from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import SAM


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "sam"

COLORS = [
    (46, 111, 168),
    (224, 159, 62),
    (84, 130, 53),
    (192, 80, 77),
    (112, 48, 160),
    (64, 170, 176),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Segment YOLO detections using SAM box prompts"
    )
    parser.add_argument("image", type=Path)
    parser.add_argument(
        "--detections",
        type=Path,
        required=True,
        help="YOLO JSON file containing detections",
    )
    parser.add_argument("--model", default="sam2.1_t.pt")
    parser.add_argument("--min-confidence", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")
    if not args.detections.exists():
        raise FileNotFoundError(f"Detection JSON not found: {args.detections}")

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {args.image}")

    detection_data = json.loads(args.detections.read_text(encoding="utf-8"))
    detections = [
        detection
        for detection in detection_data["detections"]
        if float(detection["confidence"]) >= args.min_confidence
    ]

    if not detections:
        raise ValueError("No detections passed the confidence threshold.")

    bboxes = [detection["bbox_xyxy"] for detection in detections]
    device = "0" if torch.cuda.is_available() else "cpu"

    print(f"Image      : {args.image}")
    print(f"Prompts    : {len(bboxes)}")
    print(f"Device     : {device}")
    print(f"SAM model  : {args.model}")

    model = SAM(args.model)
    results = model.predict(
        source=str(args.image),
        bboxes=bboxes,
        device=device,
        verbose=False,
    )
    result = results[0]

    if result.masks is None or result.masks.data is None:
        raise RuntimeError("SAM did not return any masks.")

    masks = result.masks.data.cpu().numpy()
    if masks.ndim == 2:
        masks = masks[None, ...]

    mask_scores = (
        result.boxes.conf.cpu().tolist()
        if result.boxes is not None and result.boxes.conf is not None
        else []
    )

    output_dir = args.output_dir
    mask_dir = output_dir / "masks"
    mask_dir.mkdir(parents=True, exist_ok=True)

    overlay = image.copy()
    records: list[dict[str, object]] = []

    for index, detection in enumerate(detections):
        if index >= len(masks):
            break

        mask = masks[index].astype(np.float32)
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(
                mask,
                (image.shape[1], image.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        binary_mask = (mask > 0.5).astype(np.uint8) * 255
        color = COLORS[index % len(COLORS)]
        colored_region = np.full_like(image, color)
        region = binary_mask > 0

        overlay[region] = cv2.addWeighted(
            overlay[region],
            0.55,
            colored_region[region],
            0.45,
            0,
        )

        contours, _ = cv2.findContours(
            binary_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        cv2.drawContours(overlay, contours, -1, color, 2)

        bbox = [int(round(value)) for value in detection["bbox_xyxy"]]
        x1, y1, _, _ = bbox
        class_name = str(detection["class_name"])
        yolo_confidence = float(detection["confidence"])
        mask_score = (
            float(mask_scores[index])
            if index < len(mask_scores)
            else None
        )

        label = f"{class_name} Y:{yolo_confidence:.2f}"
        if mask_score is not None:
            label += f" S:{mask_score:.2f}"

        text_x = max(0, x1)
        text_y = max(24, y1 - 8)
        cv2.putText(
            overlay,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            4,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        mask_path = mask_dir / f"{args.image.stem}_{index:02d}_{class_name}.png"
        cv2.imwrite(str(mask_path), binary_mask)

        mask_area = int(np.count_nonzero(binary_mask))
        records.append(
            {
                "index": index,
                "class_name": class_name,
                "yolo_confidence": round(yolo_confidence, 4),
                "sam_mask_score": (
                    round(mask_score, 4)
                    if mask_score is not None
                    else None
                ),
                "bbox_xyxy": detection["bbox_xyxy"],
                "mask_area_pixels": mask_area,
                "mask_ratio": round(mask_area / binary_mask.size, 6),
                "mask_file": str(mask_path.relative_to(PROJECT_ROOT)),
            }
        )

    image_output = output_dir / f"{args.image.stem}_sam.jpg"
    json_output = output_dir / f"{args.image.stem}_sam.json"

    if not cv2.imwrite(str(image_output), overlay):
        raise RuntimeError(f"Failed to save image: {image_output}")

    output_json = {
        "image": str(args.image),
        "detection_json": str(args.detections),
        "sam_model": args.model,
        "device": device,
        "min_confidence": args.min_confidence,
        "object_count": len(records),
        "objects": records,
    }
    json_output.write_text(
        json.dumps(output_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nSegmented objects: {len(records)}")
    for item in records:
        print(
            f"  {item['class_name']}: "
            f"YOLO={item['yolo_confidence']:.3f}, "
            f"SAM={item['sam_mask_score']}, "
            f"area={item['mask_area_pixels']}"
        )

    print(f"\nAnnotated image: {image_output}")
    print(f"Segmentation JSON: {json_output}")
    print(f"Individual masks : {mask_dir}")


if __name__ == "__main__":
    main()