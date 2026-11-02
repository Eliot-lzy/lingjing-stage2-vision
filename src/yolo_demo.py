from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "yolo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO on one desk image")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {args.image}")

    height, width = image.shape[:2]
    device = "0" if torch.cuda.is_available() else "cpu"

    print(f"Image : {args.image}")
    print(f"Size  : {width} x {height}")
    print(f"Device: {device}")

    model = YOLO(args.model)
    results = model.predict(
        source=str(args.image),
        conf=args.confidence,
        device=device,
        verbose=False,
    )
    result = results[0]

    annotated_image = result.plot()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    image_output = output_dir / f"{args.image.stem}_yolo.jpg"
    json_output = output_dir / f"{args.image.stem}_yolo.json"

    if not cv2.imwrite(str(image_output), annotated_image):
        raise RuntimeError(f"Failed to save image: {image_output}")

    detections: list[dict[str, object]] = []

    for box in result.boxes:
        class_id = int(box.cls[0].item())
        class_name = result.names[class_id]
        confidence = float(box.conf[0].item())
        coordinates = [round(float(value), 2) for value in box.xyxy[0].tolist()]

        detections.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox_xyxy": coordinates,
            }
        )

    counts = Counter(str(item["class_name"]) for item in detections)

    result_json = {
        "image": str(args.image),
        "image_width": width,
        "image_height": height,
        "model": args.model,
        "device": device,
        "confidence_threshold": args.confidence,
        "counts": dict(counts),
        "detections": detections,
    }

    json_output.write_text(
        json.dumps(result_json, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"\nDetected objects: {len(detections)}")

    for object_name, count in counts.items():
        print(f"  {object_name}: {count}")

    print(f"\nAnnotated image: {image_output}")
    print(f"Detection JSON : {json_output}")


if __name__ == "__main__":
    main()