from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a side-by-side comparison for one desk image"
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--sam-image", type=Path, required=True)
    parser.add_argument("--yolo-json", type=Path, required=True)
    parser.add_argument("--sam-json", type=Path, required=True)
    parser.add_argument("--vlm-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_rgb(image_path: Path):
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_vlm_text(vlm_data: dict) -> str:
    result = vlm_data.get("result", {})
    lines = [
        f"Model: {vlm_data.get('model', 'unknown')}",
        "",
        f"Scene: {result.get('scene_summary', '')}",
        "",
    ]

    for index, item in enumerate(result.get("items", []), start=1):
        name = item.get("name", "unknown")
        count = item.get("count", "?")
        location = item.get("location", "")
        lines.append(f"{index}. {name} x{count}")

        if location:
            lines.append(f"   {location}")

    declared_total = result.get("total_objects", "unknown")
    lines.extend(
        [
            "",
            f"Declared total: {declared_total}",
            f"Item count sum: {sum(int(item.get('count', 0)) for item in result.get('items', []))}",
            f"Latency: {vlm_data.get('elapsed_seconds', 'unknown')} s",
        ]
    )

    return "\n".join(lines)


def build_summary(
    yolo_counts: dict[str, int],
    sam_count: int,
    vlm_result: dict,
) -> str:
    items = vlm_result.get("items", [])
    vlm_count_sum = sum(int(item.get("count", 0)) for item in items)
    declared_total = int(vlm_result.get("total_objects", 0))
    yolo_total = sum(yolo_counts.values())

    findings = [
        f"YOLO + SAM: {yolo_total} detections, {sam_count} masks",
        f"VLM declared total: {declared_total}",
        f"VLM item count sum: {vlm_count_sum}",
        "",
        "Key observations:",
    ]

    if declared_total != vlm_count_sum:
        findings.append(
            "- VLM total_objects does not equal the sum of item counts."
        )

    if "tv" in yolo_counts and any(
        item.get("name") == "显示器" for item in items
    ):
        findings.append(
            "- YOLO uses COCO label 'tv'; VLM calls the same object '显示器'."
        )

    if "cup" in yolo_counts and any(
        item.get("name") in {"水瓶", "杯子", "水杯"} for item in items
    ):
        findings.append(
            "- YOLO uses 'cup'; VLM uses a more specific bottle/cup description."
        )

    yolo_names = set(yolo_counts)
    semantic_aliases = {
        "tv": {"显示器", "电视", "屏幕"},
        "cup": {"水瓶", "杯子", "水杯"},
        "mouse": {"鼠标", "鼠标（联想）", "鼠标（黑色）"},
        "laptop": {"笔记本电脑", "电脑"},
        "keyboard": {"键盘"},
    }

    vlm_names = {str(item.get("name", "")) for item in items}
    possible_extra: list[str] = []

    for name in vlm_names:
        matched = any(
            name in aliases or any(alias in name for alias in aliases)
            for yolo_name, aliases in semantic_aliases.items()
            if yolo_name in yolo_names
        )
        if not matched:
            possible_extra.append(name)

    if possible_extra:
        findings.append(
            "- VLM additionally mentions: " + "、".join(sorted(possible_extra))
        )

    return "\n".join(findings)


def main() -> None:
    args = parse_args()

    for path in (
        args.image,
        args.sam_image,
        args.yolo_json,
        args.sam_json,
        args.vlm_json,
    ):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    original = read_rgb(args.image)
    sam_image = read_rgb(args.sam_image)

    yolo_data = load_json(args.yolo_json)
    sam_data = load_json(args.sam_json)
    vlm_data = load_json(args.vlm_json)

    yolo_detections = yolo_data.get("detections", [])
    yolo_counts = dict(
        Counter(str(item["class_name"]) for item in yolo_detections)
    )
    sam_objects = sam_data.get("objects", [])
    vlm_result = vlm_data.get("result", {})

    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    figure = plt.figure(figsize=(18, 11), facecolor="#F4F7FA")
    grid = figure.add_gridspec(
        2,
        3,
        height_ratios=[3.8, 1.35],
        hspace=0.12,
        wspace=0.08,
    )

    figure.suptitle(
        "同一张图片的传统视觉方案与多模态大模型方案对比",
        fontsize=20,
        fontweight="bold",
        color="#172033",
    )

    original_axis = figure.add_subplot(grid[0, 0])
    original_axis.imshow(original)
    original_axis.set_title("输入原图", fontsize=14, fontweight="bold")
    original_axis.axis("off")

    sam_axis = figure.add_subplot(grid[0, 1])
    sam_axis.imshow(sam_image)
    sam_axis.set_title(
        "YOLO 检测 + SAM 分割",
        fontsize=14,
        fontweight="bold",
    )
    sam_axis.axis("off")

    vlm_axis = figure.add_subplot(grid[0, 2])
    vlm_axis.set_facecolor("white")
    vlm_axis.set_title(
        "视觉语言模型物品清单",
        fontsize=14,
        fontweight="bold",
    )
    vlm_axis.text(
        0.03,
        0.97,
        build_vlm_text(vlm_data),
        transform=vlm_axis.transAxes,
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.45,
        color="#172033",
        wrap=True,
    )
    vlm_axis.set_xlim(0, 1)
    vlm_axis.set_ylim(0, 1)
    vlm_axis.set_xticks([])
    vlm_axis.set_yticks([])

    summary_axis = figure.add_subplot(grid[1, :])
    summary_axis.set_facecolor("white")
    summary_axis.axis("off")
    summary_axis.text(
        0.02,
        0.93,
        "结果对比",
        transform=summary_axis.transAxes,
        fontsize=14,
        fontweight="bold",
        va="top",
        color="#172033",
    )
    summary_axis.text(
        0.02,
        0.75,
        build_summary(yolo_counts, len(sam_objects), vlm_result),
        transform=summary_axis.transAxes,
        fontsize=11,
        va="top",
        linespacing=1.55,
        color="#334155",
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.image.stem
    image_output = output_dir / f"{stem}_comparison.png"
    json_output = output_dir / f"{stem}_comparison.json"

    figure.savefig(
        image_output,
        dpi=180,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)

    vlm_count_sum = sum(
        int(item.get("count", 0))
        for item in vlm_result.get("items", [])
    )
    declared_total = int(vlm_result.get("total_objects", 0))

    comparison = {
        "image": str(args.image),
        "yolo_counts": yolo_counts,
        "yolo_total": sum(yolo_counts.values()),
        "sam_mask_count": len(sam_objects),
        "vlm_declared_total": declared_total,
        "vlm_item_count_sum": vlm_count_sum,
        "vlm_total_consistent": declared_total == vlm_count_sum,
        "vlm_latency_seconds": vlm_data.get("elapsed_seconds"),
        "vlm_model": vlm_data.get("model"),
        "vlm_items": vlm_result.get("items", []),
        "comparison_image": str(image_output),
    }

    json_output.write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"YOLO objects : {sum(yolo_counts.values())}")
    print(f"SAM masks    : {len(sam_objects)}")
    print(f"VLM declared : {declared_total}")
    print(f"VLM item sum : {vlm_count_sum}")
    print(f"Consistent   : {declared_total == vlm_count_sum}")
    print(f"Comparison image: {image_output}")
    print(f"Comparison JSON : {json_output}")


if __name__ == "__main__":
    main()