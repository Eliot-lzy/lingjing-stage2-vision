from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
COMPARISON_DIR = PROJECT_ROOT / "outputs" / "comparison"
TIMING_DIR = PROJECT_ROOT / "outputs" / "timing"
VLM_DIR = PROJECT_ROOT / "outputs" / "vlm"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "summary"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required JSON not found: {path}")
    return json.loads(path.read_text(encoding="utf-8-sig"))

def collect_rows() -> pd.DataFrame:
    comparison_files = sorted(COMPARISON_DIR.glob("*_comparison.json"))
    if not comparison_files:
        raise FileNotFoundError(
            f"No comparison JSON files found in: {COMPARISON_DIR}"
        )

    rows: list[dict[str, object]] = []

    for comparison_path in comparison_files:
        comparison = load_json(comparison_path)
        stem = comparison_path.name.removesuffix("_comparison.json")

        timing_path = TIMING_DIR / f"{stem}_timing.json"
        vlm_path = VLM_DIR / f"{stem}_vlm.json"

        timing = load_json(timing_path)
        vlm_data = load_json(vlm_path)
        usage = vlm_data.get("usage") or {}
        completion_details = usage.get("completion_tokens_details") or {}

        traditional_seconds = float(
            timing.get("traditional_total_seconds", 0.0)
        )
        vlm_seconds = float(
            comparison.get("vlm_latency_seconds", 0.0)
        )

        rows.append(
            {
                "image": stem,
                "yolo_total": int(comparison.get("yolo_total", 0)),
                "sam_mask_count": int(
                    comparison.get("sam_mask_count", 0)
                ),
                "vlm_declared_total": int(
                    comparison.get("vlm_declared_total", 0)
                ),
                "vlm_item_count_sum": int(
                    comparison.get("vlm_item_count_sum", 0)
                ),
                "vlm_total_consistent": bool(
                    comparison.get("vlm_total_consistent", False)
                ),
                "traditional_seconds": traditional_seconds,
                "vlm_seconds": vlm_seconds,
                "latency_ratio": (
                    round(vlm_seconds / traditional_seconds, 3)
                    if traditional_seconds > 0
                    else None
                ),
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(
                    usage.get("completion_tokens", 0) or 0
                ),
                "reasoning_tokens": int(
                    completion_details.get("reasoning_tokens", 0) or 0
                ),
                "total_tokens": int(
                    usage.get("total_tokens", 0) or 0
                ),
            }
        )

    dataframe = pd.DataFrame(rows)
    dataframe["count_difference"] = (
        dataframe["vlm_item_count_sum"]
        - dataframe["yolo_total"]
    )
    return dataframe


def create_overview_chart(dataframe: pd.DataFrame) -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Arial",
    ]
    plt.rcParams["axes.unicode_minus"] = False

    figure, axes = plt.subplots(
        1,
        3,
        figsize=(19, 5.8),
        facecolor="#F5F7FA",
    )

    figure.suptitle(
        "五张桌面图片的传统视觉方案与多模态大模型对比",
        fontsize=19,
        fontweight="bold",
        color="#172033",
    )

    positions = np.arange(len(dataframe))
    width = 0.36

    axes[0].bar(
        positions - width / 2,
        dataframe["yolo_total"],
        width,
        label="YOLO 检测数量",
        color="#2A6F97",
    )
    axes[0].bar(
        positions + width / 2,
        dataframe["vlm_item_count_sum"],
        width,
        label="大模型物品数量",
        color="#E09F3E",
    )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels(dataframe["image"], rotation=20)
    axes[0].set_ylabel("数量")
    axes[0].set_title("检测物品数量", fontweight="bold")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=0.25)

    for positions_group, values in (
        (positions - width / 2, dataframe["yolo_total"]),
        (positions + width / 2, dataframe["vlm_item_count_sum"]),
    ):
        for position, value in zip(positions_group, values):
            axes[0].text(
                position,
                value + 0.15,
                str(int(value)),
                ha="center",
                fontsize=9,
            )

    axes[1].bar(
        positions - width / 2,
        dataframe["traditional_seconds"],
        width,
        label="YOLO + SAM",
        color="#2A6F97",
    )
    axes[1].bar(
        positions + width / 2,
        dataframe["vlm_seconds"],
        width,
        label="视觉大模型",
        color="#E09F3E",
    )
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(dataframe["image"], rotation=20)
    axes[1].set_ylabel("秒")
    axes[1].set_title("单张图片耗时", fontweight="bold")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    axes[2].bar(
        positions - width / 2,
        dataframe["prompt_tokens"],
        width,
        label="输入 Token",
        color="#5C4D7D",
    )
    axes[2].bar(
        positions + width / 2,
        dataframe["completion_tokens"],
        width,
        label="输出 Token",
        color="#64AAB0",
    )
    axes[2].set_xticks(positions)
    axes[2].set_xticklabels(dataframe["image"], rotation=20)
    axes[2].set_ylabel("Token")
    axes[2].set_title("大模型 Token 用量", fontweight="bold")
    axes[2].legend(frameon=False)
    axes[2].grid(axis="y", alpha=0.25)

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    figure.tight_layout(rect=[0, 0, 1, 0.93])
    figure.savefig(
        OUTPUT_DIR / "summary_overview.png",
        dpi=180,
        bbox_inches="tight",
        facecolor=figure.get_facecolor(),
    )
    plt.close(figure)


def create_markdown_summary(
    dataframe: pd.DataFrame,
    summary: dict,
) -> None:
    table_lines = [
        "| 图片 | YOLO数量 | SAM掩膜 | 大模型数量 | 传统耗时 | 大模型耗时 | Token | 总数一致 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]

    for row in dataframe.itertuples(index=False):
        table_lines.append(
            f"| {row.image} | {row.yolo_total} | {row.sam_mask_count} | "
            f"{row.vlm_item_count_sum} | {row.traditional_seconds:.2f}s | "
            f"{row.vlm_seconds:.2f}s | {row.total_tokens} | "
            f"{'是' if row.vlm_total_consistent else '否'} |"
        )

    markdown = f"""# 多模态视觉对比汇总

## 总体指标

- 图片数量：{summary["image_count"]}
- 传统方案平均耗时：{summary["mean_traditional_seconds"]:.2f} 秒
- 大模型平均耗时：{summary["mean_vlm_seconds"]:.2f} 秒
- 大模型平均慢 {summary["mean_latency_ratio"]:.2f} 倍
- 大模型总 Token：{summary["total_tokens"]}
- 大模型平均 Token：{summary["mean_tokens"]:.0f}
- JSON 总数一致性：{summary["consistent_images"]}/{summary["image_count"]}

## 逐图结果

{chr(10).join(table_lines)}

## 主要发现

1. YOLO 使用固定的 COCO 类别集合，可解释性和位置信息更强，但无法识别类别表之外的物品。
2. 视觉大模型能够描述鼠标垫、桌面支架、插座、装饰物等更丰富的物品，但缺少稳定的边界框。
3. 传统方案平均约 9.4 秒，大模型平均约 39 秒。传统方案更适合实时和批量处理。
4. 免费视觉大模型适合复杂语义描述、物品清点和自然语言问答，但需要注意限流、Token 上限和输出格式约束。
"""

    (OUTPUT_DIR / "summary.md").write_text(
        markdown,
        encoding="utf-8",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataframe = collect_rows()

    dataframe.to_csv(OUTPUT_DIR / "summary.csv", index=False)

    summary = {
        "image_count": len(dataframe),
        "mean_traditional_seconds": float(
            dataframe["traditional_seconds"].mean()
        ),
        "mean_vlm_seconds": float(dataframe["vlm_seconds"].mean()),
        "mean_latency_ratio": float(dataframe["latency_ratio"].mean()),
        "total_tokens": int(dataframe["total_tokens"].sum()),
        "mean_tokens": float(dataframe["total_tokens"].mean()),
        "consistent_images": int(
            dataframe["vlm_total_consistent"].sum()
        ),
    }

    (OUTPUT_DIR / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    create_overview_chart(dataframe)
    create_markdown_summary(dataframe, summary)

    print(dataframe.to_string(index=False))
    print("\nOverall summary:")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()