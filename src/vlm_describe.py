from __future__ import annotations

import argparse
import base64
import json
import os
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv
from json_repair import repair_json
from openai import (
    APIConnectionError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "vlm"

DEFAULT_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
DEFAULT_MODEL = "glm-4.6v-flash"

DEFAULT_PROMPT = """
你是一个桌面物品清点助手。请仔细观察图片，完成以下任务：
1. 识别桌面上的主要物品；
2. 统计每种物品的数量；
3. 描述每个物品在图片中的大致位置；
4. 标出不确定或可能被遮挡的物品。

只输出一个 JSON 对象，不要输出 Markdown 或额外解释。
JSON 格式必须为：
{
  "scene_summary": "一句话描述场景",
  "items": [
    {
      "name": "物品名称",
      "count": 1,
      "location": "画面左上方",
      "evidence": "判断依据"
    }
  ],
  "total_objects": 6,
  "uncertain_items": [],
  "notes": []
}

输出约束：
- items 最多包含 10 个不同物品；
- 每个 evidence 不超过 25 个汉字；
- 每个 location 不超过 15 个汉字；
- 字符串中不要使用未转义的双引号或换行；
- 所有字段必须完整，不得省略逗号、括号或字符串引号。
- total_objects 必须等于 items 中所有 count 字段之和。
""".strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Describe a desk image with a vision language model"
    )
    parser.add_argument("image", type=Path)
    parser.add_argument("--question", default="")
    parser.add_argument("--model", default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-side", type=int, default=1280)
    parser.add_argument("--request-attempts", type=int, default=5)
    return parser.parse_args()


def encode_image(image_path: Path, max_side: int) -> tuple[str, int, int]:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"OpenCV could not read image: {image_path}")

    height, width = image.shape[:2]
    scale = min(1.0, max_side / max(height, width))

    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(width * scale), round(height * scale)),
            interpolation=cv2.INTER_AREA,
        )

    height, width = image.shape[:2]
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 90],
    )

    if not success:
        raise RuntimeError("Failed to encode image as JPEG.")

    image_base64 = base64.b64encode(encoded.tobytes()).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{image_base64}"
    return data_url, width, height


def extract_json(text: str) -> dict:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    if start == -1:
        raise ValueError(
            "VLM response does not contain a JSON object:\n"
            f"{text}"
        )

    end = cleaned.rfind("}")
    candidate = (
        cleaned[start : end + 1]
        if end > start
        else cleaned[start:]
    )

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        repaired = repair_json(candidate)
        parsed = json.loads(repaired)

    if not isinstance(parsed, dict):
        raise ValueError(
            f"Expected a JSON object, received: {type(parsed).__name__}"
        )

    return parsed


def get_message_text(content) -> str:
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts: list[str] = []

        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    texts.append(str(part.get("text", "")))
            elif hasattr(part, "text"):
                texts.append(str(part.text))

        return "\n".join(texts).strip()

    return str(content)


def get_usage_dict(usage) -> dict | None:
    if usage is None:
        return None

    if hasattr(usage, "model_dump"):
        return usage.model_dump()

    if isinstance(usage, dict):
        return usage

    return None


def request_with_backoff(
    client: OpenAI,
    *,
    model_name: str,
    image_data_url: str,
    prompt: str,
    attempts: int,
):
    last_error: Exception | None = None

    for attempt in range(1, attempts + 1):
        try:
            print(f"Request attempt {attempt}/{attempts}...")
            return client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是严谨的图像观察助手。"
                            "只根据图片中清晰可见的内容回答，"
                            "不要编造图片中不存在的物品。"
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": image_data_url,
                                },
                            },
                            {
                                "type": "text",
                                "text": prompt,
                            },
                        ],
                    },
                ],
                temperature=0.1,
                max_tokens=4096,
            )
        except (RateLimitError, APITimeoutError, APIConnectionError) as error:
            last_error = error

            if attempt >= attempts:
                break

            wait_seconds = min(60, 5 * (2 ** (attempt - 1)))
            print(
                f"Temporary API error: {type(error).__name__}. "
                f"Waiting {wait_seconds} seconds..."
            )
            time.sleep(wait_seconds)

    if last_error is not None:
        raise last_error

    raise RuntimeError("API request failed without a captured exception.")


def main() -> None:
    args = parse_args()
    load_dotenv(PROJECT_ROOT / ".env")

    if not args.image.exists():
        raise FileNotFoundError(f"Image not found: {args.image}")

    if args.request_attempts <= 0:
        raise ValueError("--request-attempts must be greater than zero.")

    api_key = os.getenv("ZHIPU_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "ZHIPU_API_KEY is missing. Fill it in the project .env file."
        )

    if not api_key.isascii():
        raise ValueError(
            "ZHIPU_API_KEY contains non-ASCII characters. "
            "Replace the placeholder with the real API key."
        )

    base_url = os.getenv("ZHIPU_BASE_URL", DEFAULT_BASE_URL).strip()
    model_name = (
        args.model
        or os.getenv("ZHIPU_VLM_MODEL", DEFAULT_MODEL).strip()
    )

    print(f"Image    : {args.image}")
    print(f"Base URL : {base_url}")
    print(f"Model    : {model_name}")

    image_data_url, sent_width, sent_height = encode_image(
        args.image,
        args.max_side,
    )

    prompt = DEFAULT_PROMPT
    question = args.question.strip()
    if question:
        prompt += f"\n\n另外请回答这个问题：{question}"

    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=120.0,
        max_retries=0,
    )

    start_time = time.perf_counter()

    response = request_with_backoff(
        client,
        model_name=model_name,
        image_data_url=image_data_url,
        prompt=prompt,
        attempts=args.request_attempts,
    )

    elapsed_seconds = time.perf_counter() - start_time
    raw_text = get_message_text(response.choices[0].message.content)
    finish_reason = response.choices[0].finish_reason

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = args.image.stem
    raw_path = output_dir / f"{stem}_vlm_raw.txt"
    json_path = output_dir / f"{stem}_vlm.json"

    raw_path.write_text(raw_text + "\n", encoding="utf-8")

    if finish_reason == "length":
        print("Warning: model output reached the token limit.")

    try:
        parsed = extract_json(raw_text)
    except Exception as error:
        raise RuntimeError(
            f"Failed to parse VLM JSON. "
            f"Raw response saved to: {raw_path}"
        ) from error

    output = {
        "image": str(args.image),
        "image_width_sent": sent_width,
        "image_height_sent": sent_height,
        "base_url": base_url,
        "model": model_name,
        "question": question,
        "elapsed_seconds": round(elapsed_seconds, 3),
        "finish_reason": finish_reason,
        "usage": get_usage_dict(response.usage),
        "result": parsed,
    }

    json_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(f"Latency  : {elapsed_seconds:.2f} seconds")
    print(f"Finish   : {finish_reason}")
    print(f"Summary  : {parsed.get('scene_summary', '')}")
    print(f"Objects  : {parsed.get('total_objects', 'unknown')}")

    items = parsed.get("items", [])
    for item in items:
        print(
            f"  {item.get('name', 'unknown')}: "
            f"{item.get('count', '?')}，"
            f"{item.get('location', '')}"
        )

    uncertain_items = parsed.get("uncertain_items", [])
    if uncertain_items:
        print(
            "Uncertain:",
            json.dumps(uncertain_items, ensure_ascii=False),
        )

    print(f"\nRaw response: {raw_path}")
    print(f"Parsed JSON : {json_path}")


if __name__ == "__main__":
    main()