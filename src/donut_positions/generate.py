"""Synthetic document pages in the Donut pretraining format: words in reading order, plus quads."""

from __future__ import annotations

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from donut_positions.targets import quad_to_xyxy

WORDS = (
    "Tebu", "Lemon", "receipt", "total", "cash", "change", "tax", "menu",
    "coffee", "tea", "bread", "rice", "chicken", "soup", "salad", "water",
    "invoice", "date", "table", "guest", "qty", "price", "amount", "subtotal",
    "visa", "cash", "discount", "service", "orange", "apple", "mango", "milk",
    "sugar", "large", "small", "hot", "iced", "cheese", "toast", "noodle",
    "please", "thank", "order", "item", "note", "store", "open", "daily",
    "morning", "evening", "special", "combo", "extra", "sauce", "spicy", "mild",
)


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    for path in candidates:
        if Path(path).is_file():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default(size=size)


def _quad(box: tuple[int, int, int, int]) -> dict:
    x1, y1, x2, y2 = box
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y1, "x3": x2, "y3": y2, "x4": x1, "y4": y2}


def render_page(rng: random.Random, min_words: int = 8, max_words: int = 36) -> tuple[Image.Image, list[dict]]:
    width = rng.randint(720, 1100)
    height = rng.randint(960, 1500)
    image = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(image)
    font = _font(rng.randint(22, 36))
    target = rng.randint(min_words, max_words)
    words: list[dict] = []
    y = rng.randint(36, 72)
    margin = 40
    while len(words) < target and y < height - 70:
        x = margin
        for _ in range(rng.randint(2, 6)):
            if len(words) >= target:
                break
            text = rng.choice(WORDS)
            left, top, right, bottom = draw.textbbox((x, y), text, font=font)
            if right > width - margin:
                break
            draw.text((x, y), text, fill=(20, 20, 20), font=font)
            box = (int(left), int(top), int(right), int(bottom))
            words.append({"text": text, "quad": _quad(box)})
            x = right + rng.randint(14, 32)
        y += rng.randint(34, 52)
    if not words:
        raise RuntimeError("page renderer produced no words")
    return image, words


def generate_split(folder: Path, count: int, seed: int) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        json_path = folder / f"{index:06d}.json"
        image_path = folder / f"{index:06d}.png"
        if json_path.is_file() and image_path.is_file():
            continue
        image, words = render_page(random.Random(seed + index))
        image.save(image_path)
        payload = {"width": image.width, "height": image.height, "words": words}
        json_path.write_text(json.dumps(payload))
        # Touch the quad helper so a bad quad fails at generation time.
        quad_to_xyxy(words[0]["quad"])


def ensure_synth(root: Path, train_samples: int, val_samples: int, seed: int) -> None:
    generate_split(root / "train", train_samples, seed)
    generate_split(root / "val", val_samples, seed + 100_000)


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write synthetic document pages")
    parser.add_argument("--out", default="data/synthdog")
    parser.add_argument("--train-samples", type=int, default=8000)
    parser.add_argument("--val-samples", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    ensure_synth(Path(args.out), args.train_samples, args.val_samples, args.seed)
    print(f"wrote pages under {args.out}")


if __name__ == "__main__":
    main()
