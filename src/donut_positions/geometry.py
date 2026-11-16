"""Map word boxes onto the canvas the Donut encoder actually sees.

Donut aspect-fits an image into a fixed canvas and then pads. Position bins are
computed after that same resize and pad, including the random training pad, so a
token refers to a region of the pixels the model receives.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

POS_BINS = 512


@dataclass(frozen=True)
class Canvas:
    height: int
    width: int
    scale: float
    pad_left: int
    pad_top: int
    resized_width: int
    resized_height: int


def fit_canvas(
    src_width: int,
    src_height: int,
    canvas_height: int,
    canvas_width: int,
    random_pad: bool,
    rng: random.Random,
) -> Canvas:
    """Aspect-fit ``src`` into the canvas and record the pad offset."""
    if src_width <= 0 or src_height <= 0:
        raise ValueError("source image size must be positive")
    scale = min(canvas_width / src_width, canvas_height / src_height)
    resized_width = max(1, min(canvas_width, int(round(src_width * scale))))
    resized_height = max(1, min(canvas_height, int(round(src_height * scale))))
    pad_w = canvas_width - resized_width
    pad_h = canvas_height - resized_height
    if random_pad:
        pad_left = rng.randrange(pad_w + 1) if pad_w > 0 else 0
        pad_top = rng.randrange(pad_h + 1) if pad_h > 0 else 0
    else:
        pad_left = pad_w // 2
        pad_top = pad_h // 2
    return Canvas(
        height=canvas_height,
        width=canvas_width,
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
        resized_width=resized_width,
        resized_height=resized_height,
    )


def map_xyxy(box: list[float] | tuple[float, ...], canvas: Canvas) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        x1 * canvas.scale + canvas.pad_left,
        y1 * canvas.scale + canvas.pad_top,
        x2 * canvas.scale + canvas.pad_left,
        y2 * canvas.scale + canvas.pad_top,
    )


def bin_coord(value: float, size: int, bins: int = POS_BINS) -> int:
    if size <= 0:
        return 0
    return int(max(0, min(bins - 1, (value / size) * bins)))


def bin_box(box: list[float] | tuple[float, ...], canvas: Canvas, bins: int = POS_BINS) -> list[int]:
    x1, y1, x2, y2 = map_xyxy(box, canvas)
    return [
        bin_coord(x1, canvas.width, bins),
        bin_coord(y1, canvas.height, bins),
        bin_coord(x2, canvas.width, bins),
        bin_coord(y2, canvas.height, bins),
    ]


def scale_box(
    box: list[float] | tuple[float, ...],
    src_width: int,
    src_height: int,
    dst_width: int,
    dst_height: int,
) -> list[float]:
    """Map a box from annotation pixel space into the loaded image's pixels."""
    x1, y1, x2, y2 = box
    sx = dst_width / max(src_width, 1)
    sy = dst_height / max(src_height, 1)
    return [x1 * sx, y1 * sy, x2 * sx, y2 * sy]


def canvas_tensor(image, canvas: Canvas):
    """Resize, pad, and normalize an image the way Donut does (mean/std 0.5)."""
    import numpy as np
    import torch
    from PIL import Image

    image = image.convert("RGB")
    resized = image.resize((canvas.resized_width, canvas.resized_height), Image.BILINEAR)
    page = Image.new("RGB", (canvas.width, canvas.height), (255, 255, 255))
    page.paste(resized, (canvas.pad_left, canvas.pad_top))
    array = np.asarray(page, dtype=np.float32) / 255.0
    array = (array - 0.5) / 0.5
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()
