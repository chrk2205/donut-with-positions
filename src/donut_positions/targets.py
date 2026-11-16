"""Decoder targets: word-level boxes for pretraining, field boxes before each closing key."""

from __future__ import annotations

import re
from collections import defaultdict
from typing import Any, Iterable

SYNTH_TASK = "<s_synthdog>"
CORD_TASK = "<s_cord-v2>"

_POS_BOX = re.compile(r"<pos_(\d+)><pos_(\d+)><pos_(\d+)><pos_(\d+)>")
_POS_TOKEN = re.compile(r"<pos_\d+>")


def box_to_tokens(bins: Iterable[int]) -> str:
    return "".join(f"<pos_{int(value)}>" for value in bins)


def quad_to_xyxy(quad: dict) -> list[float]:
    xs = [float(quad[f"x{i}"]) for i in range(1, 5)]
    ys = [float(quad[f"y{i}"]) for i in range(1, 5)]
    return [min(xs), min(ys), max(xs), max(ys)]


def is_box(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 4
        and all(isinstance(item, (int, float)) and not isinstance(item, bool) for item in value)
    )


def words_to_target(
    words: list[dict],
    task_token: str = SYNTH_TASK,
    eos_token: str = "</s>",
) -> str:
    """`<s_synthdog>word<pos_x1><pos_y1><pos_x2><pos_y2> next<pos_...></s>`."""
    parts = [task_token]
    for index, word in enumerate(words):
        if index:
            parts.append(" ")
        parts.append(str(word["text"]))
        parts.append(box_to_tokens(word["bins"]))
    parts.append(eos_token)
    return "".join(parts)


def parse_category(category: str) -> list[str]:
    """Split a CORD category into the gt_parse path.

    Hugging Face rows use ``menu.sub.nm``. The official json zip uses
    ``menu.sub_nm``. Both become ``["menu", "sub", "nm"]``.
    """
    parts = category.split(".")
    if len(parts) == 2 and parts[0] == "menu" and parts[1].startswith("sub_"):
        return ["menu", "sub", parts[1][4:]]
    return parts


def _line_box(line: dict) -> list[float]:
    boxes = [quad_to_xyxy(word["quad"]) for word in (line.get("words") or []) if word.get("quad")]
    if not boxes:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def _add_field(container: dict, field: str, box: list[float]) -> None:
    if field not in container:
        container[field] = box
        return
    previous = container[field]
    if is_box(previous):
        container[field] = [previous, box]
    else:
        previous.append(box)


def _consume_group(lines: list[dict]) -> dict:
    item: dict[str, Any] = {}
    subs: list[dict] = []
    current: dict | None = None
    for line in lines:
        path = parse_category(line.get("category") or "")
        rest = path[1:]
        box = _line_box(line)
        if rest[:1] == ["sub"]:
            field = rest[1] if len(rest) > 1 else "nm"
            if current is None or field in current:
                if current:
                    subs.append(current)
                current = {}
            _add_field(current, field, box)
        else:
            field = rest[0] if rest else "value"
            _add_field(item, field, box)
    if current:
        subs.append(current)
    if len(subs) == 1:
        item["sub"] = subs[0]
    elif len(subs) > 1:
        item["sub"] = subs
    return item


def structure_from_valid_line(valid_line: list[dict]) -> dict:
    """Box tree shaped like CORD ``gt_parse``. Leaves are ``[xmin, ymin, xmax, ymax]``."""
    groups: dict[int, list[dict]] = defaultdict(list)
    order: list[int] = []
    for line in valid_line:
        category = line.get("category") or ""
        if "." not in category:
            continue
        group_id = int(line.get("group_id", 0))
        if group_id not in groups:
            order.append(group_id)
        groups[group_id].append(line)

    section_items: dict[str, list] = defaultdict(list)
    for group_id in order:
        by_section: dict[str, list[dict]] = defaultdict(list)
        section_order: list[str] = []
        for line in groups[group_id]:
            section = parse_category(line["category"])[0]
            if section not in by_section:
                section_order.append(section)
            by_section[section].append(line)
        for section in section_order:
            section_items[section].append(_consume_group(by_section[section]))

    tree: dict[str, Any] = {}
    for section, items in section_items.items():
        tree[section] = items if len(items) > 1 else items[0]
    return tree


def align_boxes(gt: Any, built: Any) -> Any:
    """Reindex a box tree so it follows ``gt`` even if a single object was wrapped."""
    if isinstance(gt, dict):
        source = built if isinstance(built, dict) else {}
        return {key: align_boxes(value, source.get(key)) for key, value in gt.items()}
    if isinstance(gt, list):
        if isinstance(built, list):
            source_list = built
        elif isinstance(built, dict):
            source_list = [built]
        else:
            source_list = []
        return [
            align_boxes(item, source_list[index] if index < len(source_list) else None)
            for index, item in enumerate(gt)
        ]
    if is_box(built):
        return [float(value) for value in built]
    return None


def map_boxes(tree: Any, fn) -> Any:
    if is_box(tree):
        return fn(list(tree))
    if isinstance(tree, dict):
        return {key: map_boxes(value, fn) for key, value in tree.items()}
    if isinstance(tree, list):
        return [map_boxes(value, fn) for value in tree]
    return tree


def json2token(obj: Any, boxes: Any = None, keys: set[str] | None = None) -> str:
    """Donut JSON encoding. A string leaf is followed by its box, then the key closes."""
    if isinstance(obj, dict):
        parts = []
        source = boxes if isinstance(boxes, dict) else {}
        for key in sorted(obj.keys(), reverse=True):
            if keys is not None:
                keys.add(f"<s_{key}>")
                keys.add(f"</s_{key}>")
            inner = json2token(obj[key], source.get(key), keys)
            parts.append(f"<s_{key}>{inner}</s_{key}>")
        return "".join(parts)
    if isinstance(obj, list):
        source_list = boxes if isinstance(boxes, list) else []
        parts = []
        for index, item in enumerate(obj):
            child = source_list[index] if index < len(source_list) else None
            parts.append(json2token(item, child, keys))
        return "<sep/>".join(parts)
    text = str(obj)
    if is_box(boxes):
        return text + box_to_tokens(boxes)
    return text


def cord_target(
    gt_parse: dict,
    box_tree: Any,
    task_token: str = CORD_TASK,
    eos_token: str = "</s>",
    keys: set[str] | None = None,
) -> str:
    aligned = align_boxes(gt_parse, box_tree)
    return task_token + json2token(gt_parse, aligned, keys) + eos_token


def collect_special_tokens(obj: Any) -> set[str]:
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                found.add(f"<s_{key}>")
                found.add(f"</s_{key}>")
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(obj)
    return found


def leaf_texts(obj: Any) -> frozenset[str]:
    texts: list[str] = []
    if isinstance(obj, dict):
        for value in obj.values():
            texts.extend(leaf_texts(value))
    elif isinstance(obj, list):
        for value in obj:
            texts.extend(leaf_texts(value))
    else:
        text = str(obj).strip()
        if text:
            texts.append(text)
    return frozenset(texts)


def line_texts(valid_line: list[dict]) -> frozenset[str]:
    texts = []
    for line in valid_line:
        text = " ".join(word.get("text", "") for word in (line.get("words") or [])).strip()
        if text:
            texts.append(text)
    return frozenset(texts)


def strip_pos_tokens(text: str) -> str:
    return _POS_TOKEN.sub("", text)


def extract_pos_boxes(text: str) -> list[list[int]]:
    return [[int(value) for value in match] for match in _POS_BOX.findall(text)]


def boxes_from_ids(token_ids: list[int], pos_bins: dict[int, int]) -> list[list[int]]:
    """Read boxes from four position-token ids in a row. Other tokens reset the group."""
    boxes: list[list[int]] = []
    current: list[int] = []
    for token_id in token_ids:
        bin_index = pos_bins.get(token_id)
        if bin_index is None:
            current = []
            continue
        current.append(bin_index)
        if len(current) == 4:
            boxes.append(current)
            current = []
    return boxes


def box_iou(left: Iterable[float], right: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = left
    bx1, by1, bx2, by2 = right
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_left = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_right = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_left + area_right - intersection
    if union <= 0:
        return 0.0
    return float(intersection / union)


def aligned_mean_iou(predicted: list[list[int]], gold: list[list[int]]) -> float:
    denominator = max(len(predicted), len(gold), 1)
    total = 0.0
    for pred_box, gold_box in zip(predicted, gold):
        total += box_iou(pred_box, gold_box)
    return total / denominator


def sequence_accuracy(predicted: list[int], gold: list[int]) -> float:
    """1 - normalized edit distance, the sequence score Donut reports."""
    from nltk.metrics.distance import edit_distance

    denominator = max(len(predicted), len(gold), 1)
    return 1.0 - edit_distance(predicted, gold) / denominator
