"""Datasets for synthetic pretraining and CORD finetuning."""

from __future__ import annotations

import json
import random
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import torch
from torch.utils.data import Dataset

from donut_positions.geometry import bin_box, canvas_tensor, fit_canvas, scale_box
from donut_positions.targets import (
    CORD_TASK,
    cord_target,
    extract_pos_boxes,
    leaf_texts,
    line_texts,
    map_boxes,
    structure_from_valid_line,
    words_to_target,
)

CORD_URLS = {
    "train": "https://github.com/mindee/doctr/releases/download/v0.1.1/cord_train.zip",
    "test": "https://github.com/mindee/doctr/releases/download/v0.1.1/cord_test.zip",
}

KNOWN_ANNOTATION_ZIPS = [
    Path("data/cord_annotations.zip"),
    Path("/Users/rk/repo/AttentionIsAllYouNeed/Donut/data/cord_annotations.zip"),
]


def encode_target(tokenizer, text: str, max_length: int) -> torch.Tensor:
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).input_ids.squeeze(0)
    labels = token_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    return labels


class SynthDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        tokenizer,
        image_size: list[int],
        max_length: int,
        limit: int | None = None,
        random_pad: bool = False,
        task_token: str = "<s_synthdog>",
    ):
        folder = Path(root) / split
        paths = sorted(folder.glob("*.json"))
        if limit is not None:
            paths = paths[:limit]
        if not paths:
            raise FileNotFoundError(f"no synthetic pages in {folder}")
        self.paths = paths
        self.tokenizer = tokenizer
        self.canvas_height, self.canvas_width = int(image_size[0]), int(image_size[1])
        self.max_length = max_length
        self.random_pad = random_pad
        self.task_token = task_token
        self.eos = tokenizer.eos_token or "</s>"

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict:
        from PIL import Image

        meta_path = self.paths[index]
        meta = json.loads(meta_path.read_text())
        image = Image.open(meta_path.with_suffix(".png")).convert("RGB")
        canvas = fit_canvas(
            image.width,
            image.height,
            self.canvas_height,
            self.canvas_width,
            self.random_pad,
            random.Random(),
        )
        words = []
        for word in meta["words"]:
            bins = bin_box(quad_box(word["quad"]), canvas)
            words.append({"text": word["text"], "bins": bins})
        target = words_to_target(words, task_token=self.task_token, eos_token=self.eos)
        return {
            "pixel_values": canvas_tensor(image, canvas),
            "labels": encode_target(self.tokenizer, target, self.max_length),
            "target": target,
            "boxes": extract_pos_boxes(target),
        }


def quad_box(quad: dict) -> list[float]:
    from donut_positions.targets import quad_to_xyxy

    return quad_to_xyxy(quad)


class CordDataset(Dataset):
    def __init__(
        self,
        records: list[dict],
        images,
        tokenizer,
        image_size: list[int],
        max_length: int,
        random_pad: bool = False,
        task_token: str = CORD_TASK,
    ):
        self.records = records
        self.images = images
        self.tokenizer = tokenizer
        self.canvas_height, self.canvas_width = int(image_size[0]), int(image_size[1])
        self.max_length = max_length
        self.random_pad = random_pad
        self.task_token = task_token
        self.eos = tokenizer.eos_token or "</s>"

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = self.images[record["index"]].convert("RGB")
        tree = record["box_tree"]
        ann_w = record.get("ann_width") or image.width
        ann_h = record.get("ann_height") or image.height
        if ann_w != image.width or ann_h != image.height:
            tree = map_boxes(tree, lambda box: scale_box(box, ann_w, ann_h, image.width, image.height))
        canvas = fit_canvas(
            image.width,
            image.height,
            self.canvas_height,
            self.canvas_width,
            self.random_pad,
            random.Random(),
        )
        binned = map_boxes(tree, lambda box: bin_box(box, canvas))
        target = cord_target(record["gt_parse"], binned, task_token=self.task_token, eos_token=self.eos)
        return {
            "pixel_values": canvas_tensor(image, canvas),
            "labels": encode_target(self.tokenizer, target, self.max_length),
            "target": target,
            "boxes": extract_pos_boxes(target),
        }


def collate_batch(batch: list[dict]) -> dict:
    return {
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "target": [item["target"] for item in batch],
        "boxes": [item["boxes"] for item in batch],
    }


def _normalize_signature(texts: frozenset[str]) -> frozenset[str]:
    return frozenset(" ".join(text.lower().split()) for text in texts)


def load_annotation_zip(path: Path) -> list[dict]:
    annotations = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".json"):
                continue
            data = json.loads(archive.read(name))
            if "valid_line" not in data:
                continue
            annotations.append(data)
    return annotations


def download_annotation_zip(dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    import tempfile

    tmp = Path(tempfile.mkdtemp(prefix="cord_ann_"))
    with zipfile.ZipFile(dest, "w", compression=zipfile.ZIP_DEFLATED) as out:
        for split, url in CORD_URLS.items():
            local = tmp / f"{split}.zip"
            print(f"downloading CORD boxes: {url}")
            urlretrieve(url, local)
            with zipfile.ZipFile(local) as archive:
                for name in archive.namelist():
                    if name.endswith(".json"):
                        out.writestr(f"{split}/{Path(name).name}", archive.read(name))
    return dest


def resolve_annotation_zip(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    for candidate in KNOWN_ANNOTATION_ZIPS:
        if candidate.is_file():
            return candidate
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        found = sorted(kaggle_input.glob("**/*.zip"))
        for path in found:
            if "cord" in path.name.lower():
                return path
    return None


def build_annotation_index(annotations: list[dict]) -> tuple[dict, dict]:
    exact: dict[frozenset, dict] = {}
    normalized: dict[frozenset, dict] = {}
    for annotation in annotations:
        signature = line_texts(annotation["valid_line"])
        if not signature:
            continue
        exact[signature] = annotation
        normalized[_normalize_signature(signature)] = annotation
    return exact, normalized


def cord_records(dataset_split, index: tuple[dict, dict] | None) -> list[dict]:
    exact, normalized = index if index is not None else ({}, {})
    records = []
    missing = 0
    inline = 0
    matched = 0
    for row_index, sample in enumerate(dataset_split):
        ground_truth = json.loads(sample["ground_truth"])
        gt_parse = ground_truth.get("gt_parse")
        if gt_parse is None:
            missing += 1
            continue
        if "valid_line" in ground_truth:
            valid_line = ground_truth["valid_line"]
            meta = ground_truth.get("meta") or {}
            inline += 1
        else:
            signature = leaf_texts(gt_parse)
            annotation = exact.get(signature) or normalized.get(_normalize_signature(signature))
            if annotation is None:
                missing += 1
                continue
            valid_line = annotation["valid_line"]
            meta = annotation.get("meta") or {}
            matched += 1
        image_size = meta.get("image_size") or {}
        records.append(
            {
                "index": row_index,
                "gt_parse": gt_parse,
                "box_tree": structure_from_valid_line(valid_line),
                "ann_width": image_size.get("width"),
                "ann_height": image_size.get("height"),
            }
        )
    print(f"CORD rows: inline={inline} zip={matched} skipped={missing} kept={len(records)}")
    if not records:
        raise RuntimeError("no CORD rows had both a parse and word boxes")
    return records


def load_cord_split(split: str, annotation_zip: str | None):
    from datasets import load_dataset

    dataset = load_dataset("naver-clova-ix/cord-v2", split=split)
    probe = json.loads(dataset[0]["ground_truth"])
    index = None
    if "valid_line" not in probe:
        path = resolve_annotation_zip(annotation_zip)
        if path is None:
            path = download_annotation_zip(Path("data/cord_annotations.zip"))
        print(f"using CORD box zip {path}")
        index = build_annotation_index(load_annotation_zip(path))
    records = cord_records(dataset, index)
    return records, dataset["image"]
