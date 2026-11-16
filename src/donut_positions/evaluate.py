"""Generation metrics: text edit distance with boxes stripped, and box IoU."""

from __future__ import annotations

import json
from pathlib import Path

import torch

from donut_positions.targets import aligned_mean_iou, boxes_from_ids, sequence_accuracy


def _content_ids(token_ids: list[int], drop: set[int]) -> list[int]:
    return [token_id for token_id in token_ids if token_id not in drop and token_id != -100]


def pos_token_ids(tokenizer, bins: int) -> set[int]:
    ids = set()
    for index in range(bins):
        token_id = tokenizer.convert_tokens_to_ids(f"<pos_{index}>")
        if token_id != tokenizer.unk_token_id:
            ids.add(token_id)
    return ids


@torch.no_grad()
def evaluate(
    model,
    tokenizer,
    dataset,
    task_token: str,
    max_samples: int,
    max_new_tokens: int,
    pos_bins: int,
    device: torch.device,
) -> dict:
    model.eval()
    model.config.use_cache = True
    count = min(max_samples, len(dataset))
    drop = pos_token_ids(tokenizer, pos_bins)
    id_to_bin = {}
    for index in range(pos_bins):
        token_id = tokenizer.convert_tokens_to_ids(f"<pos_{index}>")
        if token_id != tokenizer.unk_token_id:
            id_to_bin[token_id] = index
    for token in (task_token, tokenizer.eos_token, tokenizer.pad_token):
        if token:
            drop.add(tokenizer.convert_tokens_to_ids(token))
    accuracies = []
    ious = []
    example = None
    prompt_id = tokenizer.convert_tokens_to_ids(task_token)
    for index in range(count):
        item = dataset[index]
        pixel_values = item["pixel_values"].unsqueeze(0).to(device)
        decoder_input_ids = torch.tensor([[prompt_id]], device=device)
        generated = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
            bad_words_ids=[[tokenizer.unk_token_id]] if tokenizer.unk_token_id is not None else None,
        )
        predicted_ids = generated[0].tolist()
        gold_ids = [token_id for token_id in item["labels"].tolist() if token_id != -100]
        accuracies.append(sequence_accuracy(_content_ids(predicted_ids, drop), _content_ids(gold_ids, drop)))
        predicted_text = tokenizer.decode(predicted_ids, skip_special_tokens=False)
        ious.append(aligned_mean_iou(boxes_from_ids(predicted_ids, id_to_bin), item["boxes"]))
        if example is None:
            example = {"prediction": predicted_text, "target": item["target"]}
        print(f"eval {index + 1}/{count} text={accuracies[-1]:.3f} iou={ious[-1]:.3f}")
    text_accuracy = sum(accuracies) / max(len(accuracies), 1)
    mean_iou = sum(ious) / max(len(ious), 1)
    return {
        "n": count,
        "text_accuracy": text_accuracy,
        "mean_iou": mean_iou,
        "example": example,
    }


def write_metrics(path: Path, metrics: dict) -> None:
    path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(
        f"text accuracy {metrics['text_accuracy']:.3f}  "
        f"mean IoU {metrics['mean_iou']:.3f}  n={metrics['n']}"
    )
