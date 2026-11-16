"""Continued pretraining and CORD finetuning for Donut with position tokens."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
import yaml
from torch.utils.data import DataLoader

from donut_positions.data import CordDataset, SynthDataset, collate_batch, load_cord_split
from donut_positions.evaluate import evaluate, write_metrics
from donut_positions.generate import ensure_synth
from donut_positions.geometry import POS_BINS
from donut_positions.targets import CORD_TASK, SYNTH_TASK, collect_special_tokens


def load_config(path: Path) -> dict:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return config


def apply_overrides(config: dict, args: argparse.Namespace) -> dict:
    mapping = {
        "max_steps": args.max_steps,
        "train_samples": args.train_samples,
        "val_samples": args.val_samples,
        "max_epochs": args.max_epochs,
        "eval_max_samples": args.eval_max_samples,
        "gen_max_new_tokens": args.gen_max_new_tokens,
        "result_dir": args.result_dir,
        "num_workers": args.num_workers,
        "init_checkpoint": args.init_checkpoint,
        "max_length": args.max_length,
    }
    for key, value in mapping.items():
        if value is not None:
            config[key] = value
    if args.image_height is not None or args.image_width is not None:
        height, width = config.get("image_size", [1280, 960])
        config["image_size"] = [
            args.image_height if args.image_height is not None else height,
            args.image_width if args.image_width is not None else width,
        ]
    return config


def check_image_size(image_size: list[int]) -> None:
    """Last Swin stage is the image divided by 32 and must be larger than the window (10)."""
    height, width = int(image_size[0]), int(image_size[1])
    if height % 32 or width % 32 or height < 384 or width < 384:
        raise SystemExit(
            f"image_size [{height}, {width}] does not fit Donut's Swin windows. "
            "Use at least [384, 384]. The training configs use [1280, 960]."
        )


def accelerator_and_precision(config: dict) -> tuple[str, str]:
    if config.get("precision"):
        precision = str(config["precision"])
    elif torch.cuda.is_available():
        precision = "16-mixed"
    else:
        precision = "32-true"
    if torch.cuda.is_available():
        return "gpu", precision
    if torch.backends.mps.is_available():
        return "mps", precision
    return "cpu", precision


def resolve_init(config: dict) -> str:
    requested = config.get("init_checkpoint")
    if requested and Path(requested).exists():
        return str(requested)
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.is_dir():
        for config_path in sorted(kaggle_input.glob("**/config.json")):
            folder = config_path.parent
            if (folder / "model.safetensors").exists() or (folder / "pytorch_model.bin").exists():
                print(f"using Kaggle checkpoint {folder}")
                return str(folder)
    if requested:
        print(f"checkpoint {requested} is missing; starting from {config['model_name']}")
    return config["model_name"]


def load_model(source: str, image_size: list[int], max_length: int):
    from transformers import AutoTokenizer, VisionEncoderDecoderConfig, VisionEncoderDecoderModel

    model_config = VisionEncoderDecoderConfig.from_pretrained(source)
    model_config.encoder.image_size = [int(image_size[0]), int(image_size[1])]
    model_config.decoder.max_length = int(max_length)
    model = VisionEncoderDecoderModel.from_pretrained(
        source,
        config=model_config,
        ignore_mismatched_sizes=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(source)
    return model, tokenizer


def add_tokens(tokenizer, model, tokens: list[str]) -> int:
    added = tokenizer.add_special_tokens({"additional_special_tokens": tokens})
    if added > 0:
        model.decoder.resize_token_embeddings(len(tokenizer))
    return added


def position_tokens(bins: int) -> list[str]:
    return [f"<pos_{index}>" for index in range(bins)]


def extend_decoder_positions(model, max_length: int) -> None:
    """Donut's decoder is trained with 768 positions. Pretraining needs more, because each word carries four box tokens."""
    decoder = model.decoder.model.decoder
    positions = decoder.embed_positions
    offset = int(getattr(positions, "offset", 0))
    current = positions.num_embeddings - offset
    if max_length <= current:
        return
    grown = positions.__class__(max_length, positions.embedding_dim)
    with torch.no_grad():
        grown.weight[: positions.num_embeddings] = positions.weight
        grown.weight[positions.num_embeddings :] = positions.weight[-1]
    decoder.embed_positions = grown
    decoder.max_target_positions = max_length
    model.decoder.config.max_position_embeddings = max_length
    model.config.decoder.max_position_embeddings = max_length
    print(f"extended decoder positions {current} -> {max_length}")


class DonutModule(pl.LightningModule):
    def __init__(self, model, learning_rate: float):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def training_step(self, batch, batch_idx):
        outputs = self.model(pixel_values=batch["pixel_values"], labels=batch["labels"])
        self.log("train_loss", outputs.loss, prog_bar=True)
        return outputs.loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


def build_datasets(config: dict, tokenizer):
    image_size = config["image_size"]
    max_length = int(config["max_length"])
    stage = config["stage"]
    if stage == "pretrain":
        root = Path(config["data_dir"])
        if config.get("generate_if_missing", True):
            ensure_synth(
                root,
                int(config["train_samples"]),
                int(config["val_samples"]),
                int(config.get("seed", 0)),
            )
        train = SynthDataset(
            root, "train", tokenizer, image_size, max_length,
            limit=int(config["train_samples"]), random_pad=True,
        )
        val = SynthDataset(
            root, "val", tokenizer, image_size, max_length,
            limit=int(config["val_samples"]), random_pad=False,
        )
        return train, val, SYNTH_TASK
    if stage != "finetune":
        raise ValueError(f"unknown stage {stage}")
    train_records, train_images = load_cord_split(config.get("train_split", "train"), config.get("annotation_zip"))
    val_records, val_images = load_cord_split(config.get("val_split", "validation"), config.get("annotation_zip"))
    specials = {CORD_TASK, "<sep/>"}
    for record in train_records + val_records:
        specials.update(collect_special_tokens(record["gt_parse"]))
    return (train_records, train_images, val_records, val_images, sorted(specials))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Donut with position tokens")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--train-samples", type=int)
    parser.add_argument("--val-samples", type=int)
    parser.add_argument("--eval-max-samples", type=int)
    parser.add_argument("--gen-max-new-tokens", type=int)
    parser.add_argument("--result-dir")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--init-checkpoint")
    parser.add_argument("--image-height", type=int)
    parser.add_argument("--image-width", type=int)
    parser.add_argument("--max-length", type=int)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = apply_overrides(load_config(Path(args.config)), args)
    pl.seed_everything(int(config.get("seed", 0)), workers=True)
    check_image_size(config["image_size"])
    bins = int(config.get("pos_bins", POS_BINS))
    source = resolve_init(config)
    print(f"loading {source}")
    model, tokenizer = load_model(source, config["image_size"], int(config["max_length"]))
    added = add_tokens(tokenizer, model, position_tokens(bins) + [SYNTH_TASK, "<sep/>"])
    print(f"added {added} tokens; vocab {len(tokenizer)}")
    probe = tokenizer("<pos_0><pos_511>", add_special_tokens=False).input_ids
    if len(probe) != 2:
        raise RuntimeError("position tokens were split by the tokenizer")
    extend_decoder_positions(model, int(config["max_length"]))

    stage = config["stage"]
    if stage == "pretrain":
        train_data, val_data, task_token = build_datasets(config, tokenizer)
    else:
        built = build_datasets(config, tokenizer)
        train_records, train_images, val_records, val_images, specials = built
        added_fields = add_tokens(tokenizer, model, specials)
        print(f"added {added_fields} CORD tokens; vocab {len(tokenizer)}")
        image_size = config["image_size"]
        max_length = int(config["max_length"])
        train_data = CordDataset(train_records, train_images, tokenizer, image_size, max_length, random_pad=True)
        val_data = CordDataset(val_records, val_images, tokenizer, image_size, max_length, random_pad=False)
        task_token = CORD_TASK

    model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(task_token)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = False
    if torch.cuda.is_available():
        model.gradient_checkpointing_enable()

    workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(
        train_data,
        batch_size=int(config.get("train_batch_size", 1)),
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_batch,
    )
    accelerator, precision = accelerator_and_precision(config)
    max_steps = int(config.get("max_steps", -1))
    result_dir = Path(config["result_dir"])
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "train_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))

    trainer = pl.Trainer(
        max_epochs=int(config.get("max_epochs", 1)),
        max_steps=max_steps,
        accelerator=accelerator,
        devices=1,
        precision=precision,
        gradient_clip_val=1.0,
        enable_checkpointing=False,
        logger=False,
        num_sanity_val_steps=0,
        enable_model_summary=False,
    )
    module = DonutModule(model, float(config.get("lr", 3e-5)))
    module.train()
    trainer.fit(module, train_loader)

    model_dir = result_dir / "model"
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    print(f"saved {model_dir}")

    eval_samples = int(config.get("eval_max_samples") or 0)
    if eval_samples > 0:
        device = module.device
        metrics = evaluate(
            model,
            tokenizer,
            val_data,
            task_token,
            eval_samples,
            int(config.get("gen_max_new_tokens", 128)),
            bins,
            device,
        )
        write_metrics(result_dir / "metrics.json", metrics)
        (result_dir / "example.json").write_text(json.dumps(metrics.get("example"), indent=2, ensure_ascii=False))
    # MPS and Lightning can hang while the interpreter tears the model down.
    os._exit(0)


if __name__ == "__main__":
    main()
