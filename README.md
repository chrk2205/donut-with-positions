# Donut with position tokens

Donut reads a document image and emits the text as a token sequence. Nothing in that sequence says where a word or a field was on the page. This project adds that, as ordinary decoder tokens, and trains it in two stages.

The position vocabulary is `<pos_0>` … `<pos_511>`. A box is four tokens, xmin ymin xmax ymax, measured on the canvas the encoder actually sees: the image is aspect-fit into Donut's CORD resolution (height 1280, width 960) and then padded. The training-time random pad moves the box with the pixels.

Continued pretraining starts from [`naver-clova-ix/donut-base`](https://huggingface.co/naver-clova-ix/donut-base). Each synthetic word is followed by its box:

```text
<s_synthdog>Tebu<pos_12><pos_40><pos_80><pos_55> Lemon<pos_90><pos_40><pos_140><pos_55></s>
```

CORD finetuning keeps the original parse text. The same four tokens are inserted immediately before the leaf key closes. Parent keys are unchanged.

```text
<s_nm>Tebu Lemon<pos_12><pos_40><pos_140><pos_55></s_nm>
```

There is no Clova SynthDoG asset pack here, so pretraining pages are generated in that task format: English words in reading order, each with a quad. CORD text comes from `naver-clova-ix/cord-v2`. Current releases of that dataset already store `valid_line` word boxes inside `ground_truth`; the tree built from those lines matches `gt_parse` on all 100 validation receipts. The [official CORD json zip](https://github.com/mindee/doctr/releases/tag/v0.1.1) is the fallback when a copy of the dataset has no boxes. Categories `menu.sub.nm` and `menu.sub_nm` both become a nested `sub` object, which is the shape Donut trains on.

## Setup

```bash
uv sync
```

Python 3.11. PyTorch is the default build, so a laptop gets CPU or MPS and a CUDA machine gets CUDA wheels if that is how `uv` resolves the platform.

## Train

Local pretraining, then CORD:

```bash
uv run python -m donut_positions.train --config configs/pretrain.yaml
uv run python -m donut_positions.train --config configs/finetune_cord.yaml
```

A short run that only checks the loop. The configs stay at 1280×960; this override is small enough for a laptop.

```bash
uv run python -m donut_positions.train --config configs/pretrain.yaml \
  --train-samples 8 --val-samples 2 --max-steps 2 \
  --eval-max-samples 1 --gen-max-new-tokens 8 \
  --max-length 64 --image-height 384 --image-width 384 \
  --result-dir outputs/smoke
```

Finetune reads `outputs/pretrain/model` when that folder exists. If it does not, it starts from `donut-base` and adds the position tokens there.

The same entrypoint runs on Kaggle. The API token is taken from `KAGGLE_API_TOKEN`, or from the knee repo `.env` on this machine. It is not stored in this repository.

```bash
uv run python scripts/kaggle_launch.py pretrain
uv run python scripts/kaggle_launch.py finetune
```

Kernels: `violet2205/donut-pos-pretrain` and `violet2205/donut-pos-cord`, on a T4. After pretraining, add the saved model as a Kaggle dataset. The finetune kernel uses `init_checkpoint` when that path exists, and otherwise the first model folder under `/kaggle/input`.

Checkpoints land in `outputs/pretrain/model` and `outputs/finetune/model`. `metrics.json` next to them has two numbers: text accuracy is `1 - normalized edit distance` after the position tokens and the task prompt are removed, and mean IoU is the overlap of the four-token boxes in bin coordinates, in the order they appear. A missing box scores 0.

## Results

No full training run is recorded here yet. The commands above are the run that fills this table.

| stage | text accuracy | mean IoU | notes |
| --- | --- | --- | --- |
| pretrain | | | 8,000 synthetic pages, 3 epochs |
| CORD | | | 10 epochs from the pretrain checkpoint |
