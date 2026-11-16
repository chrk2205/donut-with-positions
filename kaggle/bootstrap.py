"""Install Kaggle-only packages, then call the same trainer as a local run."""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ensure_deps() -> None:
    required = [
        ("pytorch_lightning", "pytorch-lightning>=2.2"),
        ("transformers", "transformers>=4.44"),
        ("datasets", "datasets>=2.19"),
        ("yaml", "pyyaml"),
        ("nltk", "nltk"),
        ("PIL", "pillow"),
        ("sentencepiece", "sentencepiece"),
    ]
    missing = []
    for module, package in required:
        try:
            importlib.import_module(module)
        except ImportError:
            missing.append(package)
    if missing:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


if __name__ == "__main__":
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "src"))
    ensure_deps()
    from donut_positions.train import main

    config = os.environ["DONUT_CONFIG"]
    main(["--config", config])
