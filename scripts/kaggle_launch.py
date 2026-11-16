"""Stage a Kaggle kernel and push it, using the token already on this machine.

The token is read from KAGGLE_API_TOKEN or from the knee repo .env. It is never
written into this project.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ENV = Path("/Users/rk/repo/kaggle/knee_abnormility/.env")


def load_kaggle_token() -> str:
    current = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if current:
        return current
    path = Path(os.environ.get("KAGGLE_ENV_FILE", DEFAULT_ENV))
    if not path.is_file():
        raise SystemExit(f"KAGGLE_API_TOKEN is not set and {path} was not found")
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        if key.strip() == "KAGGLE_API_TOKEN":
            token = value.strip().strip("'").strip('"')
            if token:
                return token
    raise SystemExit(f"KAGGLE_API_TOKEN is missing from {path}")


def stage(name: str) -> Path:
    source = ROOT / "kaggle" / name
    if not (source / "kernel-metadata.json").is_file():
        raise SystemExit(f"unknown kernel {name}")
    dest = ROOT / "build" / "kaggle" / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(ROOT / "src" / "donut_positions", dest / "src" / "donut_positions", ignore=ignore)
    shutil.copytree(ROOT / "configs", dest / "configs", ignore=ignore)
    shutil.copy(ROOT / "kaggle" / "bootstrap.py", dest / "bootstrap.py")
    shutil.copy(source / "script.py", dest / "script.py")
    shutil.copy(source / "kernel-metadata.json", dest / "kernel-metadata.json")
    return dest


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Push a Donut position-token Kaggle kernel")
    parser.add_argument("stage", choices=["pretrain", "finetune"])
    parser.add_argument("--dry-run", action="store_true", help="stage files and check the token only")
    args = parser.parse_args(argv)
    dest = stage(args.stage)
    print(f"staged {dest}")
    token = load_kaggle_token()
    env = os.environ.copy()
    env["KAGGLE_API_TOKEN"] = token
    subprocess.check_call(["uv", "run", "kaggle", "config", "view"], cwd=ROOT, env=env)
    if args.dry_run:
        return
    subprocess.check_call(["uv", "run", "kaggle", "kernels", "push", "-p", str(dest)], cwd=ROOT, env=env)


if __name__ == "__main__":
    main()
