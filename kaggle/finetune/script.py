import os
from pathlib import Path
import runpy

os.environ["DONUT_CONFIG"] = str(Path(__file__).resolve().parent / "configs" / "finetune_cord.yaml")
runpy.run_path(str(Path(__file__).resolve().parent / "bootstrap.py"), run_name="__main__")
