"""Loads configs/config.yaml and makes sure the data folders exist."""
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[1]

def load_config():
    with open(ROOT / "configs" / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    for name, folder in cfg["paths"].items():
        cfg["paths"][name] = ROOT / folder
        cfg["paths"][name].mkdir(parents=True, exist_ok=True)
    cfg["mlflow"]["tracking_uri"] = "sqlite:///" + (ROOT / cfg["mlflow"]["db_file"]).as_posix()
    return cfg
