"""Runs the pipeline WITHOUT Prefect (simple option).
  python run_pipeline.py                 -> all stages
  python run_pipeline.py --stage train   -> one stage (ingest, validate, prepare,
                                            transform, features, train, evaluate)
"""
import argparse

from src.config import load_config
from src.ingestion import ingest
from src.validation import validate
from src.preparation import prepare
from src.transformation import transform
from src.features import build_features
from src.training import train
from src.evaluation import evaluate

STAGES = {"ingest": ingest, "validate": validate, "prepare": prepare, "transform": transform,
          "features": build_features, "train": train, "evaluate": evaluate}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all"] + list(STAGES))
    args = parser.parse_args()
    cfg = load_config()
    for name, module in STAGES.items():
        if args.stage in ("all", name):
            print(f"\n=== {name} ===")
            module.run(cfg)
    print("\nDone.")
