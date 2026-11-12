"""Prefect workflow: Ingestion -> Validation -> Preparation -> Transformation
-> Features -> Training -> Evaluation.

Each task receives the result of the previous task, so Prefect runs them in this exact order.
Usage (from the project root):  python dags/recommendation_flow.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # so "src" can be imported

from prefect import flow, task

from src.config import load_config
from src.ingestion import ingest
from src.validation import validate
from src.preparation import prepare
from src.transformation import transform
from src.features import build_features
from src.training import train
from src.evaluation import evaluate


@task(name="1-ingest-data")
def ingest_task():
    return ingest.run(load_config())


@task(name="2-validate-data")
def validate_task(previous):
    return validate.run(load_config())


@task(name="3-prepare-data")
def prepare_task(previous):
    return prepare.run(load_config())


@task(name="4-transform-data")
def transform_task(previous):
    return transform.run(load_config())


@task(name="5-build-features")
def features_task(previous):
    return build_features.run(load_config())


@task(name="6-train-models")
def train_task(previous):
    return train.run(load_config())


@task(name="7-evaluate-models")
def evaluate_task(previous):
    return evaluate.run(load_config())


@flow(name="recommendation-pipeline", log_prints=True)
def recommendation_pipeline():
    a = ingest_task()
    b = validate_task(a)
    c = prepare_task(b)
    d = transform_task(c)
    e = features_task(d)
    f = train_task(e)
    evaluate_task(f)


if __name__ == "__main__":
    recommendation_pipeline()
