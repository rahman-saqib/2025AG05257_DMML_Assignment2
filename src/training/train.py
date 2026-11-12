"""Stage 6 - Model Training.
Trains the popularity baseline and the item-based CF model and logs them to MLflow.
"""
import json

import joblib
import mlflow
import pandas as pd

from src.training.models import ItemCFModel, PopularityModel


def log_to_mlflow(run_name, params, model_path, run_ids, feature_catalog):
    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.set_tag("dataset", "synthetic_bookstore")
        mlflow.log_params(params)
        mlflow.log_artifact(str(model_path))               # the trained model file
        mlflow.log_artifact(str(feature_catalog))          # feature documentation used for training
        run_ids[run_name] = active_run.info.run_id


def run(cfg):
    proc, feat, models_dir = cfg["paths"]["processed"], cfg["paths"]["features"], cfg["paths"]["models"]
    train = pd.read_csv(proc / "train.csv")
    item_features = pd.read_csv(feat / "item_features.csv")
    matrix = pd.read_csv(proc / "user_item_matrix.csv", index_col=0)
    matrix.columns = matrix.columns.astype(int)

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    run_ids = {}
    catalog = cfg["paths"]["reports"] / "feature_catalog.csv"

    popularity = PopularityModel().fit(item_features, train)
    path = models_dir / "popularity_model.joblib"
    joblib.dump(popularity, path)
    log_to_mlflow("popularity_baseline",
                  {"model_type": "popularity", "ranked_by": "item_popularity",
                   "train_rows": len(train), "dataset_seed": cfg["dataset"]["seed"]}, path, run_ids, catalog)

    item_cf = ItemCFModel().fit(matrix)
    path = models_dir / "item_cf_model.joblib"
    joblib.dump(item_cf, path)
    log_to_mlflow("item_based_cf",
                  {"model_type": "item_based_cf", "similarity": "cosine",
                   "n_users": matrix.shape[0], "n_items": matrix.shape[1],
                   "train_rows": len(train), "dataset_seed": cfg["dataset"]["seed"]}, path, run_ids, catalog)

    (models_dir / "run_ids.json").write_text(json.dumps(run_ids))
    print("Models trained and logged to MLflow:", list(run_ids))
    return "training done"
