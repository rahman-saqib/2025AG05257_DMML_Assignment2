"""Stage 7 - Model Evaluation.
Computes Precision@K, Recall@K, NDCG@K, Coverage and Diversity on the test set,
logs the metrics to the MLflow runs created in training and writes a results table.
"""
import json

import joblib
import mlflow
import numpy as np
import pandas as pd

from src.training.models import ItemCFModel, PopularityModel  # needed to load the saved models


def intra_list_diversity(recs, category_of):
    """Share of item pairs in the list that belong to DIFFERENT categories (0 = all same, 1 = all different)."""
    if len(recs) < 2:
        return 0.0
    cats = category_of.loc[recs].to_numpy()
    n = len(recs)
    same_pairs = (cats[:, None] == cats[None, :]).sum() - n      # remove the diagonal
    return 1 - same_pairs / (n * (n - 1))


def evaluate_model(model, relevant, k, category_of, n_catalog):
    precision, recall, ndcg, diversity, recommended = [], [], [], [], set()
    for user_id, liked in relevant.items():
        recs = model.recommend(user_id, k)
        recommended.update(recs)
        hits = [1 if item in liked else 0 for item in recs]
        precision.append(sum(hits) / k)
        recall.append(sum(hits) / len(liked))
        dcg = sum(h / np.log2(pos + 2) for pos, h in enumerate(hits))
        ideal = sum(1 / np.log2(pos + 2) for pos in range(min(len(liked), k)))
        ndcg.append(dcg / ideal)
        diversity.append(intra_list_diversity(recs, category_of))
    return {f"precision_at_{k}": np.mean(precision), f"recall_at_{k}": np.mean(recall),
            f"ndcg_at_{k}": np.mean(ndcg), "coverage": len(recommended) / n_catalog,
            "diversity": np.mean(diversity), "users_evaluated": len(relevant)}


def run(cfg):
    proc, models_dir = cfg["paths"]["processed"], cfg["paths"]["models"]
    k = cfg["model"]["top_k"]
    train = pd.read_csv(proc / "train.csv")
    test = pd.read_csv(proc / "test.csv")
    items = pd.read_csv(proc / "items_clean.csv")
    category_of = items.set_index("item_id")["category"]

    # users who liked something in the test period and also exist in the training data
    liked_rows = test[(test["rating"] >= cfg["model"]["relevance_threshold"]) &
                      test["user_id"].isin(train["user_id"])]
    relevant = liked_rows.groupby("user_id")["item_id"].apply(set).to_dict()

    models = {"popularity_baseline": joblib.load(models_dir / "popularity_model.joblib"),
              "item_based_cf": joblib.load(models_dir / "item_cf_model.joblib")}
    run_ids = json.loads((models_dir / "run_ids.json").read_text())

    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])
    rows = []
    for name, model in models.items():
        metrics = evaluate_model(model, relevant, k, category_of, len(items))
        with mlflow.start_run(run_id=run_ids[name]):
            mlflow.log_metrics({m: float(v) for m, v in metrics.items()})
            mlflow.set_tag("stage", "evaluated")
        rows.append({"model": name, **{m: round(float(v), 4) for m, v in metrics.items()}})

    results = pd.DataFrame(rows)
    results.to_csv(cfg["paths"]["reports"] / "model_results.csv", index=False)
    header = "| " + " | ".join(results.columns) + " |"
    divider = "|" + "---|" * len(results.columns)
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in results.to_numpy()]
    text = f"# Model Evaluation Results\n\nK = {k}\n\n" + "\n".join([header, divider] + body)
    (cfg["paths"]["reports"] / "model_results.md").write_text(text, encoding="utf-8")
    for name in models:                      # attach the comparison table to each MLflow run
        with mlflow.start_run(run_id=run_ids[name]):
            mlflow.log_artifact(str(cfg["paths"]["reports"] / "model_results.csv"))
    print(results.to_string(index=False))
    return "evaluation done"
