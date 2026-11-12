"""Stage 4 - Data Transformation.
Builds derived datasets from the TRAIN data: user activity summary,
item popularity and the user-item interaction matrix.
"""
import pandas as pd


def run(cfg):
    proc = cfg["paths"]["processed"]
    train = pd.read_csv(proc / "train.csv", parse_dates=["timestamp"])
    items = pd.read_csv(proc / "items_clean.csv")

    user_activity = train.groupby("user_id").agg(
        n_interactions=("item_id", "count"),
        avg_rating=("rating", "mean"),
        last_interaction=("timestamp", "max")).reset_index()

    item_popularity = train.groupby("item_id").agg(
        n_interactions=("user_id", "count"),
        avg_rating=("rating", "mean")).reset_index()

    # rows = users, columns = ALL items of the catalogue, value = rating (0 = not rated)
    matrix = train.pivot_table(index="user_id", columns="item_id", values="rating", fill_value=0)
    matrix = matrix.reindex(columns=items["item_id"], fill_value=0)

    user_activity.to_csv(proc / "user_activity_summary.csv", index=False)
    item_popularity.to_csv(proc / "item_popularity.csv", index=False)
    matrix.to_csv(proc / "user_item_matrix.csv")
    print("Matrix shape (users x items):", matrix.shape)
    return "transformation done"
