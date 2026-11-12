"""Stage 3 - Data Preparation.
Cleans the validated tables, standardises formats and creates a time-based train/test split.
"""
import pandas as pd


def clean_users(users):
    users["age"] = users["age"].fillna(users["age"].median())          # safety net for missing ages
    users["country"] = users["country"].fillna("Unknown").str.strip().str.title()
    return users


def clean_items(items):
    items["title"] = items["title"].str.strip()
    items["category"] = items["category"].fillna("Unknown").str.strip().str.title()
    items["price"] = items["price"].astype(float).round(2)
    items["publish_year"] = items["publish_year"].fillna(items["publish_year"].median()).astype(int)
    return items


def time_split(inter, test_fraction):
    """Oldest interactions -> train, newest interactions -> test."""
    inter = inter.sort_values("timestamp").reset_index(drop=True)
    cutoff = inter["timestamp"].iloc[int(len(inter) * (1 - test_fraction)) - 1]
    train = inter[inter["timestamp"] <= cutoff]
    test = inter[inter["timestamp"] > cutoff]
    return train, test, cutoff


def run(cfg):
    proc = cfg["paths"]["processed"]
    users = clean_users(pd.read_csv(proc / "users_valid.csv"))
    items = clean_items(pd.read_csv(proc / "items_valid.csv"))
    inter = pd.read_csv(proc / "interactions_valid.csv")

    inter = inter.drop_duplicates(["user_id", "item_id"], keep="last")
    inter["timestamp"] = pd.to_datetime(inter["timestamp"], unit="s")

    train, test, cutoff = time_split(inter, cfg["split"]["test_fraction"])
    print(f"Split at {cutoff}: train={len(train)} rows, test={len(test)} rows")

    users.to_csv(proc / "users_clean.csv", index=False)
    items.to_csv(proc / "items_clean.csv", index=False)
    train.to_csv(proc / "train.csv", index=False)
    test.to_csv(proc / "test.csv", index=False)
    return "preparation done"
