"""Stage 1 - Data Ingestion.
Generates a documented SYNTHETIC online-bookstore dataset (users, items, ratings)
and saves it as CSV files in data/raw/. The same seed always gives the same data.
"""
import numpy as np
import pandas as pd

CATEGORIES = ["Fiction", "Mystery", "Sci-Fi", "Romance", "History", "Science", "Biography", "Self-Help"]
COUNTRIES = ["India", "USA", "UK", "Germany", "Canada", "Australia"]
START = pd.Timestamp("2022-01-01").timestamp()
END = pd.Timestamp("2024-12-31").timestamp()


def make_users(n_users, rng):
    return pd.DataFrame({"user_id": np.arange(1, n_users + 1),
                         "age": rng.integers(16, 76, n_users),
                         "country": rng.choice(COUNTRIES, n_users)})


def make_items(n_items, rng):
    return pd.DataFrame({"item_id": np.arange(1, n_items + 1),
                         "title": [f"Book {i:04d}" for i in range(1, n_items + 1)],
                         "category": rng.choice(CATEGORIES, n_items),
                         "publish_year": rng.integers(1980, 2024, n_items),
                         "price": rng.uniform(5, 60, n_items).round(2)})


def make_interactions(users, items, rng):
    """Each user has 1-2 favourite categories and rates 15-80 books.
    Popular books and books from favourite categories are more likely to be rated,
    and favourite categories get higher ratings."""
    popularity = rng.pareto(1.5, len(items)) + 1                      # a few very popular books
    item_cat = items["category"].to_numpy()
    rows = []
    for user_id in users["user_id"]:
        favourites = rng.choice(CATEGORIES, rng.integers(1, 3), replace=False)
        is_fav = np.isin(item_cat, favourites)
        prob = popularity * np.where(is_fav, 4, 1)
        n_rated = rng.integers(15, 81)
        chosen = rng.choice(len(items), n_rated, replace=False, p=prob / prob.sum())
        ratings = np.clip(np.round(3 + is_fav[chosen] * 1.0 + rng.normal(0, 0.8, n_rated)), 1, 5)
        times = rng.integers(START, END, n_rated)
        for idx, r, t in zip(chosen, ratings, times):
            rows.append((user_id, items["item_id"].iloc[idx], int(r), int(t)))
    return pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "timestamp"])


def add_demo_errors(inter):
    """Append 7 deliberately bad rows so the validation step can be demonstrated."""
    existing = set(zip(inter["user_id"], inter["item_id"]))
    free = [(u, i) for u in range(1, 6) for i in range(1, 50) if (u, i) not in existing]
    ts = int(START) + 1000
    bad_rows = [
        inter.iloc[0].to_dict(),                                                  # duplicate
        {"user_id": free[0][0], "item_id": free[0][1], "rating": 9, "timestamp": ts},        # invalid rating
        {"user_id": free[1][0], "item_id": free[1][1], "rating": None, "timestamp": ts},     # missing value
        {"user_id": free[2][0], "item_id": free[2][1], "rating": 4, "timestamp": -100},      # invalid timestamp
        {"user_id": free[3][0], "item_id": free[3][1], "rating": "five", "timestamp": ts},   # wrong data type
        {"user_id": 99999, "item_id": 1, "rating": 4, "timestamp": ts},           # user does not exist
        {"user_id": 1, "item_id": 99999, "rating": 3, "timestamp": ts},           # item does not exist
    ]
    return pd.concat([inter, pd.DataFrame(bad_rows)], ignore_index=True)


def run(cfg):
    d = cfg["dataset"]
    rng = np.random.default_rng(d["seed"])
    users = make_users(d["n_users"], rng)
    items = make_items(d["n_items"], rng)
    inter = make_interactions(users, items, rng)
    if d["inject_demo_errors"]:
        inter = add_demo_errors(inter)
    raw = cfg["paths"]["raw"]
    users.to_csv(raw / "users.csv", index=False)
    items.to_csv(raw / "items.csv", index=False)
    inter.to_csv(raw / "interactions.csv", index=False)
    print(f"Ingested: {len(users)} users, {len(items)} items, {len(inter)} interactions")
    return "ingestion done"
