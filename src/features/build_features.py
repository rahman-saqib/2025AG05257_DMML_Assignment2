"""Stage 5 - Feature Management.
Builds user, item and interaction features and a feature catalog.
The functions only need an interactions table, so the same code can be reused
at prediction time on fresh data.
"""
import pandas as pd

FEATURE_CATALOG = [
    # (feature, entity, description, how it is computed)
    ("user_activity_count", "user", "Number of ratings given by the user", "count of train rows per user"),
    ("user_avg_rating", "user", "Average rating given by the user", "mean rating per user"),
    ("days_since_last_interaction", "user", "Days between the user's last rating and the reference date", "reference_date - max(timestamp)"),
    ("preferred_category", "user", "Category the user has rated most often", "most frequent category among the user's rated items"),
    ("item_popularity", "item", "Number of ratings the item received", "count of train rows per item (0 if never rated)"),
    ("item_avg_rating", "item", "Average rating of the item", "mean rating per item (global mean if never rated)"),
    ("publish_year", "item", "Year the book was published", "copied from the items table"),
    ("price", "item", "Price of the book", "copied from the items table"),
    ("category", "item", "Category of the book", "copied from the items table"),
    ("days_since_interaction", "interaction", "Age of the rating at the reference date", "reference_date - timestamp"),
    ("rating_minus_user_avg", "interaction", "How much higher/lower than the user's usual rating", "rating - user_avg_rating"),
    ("is_preferred_category", "interaction", "1 if the item is in the user's preferred category", "item category == user's preferred_category"),
]


def build_user_features(train, items, user_activity, ref_time):
    uf = user_activity.rename(columns={"n_interactions": "user_activity_count",
                                       "avg_rating": "user_avg_rating"})
    uf["days_since_last_interaction"] = (ref_time - uf["last_interaction"]).dt.days
    rated = train.merge(items[["item_id", "category"]], on="item_id")
    counts = rated.groupby(["user_id", "category"]).size().unstack(fill_value=0)
    uf["preferred_category"] = uf["user_id"].map(counts.idxmax(axis=1))
    return uf.drop(columns=["last_interaction"])


def build_item_features(items, item_popularity):
    itf = items[["item_id", "category", "publish_year", "price"]].merge(item_popularity, on="item_id", how="left")
    itf = itf.rename(columns={"n_interactions": "item_popularity", "avg_rating": "item_avg_rating"})
    itf["item_popularity"] = itf["item_popularity"].fillna(0).astype(int)
    itf["item_avg_rating"] = itf["item_avg_rating"].fillna(itf["item_avg_rating"].mean())
    return itf[["item_id", "item_popularity", "item_avg_rating", "publish_year", "price", "category"]]


def build_interaction_features(train, items, user_features, ref_time):
    inter = train.merge(user_features[["user_id", "user_avg_rating", "preferred_category"]], on="user_id")
    inter = inter.merge(items[["item_id", "category"]], on="item_id")
    inter["days_since_interaction"] = (ref_time - inter["timestamp"]).dt.days
    inter["rating_minus_user_avg"] = inter["rating"] - inter["user_avg_rating"]
    inter["is_preferred_category"] = (inter["category"] == inter["preferred_category"]).astype(int)
    return inter[["user_id", "item_id", "rating", "days_since_interaction",
                  "rating_minus_user_avg", "is_preferred_category"]]


def run(cfg):
    proc, feat = cfg["paths"]["processed"], cfg["paths"]["features"]
    train = pd.read_csv(proc / "train.csv", parse_dates=["timestamp"])
    items = pd.read_csv(proc / "items_clean.csv")
    user_activity = pd.read_csv(proc / "user_activity_summary.csv", parse_dates=["last_interaction"])
    item_popularity = pd.read_csv(proc / "item_popularity.csv")
    ref_time = train["timestamp"].max()          # "today" for all time-based features

    user_features = build_user_features(train, items, user_activity, ref_time)
    item_features = build_item_features(items, item_popularity)
    inter_features = build_interaction_features(train, items, user_features, ref_time)

    user_features.to_csv(feat / "user_features.csv", index=False)
    item_features.to_csv(feat / "item_features.csv", index=False)
    inter_features.to_csv(feat / "interaction_features.csv", index=False)
    pd.DataFrame(FEATURE_CATALOG, columns=["feature", "entity", "description", "computation"]
                 ).to_csv(cfg["paths"]["reports"] / "feature_catalog.csv", index=False)
    print("Features saved:", len(user_features), "users,", len(item_features), "items,",
          len(inter_features), "interactions")
    return "features done"
