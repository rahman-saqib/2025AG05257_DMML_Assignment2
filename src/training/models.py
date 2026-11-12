"""The two recommenders. Both have fit(...) and recommend(user_id, k)."""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class PopularityModel:
    """Baseline: recommend the most popular items the user has not seen yet."""

    def fit(self, item_features, train):
        self.ranked_items = item_features.sort_values("item_popularity", ascending=False)["item_id"].tolist()
        self.seen = train.groupby("user_id")["item_id"].apply(set).to_dict()
        return self

    def recommend(self, user_id, k):
        seen = self.seen.get(user_id, set())
        return [i for i in self.ranked_items if i not in seen][:k]


class ItemCFModel:
    """Item-based collaborative filtering using cosine similarity between item columns."""

    def fit(self, matrix):
        self.item_ids = matrix.columns.to_numpy()
        self.user_pos = {u: pos for pos, u in enumerate(matrix.index)}
        self.ratings = matrix.to_numpy(dtype=float)
        self.item_sim = cosine_similarity(self.ratings.T)      # items x items
        np.fill_diagonal(self.item_sim, 0)
        return self

    def recommend(self, user_id, k):
        if user_id not in self.user_pos:
            return []
        user_row = self.ratings[self.user_pos[user_id]]
        scores = user_row @ self.item_sim                      # similar to what the user rated
        scores[user_row > 0] = -np.inf                         # never recommend already-rated items
        top = np.argsort(scores)[::-1][:k]
        return [int(self.item_ids[i]) for i in top if scores[i] > 0]
