import numpy as np
import time
class BaselineModel:
    def __init__(self):
        self.global_mean = 0.0
        self._known_ids = np.array([], dtype=np.int64)
        self._known_means = np.array([], dtype=np.float64)

    def fit(self, data: np.ndarray):
        if data.size == 0:
            self.global_mean = 0.0
            self._known_ids = np.array([], dtype=np.int64)
            self._known_means = np.array([], dtype=np.float64)
            return

        items = data[:, 1].astype(np.int64)
        targets = data[:, 2].astype(np.float64)
        precomputed = data[:, 4].astype(np.float64) if data.ndim == 2 and data.shape[1] >= 5 else None

        if precomputed is not None:
            self._known_ids, first_pos = np.unique(items, return_index=True)
            self._known_means = precomputed[first_pos]
            self.global_mean = float(self._known_means.mean()) if self._known_means.size else 0.0
            return

        self.global_mean = float(targets.mean()) if targets.size else 0.0
        if items.size == 0:
            self._known_ids = np.array([], dtype=np.int64)
            self._known_means = np.array([], dtype=np.float64)
            return

        unique_items, counts = np.unique(items, return_counts=True)
        accumulator = np.zeros(unique_items.max() + 1, dtype=np.float64)
        np.add.at(accumulator, items, targets)
        self._known_ids = unique_items
        self._known_means = accumulator[unique_items] / counts

    def predict_vector(self, product_ids: np.ndarray):
        if self._known_ids.size == 0:
            return np.full(product_ids.shape, self.global_mean, dtype=np.float64)
        pos = np.searchsorted(self._known_ids, product_ids)
        result = np.full(product_ids.shape, self.global_mean, dtype=np.float64)
        valid_mask = pos < self._known_ids.size
        if not np.any(valid_mask):
            return result
        valid_positions = pos[valid_mask]
        match_mask = self._known_ids[valid_positions] == product_ids[valid_mask]
        if np.any(match_mask):
            target_idx = np.where(valid_mask)[0][match_mask]
            result[target_idx] = self._known_means[valid_positions[match_mask]]
        return result

    def predict(self, _, product_id):
        product_id = int(product_id)
        if self._known_ids.size == 0:
            return self.global_mean
        idx = np.searchsorted(self._known_ids, product_id)
        if idx < self._known_ids.size and self._known_ids[idx] == product_id:
            return float(self._known_means[idx])
        return self.global_mean

    def evaluate(self, test_data: np.ndarray):
        if test_data.size == 0:
            return 0.0
        items = test_data[:, 1].astype(np.int64)
        preds = self.predict_vector(items)
        err = test_data[:, 2].astype(np.float64) - preds
        return float(np.sqrt(np.mean(err * err)))

class MatrixFactorizationModel:
    def __init__(self, n_users, n_products, n_factors=64, learning_rate=0.007, regularization=0.03, n_epochs=150, early_stopping=True, patience=15, use_standardized_targets=True, target_mean: float | None = None, target_std: float | None = None, base_user_bias: np.ndarray | None = None, base_item_bias: np.ndarray | None = None, rating_baseline_mean: float | None = None):
        self.n_users = n_users
        self.n_products = n_products
        self.K = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.epochs = n_epochs
        self.early_stopping = early_stopping
        self.patience = patience
        self.use_standardized = use_standardized_targets
        self._manual_target_stats = target_mean is not None and target_std is not None
        self.target_mean = float(target_mean) if target_mean is not None else 0.0
        self.target_std = float(target_std) if (target_std is not None and abs(target_std) > 1e-8) else 1.0
        self.base_user_bias = (
            np.asarray(base_user_bias, dtype=np.float64)
            if base_user_bias is not None else None
        )
        self.base_item_bias = (
            np.asarray(base_item_bias, dtype=np.float64)
            if base_item_bias is not None else None
        )
        self.rating_baseline_mean = float(rating_baseline_mean) if rating_baseline_mean is not None else 0.0
        self._use_baseline = (
            self.base_user_bias is not None and self.base_item_bias is not None
        )

        scale = 1.0 / np.sqrt(self.K)
        self.P = np.random.normal(0, scale=scale, size=(self.n_users, self.K))
        self.Q = np.random.normal(0, scale=scale, size=(self.n_products, self.K))

        self.b_u = np.zeros(self.n_users)
        self.b_i = np.zeros(self.n_products)
        self.global_mean = 0.0

        self.train_rmse_history = []
        self.test_rmse_history = []

    def denormalize(self, values):
        if not self.use_standardized:
            return values
        return values * self.target_std + self.target_mean

    def baseline(self, users, products):
        if not self._use_baseline:
            return 0.0
        return self.rating_baseline_mean + self.base_user_bias[users] + self.base_item_bias[products]

    def prepare_targets(self, ratings, users=None, products=None):
        targets = ratings.astype(np.float64)
        if self._use_baseline and users is not None and products is not None:
            targets = targets - self.baseline(users, products)
        if not self.use_standardized:
            return targets
        if not self._manual_target_stats:
            self.target_mean = float(targets.mean())
            std = float(targets.std())
            self.target_std = std if std > 1e-8 else 1.0
        return (targets - self.target_mean) / self.target_std

    def fit(self, train_data, test_data=None, verbose=True):
        if verbose:
            print("\n" + "─" * 72)
            print(f"» Huấn luyện Matrix Factorization | k={self.K}, lr={self.lr}, reg={self.reg}")
            print("─" * 72)
            header = f"{'epoch':>7} | {'train':>8} | {'test':>8} | {'Δt(s)':>7}"
            print(header)
            print("-" * len(header))
        users = train_data[:, 0].astype(np.int32)
        items = train_data[:, 1].astype(np.int32)
        ratings_raw = train_data[:, 2].astype(np.float64)
        if train_data.shape[1] > 5:
            weights = train_data[:, 5].astype(np.float64)
        elif train_data.shape[1] > 4:
            weights = train_data[:, 4].astype(np.float64)
        else:
            weights = np.ones_like(ratings_raw)
        targets = self.prepare_targets(ratings_raw, users, items)
        self.global_mean = float(targets.mean()) if targets.size else 0.0
        count = train_data.shape[0]
        best_state = None
        best_metric = float("inf")
        patience_left = self.patience

        for epoch in range(self.epochs):
            start = time.time()
            order = np.arange(count)
            np.random.shuffle(order)
            for idx in order:
                u = users[idx]
                i = items[idx]
                err = targets[idx] - (self.global_mean + self.b_u[u] + self.b_i[i] + np.dot(self.P[u], self.Q[i]))
                err *= weights[idx]
                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])
                prev = self.P[u].copy()
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * prev)
                self.Q[i] += self.lr * (err * prev - self.reg * self.Q[i])

            train_rmse = self.evaluate(train_data)
            self.train_rmse_history.append(train_rmse)
            score = None
            if test_data is not None:
                score = self.evaluate(test_data)
                self.test_rmse_history.append(score)
            if verbose:
                test_disp = f"{score:.4f}" if score is not None else "—"
                print(f"{epoch + 1:7d}/{self.epochs:<3} | {train_rmse:>8.4f} | {test_disp:>8} | {time.time() - start:>7.2f}")

            if self.early_stopping and test_data is not None:
                metric = score if score is not None else train_rmse
                if metric < best_metric - 1e-4:
                    best_metric = metric
                    patience_left = self.patience
                    best_state = (self.P.copy(), self.Q.copy(), self.b_u.copy(), self.b_i.copy(), self.global_mean)
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        if verbose:
                            print("-" * len(header))
                            print(f"↳ Early stopping tại epoch {epoch + 1}")
                        if best_state is not None:
                            self.P, self.Q, self.b_u, self.b_i, self.global_mean = best_state
                        break
        if verbose:
            print("─" * 72)

    def evaluate(self, data):
        if data.size == 0:
            return 0.0
        users = data[:, 0].astype(np.int32)
        items = data[:, 1].astype(np.int32)
        targets = data[:, 2].astype(np.float64)
        preds = self.global_mean + self.b_u[users] + self.b_i[items] + np.sum(self.P[users] * self.Q[items], axis=1)
        preds = self.denormalize(preds)
        if self._use_baseline:
            preds += self.baseline(users, items)
        err = targets - preds
        return float(np.sqrt(np.mean(err * err)))

    def predict(self, user_id, product_id):
        if user_id >= self.n_users or product_id >= self.n_products:
            return float(self.denormalize(self.global_mean))
        value = self.global_mean + self.b_u[user_id] + self.b_i[product_id] + np.dot(self.P[user_id], self.Q[product_id])
        value = float(self.denormalize(value))
        if self._use_baseline:
            value += float(self.rating_baseline_mean + self.base_user_bias[user_id] + self.base_item_bias[product_id])
        return value

    def recommend(self, user_id, top_k=5):
        if user_id >= self.n_users:
            return []
        scores = self.global_mean + self.b_u[user_id] + self.b_i + np.dot(self.P[user_id], self.Q.T)
        scores = self.denormalize(scores)
        if self._use_baseline:
            scores = scores + self.rating_baseline_mean + self.base_user_bias[user_id] + self.base_item_bias
        return np.argsort(scores)[-top_k:][::-1]


def build_exclude_dict(user_indices: np.ndarray, item_indices: np.ndarray) -> dict:
    user_indices = np.asarray(user_indices, dtype=np.int32)
    item_indices = np.asarray(item_indices, dtype=np.int32)
    if user_indices.size == 0:
        return {}
    order = np.argsort(user_indices)
    u_sorted = user_indices[order]
    i_sorted = item_indices[order]
    uniq, counts = np.unique(u_sorted, return_counts=True)
    ends = np.cumsum(counts)
    starts = np.concatenate(([0], ends[:-1]))
    out = {}
    for u, s, e in zip(uniq, starts, ends):
        out[int(u)] = set(map(int, i_sorted[s:e]))
    return out

def temporal_split(matrix: np.ndarray, train_ratio: float = 0.8):
    if matrix.size == 0:
        return np.array([], dtype=bool), np.array([], dtype=bool)
    n = len(matrix)
    train_ratio = float(np.clip(train_ratio, 0.0, 1.0))
    train_cut = int(round(n * train_ratio))
    timestamps = matrix[:, 3].astype(np.int64)
    order = np.argsort(timestamps)
    train_idx = order[:train_cut]
    train_mask = np.zeros(n, dtype=bool)
    train_mask[train_idx] = True
    test_mask = ~train_mask
    return train_mask, test_mask


def temporal_split_blocks(payload: dict, train_ratio: float = 0.8, verbose: bool = False):
    interactions = payload['interactions']
    train_mask, test_mask = temporal_split(interactions, train_ratio)

    def subset(mask):
        return {
            'users': interactions[mask, 0].astype(np.int32),
            'items': interactions[mask, 1].astype(np.int32),
            'timestamps': interactions[mask, 3].astype(np.int64),
            'weights': interactions[mask, 4].astype(np.float32),
            'ratings_raw': payload['ratings_raw'][mask],
            'product_means': payload['product_means'][mask],
        }

    train_block = subset(train_mask)
    test_block = subset(test_mask)

    stats = {
        'train_interactions': train_block['ratings_raw'].size,
        'test_interactions': test_block['ratings_raw'].size,
        'train_users': len(np.unique(train_block['users'])),
        'test_users': len(np.unique(test_block['users'])),
        'train_items': len(np.unique(train_block['items'])),
        'test_items': len(np.unique(test_block['items'])),
        'train_weight_min': float(train_block['weights'].min()) if train_block['weights'].size else 0.0,
        'train_weight_max': float(train_block['weights'].max()) if train_block['weights'].size else 0.0,
        'test_weight_min': float(test_block['weights'].min()) if test_block['weights'].size else 0.0,
        'test_weight_max': float(test_block['weights'].max()) if test_block['weights'].size else 0.0,
        'train_ratio': train_ratio,
    }

    if verbose:
        train_pct = int(round(train_ratio * 100))
        print(f"Split stats (temporal {train_pct}/{100 - train_pct}):")
        print(f"  Train interactions: {stats['train_interactions']:,}")
        print(f"  Test interactions:  {stats['test_interactions']:,}")
        print(f"  Users train/test: {stats['train_users']:,} / {stats['test_users']:,}")
        print(f"  Items train/test: {stats['train_items']:,} / {stats['test_items']:,}")
        print(f"  Time weight range (train): {stats['train_weight_min']:.4f} – {stats['train_weight_max']:.4f}")
        print(f"  Time weight range (test):  {stats['test_weight_min']:.4f} – {stats['test_weight_max']:.4f}")

    return train_block, test_block, stats

__all__ = [
    "BaselineModel",
    "MatrixFactorizationModel",
    "build_exclude_dict",
    "temporal_split",
    "temporal_split_blocks",
]
