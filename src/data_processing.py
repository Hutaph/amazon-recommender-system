import numpy as np


def load_raw_data(path: str):
    with open(path, "r", encoding="utf-8") as f:
        headers = f.readline().strip().split(",")

    raw = np.genfromtxt(path, delimiter=",", dtype=str, skip_header=1)

    if raw.ndim != 2 or raw.shape[1] < 4:
        raise ValueError(f"Invalid raw shape: {raw.shape}")

    return raw, tuple(headers[:4])


def load_processed_data(processed_dir: str):
    ratings_path = processed_dir + "/ratings_processed.csv"
    user_map_path = processed_dir + "/user_mapping.csv"
    product_map_path = processed_dir + "/product_mapping.csv"

    with open(ratings_path, "r", encoding="utf-8") as f:
        headers = f.readline().strip().split(",")
        header_idx = {name: idx for idx, name in enumerate(headers)}
        required = ["UserId", "ProductId", "Rating", "Timestamp"]

        for col in required:
            if col not in header_idx:
                raise ValueError(f'Missing column "{col}" in processed ratings file.')

        has_cols = {
            "TimestampDT": "TimestampDT" in header_idx,
            "TimeWeight": "TimeWeight" in header_idx,
            "ProductMean": "ProductMean" in header_idx,
        }

        rows = [line.strip().split(",") for line in f if line.strip()]

    n = len(rows)

    if n == 0:
        raise ValueError("Processed ratings file is empty.")

    user_ids = np.empty(n, dtype="U50")
    product_ids = np.empty(n, dtype="U50")
    ratings = np.zeros(n, dtype=np.float64)
    timestamps = np.zeros(n, dtype=np.int64)
    timestamp_dt = np.empty(n, dtype="U32")
    time_weights = np.ones(n, dtype=np.float64)
    product_means = np.zeros(n, dtype=np.float64)

    missing_tsdt = not has_cols["TimestampDT"]
    missing_timeweight = not has_cols["TimeWeight"]
    missing_productmean = not has_cols["ProductMean"]

    for i, row in enumerate(rows):
        user_ids[i] = row[header_idx["UserId"]]
        product_ids[i] = row[header_idx["ProductId"]]
        ratings[i] = float(row[header_idx["Rating"]])
        timestamps[i] = int(float(row[header_idx["Timestamp"]]))

        if has_cols["TimestampDT"]:
            timestamp_dt[i] = row[header_idx["TimestampDT"]]

        if has_cols["TimeWeight"]:
            time_weights[i] = float(row[header_idx["TimeWeight"]])

        if has_cols["ProductMean"]:
            product_means[i] = float(row[header_idx["ProductMean"]])

    timestamp_epoch = timestamps.astype(np.int64)

    if missing_tsdt:
        timestamp_dt = np.datetime_as_string(
            timestamp_epoch.astype("datetime64[s]"),
            unit="s",
        )

    if missing_timeweight:
        time_weights = _compute_time_weights(timestamp_epoch, 0.2, 1.0)

    if missing_productmean:
        product_ids_unique, product_indices = np.unique(
            product_ids,
            return_inverse=True,
        )

        sums = np.bincount(
            product_indices,
            weights=ratings,
            minlength=len(product_ids_unique),
        ).astype(np.float64)

        counts = np.bincount(
            product_indices,
            minlength=len(product_ids_unique),
        ).astype(np.float64)

        global_mean = float(ratings.mean()) if ratings.size else 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            product_lookup = np.divide(
                sums,
                np.maximum(counts, 1.0),
                out=np.full(len(product_ids_unique), global_mean, dtype=np.float64),
                where=counts > 0,
            )

        product_means = product_lookup[product_indices]

    minimal_header = [
        "UserId",
        "ProductId",
        "Rating",
        "Timestamp",
        "TimestampDT",
        "TimeWeight",
        "ProductMean",
    ]

    needs_rewrite = (
        missing_tsdt
        or missing_timeweight
        or missing_productmean
        or headers != minimal_header
    )

    dtype = np.dtype(
        [
            ("UserId", "U50"),
            ("ProductId", "U50"),
            ("Rating", "f8"),
            ("Timestamp", "i8"),
            ("TimestampDT", "U32"),
            ("TimeWeight", "f8"),
            ("ProductMean", "f8"),
        ]
    )

    data = np.zeros(n, dtype=dtype)
    data["UserId"] = user_ids
    data["ProductId"] = product_ids
    data["Rating"] = ratings
    data["Timestamp"] = timestamps
    data["TimestampDT"] = timestamp_dt
    data["TimeWeight"] = time_weights
    data["ProductMean"] = product_means

    if needs_rewrite:
        with open(ratings_path, "w", encoding="utf-8") as f:
            f.write(",".join(minimal_header) + "\n")

            for i in range(n):
                f.write(
                    f"{user_ids[i]},"
                    f"{product_ids[i]},"
                    f"{ratings[i]:.6f},"
                    f"{int(timestamps[i])},"
                    f"{timestamp_dt[i]},"
                    f"{time_weights[i]:.6f},"
                    f"{product_means[i]:.6f}\n"
                )

    user_mapping = {}

    with open(user_map_path, "r", encoding="utf-8") as f:
        f.readline()

        for line in f:
            parts = line.strip().split(",")

            if len(parts) == 2:
                user_mapping[int(parts[0])] = parts[1]

    product_mapping = {}

    with open(product_map_path, "r", encoding="utf-8") as f:
        f.readline()

        for line in f:
            parts = line.strip().split(",")

            if len(parts) == 2:
                product_mapping[int(parts[0])] = parts[1]

    return data, user_mapping, product_mapping


def clean_raw(raw: np.ndarray) -> np.ndarray:
    user_col = np.char.strip(raw[:, 0])
    product_col = np.char.strip(raw[:, 1])
    rating_col = np.char.strip(raw[:, 2])
    ts_col = np.char.strip(raw[:, 3])

    valid_user = user_col != ""
    valid_product = product_col != ""

    rating_base = np.char.replace(np.char.lstrip(rating_col, "+-"), ".", "")
    rating_digits = np.char.isdecimal(rating_base)

    ratings = np.full(rating_col.shape, np.nan, dtype=np.float64)

    if rating_digits.any():
        ratings[rating_digits] = rating_col[rating_digits].astype(np.float64)

    valid_rating = rating_digits & (ratings >= 1.0) & (ratings <= 5.0)

    ts_base = np.char.lstrip(ts_col, "+")
    ts_digits = np.char.isdecimal(ts_base)

    timestamps = np.zeros(ts_col.shape, dtype=np.int64)

    if ts_digits.any():
        timestamps[ts_digits] = ts_col[ts_digits].astype(np.int64)

    valid_ts = ts_digits & (timestamps > 0)

    mask = valid_user & valid_product & valid_rating & valid_ts

    dtype = [
        ("UserId", "U32"),
        ("ProductId", "U32"),
        ("Rating", "i4"),
        ("Timestamp", "i8"),
    ]

    out = np.zeros(mask.sum(), dtype=dtype)
    out["UserId"] = user_col[mask]
    out["ProductId"] = product_col[mask]
    out["Rating"] = ratings[mask].astype("i4")
    out["Timestamp"] = timestamps[mask]

    return out


def deduplicate_latest(data: np.ndarray) -> np.ndarray:
    keys = np.char.add(np.char.add(data["UserId"], "_"), data["ProductId"])
    ts = data["Timestamp"].astype(np.int64)

    order = np.lexsort((-ts, keys))
    keys_sorted = keys[order]

    _, first_idx = np.unique(keys_sorted, return_index=True)

    return data[order][first_idx]


def kcore_filter(data: np.ndarray, k: int = 5, max_iterations: int = 10) -> np.ndarray:
    print(f"  K-core filtering (k={k})...")

    current = data.copy()

    for iteration in range(max_iterations):
        before_count = len(current)

        users, user_counts = np.unique(current["UserId"], return_counts=True)
        items, item_counts = np.unique(current["ProductId"], return_counts=True)

        valid_users = set(users[user_counts >= k])
        valid_items = set(items[item_counts >= k])

        user_mask = np.isin(current["UserId"], list(valid_users))
        item_mask = np.isin(current["ProductId"], list(valid_items))
        current = current[user_mask & item_mask]

        removed = before_count - len(current)

        print(
            f"    Iteration {iteration + 1}: "
            f"{before_count:,} -> {len(current):,} (-{removed:,} ratings)"
        )

        if removed == 0:
            print(f"    Converged after {iteration + 1} iterations")
            break

    final_users = len(np.unique(current["UserId"]))
    final_items = len(np.unique(current["ProductId"]))

    print(
        f"  Final: {len(current):,} ratings, {final_users:,} users, {final_items:,} items"
    )

    print(
        f"  Removed: {len(data) - len(current):,} ratings "
        f"({100 * (len(data) - len(current)) / len(data):.1f}%)"
    )

    return current


def to_datetime_minimal(data: np.ndarray) -> np.ndarray:
    timestamps_dt = data["Timestamp"].astype("int64").astype("datetime64[s]")

    dtype = [
        ("UserId", "U32"),
        ("ProductId", "U32"),
        ("Rating", "i4"),
        ("Timestamp", "U32"),
        ("TimestampDT", "datetime64[s]"),
        ("Year", "i4"),
        ("Month", "i4"),
        ("Day", "i4"),
    ]

    out = np.zeros(len(data), dtype=dtype)
    out["UserId"] = data["UserId"]
    out["ProductId"] = data["ProductId"]
    out["Rating"] = data["Rating"]
    out["Timestamp"] = data["Timestamp"]
    out["TimestampDT"] = timestamps_dt
    out["Year"] = timestamps_dt.astype("datetime64[Y]").astype(int) + 1970
    out["Month"] = timestamps_dt.astype("datetime64[M]").astype(int) % 12 + 1
    out["Day"] = (
        timestamps_dt.astype("datetime64[D]").astype(int)
        - timestamps_dt.astype("datetime64[M]").astype(int)
        + 1
    )

    return out


def _minmax_scale(values: np.ndarray, min_value: float = 0.0, max_value: float = 1.0) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return np.array([], dtype=np.float64)

    vmin = float(values.min()) if values.size else 0.0
    vmax = float(values.max()) if values.size else 0.0

    if vmax - vmin < 1e-9:
        return np.full_like(values, (min_value + max_value) / 2.0, dtype=np.float64)

    norm = (values - vmin) / (vmax - vmin)

    return min_value + (max_value - min_value) * norm


def _compute_time_weights(timestamps_epoch: np.ndarray, min_weight: float = 0.2, max_weight: float = 1.0) -> np.ndarray:
    if timestamps_epoch.size == 0:
        return np.array([], dtype=np.float64)

    return _minmax_scale(timestamps_epoch.astype(np.float64), min_weight, max_weight)


def encode_ids(data_minimal: np.ndarray):
    if not {"UserId", "ProductId", "Rating", "TimestampDT"}.issubset(
        data_minimal.dtype.names
    ):
        raise ValueError("Unexpected schema for encode_ids.")

    ts_epoch = data_minimal["TimestampDT"].astype("datetime64[s]").astype("int64")
    user_ids, user_inv = np.unique(data_minimal["UserId"], return_inverse=True)
    product_ids, product_inv = np.unique(
        data_minimal["ProductId"],
        return_inverse=True,
    )

    dtype = [
        ("UserId", "U32"),
        ("ProductId", "U32"),
        ("UserIndex", "int32"),
        ("ProductIndex", "int32"),
        ("Rating", "i4"),
        ("Timestamp", "U32"),
        ("TimestampDT", "datetime64[s]"),
        ("TimestampEpoch", "int64"),
    ]

    mapped = np.zeros(len(data_minimal), dtype=dtype)
    mapped["UserId"] = data_minimal["UserId"]
    mapped["ProductId"] = data_minimal["ProductId"]
    mapped["UserIndex"] = user_inv.astype("int32")
    mapped["ProductIndex"] = product_inv.astype("int32")
    mapped["Rating"] = data_minimal["Rating"].astype("i4")
    mapped["Timestamp"] = data_minimal["Timestamp"]
    mapped["TimestampDT"] = data_minimal["TimestampDT"]
    mapped["TimestampEpoch"] = ts_epoch

    return mapped, user_ids, product_ids


def encode_interactions_for_modeling(data: np.ndarray, idx_to_user_id: dict[int, str], idx_to_product_id: dict[int, str]):
    user_id_to_idx = {uid: idx for idx, uid in idx_to_user_id.items()}
    product_id_to_idx = {pid: idx for idx, pid in idx_to_product_id.items()}

    user_indices = np.array([user_id_to_idx[uid] for uid in data["UserId"]], dtype=np.int32)
    item_indices = np.array([product_id_to_idx[pid] for pid in data["ProductId"]], dtype=np.int32)

    ratings = data["Rating"].astype(np.float32)
    timestamps = data["Timestamp"].astype(np.int64)
    time_weights = data["TimeWeight"].astype(np.float32)
    product_means = data["ProductMean"].astype(np.float32)

    interactions = np.column_stack(
        [user_indices, item_indices, ratings, timestamps, time_weights]
    ).astype(np.float64)

    stats = {
        "rating_mean": float(ratings.mean()) if ratings.size else 0.0,
        "rating_std": float(ratings.std()) if ratings.size else 0.0,
        "time_weight_min": float(time_weights.min()) if time_weights.size else 0.0,
        "time_weight_max": float(time_weights.max()) if time_weights.size else 0.0,
        "product_mean_min": float(product_means.min()) if product_means.size else 0.0,
        "product_mean_max": float(product_means.max()) if product_means.size else 0.0,
    }

    payload = {
        "interactions": interactions,
        "ratings_raw": ratings,
        "time_weights": time_weights,
        "product_means": product_means,
        "n_users": len(idx_to_user_id),
        "n_items": len(idx_to_product_id),
    }

    return payload, stats


def standardize_blocks(train_block: dict, test_block: dict, n_items: int):
    rating_mean = (
        float(train_block["ratings_raw"].mean())
        if train_block["ratings_raw"].size
        else 0.0
    )

    rating_std = (
        float(train_block["ratings_raw"].std())
        if train_block["ratings_raw"].size
        else 1.0
    )

    if rating_std < 1e-8:
        rating_std = 1.0

    train_std = (train_block["ratings_raw"] - rating_mean) / rating_std
    test_std = (test_block["ratings_raw"] - rating_mean) / rating_std

    item_sum = np.bincount(
        train_block["items"],
        weights=train_std,
        minlength=n_items,
    ).astype(np.float32)

    item_count = np.bincount(train_block["items"], minlength=n_items).astype(np.int32)

    item_means = np.zeros(n_items, dtype=np.float32)
    valid = item_count > 0
    item_means[valid] = item_sum[valid] / item_count[valid]

    train_prod_std = item_means[train_block["items"]]
    test_prod_std = item_means[test_block["items"]]

    def build_matrix(block, std_ratings):
        return np.column_stack(
            [block["users"], block["items"], std_ratings, block["timestamps"], block["weights"]]
        ).astype(np.float64)

    return {
        "rating_mean": rating_mean,
        "rating_std": rating_std,
        "train_ratings_std": train_std,
        "test_ratings_std": test_std,
        "train_product_means_std": train_prod_std,
        "test_product_means_std": test_prod_std,
        "train_matrix": build_matrix(train_block, train_std),
        "test_matrix": build_matrix(test_block, test_std),
    }


def split_train_test(mapped: np.ndarray, train_ratio: float = 0.8, return_masks: bool = False):
    if "Timestamp" not in mapped.dtype.names:
        raise ValueError("Mapped array missing Timestamp field for split.")

    n = len(mapped)

    if n == 0:
        if return_masks:
            return mapped[:0], mapped[:0], np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)

        return mapped[:0], mapped[:0]

    train_ratio = float(np.clip(train_ratio, 0.0, 1.0))
    num_train = int(round(n * train_ratio))

    timestamps = mapped["Timestamp"].astype(np.int64)
    sorted_idx = np.argsort(timestamps)
    data_sorted = mapped[sorted_idx]

    train_data = data_sorted[:num_train]
    test_data = data_sorted[num_train:]

    if not return_masks:
        return train_data, test_data

    train_mask = np.zeros(n, dtype=bool)
    train_mask[sorted_idx[:num_train]] = True
    test_mask = ~train_mask

    return train_data, test_data, train_mask, test_mask


def save_processed(mapped: np.ndarray, user_ids: np.ndarray, product_ids: np.ndarray, output_dir: str, use_datetime: bool = True):
    ratings_fp = output_dir.rstrip("/") + "/ratings_processed.csv"
    user_fp = output_dir.rstrip("/") + "/user_mapping.csv"
    product_fp = output_dir.rstrip("/") + "/product_mapping.csv"

    if "Timestamp" not in mapped.dtype.names:
        raise ValueError("Mapped array missing Timestamp field for serialization.")

    if use_datetime and "TimestampDT" not in mapped.dtype.names:
        raise ValueError("Mapped array missing TimestampDT field for serialization.")

    if "TimestampEpoch" not in mapped.dtype.names:
        raise ValueError("Mapped array missing TimestampEpoch for feature computation.")

    timestamp_epoch = mapped["TimestampEpoch"].astype(np.int64)
    ratings = mapped["Rating"].astype(np.float64)

    rating_mean = float(ratings.mean()) if ratings.size else 0.0
    rating_std = float(ratings.std()) if ratings.size else 1.0

    time_weights = _compute_time_weights(timestamp_epoch, 0.2, 1.0)

    if "ProductIndex" not in mapped.dtype.names:
        raise ValueError("Mapped array missing ProductIndex for product-level features.")

    product_indices = mapped["ProductIndex"].astype(np.int64)
    product_count = len(product_ids)

    product_sum = np.bincount(
        product_indices,
        weights=ratings,
        minlength=product_count,
    ).astype(np.float64)

    product_cnt = np.bincount(product_indices, minlength=product_count).astype(np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):
        product_mean_lookup = np.divide(
            product_sum,
            np.maximum(product_cnt, 1.0),
            out=np.full(product_count, rating_mean, dtype=np.float64),
            where=product_cnt > 0,
        )

    product_mean_per_entry = product_mean_lookup[product_indices]

    if use_datetime:
        timestamps_dt = mapped["TimestampDT"].astype("datetime64[s]")
        dt_strings = np.datetime_as_string(timestamps_dt, unit="s")
    else:
        dt_strings = mapped["Timestamp"].astype(str)
        timestamps_dt = mapped["Timestamp"].astype("datetime64[s]")

    with open(ratings_fp, "w", encoding="utf-8") as f:
        f.write("UserId,ProductId,Rating,Timestamp,TimestampDT,TimeWeight,ProductMean\n")

        batch = 250000
        total = len(mapped)

        for start in range(0, total, batch):
            end = min(start + batch, total)

            lines = [
                f"{mapped['UserId'][i]},"
                f"{mapped['ProductId'][i]},"
                f"{ratings[i]:.6f},"
                f"{int(timestamp_epoch[i])},"
                f"{dt_strings[i]},"
                f"{time_weights[i]:.6f},"
                f"{product_mean_per_entry[i]:.6f}\n"
                for i in range(start, end)
            ]

            f.writelines(lines)

    with open(user_fp, "w", encoding="utf-8") as f:
        f.write("UserIndex,UserId\n")

        for idx, uid in enumerate(user_ids):
            f.write(f"{idx},{uid}\n")

    with open(product_fp, "w", encoding="utf-8") as f:
        f.write("ProductIndex,ProductId\n")

        for idx, pid in enumerate(product_ids):
            f.write(f"{idx},{pid}\n")

    files = {
        "ratings": ratings_fp,
        "user_mapping": user_fp,
        "product_mapping": product_fp,
    }

    files["stats"] = {
        "rating_mean": rating_mean,
        "rating_std": rating_std,
        "time_weight_mean": float(time_weights.mean()) if time_weights.size else 0.0,
        "product_mean_std": float(product_mean_per_entry.std())
        if product_mean_per_entry.size
        else 0.0,
    }

    return files
