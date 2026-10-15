import numpy as np


def known_e_recall(ranking, exact_ids, k):
    exact = set(map(int, exact_ids))
    if not exact:
        return None
    return len(set(map(int, ranking[:k])) & exact) / len(exact)


def evaluate_retrieval(frame, rankings):
    rows = []
    for query_group, group in frame.groupby("query_group", sort=True):
        exact = group.loc[group.label == 3, "catalog_row"].to_numpy()
        if not len(exact):
            continue
        ranking = rankings[query_group]
        rows.append({"query_group": query_group, "query": str(group.iloc[0]["query"]),
                     **{f"recall_at_{k}": known_e_recall(ranking, exact, k) for k in (50, 100, 200)}})
    return {"queries_with_E": len(rows), "queries_without_E": int(frame.query_group.nunique())-len(rows),
            **{f"recall_at_{k}": float(np.mean([r[f'recall_at_{k}'] for r in rows])) if rows else 0.0 for k in (50, 100, 200)}}, rows


def paired_interval(left, right, seed=42):
    keys = sorted(set(left) & set(right))
    delta = np.array([left[key]-right[key] for key in keys])
    rng = np.random.default_rng(seed)
    sample = rng.choice(delta, (2000, len(delta)), replace=True).mean(axis=1)
    return {"queries": len(keys), "mean_delta": float(delta.mean()),
            "ci95": np.quantile(sample, [.025, .975]).tolist(), "resamples": 2000, "seed": seed}
