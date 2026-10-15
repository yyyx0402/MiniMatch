"""Retrieval-aligned experimental helpers; dense_v1 remains immutable."""
import numpy as np
import torch
import torch.nn.functional as F

from .dense import top_k
from .metrics import GAINS, ndcg


def freeze_encoder(model, bottom_layers=0):
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if bottom_layers:
        for parameter in model.embeddings.parameters():
            parameter.requires_grad_(False)
        for layer in model.encoder.layer[:bottom_layers]:
            for parameter in layer.parameters():
                parameter.requires_grad_(False)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def contrastive_weights(query_keys, product_rows, judgments, unknown_weight=.1):
    """Deduplicated documents; all known E positives, known S/C masked.

    Unknown query/document relations are explicitly weak training assumptions.
    They never become evaluation labels or mined supervised I judgments.
    """
    weights = np.full((len(query_keys), len(product_rows)), unknown_weight, dtype=np.float32)
    positives = np.zeros_like(weights, dtype=bool)
    for i, key in enumerate(query_keys):
        labels = judgments[key]
        for j, row in enumerate(product_rows):
            label = labels.get(int(row))
            if label == 3:
                positives[i, j] = True
                weights[i, j] = 1
            elif label in (1, 2):
                weights[i, j] = 0
            elif label == 0:
                weights[i, j] = 1
    return positives, weights


def multi_positive_loss(query, documents, positive_mask, denominator_weights, temperature=.1):
    logits = query.float() @ documents.float().T / temperature
    positives = positive_mask.to(device=logits.device, dtype=torch.bool)
    weights = denominator_weights.to(device=logits.device, dtype=torch.float32)
    if not torch.all(positives.any(dim=1)):
        raise ValueError("Each query must have a known E positive in the batch")
    if torch.any(weights[positives] != 1):
        raise ValueError("Positive denominator weight must equal one")
    weighted = logits + weights.clamp_min(1e-30).log()
    weighted = weighted.masked_fill(weights == 0, -torch.inf)
    log_normalizer = torch.logsumexp(weighted, dim=1, keepdim=True)
    # Mean positive log likelihood treats all E answers as positives.
    positive_loss = (log_normalizer - logits).masked_fill(~positives, 0)
    return (positive_loss.sum(dim=1) / positives.sum(dim=1)).mean()


def anchor_loss(current, original):
    # Clone outside inference_mode so autograd may save the teacher target.
    return (1 - (current.float() * original.detach().clone().float()).sum(dim=1)).clamp_min(0).mean()


class RecallSelection:
    """Epoch zero is eligible; three non-improving checks stop training."""
    def __init__(self, baseline, patience=3, minimum_improvement=.0005):
        self.best = float(baseline)
        self.step = 0
        self.stale = 0
        self.patience = patience
        self.minimum_improvement = minimum_improvement

    def update(self, score, step):
        improved = float(score) > self.best + self.minimum_improvement
        if improved:
            self.best, self.step, self.stale = float(score), int(step), 0
        else:
            self.stale += 1
        return improved, self.stale >= self.patience


def retrieval_summary(frame, rankings):
    """Known-E recall and MRR; never treat unjudged products as irrelevant."""
    records = []
    for key, group in frame.groupby("query_group", sort=True):
        exact = set(map(int, group.loc[group.label == 3, "catalog_row"]))
        if not exact:
            continue
        ranking = list(map(int, rankings[key]))
        first = next((i for i, row in enumerate(ranking, 1) if row in exact), None)
        records.append({"query_group": key, "query": str(group.iloc[0]["query"]),
                        **{f"recall_at_{k}": len(set(ranking[:k]) & exact) / len(exact) for k in (50, 100, 200)},
                        "mrr_at_200": 0. if first is None else 1 / first})
    fields = ["recall_at_50", "recall_at_100", "recall_at_200", "mrr_at_200"]
    return {"queries_with_E": len(records), "queries_without_E": int(frame.query_group.nunique()) - len(records),
            **{field: float(np.mean([r[field] for r in records])) if records else 0. for field in fields}}, records


def candidate_summary(frame, qvectors, vectors):
    values = {}
    for i, (key, group) in enumerate(frame.groupby("query_group", sort=True)):
        values[key] = ndcg(group.label, vectors[group.catalog_row.to_numpy()] @ qvectors[i], group.product_id)
    return float(np.mean(list(values.values()))), values


def mine_ordinal(frame, bm25, config):
    rng = np.random.default_rng(config["split_seed"])
    pairs = []
    for qi, (key, group) in enumerate(frame.groupby("query_group", sort=True)):
        rows = group.product_row.to_numpy()
        scores = bm25.scores(str(group.iloc[0]["query"]), rows)
        labels = group.label.to_numpy()
        positives, negatives = np.flatnonzero(labels == 3), np.flatnonzero(labels == 0)
        hard = negatives[np.argsort(-scores[negatives], kind="stable")[:3]]
        local = [(int(pos), int(rng.choice(hard))) for pos in rng.permutation(positives)[:4]] if len(hard) else []
        for high, low in [(3, 2), (2, 0), (3, 1), (2, 1), (1, 0)]:
            p, n = np.flatnonzero(labels == high), np.flatnonzero(labels == low)
            if len(p) and len(n):
                local.append((int(rng.choice(p)), int(n[np.argmax(scores[n])])))
        for pos, neg in local[:config["pairs_per_query"]]:
            pairs.append({"query_group": key, "query_index": qi, "positive": int(rows[pos]), "negative": int(rows[neg]),
                          "positive_label": int(labels[pos]), "negative_label": int(labels[neg]),
                          "weight": float(GAINS[labels[pos]] - GAINS[labels[neg]])})
    import pandas as pd
    result = pd.DataFrame(pairs).drop_duplicates(["query_group", "positive", "negative"])
    if result.empty:
        raise ValueError("No ordinal pairs")
    return result.reset_index(drop=True)


def dense_rankings(keys, qvectors, vectors, depth=200, device="cuda"):
    result = {}
    catalog = torch.from_numpy(np.array(vectors, copy=True)).to(device)
    with torch.inference_mode():
        for start in range(0, len(keys), 32):
            scores = (torch.from_numpy(qvectors[start:start + 32]).to(device) @ catalog.T).cpu().numpy()
            result.update((key, top_k(score, depth)) for key, score in zip(keys[start:start + 32], scores))
    del catalog
    return result
