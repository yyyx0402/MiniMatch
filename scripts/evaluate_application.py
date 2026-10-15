"""Evaluate the frozen serving model, without training or test-guided selection."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from productsearch.data import ROOT, normalize_query
from productsearch.dense import Encoder, reciprocal_rank_fusion, top_k
from productsearch.dense_v2 import candidate_summary, dense_rankings, retrieval_summary
from productsearch.release import MANIFEST, sha256, verify_release
from productsearch.retrieval_metrics import paired_interval


def main():
    output = ROOT / "reports/application"
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((ROOT / MANIFEST).read_text("utf-8"))
    manifest_hash = sha256(ROOT / MANIFEST)
    verify_release(ROOT, manifest)
    if (output / "metrics.json").exists():
        existing = json.loads((output / "metrics.json").read_text("utf-8"))
        if existing["manifest_sha256"] != manifest_hash:
            raise ValueError("Evaluation does not match the locked release")
        print("Frozen model evaluation already complete; no repeated selection")
        return
    torch.set_num_threads(4)
    # Previously opened official test queries are used as a disclosed regression
    # benchmark, not presented as a newly untouched holdout.
    test = pd.read_parquet(ROOT / "data/dense_v2/test.parquet")
    protocol = json.loads((ROOT / "data/dense_v2/protocol.json").read_text("utf-8"))
    excluded = set(protocol["queries"]["train_small"]) | set(protocol["queries"]["train_large"]) | set(protocol["queries"]["validation"])
    if set(test["query"].map(normalize_query)) & excluded:
        raise ValueError("Evaluation queries overlap training or validation")
    queries = test.groupby("query_group", sort=True).agg(query=("query", "first")).reset_index()
    bm25 = joblib.load(ROOT / manifest["bm25"])
    depth = manifest["retrieval_depth"]
    sparse = {}
    for row in queries.itertuples():
        scores = bm25.scores(row.query)
        matched = np.flatnonzero(scores > 0)
        sparse[row.query_group] = matched[top_k(scores[matched], depth)]
    report = {"model_name": manifest["model_name"], "manifest_sha256": manifest_hash,
              "test_sha256": sha256(ROOT / "data/dense_v2/test.parquet"),
              "evaluation_type": "previously_seen_test_regression",
              "evaluation_note": "These 1,000 official test queries were evaluated in earlier project work. Training and validation are disjoint, but this is not a newly untouched holdout. Model and fusion weights remain locked.",
              "query_count": len(queries), "catalog_products": bm25.matrix.shape[0],
              "retrieval": {}, "candidate_ndcg_at_10": {}, "parameters_changed": False,
              "inference": "CPU FP32 query encoding; saved normalized product vectors; full-catalog retrieval",
              "metrics_note": "Recall is against known E labels only. Candidate NDCG measures supplied labeled-list ranking, separately from full-catalog retrieval."}
    records = []
    def record(name, rankings):
        summary, rows = retrieval_summary(test, rankings)
        report["retrieval"][name] = summary
        records.extend({"model": name, **row} for row in rows)
        print(f"{name}: Recall@100={summary['recall_at_100']:.6f}", flush=True)
    record("keyword", sparse)
    candidate_rows = []
    keyword_scores = []
    from productsearch.metrics import ndcg
    for key, group in test.groupby("query_group", sort=True):
        value = ndcg(group.label, bm25.scores(str(group.iloc[0]["query"]), group.catalog_row.to_numpy()), group.product_id)
        keyword_scores.append(value)
        candidate_rows.append({"model": "keyword", "query_group": key, "ndcg_at_10": value})
    report["candidate_ndcg_at_10"]["keyword"] = float(np.mean(keyword_scores))
    for prefix, name in [("pretrained_", "minilm_pretrained"), ("", "minilm")]:
        print(f"Encoding queries for {name}", flush=True)
        encoder = Encoder(ROOT / manifest[prefix + "encoder"], device="cpu", max_length=manifest["max_length"])
        qvectors = encoder.encode(queries["query"].tolist(), batch_size=32)
        np.save(output / f"{name}_queries.npy", qvectors)
        vectors = np.load(ROOT / manifest[prefix + "vectors"], mmap_mode="r")
        rankings = dense_rankings(queries.query_group.tolist(), qvectors, vectors, depth, device="cpu")
        score, per_query = candidate_summary(test, qvectors, vectors)
        report["candidate_ndcg_at_10"][name] = score
        candidate_rows.extend({"model": name, "query_group": key, "ndcg_at_10": value} for key, value in per_query.items())
        record(name, rankings)
        alpha = manifest["pretrained_alpha"] if prefix else manifest["alpha"]
        hybrid = {key: reciprocal_rank_fusion(sparse[key], ranks, alpha, manifest["rrf_k"], depth)[0] for key, ranks in rankings.items()}
        record("hybrid_pretrained" if prefix else "hybrid", hybrid)
        del encoder, vectors, qvectors
    table = pd.DataFrame(records)
    pivot = table.pivot(index="query_group", columns="model", values="recall_at_100")
    report["paired_intervals"] = {
        "minilm_minus_pretrained": paired_interval(pivot.minilm.to_dict(), pivot.minilm_pretrained.to_dict()),
        "hybrid_minus_pretrained_hybrid": paired_interval(pivot.hybrid.to_dict(), pivot.hybrid_pretrained.to_dict())}
    if sha256(ROOT / MANIFEST) != manifest_hash:
        raise ValueError("Serving selection changed during evaluation")
    verify_release(ROOT, manifest)
    table.to_csv(output / "per_query_retrieval.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output / "per_query_ranking.csv", index=False)
    (output / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# 当前 MiniLM 固定模型评测", "", report["evaluation_note"], "", "| 方法 | Recall@100 |", "|---|---:|"]
    lines.extend(f"|{name}|{value['recall_at_100']:.4%}|" for name, value in report["retrieval"].items())
    for name, title in [("minilm_minus_pretrained", "微调 MiniLM − 未微调"),
                        ("hybrid_minus_pretrained_hybrid", "智能搜索 − 未微调混合")]:
        interval = report["paired_intervals"][name]
        low, high = interval["ci95"]
        lines.extend(["", f"{title}：{interval['mean_delta'] * 100:.4f} 个百分点，95% 配对区间 [{low * 100:.4f}, {high * 100:.4f}] 个百分点。"
                      + (" 区间包含零，不能声称增益具有明确统计证据。" if low <= 0 <= high else "")])
    lines.extend(["", "召回仅针对已标注 E 商品；候选列表 NDCG 另存，不代表全库排序质量。评测没有改变模型、参数、融合权重或训练产物。"])
    (output / "EVALUATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    with threadpool_limits(limits=4):
        main()
