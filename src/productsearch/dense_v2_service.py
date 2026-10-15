"""Serve the validation-selected v2 artifact while preserving the v1 comparison."""
import json
import threading
import time

import joblib
import numpy as np
import pandas as pd

from .dense import Encoder, reciprocal_rank_fusion, top_k
from .dense_service import DenseSearchService
from .metrics import ndcg


class DenseV2SearchService:
    product = DenseSearchService.product

    def __init__(self, root):
        self.root = root
        self.selection = json.loads((root / "models/dense_v2/selection.json").read_text("utf-8"))
        self.config = self.selection["config"]
        self.catalog = pd.read_parquet(root / "data/dense_v2/catalog.parquet")
        self.bm25 = joblib.load(root / "models/dense_v2/bm25.joblib")
        self.test = pd.read_parquet(root / "data/dense_v2/test.parquet")
        self.queries = self.test.groupby("query_group", sort=True).agg(query=("query", "first"), candidates=("product_id", "size")).reset_index()
        self.positions = {key: i for i, key in enumerate(self.queries.query_group)}
        self.vectors, self.encoders, self.test_vectors = {}, {}, {}
        self.lock = threading.Lock()

    def artifacts(self, key):
        if key not in self.vectors:
            self.vectors[key] = np.load(self.root / self.selection["vector_paths"][key], mmap_mode="r")
            if len(self.vectors[key]) != len(self.catalog):
                raise ValueError("改进实验索引与商品库不一致。")
            self.encoders[key] = Encoder(self.root / self.selection["encoder_paths"][key], device="cpu", max_length=self.config["max_length"])
            self.test_vectors[key] = np.load(self.root / f"models/dense_v2/{key}_test_queries.npy")
        return self.vectors[key], self.encoders[key]

    def search(self, query, model, limit=12):
        started = time.perf_counter()
        key = "pretrained" if model == "dense_v2_pretrained" else self.selection["serving_key"] if model == "hybrid_v2" else self.selection["trained_key"]
        with self.lock:
            vectors, encoder = self.artifacts(key)
            scores = vectors @ encoder.encode([query])[0]
        ids = top_k(scores, self.config["retrieval_depth"])
        values = scores[ids]
        if model == "hybrid_v2":
            sparse_scores = self.bm25.scores(query)
            matched = np.flatnonzero(sparse_scores > 0)
            sparse = matched[top_k(sparse_scores[matched], self.config["retrieval_depth"])]
            alpha = self.selection["policies"][key]["alpha"]
            ids, values = reciprocal_rank_fusion(sparse, ids, alpha, self.config["rrf_k"], self.config["retrieval_depth"])
            protocol = f"v2 固定商品库，BM25 + {'未微调' if key == 'pretrained' else '改进微调'}编码器，RRF α={alpha:g}；方案只按验证集召回选择。"
        else:
            protocol = f"v2 固定商品库，{'未微调' if key == 'pretrained' else '改进微调'}编码器余弦检索；与 v1 商品库不同。"
        return {"query": query, "model": model, "encoder_key": key,
                "results": [self.product(row, query, score, i + 1) for i, (row, score) in enumerate(zip(ids[:limit], values[:limit]))],
                "matched_products": len(self.catalog), "reranked_candidates": len(ids), "catalog_products": len(self.catalog),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1), "protocol": protocol}

    def benchmark(self, query_group):
        group = self.test[self.test.query_group == query_group]
        if group.empty:
            raise KeyError(query_group)
        query, ids = str(group.iloc[0]["query"]), group.catalog_row.to_numpy()
        scores = {"bm25": self.bm25.scores(query, ids)}
        with self.lock:
            for model, key in [("dense_v2_pretrained", "pretrained"), ("dense_v2", self.selection["trained_key"])]:
                vectors, _ = self.artifacts(key)
                scores[model] = vectors[ids] @ self.test_vectors[key][self.positions[query_group]]
        rankings = {}
        for model, values in scores.items():
            products = []
            for rank, i in enumerate(top_k(values, 10), 1):
                p = self.product(ids[i], query, values[i], rank)
                p["label"] = str(group.iloc[i].esci_label)
                products.append(p)
            rankings[model] = {"results": products, "ndcg_at_10": ndcg(group.label, values, group.product_id)}
        return {"query": query, "query_group": query_group, "candidates": len(group), "rankings": rankings,
                "protocol": "v2 新测试查询的标注候选排序，与全库 Recall 分开。"}
