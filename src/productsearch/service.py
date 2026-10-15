import json
import re
import time
import threading

import joblib
import numpy as np
import pandas as pd

from .data import ROOT
from .features import tokens
from .metrics import GAINS, ndcg

MODEL_NAMES = ("bm25", "linear", "lambdamart")


class SearchService:
    def __init__(self, root=ROOT):
        self.root = root
        self._dense = None
        self._dense_lock = threading.Lock()
        self._dense_v2 = None
        self._dense_v2_lock = threading.Lock()
        self.bundle = joblib.load(root / "models/baseline/bundle.joblib")
        self.features = self.bundle["features"]
        self.products = self.features.products
        self.test = pd.read_parquet(root / "data/processed/test.parquet")
        self.report = json.loads((root / "reports/baseline/metrics.json").read_text("utf-8"))
        self.queries = self.test.groupby("query_group", sort=True).agg(query=("query", "first"), candidates=("product_id", "size")).reset_index()

    def dense_available(self):
        return (self.root / "reports/dense_v1/metrics.json").exists()

    def dense(self):
        if not self.dense_available():
            raise ValueError("语义模型实验尚未完成，请先运行 scripts/train_dense.py。")
        with self._dense_lock:
            if self._dense is None:
                from .dense_service import DenseSearchService
                self._dense = DenseSearchService(self.root)
        return self._dense

    def scores(self, x, model):
        if model == "bm25":
            return x.bm25_all.to_numpy()
        if model == "linear":
            linear = self.bundle["linear"]
            return linear.predict_proba(x) @ GAINS[linear[-1].classes_]
        return self.bundle["ranker"].predict(x)

    def dense_v2_available(self):
        return (self.root / "reports/dense_v2/metrics.json").exists()

    def dense_v2(self):
        if not self.dense_v2_available():
            raise ValueError("改进实验尚未完成，请查看实验进度。")
        with self._dense_v2_lock:
            if self._dense_v2 is None:
                from .dense_v2_service import DenseV2SearchService
                self._dense_v2 = DenseV2SearchService(self.root)
        return self._dense_v2

    def product(self, row, query=""):
        p = self.products.iloc[int(row)]
        terms = set(tokens(query))
        return {"id": str(p.product_id), "title": p.product_title, "brand": p.product_brand,
                "color": p.product_color, "description": p.product_description, "bullets": p.product_bullet_point,
                "matched_terms": sorted(terms & self.features.document_tokens[int(row)])}

    def rank(self, query, ids, model, limit=20, labels=None):
        x = self.features.transform(query, ids)
        scores = self.scores(x, model)
        product_ids = self.products.iloc[ids].product_id.to_numpy()
        order = np.lexsort((product_ids, -scores))
        result = []
        for i in order[:limit]:
            product = self.product(ids[i], query)
            product.update(score=round(float(scores[i]), 5), rank=len(result)+1,
                           coverage=round(float(x.iloc[i].coverage_all), 3))
            if labels is not None:
                product["label"] = str(labels[i])
            result.append(product)
        return result, scores

    def search(self, query, model="lambdamart", limit=12):
        started = time.perf_counter()
        if re.search(r"[\u3400-\u9fff]", query):
            raise ValueError("当前模型只支持英文检索，请输入英文商品需求，例如 wireless mouse。")
        if not tokens(query):
            raise ValueError("请输入有效的英文商品关键词。")
        if model in ("dense_v2", "dense_v2_pretrained", "hybrid_v2"):
            return self.dense_v2().search(query, model, limit)
        if model in ("dense_pretrained", "dense_finetuned", "dense_hybrid_pretrained", "dense_hybrid"):
            return self.dense().search(query, model, limit)
        retrieval = self.features.all_bm25.scores(query)
        matched = np.flatnonzero(retrieval > 0)
        # Apply the same deterministic product-id tie break to retrieval and ranking.
        order = np.lexsort((self.products.iloc[matched].product_id.to_numpy(), -retrieval[matched]))
        ids = matched[order[:200]]
        results = self.rank(query, ids, model, limit)[0] if len(ids) else []
        return {"query": query, "model": model, "results": results, "matched_products": len(matched),
                "catalog_products": len(self.products),
                "reranked_candidates": len(ids), "elapsed_ms": round((time.perf_counter()-started)*1000, 1),
                "protocol": "子集商品库 BM25 召回前 200 件，再按所选模型排序；尚未评测全库召回率。"}

    def benchmark(self, query_group):
        group = self.test[self.test.query_group == query_group]
        if group.empty:
            raise KeyError(query_group)
        query = str(group.iloc[0]["query"])
        ids = group.catalog_row.to_numpy()
        rankings = {}
        for model in MODEL_NAMES:
            results, scores = self.rank(query, ids, model, 10, group.esci_label.to_numpy())
            rankings[model] = {"results": results, "ndcg_at_10": ndcg(group.label, scores, group.product_id)}
        return {"query": query, "query_group": query_group, "candidates": len(group), "rankings": rankings,
                "protocol": "官方测试集中的指定候选商品排序；E 精确匹配 / S 替代品 / C 配套品 / I 不相关。"}
