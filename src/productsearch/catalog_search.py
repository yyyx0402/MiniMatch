"""User-facing search over one catalog and the locked fine-tuned MiniLM."""
from collections import OrderedDict
import json
import re
import threading
import time

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from .data import ROOT
from .dense import Encoder, reciprocal_rank_fusion, top_k
from .features import tokens
from .release import MANIFEST, artifact_path, verify_release

SEARCH_MODELS = ("hybrid", "minilm", "keyword", "minilm_pretrained")


class CatalogSearchService:
    def __init__(self, root=ROOT):
        self.root = root
        path = root / MANIFEST
        if not path.exists():
            raise ValueError("请先运行 scripts/prepare_application.py 准备搜索模型。")
        self.manifest = json.loads(path.read_text("utf-8"))
        verify_release(root, self.manifest)
        self.catalog = pd.read_parquet(root / self.manifest["catalog"]).fillna("")
        self.bm25 = joblib.load(root / self.manifest["bm25"])
        if self.bm25.matrix.shape[0] != len(self.catalog) or not self.catalog.product_id.is_monotonic_increasing:
            raise ValueError("商品索引与商品库不一致。")
        self.positions = {str(value): i for i, value in enumerate(self.catalog.product_id)}
        self.encoders, self.vectors = {}, {}
        self.cache = OrderedDict()
        self.lock = threading.RLock()
        torch.set_num_threads(4)
        self.blas_limits = threadpool_limits(limits=4, user_api="blas")

    def artifacts(self, pretrained=False):
        key = "pretrained" if pretrained else "finetuned"
        with self.lock:
            if key not in self.encoders:
                prefix = "pretrained_" if pretrained else ""
                vectors = np.load(artifact_path(self.root, self.manifest[prefix + "vectors"]), mmap_mode="r")
                encoder = Encoder(artifact_path(self.root, self.manifest[prefix + "encoder"]),
                                  device="cpu", max_length=self.manifest["max_length"])
                if vectors.shape != (len(self.catalog), encoder.model.config.hidden_size):
                    raise ValueError("语义索引与当前模型不一致。")
                self.vectors[key], self.encoders[key] = vectors, encoder
            return self.vectors[key], self.encoders[key]

    def product(self, row, query="", score=None, rank=None):
        p = self.catalog.iloc[int(row)]
        terms = set(tokens(query))
        document = " ".join(str(p[field]) for field in (
            "product_title", "product_brand", "product_color", "product_bullet_point", "product_description"))
        matched = sorted(terms & set(tokens(document)))
        result = {"id": str(p.product_id), "title": str(p.product_title), "brand": str(p.product_brand),
                  "color": str(p.product_color), "description": str(p.product_description),
                  "bullets": str(p.product_bullet_point), "matched_terms": matched}
        if score is not None:
            result.update(score=float(score), rank=int(rank), coverage=len(matched) / max(1, len(terms)))
        return result

    def detail(self, product_id):
        if product_id not in self.positions:
            raise KeyError(product_id)
        return self.product(self.positions[product_id])

    def info(self):
        return {"status": "ready", "application_name": "MiniMatch", "catalog_products": len(self.catalog), "model_name": "微调后的 MiniLM",
                "default_search": "hybrid", "language": "English", "local_only": True,
                "search_modes": [{"id": "hybrid", "name": "智能搜索"}, {"id": "minilm", "name": "语义搜索"},
                                 {"id": "keyword", "name": "关键词搜索"}],
                "data_note": "商品来自 Amazon ESCI 历史数据，不含实时价格、库存和购买入口。"}

    def rankings(self, query, model):
        cache_key = (query.lower(), model)
        with self.lock:
            if cache_key in self.cache:
                self.cache.move_to_end(cache_key)
                return self.cache[cache_key]
            depth = self.manifest["retrieval_depth"]
            if model == "keyword":
                scores = self.bm25.scores(query)
                matches = np.flatnonzero(scores > 0)
                ids = matches[top_k(scores[matches], depth)]
                values = scores[ids]
            else:
                vectors, encoder = self.artifacts(model == "minilm_pretrained")
                scores = vectors @ encoder.encode([query])[0]
                ids = top_k(scores, depth)
                values = scores[ids]
                if model == "hybrid":
                    sparse_scores = self.bm25.scores(query)
                    matched = np.flatnonzero(sparse_scores > 0)
                    sparse = matched[top_k(sparse_scores[matched], depth)]
                    ids, values = reciprocal_rank_fusion(sparse, ids, self.manifest["alpha"],
                                                        self.manifest["rrf_k"], depth)
            self.cache[cache_key] = (ids, values)
            while len(self.cache) > 32:
                self.cache.popitem(last=False)
            return ids, values

    def search(self, query, model="hybrid", limit=12, offset=0):
        started = time.perf_counter()
        query = " ".join(query.split())
        if model not in SEARCH_MODELS:
            raise ValueError("未知的搜索方式。")
        if not query or len(query) > 240:
            raise ValueError("请输入有效的英文商品需求，最多 240 个字符。")
        if re.search(r"[\u3400-\u9fff]", query):
            raise ValueError("当前支持英文搜索，例如 wireless mouse。")
        if not tokens(query):
            raise ValueError("请输入有效的英文商品需求，最多 240 个字符。")
        if not 1 <= limit <= 48 or not 0 <= offset <= 200:
            raise ValueError("分页参数不合法。")
        ids, values = self.rankings(query, model)
        page = list(zip(ids[offset:offset + limit], values[offset:offset + limit]))
        notes = {"hybrid": "结合关键词匹配与微调后的 MiniLM 语义理解。",
                 "minilm": "使用微调后的 MiniLM 按语义相关性搜索。",
                 "keyword": "按商品文字中的关键词匹配搜索。", "minilm_pretrained": "使用未微调 MiniLM。"}
        return {"query": query, "model": model,
                "results": [self.product(row, query, value, offset + i + 1) for i, (row, value) in enumerate(page)],
                "catalog_products": len(self.catalog), "matched_products": len(ids),
                "reranked_candidates": len(ids), "total_results": len(ids), "offset": offset,
                "has_more": offset + len(page) < len(ids),
                "elapsed_ms": round((time.perf_counter() - started) * 1000, 1), "protocol": notes[model]}
