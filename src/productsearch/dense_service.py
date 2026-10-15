import json
import threading
import time

import joblib
import numpy as np
import pandas as pd
import torch

from .data import ROOT
from .dense import Encoder, top_k, reciprocal_rank_fusion
from .features import tokens
from .metrics import ndcg

DENSE_MODELS = ("dense_pretrained", "dense_finetuned", "dense_hybrid_pretrained", "dense_hybrid")


def choose_serving_policy(selection):
    """Choose by the previously locked validation trials, without receiving test scores."""
    candidate=max(["pretrained",str(selection["seed"])],key=lambda key:max(row["recall_at_100"] for row in selection["policies"][key]["validation_hybrid_trials"]))
    return {"model":"dense_hybrid_pretrained" if candidate=="pretrained" else "dense_hybrid",
            "basis":"validation known-E Recall@100 only", "experiment_seed":selection["seed"],
            "encoder":"pretrained" if candidate=="pretrained" else "finetuned",
            "alpha":selection["policies"][candidate]["alpha"]}


class DenseSearchService:
    def __init__(self, root=ROOT):
        directory = root / "models/dense_v1"
        self.report = json.loads((root / "reports/dense_v1/metrics.json").read_text("utf-8"))
        self.selection = self.report["selection"]
        self.config = self.selection["config"]
        self.seed = str(self.selection["seed"])
        self.catalog = pd.read_parquet(root / "data/dense_v1/catalog.parquet")
        self.test = pd.read_parquet(root / "data/dense_v1/test.parquet")
        self.queries = self.test.groupby("query_group",sort=True).agg(query=("query","first"), candidates=("product_id","size")).reset_index()
        self.bm25 = joblib.load(directory / "bm25.joblib")
        self.vectors = {"pretrained":np.load(directory / "pretrained_vectors.npy",mmap_mode="r"),
                        "finetuned":np.load(directory / f"seed_{self.seed}/vectors.npy",mmap_mode="r")}
        self.test_vectors = {"pretrained":np.load(directory / "pretrained_test_queries.npy"),
                             "finetuned":np.load(directory / f"seed_{self.seed}/test_queries.npy")}
        self.query_positions = {key:i for i,key in enumerate(self.queries.query_group)}
        torch.set_num_threads(4)
        self.encoders = {"pretrained":Encoder(directory / "pretrained",device="cpu",max_length=self.config["max_length"]),
                         "finetuned":Encoder(directory / f"seed_{self.seed}/encoder",device="cpu",max_length=self.config["max_length"])}
        self.lock = threading.Lock()
        if any(len(v)!=len(self.catalog) for v in self.vectors.values()):
            raise ValueError("Dense index/catalog row count mismatch")

    def product(self, row, query, score, rank):
        p = self.catalog.iloc[int(row)]
        terms = set(tokens(query))
        document = " ".join([p.product_title,p.product_brand,p.product_color,p.product_bullet_point,p.product_description])
        matched = sorted(terms & set(tokens(document)))
        return {"id":str(p.product_id),"title":p.product_title,"brand":p.product_brand,"color":p.product_color,
                "description":p.product_description,"bullets":p.product_bullet_point,"matched_terms":matched,
                "coverage":len(matched)/max(len(terms),1),"score":float(score),"rank":rank}

    def dense_scores(self, query, variant):
        with self.lock:
            vector = self.encoders[variant].encode([query])[0]
            return self.vectors[variant] @ vector

    def search(self, query, model, limit=12):
        started=time.perf_counter()
        variant="pretrained" if model in ("dense_pretrained","dense_hybrid_pretrained") else "finetuned"
        scores=self.dense_scores(query,variant)
        dense=top_k(scores,self.config["retrieval_depth"])
        if model in ("dense_hybrid","dense_hybrid_pretrained"):
            sparse_scores=self.bm25.scores(query)
            matches=np.flatnonzero(sparse_scores>0)
            sparse=matches[top_k(sparse_scores[matches],self.config["retrieval_depth"])]
            alpha=self.selection["policies"]["pretrained" if variant=="pretrained" else self.seed]["alpha"]
            ids,values=reciprocal_rank_fusion(sparse,dense,alpha,self.config["rrf_k"],self.config["retrieval_depth"])
            protocol=f"语义与 BM25 各召回前 200 件，RRF 融合；语义权重 {alpha:g}，由验证集选定。"
        else:
            ids,values=dense,scores[dense]
            protocol="在固定实验商品库中计算归一化向量余弦相似度，返回前 200 件候选。"
        return {"query":query,"model":model,"results":[self.product(row,query,score,i+1) for i,(row,score) in enumerate(zip(ids[:limit],values[:limit]))],
                "matched_products":len(self.catalog),"reranked_candidates":len(ids),"elapsed_ms":round((time.perf_counter()-started)*1000,1),
                "protocol":protocol,"catalog_products":len(self.catalog)}

    def benchmark(self, query_group):
        group=self.test[self.test.query_group==query_group]
        if group.empty:
            raise KeyError(query_group)
        query=str(group.iloc[0]["query"])
        ids=group.catalog_row.to_numpy()
        position=self.query_positions[query_group]
        scores={"bm25":self.bm25.scores(query,ids),
                "dense_pretrained":self.vectors["pretrained"][ids] @ self.test_vectors["pretrained"][position],
                "dense_finetuned":self.vectors["finetuned"][ids] @ self.test_vectors["finetuned"][position]}
        rankings={}
        for model,values in scores.items():
            order=top_k(values,10)
            results=[]
            for rank,i in enumerate(order,1):
                product=self.product(ids[i],query,values[i],rank)
                product["label"]=str(group.iloc[i].esci_label)
                results.append(product)
            rankings[model]={"results":results,"ndcg_at_10":ndcg(group.label,values,group.product_id)}
        return {"query":query,"query_group":query_group,"candidates":len(group),"rankings":rankings,
                "protocol":"新封存测试集的标注候选排序；与全库 Recall 评测分开。"}
