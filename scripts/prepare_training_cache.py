"""Prepare current training inputs without running historical model experiments."""
import gc
import json
import runpy

import joblib
import numpy as np
import pandas as pd
import torch

from productsearch.data import ROOT
from productsearch.dense import Encoder, product_texts
from productsearch.dense_v2 import mine_ordinal
from productsearch.features import BM25


def main():
    data, cache, reports = ROOT / "data/dense_v2", ROOT / "models/dense_v2", ROOT / "reports/dense_v2"
    cache.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)
    required = [cache / name for name in ("product_tokens.pt", "mining_bm25.joblib", "bm25.joblib",
                "baseline_validation.json", "pretrained_vectors.npy", "pretrained_validation_queries.npy")]
    required.extend(data / f"ordinal_pairs_{pool}.parquet" for pool in ("small", "large"))
    if all(path.exists() for path in required):
        print("Training inputs already available; no files changed")
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA PyTorch is required to prepare and train this model")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    config = json.loads((ROOT / "configs/dense_v2.json").read_text("utf-8"))
    helpers = runpy.run_path(str(ROOT / "scripts/train_dense_v2.py"))
    products = pd.read_parquet(data / "products.parquet")
    catalog = pd.read_parquet(data / "catalog.parquet")
    encoder = Encoder(ROOT / "models/dense_v1/pretrained", max_length=config["max_length"])
    token_path = cache / "product_tokens.pt"
    if not token_path.exists():
        torch.save(encoder.tokenize(product_texts(products)), token_path)
    tokens = torch.load(token_path, map_location="cpu", weights_only=True)
    for name, frame in [("mining_bm25.joblib", products), ("bm25.joblib", catalog)]:
        path = cache / name
        if not path.exists():
            joblib.dump(BM25().fit(product_texts(frame)), path, compress=3)
    mining = joblib.load(cache / "mining_bm25.joblib")
    for pool in ("small", "large"):
        path = data / f"ordinal_pairs_{pool}.parquet"
        if not path.exists():
            mine_ordinal(pd.read_parquet(data / f"train_{pool}.parquet"), mining, config).to_parquet(path, index=False)
    if not (cache / "baseline_validation.json").exists():
        catalog_tokens = {key: values[catalog.product_row.to_numpy()] for key, values in tokens.items()}
        summary, rows, vectors, qvectors, _ = helpers["validate"](
            encoder, pd.read_parquet(data / "validation.parquet"), catalog_tokens, config)
        helpers["vectors_save"](cache / "pretrained_vectors.npy", vectors)
        helpers["vectors_save"](cache / "pretrained_validation_queries.npy", qvectors)
        helpers["write_json"](cache / "baseline_validation.json", summary)
        pd.DataFrame(rows).to_csv(reports / "validation_pretrained.csv", index=False)
    del encoder, tokens, mining, products, catalog
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
