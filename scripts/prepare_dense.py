"""Seal a new official test subset while preserving baseline data and model artifacts."""
import hashlib
import json

import pandas as pd
import pyarrow.parquet as pq

from productsearch.data import ROOT, clean_text, normalize_query, stable_key, LABELS


def main():
    config = json.loads((ROOT / "configs/dense_v1.json").read_text("utf-8"))
    old = ROOT / "data/processed"
    out = ROOT / "data/dense_v1"
    out.mkdir(parents=True, exist_ok=True)
    baseline = json.loads((old / "split_manifest.json").read_text("utf-8"))
    examples = pd.read_parquet(ROOT / "data/raw/shopping_queries_dataset_examples.parquet")
    examples = examples[(examples.product_locale == "us") & (examples.small_version == 1) & (examples.split == "test")].copy()
    examples["normalized_query"] = examples["query"].map(normalize_query)
    exposed = set(baseline["queries"]["test"])
    available = set(examples.normalized_query) - exposed - {""}
    queries = sorted(available, key=lambda q: stable_key(q, config["split_seed"]))[:config["test_queries"]]
    if len(queries) != config["test_queries"]:
        raise ValueError("Not enough previously uninspected official test queries")
    test = examples[examples.normalized_query.isin(queries)].copy()
    test["official_split"] = "test"
    test["query_group"] = test.normalized_query.map(lambda q: stable_key(q, config["split_seed"])[:20])
    conflicts = test.groupby(["query_group", "product_id"]).esci_label.transform("nunique") > 1
    test = test[~conflicts].drop_duplicates(["query_group", "product_id"])
    test["label"] = test.esci_label.map(LABELS)
    catalog = pd.read_parquet(old / "catalog.parquet")
    needed = set(test.product_id) - set(catalog.product_id)
    pieces = [catalog.drop(columns="catalog_row")]
    for batch in pq.ParquetFile(ROOT / "data/raw/shopping_queries_dataset_products.parquet").iter_batches(batch_size=20000):
        part = batch.to_pandas()
        part = part[(part.product_locale == "us") & part.product_id.isin(needed)]
        if len(part):
            pieces.append(part)
    catalog = pd.concat(pieces, ignore_index=True).drop_duplicates("product_id").sort_values("product_id").reset_index(drop=True)
    for field in ["product_title", "product_description", "product_bullet_point", "product_brand", "product_color"]:
        catalog[field] = catalog[field].map(clean_text)
    catalog["product_description"] = catalog.product_description.str.slice(0, 1000)
    catalog["product_bullet_point"] = catalog.product_bullet_point.str.slice(0, 800)
    catalog["catalog_row"] = catalog.index
    catalog.to_parquet(out / "catalog.parquet", index=False)
    frames = {s: pd.read_parquet(old / f"{s}.parquet").drop(columns="catalog_row") for s in ["train", "validation"]}
    frames["test"] = test.drop(columns="catalog_row", errors="ignore")
    counts = {}
    query_sets = {}
    for split, frame in frames.items():
        frame = frame.merge(catalog[["product_id", "catalog_row"]], on="product_id", how="left", validate="many_to_one")
        if frame.catalog_row.isna().any():
            raise ValueError("Missing judgment products")
        frame["split"] = split
        frame = frame.sort_values(["query_group", "product_id"]).reset_index(drop=True)
        frame.to_parquet(out / f"{split}.parquet", index=False)
        counts[split] = {"queries": int(frame.query_group.nunique()), "pairs": len(frame), "labels": frame.esci_label.value_counts().to_dict()}
        query_sets[split] = sorted(set(frame.normalized_query))
    assert set(query_sets["test"]).isdisjoint(exposed)
    assert set(query_sets["train"]).isdisjoint(set(query_sets["validation"]) | set(query_sets["test"]))
    manifest = {"protocol_version": "dense_v1-frozen-before-training", "config": config, "counts": counts,
                "queries": query_sets, "previous_test_queries_excluded": sorted(exposed), "products": len(catalog),
                "catalog_sha256": hashlib.sha256((out / "catalog.parquet").read_bytes()).hexdigest(),
                "source": baseline["data_manifest"], "conflicting_test_rows_removed": int(conflicts.sum()),
                "limits": ["Historical US subset; catalog includes previous baseline products and new sealed test products",
                           "Unlabeled catalog available to BM25 IDF; only known E judgments define retrieval recall",
                           "Retrieval evaluation ignores queries with no E judgments; supplied-candidate NDCG includes all queries"]}
    (out / "protocol.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    # A tracked digest pins the full ignored split manifest without exposing test results during development.
    report = ROOT / "reports/dense_v1"
    report.mkdir(parents=True, exist_ok=True)
    audit = {k: manifest[k] for k in ["protocol_version", "config", "counts", "products", "catalog_sha256", "limits"]}
    audit["protocol_sha256"] = hashlib.sha256((out / "protocol.json").read_bytes()).hexdigest()
    (report / "protocol.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(counts, indent=2), flush=True)
    print(f"Frozen catalog: {len(catalog):,} products; no overlap with previously inspected test queries", flush=True)


if __name__ == "__main__":
    main()
