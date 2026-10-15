"""Preserve validation groups, enlarge training only, seal a fresh official test."""
import hashlib
import json

import pandas as pd
import pyarrow.parquet as pq

from productsearch.data import ROOT, LABELS, clean_text, normalize_query, stable_key


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    config = json.loads((ROOT / "configs/dense_v2.json").read_text("utf-8"))
    out, report = ROOT / "data/dense_v2", ROOT / "reports/dense_v2"
    out.mkdir(parents=True, exist_ok=True); report.mkdir(parents=True, exist_ok=True)
    if (out / "protocol.json").exists():
        manifest = json.loads((out / "protocol.json").read_text("utf-8"))
        if manifest["config"] != config:
            raise ValueError("Sealed config changed; use a new experiment")
        print("dense_v2 already sealed", flush=True)
        return
    baseline = json.loads((ROOT / "data/processed/split_manifest.json").read_text("utf-8"))
    previous = json.loads((ROOT / "data/dense_v1/protocol.json").read_text("utf-8"))
    raw = pd.read_parquet(ROOT / "data/raw/shopping_queries_dataset_examples.parquet")
    raw = raw[(raw.product_locale == "us") & (raw.small_version == 1)].copy()
    raw["normalized_query"] = raw["query"].map(normalize_query)
    all_test = set(raw.loc[raw.split == "test", "normalized_query"]) - {""}
    exposed = set(baseline["queries"]["test"]) | set(previous["queries"]["test"])
    small = set(previous["queries"]["train"])
    validation = set(previous["queries"]["validation"])
    eligible = set(raw.loc[raw.split == "train", "normalized_query"]) - all_test - validation - small - {""}
    extra_count = config["train_queries_large"] - len(small)
    large = small | set(sorted(eligible, key=lambda q: stable_key(q, config["split_seed"]))[:extra_count])
    test = set(sorted(all_test - exposed, key=lambda q: stable_key(q, config["split_seed"]))[:config["test_queries"]])
    if len(large) != config["train_queries_large"] or len(test) != config["test_queries"]:
        raise ValueError("Insufficient unexposed query groups")
    sets = {"train_small": small, "train_large": large, "validation": validation, "test": test}
    frames, removed = {}, {}
    for split, selected in sets.items():
        f = raw[raw.normalized_query.isin(selected) & (raw.split == ("test" if split == "test" else "train"))].copy()
        f["official_split"] = f["split"]
        f["split"] = split
        f["query_group"] = f.normalized_query.map(lambda q: stable_key(q, config["split_seed"])[:20])
        conflict = f.groupby(["query_group", "product_id"]).esci_label.transform("nunique") > 1
        removed[split] = int(conflict.sum())
        f = f[~conflict].drop_duplicates(["query_group", "product_id"])
        f["label"] = f.esci_label.map(LABELS)
        frames[split] = f
    old_catalog = pd.read_parquet(ROOT / "data/dense_v1/catalog.parquet").drop(columns="catalog_row")
    evaluation_ids = set(old_catalog.product_id) | set(frames["test"].product_id)
    needed = set(frames["train_large"].product_id) | evaluation_ids
    missing = needed - set(old_catalog.product_id)
    parts = [old_catalog]
    for batch in pq.ParquetFile(ROOT / "data/raw/shopping_queries_dataset_products.parquet").iter_batches(batch_size=20000):
        part = batch.to_pandas()
        part = part[(part.product_locale == "us") & part.product_id.isin(missing)]
        if len(part):
            parts.append(part)
    products = pd.concat(parts, ignore_index=True).drop_duplicates("product_id").sort_values("product_id").reset_index(drop=True)
    for field in ["product_title", "product_description", "product_bullet_point", "product_brand", "product_color"]:
        products[field] = products[field].map(clean_text)
    products["product_description"] = products.product_description.str.slice(0, 1000)
    products["product_bullet_point"] = products.product_bullet_point.str.slice(0, 800)
    products["product_row"] = products.index
    products.to_parquet(out / "products.parquet", index=False)
    catalog = products[products.product_id.isin(evaluation_ids)].copy().reset_index(drop=True)
    catalog["catalog_row"] = catalog.index
    catalog.to_parquet(out / "catalog.parquet", index=False)
    counts = {}
    for split, f in frames.items():
        f = f.merge(products[["product_id", "product_row"]], on="product_id", how="left", validate="many_to_one")
        if f.product_row.isna().any():
            raise ValueError("Missing training or judgment text")
        f = f.merge(catalog[["product_id", "catalog_row"]], on="product_id", how="left", validate="many_to_one")
        if split in ("validation", "test") and f.catalog_row.isna().any():
            raise ValueError("Missing evaluation product")
        if split in ("validation", "test"):
            f["catalog_row"] = f.catalog_row.astype("int64")
        f = f.sort_values(["query_group", "product_id"]).reset_index(drop=True)
        f.to_parquet(out / f"{split}.parquet", index=False)
        counts[split] = {"queries": int(f.query_group.nunique()), "judgments": len(f), "labels": f.esci_label.value_counts().to_dict()}
    assert small <= large and large.isdisjoint(validation | all_test)
    assert validation.isdisjoint(all_test) and test.isdisjoint(exposed)
    manifest = {"experiment": "dense_v2", "config": config, "queries": {k: sorted(v) for k, v in sets.items()},
                "counts": counts, "excluded_exposed_test_queries": sorted(exposed), "catalog_products": len(catalog),
                "training_text_products": len(products), "conflicting_rows_removed": removed,
                "source": previous["source"], "files": {p.name: sha(p) for p in out.glob("*.parquet")},
                "catalog_policy": "v1 catalog plus fresh test products, fixed across arms; expanded training texts do not enlarge the retrieval catalog",
                "validation_policy": "Original 600 normalized validation query groups are unchanged",
                "test_policy": "New official test queries excluded from all tuning; previous 2000 test groups excluded"}
    (out / "protocol.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), "utf-8")
    audit = {k: v for k, v in manifest.items() if k not in ("queries", "excluded_exposed_test_queries")}
    audit["protocol_sha256"] = sha(out / "protocol.json")
    (report / "protocol.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), "utf-8")
    # Coverage audits use training/validation only, never fresh test outcomes.
    slices = []
    for split in ("train_small", "train_large", "validation"):
        f = pd.read_parquet(out / f"{split}.parquet")
        q = f.groupby("query_group").agg(query=("normalized_query", "first"), judgments=("label", "size"))
        p = f.merge(products[["product_row", "product_brand"]], on="product_row")
        slices.append({"split": split, "queries": len(q), "mean_words": float(q["query"].str.split().str.len().mean()),
                       "fraction_with_digits": float(q["query"].str.contains(r"\d").mean()),
                       "fraction_with_negation": float(q["query"].str.contains(r"\b(?:no|not|without)\b").mean()),
                       "unique_brands": int(p.loc[p.product_brand != "", "product_brand"].nunique()),
                       "mean_judgments": float(q.judgments.mean())})
    pd.DataFrame(slices).to_csv(report / "data_coverage.csv", index=False)
    print(json.dumps({"counts": counts, "catalog": len(catalog), "training_products": len(products)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
