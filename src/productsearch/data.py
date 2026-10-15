import hashlib
import html
import json
import re
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
LABELS = {"I":0,"C":1,"S":2,"E":3}


def normalize_query(text):
    return " ".join(re.findall(r"[a-z0-9]+", str(text).lower()))


def stable_key(value, seed):
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def clean_text(value):
    if value is None or pd.isna(value):
        return ""
    return " ".join(html.unescape(re.sub(r"<[^>]*>", " ", str(value))).split())


def split_queries(examples, config):
    frame = examples.copy()
    frame["normalized_query"] = frame["query"].map(normalize_query)
    frame = frame[frame.normalized_query.str.len() > 0]
    all_test = set(frame.loc[frame.split == "test", "normalized_query"])
    train = set(frame.loc[frame.split == "train", "normalized_query"]) - all_test
    ordered = sorted(train, key=lambda value: stable_key(value, config["seed"]))
    count = config["train_queries"]
    selected = {
        "train":ordered[:count],
        "validation":ordered[count:count+config["validation_queries"]],
        "test":sorted(all_test, key=lambda value: stable_key(value, config["seed"]))[:config["test_queries"]],
    }
    mapping = {query:split for split, queries in selected.items() for query in queries}
    frame = frame[frame.normalized_query.isin(mapping)].copy()
    frame["official_split"] = frame["split"]
    frame["split"] = frame.normalized_query.map(mapping)
    frame = frame[(frame.split == "test") == (frame.official_split == "test")]
    frame["query_group"] = frame.normalized_query.map(lambda value: stable_key(value, config["seed"])[:20])
    # One query/product judgment per experiment group; discard conflicting labels.
    conflicts = frame.groupby(["query_group","product_id"]).esci_label.transform("nunique") > 1
    conflict_rows = int(conflicts.sum())
    frame = frame[~conflicts].drop_duplicates(["query_group","product_id"])
    frame["label"] = frame.esci_label.map(LABELS)
    if frame.label.isna().any():
        raise ValueError("Unknown ESCI labels")
    frame = frame.sort_values(["split","query_group","product_id"]).reset_index(drop=True)
    return frame, {"queries":selected,"conflicting_rows_removed":conflict_rows}


def prepare(config):
    raw, out = ROOT / "data/raw", ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    examples = pd.read_parquet(raw / "shopping_queries_dataset_examples.parquet")
    examples = examples[(examples.product_locale == config["locale"]) & (examples.small_version == 1)]
    pairs, manifest = split_queries(examples, config)
    needed = set(pairs.product_id)
    print(f"Selected {len(pairs):,} query/product judgments, {len(needed):,} products", flush=True)
    pieces = []
    parquet = pq.ParquetFile(raw / "shopping_queries_dataset_products.parquet")
    for index, batch in enumerate(parquet.iter_batches(batch_size=20000)):
        part = batch.to_pandas()
        part = part[(part.product_locale == config["locale"]) & part.product_id.isin(needed)]
        if len(part):
            pieces.append(part)
        if (index + 1) % 25 == 0:
            print(f"Scanned {(index+1)*20000:,} catalog records", flush=True)
    products = pd.concat(pieces, ignore_index=True).drop_duplicates(["product_id","product_locale"]).sort_values("product_id").reset_index(drop=True)
    for field in ["product_title","product_description","product_bullet_point","product_brand","product_color"]:
        products[field] = products[field].map(clean_text)
    products["product_description"] = products.product_description.str.slice(0,config["description_chars"])
    products["product_bullet_point"] = products.product_bullet_point.str.slice(0,config["bullet_chars"])
    products["catalog_row"] = products.index
    pairs = pairs.merge(products[["product_id","catalog_row"]], on="product_id", how="left", validate="many_to_one")
    if pairs.catalog_row.isna().any():
        raise ValueError("Selected products are missing from the official catalog")
    for split in ("train","validation","test"):
        subset = pairs[pairs.split == split].sort_values(["query_group","product_id"]).reset_index(drop=True)
        subset.to_parquet(out / f"{split}.parquet", index=False)
    products.to_parquet(out / "catalog.parquet", index=False)
    counts = {split:{"queries":int(group.query_group.nunique()), "pairs":len(group), "labels":group.esci_label.value_counts().to_dict()} for split,group in pairs.groupby("split")}
    manifest.update(seed=config["seed"],locale=config["locale"],counts=counts,products=len(products),
                    protocol="Official small-version US data. Query-group train/validation split; official test held out. Unlabeled catalog text is used for indexing/IDF only.",
                    data_manifest=json.loads((raw / "manifest.json").read_text("utf-8")))
    (out / "split_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),"utf-8")
    print(json.dumps(counts, indent=2), flush=True)
    return manifest
