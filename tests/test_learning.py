import numpy as np
import pandas as pd
import pytest

from productsearch.data import split_queries
from productsearch.features import BM25, FeatureExtractor
from productsearch.metrics import ndcg, evaluate, paired_bootstrap


def test_ndcg_uses_official_fractional_gains_and_deterministic_ties():
    expected = (0.1 + 1 / np.log2(3)) / (1 + 0.1 / np.log2(3))
    assert ndcg([3, 2], [0, 0], ["B", "A"]) == pytest.approx(expected)
    assert ndcg([0, 0], [2, 1], ["A", "B"]) == 0
    assert ndcg([3, 2, 1, 0], [4, 3, 2, 1], ["A", "B", "C", "D"]) == 1


def test_split_excludes_all_official_test_queries_and_conflicts():
    rows = []
    for q in ["Mouse", "keyboard", "monitor", "camera", "headphone"]:
        rows.append(dict(query=q, product_id=q, esci_label="E", split="train"))
    rows += [dict(query="MOUSE!!", product_id="mouse2", esci_label="S", split="test"),
             dict(query="Mouse", product_id="mouse2", esci_label="E", split="test"),
             dict(query="mouse", product_id="mouse3", esci_label="I", split="test")]
    config = dict(seed=42, train_queries=2, validation_queries=2, test_queries=1)
    result, manifest = split_queries(pd.DataFrame(rows), config)
    sets = {s: set(g.normalized_query) for s, g in result.groupby("split")}
    assert sets["train"].isdisjoint(sets["validation"] | sets["test"])
    assert sets["validation"].isdisjoint(sets["test"])
    assert sets["test"] == {"mouse"}
    assert manifest["conflicting_rows_removed"] == 2
    assert result[result.split == "test"].product_id.to_list() == ["mouse3"]
    shuffled, _ = split_queries(pd.DataFrame(rows).sample(frac=1, random_state=99), config)
    assert result[["product_id", "split"]].equals(shuffled[["product_id", "split"]])


def test_macro_evaluation_does_not_overweight_large_groups():
    frame = pd.DataFrame(dict(query_group=["a", "b", "b"], query=["a", "b", "b"], label=[3, 3, 0], product_id=["a", "b", "c"]))
    result, per = evaluate(frame, {"bm25": np.array([1, 0, 1]), "lambdamart": np.array([1, 1, 0])})
    assert result["bm25"]["ndcg_at_10"] == pytest.approx((1+1/np.log2(3))/2)
    assert result["lambdamart"]["mrr_exact"] == 1
    ci = paired_bootstrap(per)
    assert ci["mean_delta"] == pytest.approx((1-1/np.log2(3))/2)
    assert ci["ci95"][0] >= 0


def test_bm25_unknown_words_and_subset_scores():
    bm25 = BM25().fit(["wireless mouse", "wired keyboard", "wireless keyboard"])
    scores = bm25.scores("wireless mouse")
    assert scores.argmax() == 0
    np.testing.assert_allclose(bm25.scores("wireless mouse", [2, 0]), scores[[2, 0]])
    np.testing.assert_array_equal(bm25.scores("unfindablezz"), np.zeros(3))
    np.testing.assert_allclose(bm25.scores("mouse mouse"), bm25.scores("mouse"))


def test_pair_feature_alignment_survives_noncontiguous_groups():
    products = pd.DataFrame(dict(product_title=["red mouse", "blue keyboard"], product_bullet_point=["", ""],
                                 product_description=["wireless mouse", "wired keyboard"], product_brand=["Acme", "Other"], product_color=["red", "blue"]))
    extractor = FeatureExtractor().fit(products)
    pairs = pd.DataFrame(dict(query_group=["a", "b", "a"], query=["mouse", "keyboard", "mouse"], catalog_row=[0, 1, 1]))
    result = extractor.for_pairs(pairs)
    assert result.coverage_title.tolist() == [1, 1, 0]
    assert extractor.transform("mouse", []).empty


def test_actual_saved_splits_have_no_query_or_judgment_leakage():
    from productsearch.data import ROOT
    if not (ROOT / "data/processed/train.parquet").exists():
        pytest.skip("Download and prepare ESCI to audit real data")
    frames = {s: pd.read_parquet(ROOT / f"data/processed/{s}.parquet") for s in ["train", "validation", "test"]}
    keys = {s: set(f.normalized_query) for s, f in frames.items()}
    assert keys["train"].isdisjoint(keys["validation"] | keys["test"])
    assert keys["validation"].isdisjoint(keys["test"])
    assert set(frames["test"].official_split) == {"test"}
    for s, frame in frames.items():
        if s != "test":
            assert set(frame.official_split) == {"train"}
        assert not frame.duplicated(["query_group", "product_id"]).any()
        assert frame.query_group.is_monotonic_increasing
