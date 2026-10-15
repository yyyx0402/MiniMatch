import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from productsearch.app import create_app
from productsearch.catalog_search import CatalogSearchService
from productsearch.data import ROOT
from productsearch.release import MANIFEST, artifact_path, sha256


@pytest.fixture(scope="module")
def current():
    if not (ROOT / MANIFEST).exists():
        pytest.skip("Prepare the locked application release first")
    return CatalogSearchService()


@pytest.fixture(scope="module")
def client(current):
    with TestClient(create_app(current)) as value:
        yield value


def test_default_search_uses_current_trained_checkpoint_without_loading_legacy_models(client, current):
    response = client.get("/api/search", params={"q": "wireless mouse"})
    assert response.status_code == 200
    data = response.json()
    assert data["model"] == "hybrid" and len(data["results"]) == 12
    assert client.app.state.legacy is None
    selection = json.loads((ROOT / "reports/dense_v6/selection.json").read_text("utf-8"))
    assert current.manifest["encoder"] == f"models/dense_v6/{selection['trained_trial']}/encoder"
    reference = json.loads((ROOT / f"models/dense_v6/{selection['trained_trial']}/result.json").read_text("utf-8"))
    assert sha256(ROOT / current.manifest["encoder"] / "model.safetensors") == reference["artifact_hashes"]["encoder/model.safetensors"]
    assert current.manifest["alpha"] == selection["policies"][selection["serving_key"]]["alpha"]


def test_search_modes_share_catalog_and_pagination_preserves_rankings(client, current):
    for model in ("hybrid", "minilm", "keyword"):
        first = client.get("/api/search", params={"q": "wireless mouse", "model": model}).json()
        second = client.get("/api/search", params={"q": "wireless mouse", "model": model, "offset": 12}).json()
        whole = client.get("/api/search", params={"q": "wireless mouse", "model": model, "limit": 24}).json()
        assert first["catalog_products"] == len(current.catalog) == second["catalog_products"]
        ids = [p["id"] for p in first["results"] + second["results"]]
        assert ids == [p["id"] for p in whole["results"]] and len(ids) == len(set(ids))
        assert [p["rank"] for p in second["results"]] == list(range(13, 13 + len(second["results"])))
    empty = client.get("/api/search", params={"q": "zzzzunfindabletokenxyz", "model": "keyword"}).json()
    assert empty["results"] == [] and not empty["has_more"]


def test_product_details_are_real_catalog_fields_and_invalid_inputs_are_rejected(client, current):
    result = client.get("/api/search", params={"q": "water bottle"}).json()["results"][0]
    detail = client.get(f"/api/products/{result['id']}").json()
    row = current.catalog.iloc[current.positions[result["id"]]]
    assert detail["title"] == row.product_title and detail["description"] == row.product_description
    assert "price" not in detail and "stock" not in detail and "score" not in detail
    assert client.get("/api/products/no-such-product").status_code == 404
    for params in ({"q": "无线鼠标"}, {"q": " "}, {"q": "mouse", "offset": -1},
                   {"q": "mouse", "limit": 49}, {"q": "mouse", "offset": 201}):
        assert client.get("/api/search", params=params).status_code == 422
    assert "当前支持英文搜索" in client.get("/api/search", params={"q": "无线鼠标"}).json()["detail"]
    with pytest.raises(ValueError):
        artifact_path(ROOT, "../outside-project")


def test_fixed_evaluation_matches_current_model_and_does_not_change_selection():
    path = ROOT / "reports/application/metrics.json"
    if not path.exists():
        pytest.skip("Run fixed-model evaluation first")
    report = json.loads(path.read_text("utf-8"))
    assert report["manifest_sha256"] == sha256(ROOT / MANIFEST)
    assert report["evaluation_type"] == "previously_seen_test_regression"
    assert not report["parameters_changed"]
    rows = pd.read_csv(ROOT / "reports/application/per_query_retrieval.csv")
    for model, summary in report["retrieval"].items():
        assert summary["recall_at_100"] == pytest.approx(rows[rows.model == model].recall_at_100.mean())
    manifest = json.loads((ROOT / MANIFEST).read_text("utf-8"))
    vectors = np.load(ROOT / manifest["vectors"], mmap_mode="r")
    assert np.linalg.norm(vectors[::5000], axis=1) == pytest.approx(np.ones(len(vectors[::5000])), abs=2e-3)
