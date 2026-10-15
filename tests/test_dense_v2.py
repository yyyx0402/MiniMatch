import json

import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn.functional as F

from productsearch.data import ROOT
from productsearch.dense_v2 import RecallSelection, anchor_loss, contrastive_weights, multi_positive_loss, retrieval_summary


def test_pretrained_is_retained_when_training_recall_falls():
    selector = RecallSelection(.70, patience=3, minimum_improvement=.0005)
    assert selector.update(.69, 30) == (False, False)
    assert selector.update(.68, 60) == (False, False)
    assert selector.update(.67, 120) == (False, True)
    assert selector.step == 0 and selector.best == .70
    assert selector.update(.72, 240) == (True, False)


def test_all_known_positives_survive_and_substitutes_are_masked():
    positives, weights = contrastive_weights(["q"], [1, 2, 3, 4, 5, 6], {"q": {1: 3, 2: 3, 3: 2, 4: 1, 5: 0}}, .1)
    assert positives.tolist() == [[True, True, False, False, False, False]]
    np.testing.assert_allclose(weights, [[1, 1, 0, 0, 1, .1]])


def test_masked_product_cannot_influence_loss_or_gradient():
    q = torch.tensor([[1., 0.]], requires_grad=True)
    docs = torch.tensor([[1., 0.], [0., 1.], [-1., 0.]], requires_grad=True)
    positive = torch.tensor([[True, False, False]])
    weights = torch.tensor([[1., 0., 1.]])
    loss = multi_positive_loss(q, docs, positive, weights)
    loss.backward()
    assert torch.isfinite(loss) and q.grad is not None
    assert torch.all(docs.grad[1] == 0)
    changed = docs.detach().clone(); changed[1] = torch.tensor([100., 0.])
    torch.testing.assert_close(loss.detach(), multi_positive_loss(q.detach(), changed, positive, weights))


def test_anchor_supports_teacher_inference_tensors():
    current = F.normalize(torch.randn(3, 4), dim=1).requires_grad_()
    with torch.inference_mode():
        teacher = F.normalize(torch.randn(3, 4), dim=1)
    loss = anchor_loss(current, teacher)
    loss.backward()
    assert torch.isfinite(current.grad).all()


def test_weight_audit_tolerates_only_nonpersistent_position_buffer(tmp_path):
    import runpy
    from safetensors.torch import save_file
    audit = runpy.run_path(str(ROOT / "scripts/train_dense_v2.py"))["weight_audit"]
    old, new = tmp_path / "old.safetensors", tmp_path / "new.safetensors"
    save_file({"embeddings.position_ids": torch.tensor([[0, 1]]), "encoder.layer.0.weight": torch.ones(2)}, str(old))
    save_file({"encoder.layer.0.weight": torch.ones(2) * 2}, str(new))
    result = audit(new, 0, old)
    assert result["tensors_changed"] == 1 and result["omitted_nonpersistent_buffers"] == ["embeddings.position_ids"]
    with pytest.raises(RuntimeError, match="frozen"):
        audit(new, 1, old)
    save_file({"unrelated": torch.ones(2)}, str(new))
    with pytest.raises(RuntimeError, match="missing model tensors"):
        audit(new, 0, old)


def test_retrieval_mrr_ignores_unjudged_items_without_calling_them_irrelevant():
    frame = pd.DataFrame({"query_group": ["q", "q", "empty"], "query": ["a", "a", "b"], "label": [3, 0, 0], "catalog_row": [10, 20, 30]})
    summary, records = retrieval_summary(frame, {"q": [99, 10, 20], "empty": [30]})
    assert summary["mrr_at_200"] == .5 and summary["recall_at_100"] == 1
    assert summary["queries_without_E"] == 1 and len(records) == 1


def test_v2_split_preserves_validation_and_excludes_previously_opened_tests():
    path = ROOT / "data/dense_v2/protocol.json"
    if not path.exists():
        pytest.skip("Prepare dense_v2 first")
    protocol = json.loads(path.read_text("utf-8"))
    v1 = json.loads((ROOT / "data/dense_v1/protocol.json").read_text("utf-8"))
    groups = {k: set(v) for k, v in protocol["queries"].items()}
    assert groups["validation"] == set(v1["queries"]["validation"])
    assert groups["train_small"] == set(v1["queries"]["train"])
    assert len(groups["train_large"]) == 20000
    assert groups["train_small"] <= groups["train_large"]
    assert groups["train_large"].isdisjoint(groups["validation"] | groups["test"])
    assert groups["test"].isdisjoint(protocol["excluded_exposed_test_queries"])


def test_v2_ordinal_negatives_are_supervised_lower_grade_judgments():
    for pool in ("small", "large"):
        path = ROOT / f"data/dense_v2/ordinal_pairs_{pool}.parquet"
        if not path.exists():
            pytest.skip("Mine v2 ordinal pairs first")
        pairs = pd.read_parquet(path)
        labels = pd.read_parquet(ROOT / f"data/dense_v2/train_{pool}.parquet")[["query_group", "product_row", "label"]]
        for side in ("positive", "negative"):
            checked = pairs.merge(labels.rename(columns={"product_row": side, "label": "actual_label"}), on=["query_group", side], how="left", validate="many_to_one")
            assert checked.actual_label.notna().all()
            assert (checked.actual_label == checked[f"{side}_label"]).all()
        assert (pairs.positive_label > pairs.negative_label).all()


def test_completed_runs_respect_budget_freezes_and_validation_selection():
    path = ROOT / "reports/dense_v2/metrics.json"
    if not path.exists():
        pytest.skip("Complete dense_v2 first")
    report = json.loads(path.read_text("utf-8"))
    selection = report["selection"]
    assert selection["locked_before_test"] and "Recall@100" in selection["selection_basis"]
    assert len(selection["confirmation_trials"]) == 3
    for run in selection["runs"]:
        assert run["steps_completed"] <= selection["config"]["max_optimizer_steps"]
        assert run["weight_updates"]["tensors_changed"] > 0
        assert run["weight_updates"]["frozen_tensors_changed"] == []
        assert run["observed_training_queries"] <= run["training_pool_queries"]
        assert run["best_trained"]["recall_at_100"] == max(row["recall_at_100"] for row in run["history"] if row["step"] > 0)
    for trial in selection["confirmation_trials"]:
        vectors = np.load(ROOT / f"models/dense_v2/{trial}/vectors.npy", mmap_mode="r")
        assert vectors.shape == (report["catalog_products"], 384)
        sample = np.asarray(vectors[::700])
        assert np.isfinite(sample).all()
        np.testing.assert_allclose(np.linalg.norm(sample, axis=1), 1, atol=1e-5)
