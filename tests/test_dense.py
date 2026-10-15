import json

import numpy as np
import pandas as pd
import pytest

from productsearch.data import ROOT
from productsearch.retrieval_metrics import known_e_recall, evaluate_retrieval


def test_recall_deduplicates_results_and_excludes_no_positive_queries():
    assert known_e_recall([4, 4, 5], [4, 6], 3) == .5
    assert known_e_recall([4], [], 50) is None
    frame = pd.DataFrame(dict(query_group=["a", "a", "b"], query=["mouse", "mouse", "keyboard"], label=[3,3,0], catalog_row=[4,6,8]))
    summary, rows = evaluate_retrieval(frame, {"a": np.array([4,5]), "b": np.array([8])})
    assert summary["recall_at_100"] == .5
    assert summary["queries_without_E"] == 1
    assert len(rows) == 1


def test_sealed_test_is_disjoint_from_previous_test_and_training():
    path = ROOT / "data/dense_v1/protocol.json"
    if not path.exists():
        pytest.skip("Prepare dense data before checking sealed protocol")
    protocol = json.loads(path.read_text("utf-8"))
    queries = protocol["queries"]
    assert set(queries["test"]).isdisjoint(protocol["previous_test_queries_excluded"])
    assert set(queries["train"]).isdisjoint(set(queries["test"]) | set(queries["validation"]))
    assert set(queries["validation"]).isdisjoint(queries["test"])
    assert len(queries["test"]) == 1000


def test_stable_top_k_and_fusion_endpoints():
    pytest.importorskip("torch")
    from productsearch.dense import top_k, reciprocal_rank_fusion
    assert top_k([1,2,2,2,0], 2).tolist() == [1,2]
    assert top_k([], 10).tolist() == []
    sparse, dense = [1,2,3], [3,2,4]
    assert reciprocal_rank_fusion(sparse,dense,0)[0].tolist() == sparse
    assert reciprocal_rank_fusion(sparse,dense,1)[0].tolist() == dense
    fused, scores = reciprocal_rank_fusion(sparse,dense,.5)
    assert fused[0] in (2,3)
    assert len(set(fused)) == len(fused)
    assert np.all(scores[:-1] >= scores[1:])


def test_ordinal_loss_rewards_correct_order_and_has_gradients():
    torch = pytest.importorskip("torch")
    from productsearch.dense import pairwise_loss
    q = torch.tensor([[1.,0.]],requires_grad=True)
    p = torch.tensor([[1.,0.]])
    n = torch.tensor([[0.,1.]])
    weights = torch.tensor([1.])
    good = pairwise_loss(q,p,n,weights)
    bad = pairwise_loss(q,n,p,weights)
    assert good < bad
    good.backward()
    assert torch.isfinite(q.grad).all()


def test_mined_negatives_have_lower_observed_relevance():
    path = ROOT / "data/dense_v1/training_pairs.parquet"
    if not path.exists():
        pytest.skip("Run mining to audit ordinal pairs")
    pairs = pd.read_parquet(path)
    train = pd.read_parquet(ROOT / "data/dense_v1/train.parquet")
    judgments = train.set_index(["query_group", "catalog_row"]).label
    assert (pairs.positive_label > pairs.negative_label).all()
    assert (pairs.weight > 0).all()
    for row in pairs.itertuples():
        assert judgments.loc[(row.query_group,row.positive)] == row.positive_label
        assert judgments.loc[(row.query_group,row.negative)] == row.negative_label


def test_completed_encoder_weights_and_catalog_index_are_real_artifacts():
    import hashlib
    path=ROOT/"reports/dense_v1/metrics.json"
    if not path.exists():
        pytest.skip("Complete training before auditing model artifacts")
    report=json.loads(path.read_text("utf-8"))
    seed=report["selection"]["seed"]
    directory=ROOT/"models/dense_v1"
    trained=directory/f"seed_{seed}/encoder/model.safetensors"
    pretrained=directory/"pretrained/model.safetensors"
    assert hashlib.sha256(trained.read_bytes()).hexdigest() != hashlib.sha256(pretrained.read_bytes()).hexdigest()
    assert len(report["selection"]["runs"]) == 3
    for run in report["selection"]["runs"]:
        assert run["weight_updates"]["tensors_changed"] > 0
        assert run["weight_updates"]["mean_absolute_weight_delta"] > 0
        vectors=np.load(directory/f"seed_{run['seed']}/vectors.npy",mmap_mode="r")
        assert vectors.shape == (report["catalog_products"],384)
        sample=np.asarray(vectors[::500])
        assert np.isfinite(sample).all()
        np.testing.assert_allclose(np.linalg.norm(sample,axis=1),1,atol=1e-5)
        assert len(run["history"]) == 2


def test_serving_choice_depends_on_locked_validation_only():
    pytest.importorskip("torch")
    from productsearch.dense_service import choose_serving_policy
    selection={"seed":44,"policies":{"pretrained":{"alpha":.5,"validation_hybrid_trials":[{"recall_at_100":.8}]},
                                     "44":{"alpha":.5,"validation_hybrid_trials":[{"recall_at_100":.7}]}}}
    assert choose_serving_policy(selection)["model"] == "dense_hybrid_pretrained"
