import json
import runpy

import numpy as np
import pandas as pd
import pytest

from productsearch.data import ROOT
from productsearch.dense_v6 import StagedRecallStopping, validation_steps


def test_restored_schedule_and_all_nine_runs_are_fixed():
    config = json.loads((ROOT / "configs/dense_v6.json").read_text("utf-8"))
    assert validation_steps(config) == list(range(10, 121, 10)) + list(range(150, 601, 30))
    assert config["planned_run_count"] == 9
    assert config["learning_rates"] == [5e-6, 1e-5, 2e-5]
    assert config["confirmation_seeds"] == [42, 43, 44]
    with pytest.raises(ValueError, match="restored"):
        validation_steps({**config, "dense_validation_interval": 5})


def test_bad_trial_cannot_stop_before_post_warmup_120_updates():
    control = StagedRecallStopping(.7)
    for step in range(10, 121, 10):
        assert not control.update(.69, step)["stop"]
    decision = control.update(.69, 150)
    assert decision["stop"] and decision["steps_without_significant_improvement"] == 120


def test_improvement_near_cadence_boundary_extends_window_in_updates():
    control = StagedRecallStopping(.7)
    for step in range(10, 121, 10):
        control.update(.702 if step == 120 else .7, step)
    for step in (150, 180, 210):
        assert not control.update(.701, step)["stop"]
    assert control.update(.701, 240)["stop"]
    assert control.selection()["step"] == 120


def test_warmup_peak_is_kept_and_counter_rebases_at_30():
    control = StagedRecallStopping(.7)
    control.update(.705, 10)
    control.update(.7057, 20)
    decision = control.update(.701, 30)
    assert decision["reference_recall_at_100"] == .7057
    assert decision["patience_reset_reason"] == "warmup_completed"
    for step in range(40, 121, 10):
        control.update(.702, step)
    assert control.update(.702, 150)["stop"]
    assert control.selection()["step"] == 20


def test_off_schedule_validation_and_invalid_metrics_are_rejected():
    control = StagedRecallStopping(.7)
    with pytest.raises(ValueError, match="cadence"):
        control.update(.7, 15)
    with pytest.raises(ValueError, match="finite"):
        control.update(float("nan"), 10)
    assert control.last_validation_step == 0


def test_completed_nine_runs_keep_real_best_weights_and_all_seeds():
    runner = runpy.run_path(str(ROOT / "scripts/train_dense_v6.py"))
    assert runner["MODELS"] == ROOT / "models/dense_v6"
    assert runner["REPORTS"] == ROOT / "reports/dense_v6"
    path = ROOT / "reports/dense_v6/validation_trials.json"
    if not path.exists():
        return
    report = json.loads(path.read_text("utf-8"))
    config = report["config"]
    assert report["planned_trials"] == 9 and not report["test_opened"]
    for run in report["runs"]:
        assert [row["step"] for row in run["history"]] == [0] + [step for step in validation_steps(config) if step <= run["steps_completed"]]
        best = max(run["history"][1:], key=lambda row: row["recall_at_100"])
        assert run["best_trained"]["step"] == best["step"]
        assert run["steps_completed"] >= 150
        assert run["optimizer_attempts"] - run["amp_skipped_updates"] == run["steps_completed"]
        assert run["weight_updates"]["tensors_changed"] > 0
        assert run["weight_updates"]["frozen_tensors_changed"] == []
        assert run["observed_training_queries"] <= run["training_pool_queries"]
        if run["recipe"]["anchor"]:
            assert max(row["mean_anchor_loss"] for row in run["history"][1:]) > 0
        directory = ROOT / "models/dense_v6" / run["name"]
        for filename, expected in run["artifact_hashes"].items():
            assert runner["sha"](directory / filename) == expected
        rows = pd.read_csv(directory / "validation_per_query.csv")
        assert len(rows) == 600
        assert rows.recall_at_100.mean() == pytest.approx(best["recall_at_100"], abs=1e-12)
        vectors = np.load(directory / "vectors.npy", mmap_mode="r")
        assert vectors.shape == (140084, 384)
        assert np.allclose(np.linalg.norm(vectors[::701], axis=1), 1, atol=1e-5)
    selection_path = ROOT / "reports/dense_v6/selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text("utf-8"))
        assert len(report["runs"]) == selection["completed_trials"] == 9
        assert {run["seed"] for run in report["runs"]} == {42, 43, 44}
        assert selection["selection_locked"] and not selection["test_opened"]
        assert len(selection["confirmation_trials"]) == 3
        assert selection["serving_key"] in selection["policies"]
