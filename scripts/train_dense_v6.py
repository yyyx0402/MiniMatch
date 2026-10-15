"""Early 10-step / later 30-step full validation; fixed 120-update patience."""
import argparse
import gc
import json
import random
import runpy
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import get_linear_schedule_with_warmup

from productsearch.data import ROOT
from productsearch.dense import Encoder, pairwise_loss
from productsearch.dense_v2 import anchor_loss, candidate_summary, dense_rankings, freeze_encoder, multi_positive_loss, retrieval_summary
from productsearch.dense_v6 import StagedRecallStopping as StepEarlyStopping, validation_steps

# These helpers only read inputs or write the explicit supplied path. Do not call
# v2's main/validate/log: their output directories belong to the frozen experiment.
_legacy = runpy.run_path(str(ROOT / "scripts/train_dense_v2.py"))
sha = _legacy["sha"]
write_json = _legacy["write_json"]
vectors_save = _legacy["vectors_save"]
query_table = _legacy["query_table"]
weight_audit = _legacy["weight_audit"]
contrastive_batch = _legacy["contrastive_batch"]
sparse_rankings = _legacy["sparse_rankings"]
validation_policy = _legacy["validation_policy"]
DATA = ROOT / "data/dense_v2"
CACHE = ROOT / "models/dense_v2"
MODELS = ROOT / "models/dense_v6"
REPORTS = ROOT / "reports/dense_v6"


def log(message):
    print(message, flush=True)
    with (REPORTS / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def validate(encoder, validation, catalog_tokens, config):
    torch.cuda.synchronize()
    started = time.perf_counter()
    vectors = encoder.encode_tokens(catalog_tokens, config["encode_batch_size"], log)
    torch.cuda.synchronize()
    encode_seconds = time.perf_counter() - started
    queries = query_table(validation)
    qvectors = encoder.encode(queries["query"].tolist(), config["encode_batch_size"])
    rankings = dense_rankings(queries.query_group.tolist(), qvectors, vectors, config["retrieval_depth"])
    summary, rows = retrieval_summary(validation, rankings)
    summary["candidate_ndcg_at_10"], _ = candidate_summary(validation, qvectors, vectors)
    torch.cuda.synchronize()
    summary["catalog_encode_seconds"] = encode_seconds
    summary["validation_seconds"] = time.perf_counter() - started
    return summary, rows, vectors, qvectors


def train_trial(recipe, config, frame, pairs, tokens, catalog_tokens, validation, baseline, mining):
    name, seed = recipe["name"], config["seed"]
    directory = MODELS / name
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        result = json.loads(result_path.read_text("utf-8"))
        if result["recipe"] != recipe or result["seed"] != seed:
            raise ValueError("Completed trial does not match its frozen recipe")
        for filename, expected in result["artifact_hashes"].items():
            if sha(directory / filename) != expected:
                raise ValueError(f"Completed trial artifact changed: {name}/{filename}")
        log(f"Using completed trial {name}")
        return result
    if (directory / "history.json").exists():
        log(f"Restarting interrupted {name} from the same pretrained weights and seed; no optimizer-state resume")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    rng = np.random.default_rng(seed)
    encoder = Encoder(ROOT / "models/dense_v1/pretrained", max_length=config["max_length"])
    trainable = freeze_encoder(encoder.model, recipe["frozen_layers"])
    queries = query_table(frame)
    qtokens = encoder.tokenize(queries["query"].tolist())
    key_index = dict(zip(queries.query_group, range(len(queries))))
    groups = {key: group for key, group in frame.groupby("query_group", sort=True)}
    judgments = {key: dict(zip(group.product_row.map(int), group.label.map(int))) for key, group in groups.items()}
    eligible_keys = [key for key, labels in judgments.items() if 3 in labels.values()]
    optimizer = torch.optim.AdamW(encoder.model.parameters(), lr=recipe["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, config["warmup_steps"], config["max_optimizer_steps"])
    scaler = torch.amp.GradScaler("cuda")
    teacher = Encoder(ROOT / "models/dense_v1/pretrained", max_length=config["max_length"]) if recipe["anchor"] else None
    if teacher:
        teacher.model.eval()
        for parameter in teacher.model.parameters():
            parameter.requires_grad_(False)
    controller = StepEarlyStopping(baseline["recall_at_100"], config["warmup_steps"], config["minimum_training_steps"],
                                   config["early_stopping_window_steps"], config["minimum_improvement"])
    checks = set(validation_steps(config))
    history = [{"step": 0, "source": "pretrained", **baseline}]
    best_trained = None
    observed = set()
    accumulated = []
    order, cursor = np.empty(0, dtype=np.int64), 0
    step = attempts = skipped = consecutive_skips = 0
    stop = False
    training_seconds = checkpoint_seconds = 0.
    started = time.perf_counter()
    log(f"TRIAL {name}: {recipe}; checks={sorted(checks)}; warmup=30; patience=120 optimizer updates")
    while step < config["max_optimizer_steps"]:
        torch.cuda.synchronize()
        train_started = time.perf_counter()
        encoder.model.train()
        if recipe["frozen_layers"]:
            encoder.model.embeddings.eval()
            for layer in encoder.model.encoder.layer[:recipe["frozen_layers"]]:
                layer.eval()
        optimizer.zero_grad(set_to_none=True)
        attempt_losses = []
        for _ in range(config["gradient_accumulation"]):
            n = config["batch_size"]
            if recipe["loss"] == "ordinal":
                if cursor + n > len(order):
                    order, cursor = rng.permutation(len(pairs)), 0
                batch = pairs.iloc[order[cursor:cursor + n]]
                cursor += len(batch)
                size = len(batch)
                observed.update(batch.query_group)
                inputs = {k: torch.cat([qtokens[k][batch.query_index.to_numpy()], v[batch.positive.to_numpy()],
                                         v[batch.negative.to_numpy()]]).to("cuda", dtype=torch.long) for k, v in tokens.items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    current = encoder.forward(inputs)
                    primary = pairwise_loss(current[:size], current[size:2 * size], current[2 * size:],
                                         torch.tensor(batch.weight.to_numpy(), device="cuda", dtype=torch.float32), config["temperature"])
            else:
                keys = rng.choice(eligible_keys, size=min(n, len(eligible_keys)), replace=False).tolist()
                observed.update(keys)
                docs, positives, weights = contrastive_batch(keys, groups, judgments, mining, rng, config)
                query_ids = np.asarray([key_index[key] for key in keys])
                size = len(keys)
                inputs = {k: torch.cat([qtokens[k][query_ids], v[docs]]).to("cuda", dtype=torch.long) for k, v in tokens.items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    current = encoder.forward(inputs)
                    primary = multi_positive_loss(current[:size], current[size:], positives, weights, config["temperature"])
            anchor = current.new_tensor(0.)
            if teacher:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    original = teacher.forward(inputs)
                anchor = anchor_loss(current, original)
            loss = primary + config["anchor_weight"] * anchor
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss")
            scaler.scale(loss / config["gradient_accumulation"]).backward()
            attempt_losses.append([float(primary.detach()), float(anchor.detach()), float(loss.detach())])
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), 1.)
        previous_scale = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        attempts += 1
        torch.cuda.synchronize()
        training_seconds += time.perf_counter() - train_started
        # GradScaler may skip AdamW on overflow; a skipped update is not an
        # optimizer step and must not consume scheduler/validation/patience time.
        if scaler.get_scale() < previous_scale:
            skipped += 1
            consecutive_skips += 1
            log(f"{name} skipped AMP update (total={skipped})")
            if consecutive_skips >= 10:
                raise RuntimeError("Ten consecutive AMP overflows")
            continue
        consecutive_skips = 0
        scheduler.step()
        step += 1
        accumulated.extend(attempt_losses)
        if step % 10 == 0:
            log(f"{name} optimizer_step={step} loss={np.mean(accumulated, axis=0)[2]:.4f} observed_queries={len(observed):,}")
        if step not in checks:
            continue
        log(f"{name} validating all {len(validation.query_group.unique())} queries / full catalog at step {step}")
        summary, rows, vectors, qvectors = validate(encoder, validation, catalog_tokens, config)
        decision = controller.update(summary["recall_at_100"], step)
        entry = {"step": step, **summary, **decision, "mean_train_loss": float(np.mean(accumulated, axis=0)[2]),
                 "mean_primary_loss": float(np.mean(accumulated, axis=0)[0]),
                 "mean_anchor_loss": float(np.mean(accumulated, axis=0)[1]),
                 "next_learning_rate": float(scheduler.get_last_lr()[0]), "observed_queries": len(observed),
                 "elapsed_seconds": time.perf_counter() - started, "training_seconds_so_far": training_seconds,
                 "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / 2**20,
                 "amp_skipped_updates": skipped}
        checkpoint_started = time.perf_counter()
        if best_trained is None or entry["recall_at_100"] > best_trained["recall_at_100"]:
            best_trained = dict(entry)
            encoder.save(directory / "encoder")
            vectors_save(directory / "vectors.npy", vectors)
            vectors_save(directory / "validation_queries.npy", qvectors)
            pd.DataFrame(rows).to_csv(directory / "validation_per_query.csv", index=False)
        entry["checkpoint_seconds"] = time.perf_counter() - checkpoint_started
        checkpoint_seconds += entry["checkpoint_seconds"]
        history.append(entry)
        accumulated.clear()
        write_json(directory / "history.json", history)
        stop = decision["stop"] and step < config["max_optimizer_steps"]
        log(f"CHECK {name} step={step} R100={summary['recall_at_100']:.6f} gap={decision['steps_without_significant_improvement']} stop={stop} val_seconds={summary['validation_seconds']:.2f}")
        write_json(REPORTS / "progress.json", {"status": "training", "trial": name, "history": history, "test_opened": False})
        del vectors, qvectors, inputs, current, loss
        if stop:
            break
    result = {"name": name, "recipe": recipe, "seed": seed, "steps_completed": step,
              "optimizer_attempts": attempts, "amp_skipped_updates": skipped,
              "stop_reason": "early_stopping_window" if stop else "optimizer_step_budget",
              "best_at_budget_boundary": best_trained["step"] == config["max_optimizer_steps"],
              "early_stopping_triggered": stop,
              "trainable_parameters": trainable,
              "training_pool_queries": int(frame.query_group.nunique()), "training_pairs": len(pairs),
              "eligible_contrastive_queries": len(eligible_keys), "observed_training_queries": len(observed),
              "best_trained": best_trained, "selection_including_pretrained": controller.selection(),
              "history": history, "training_seconds": training_seconds,
              "validation_seconds": sum(row.get("validation_seconds", 0.) for row in history),
              "checkpoint_seconds": checkpoint_seconds, "elapsed_seconds": time.perf_counter() - started,
              "weight_updates": weight_audit(directory / "encoder/model.safetensors", recipe["frozen_layers"]),
              "artifact_hashes": {filename: sha(directory / filename) for filename in
                  ("encoder/model.safetensors", "vectors.npy", "validation_queries.npy", "validation_per_query.csv", "history.json")},
              "test_opened": False}
    write_json(result_path, result)
    del encoder, optimizer, scheduler, scaler, teacher, qtokens, groups, judgments
    gc.collect()
    torch.cuda.empty_cache()
    return result


def freeze_protocol(config):
    source_protocol = json.loads((DATA / "protocol.json").read_text("utf-8"))
    for key in ("model_id", "model_revision", "max_length", "batch_size", "gradient_accumulation", "temperature"):
        if config[key] != source_protocol["config"][key]:
            raise ValueError(f"Inherited protocol is incompatible: {key}")
    inputs = [DATA / name for name in ("train_small.parquet", "train_large.parquet", "validation.parquet", "catalog.parquet", "ordinal_pairs_small.parquet", "ordinal_pairs_large.parquet")]
    inputs += [DATA / "protocol.json", CACHE / "product_tokens.pt", CACHE / "mining_bm25.joblib", CACHE / "bm25.joblib", CACHE / "baseline_validation.json",
               CACHE / "pretrained_vectors.npy", CACHE / "pretrained_validation_queries.npy",
               ROOT / "models/dense_v1/pretrained/model.safetensors", ROOT / "models/dense_v1/pretrained/tokenizer.json",
               ROOT / "models/dense_v1/pretrained/tokenizer_config.json", ROOT / "models/dense_v1/pretrained/config.json"]
    sources = [Path(__file__), ROOT / "configs/dense_v6.json", ROOT / "src/productsearch/dense_v6.py", ROOT / "src/productsearch/dense_v3.py",
               ROOT / "scripts/train_dense_v2.py", ROOT / "src/productsearch/dense_v2.py", ROOT / "src/productsearch/dense.py",
               ROOT / "src/productsearch/features.py", ROOT / "src/productsearch/metrics.py"]
    fingerprints = {str(path.relative_to(ROOT)): sha(path) for path in inputs + sources}
    # Ordinal pairs were mined after v2's dataset preparation; their own v6
    # fingerprint is fixed above, while the four prepared inputs have v2 hashes.
    for path in inputs[:4]:
        if fingerprints[str(path.relative_to(ROOT))] != source_protocol["files"][path.name]:
            raise ValueError(f"Inherited data changed: {path.name}")
    frozen = MODELS / "experiment_fingerprint.json"
    if frozen.exists() and json.loads(frozen.read_text("utf-8")) != fingerprints:
        raise ValueError("Frozen experiment source/input changed; resume refused")
    write_json(frozen, fingerprints)
    write_json(REPORTS / "protocol.json", {"config": config, "fingerprints": fingerprints,
              "validation_steps": validation_steps(config),
              "read_only_inherited_data": "data/dense_v2; 4000 train queries, 600 validation queries, 140084 catalog products",
              "test_policy": "No test labels or previous test metrics loaded by this runner. Independent test performance is unmeasured.",
              "deployment_policy": "Training creates validation candidates only; deployed v2 remains its independently evaluated locked model."})
    preservation = REPORTS / "previous_experiments_preservation.json"
    if not preservation.exists():
        old = json.loads((ROOT / "reports/dense_v2/v1_preservation_hashes.json").read_text("utf-8"))
        paths = set(ROOT / name for name in old)
        paths.update(path for path in (ROOT / "reports/dense_v2").iterdir() if path.is_file())
        paths.update([ROOT / "configs/dense_v2.json", ROOT / "models/dense_v2/selection.json", ROOT / "models/dense_v2/experiment_fingerprint.json"])
        paths.update([ROOT / "configs/dense_v3.json", ROOT / "scripts/train_dense_v3.py"])
        paths.update(path for path in (ROOT / "reports/dense_v3").iterdir() if path.is_file())
        paths.update(path for path in (ROOT / "models/dense_v3").rglob("*") if path.is_file())
        paths.update([ROOT / "configs/dense_v4.json", ROOT / "scripts/train_dense_v4.py", ROOT / "src/productsearch/dense_v4.py"])
        paths.update(path for path in (ROOT / "reports/dense_v4").iterdir() if path.is_file())
        paths.update(path for path in (ROOT / "models/dense_v4").rglob("*") if path.is_file())
        paths.update([ROOT / "configs/dense_v5.json", ROOT / "scripts/train_dense_v5.py", ROOT / "src/productsearch/dense_v5.py"])
        paths.update(path for path in (ROOT / "reports/dense_v5").iterdir() if path.is_file())
        paths.update(path for path in (ROOT / "models/dense_v5").rglob("*") if path.is_file())
        write_json(preservation, {str(path.relative_to(ROOT)): sha(path) for path in sorted(paths)})
    return fingerprints


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="Compatibility alias; all nine runs are always scheduled")
    parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Use the authorized Windows CUDA runtime")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/dense_v6.json").read_text("utf-8"))
    validation_steps(config)
    freeze_protocol(config)
    write_json(REPORTS / "environment.json", {"python": sys.executable, "python_version": sys.version,
               "torch_version": torch.__version__, "torch_path": torch.__file__, "cuda_runtime": torch.version.cuda,
               "gpu": torch.cuda.get_device_name(), "transformers": transformers.__version__, "numpy": np.__version__})
    frames = {pool: pd.read_parquet(DATA / f"train_{pool}.parquet") for pool in ("small", "large")}
    pairs = {pool: pd.read_parquet(DATA / f"ordinal_pairs_{pool}.parquet") for pool in ("small", "large")}
    validation = pd.read_parquet(DATA / "validation.parquet")
    catalog = pd.read_parquet(DATA / "catalog.parquet")
    baseline = json.loads((CACHE / "baseline_validation.json").read_text("utf-8"))
    if baseline["queries_with_E"] != validation.query_group.nunique():
        raise ValueError("Cached baseline validation queries do not match")
    tokens = torch.load(CACHE / "product_tokens.pt", map_location="cpu", weights_only=True)
    catalog_tokens = {k: v[catalog.product_row.to_numpy()] for k, v in tokens.items()}
    mining = joblib.load(CACHE / "mining_bm25.joblib")
    bm25 = joblib.load(CACHE / "bm25.joblib")
    log(f"GPU={torch.cuda.get_device_name()} catalog={len(catalog):,}; inherited token/mining/baseline caches read-only")
    runs = []
    def run(name, recipe, seed=42):
        recipe = {"name": name, **recipe}
        pool = recipe["pool"]
        value = train_trial(recipe, {**config, "seed": seed}, frames[pool], pairs[pool], tokens,
                            catalog_tokens, validation, baseline, mining)
        runs.append(value)
        write_json(REPORTS / "validation_trials.json", {"config": config, "baseline": baseline, "runs": runs,
                   "completed_trials": len(runs), "planned_trials": config["planned_run_count"], "test_opened": False})
        return value
    def best_run(values):
        return max(values, key=lambda value: value["best_trained"]["recall_at_100"])
    base_recipe = {"loss": "ordinal", "learning_rate": 0., "frozen_layers": 0, "anchor": False, "pool": "small"}
    for i, rate in enumerate(config["learning_rates"]):
        run(f"ordinal_lr_{i}_seed42", {**base_recipe, "learning_rate": rate})
    lr_winner = best_run(runs)
    current = {key: value for key, value in lr_winner["recipe"].items() if key != "name"}
    run("freeze_bottom4_seed42", {**current, "frozen_layers": config["freeze_bottom_layers"]})
    contrastive = run("contrastive_seed42", {**current, "loss": "contrastive"})
    run("contrastive_anchor_seed42", {**current, "loss": "contrastive", "anchor": True})
    small_winner = best_run(runs)
    expanded_recipe = {key: value for key, value in small_winner["recipe"].items() if key != "name"}
    run("expanded_pool_seed42", {**expanded_recipe, "pool": "large"})
    recipe_winner = best_run(runs)
    confirmation_recipe = {key: value for key, value in recipe_winner["recipe"].items() if key != "name"}
    confirm = [recipe_winner]
    for seed in config["confirmation_seeds"]:
        if seed != 42:
            confirm.append(run(f"confirmation_seed{seed}", confirmation_recipe, seed))
    if len(runs) != config["planned_run_count"]:
        raise ValueError("The complete study must contain nine actual runs")
    best = best_run(confirm)
    selected = best["selection_including_pretrained"]
    sparse = sparse_rankings(validation, bm25, config)
    policies = {"pretrained": validation_policy("pretrained", np.load(CACHE / "pretrained_vectors.npy", mmap_mode="r"),
                np.load(CACHE / "pretrained_validation_queries.npy"), validation, sparse, config)}
    for value in confirm:
        key = f"seed_{value['seed']}"
        path = MODELS / value["name"]
        policies[key] = validation_policy(key, np.load(path / "vectors.npy", mmap_mode="r"),
                                         np.load(path / "validation_queries.npy"), validation, sparse, config)
    serving = max(policies, key=lambda key: policies[key]["validation_hybrid_recall_at_100"])
    if policies[serving]["validation_hybrid_recall_at_100"] <= policies["pretrained"]["validation_hybrid_recall_at_100"] + config["minimum_improvement"]:
        serving = "pretrained"
    seed_summary = {}
    for field in ("recall_at_100", "candidate_ndcg_at_10"):
        values = [value["best_trained"][field] for value in confirm]
        seed_summary[field] = {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1))}
    selection = {"experiment": config["experiment"], "config": config, "selection_basis": config["selection_metric"],
                 "key": best["name"] if selected["source"] == "finetuned" else "pretrained", **selected,
                 "test_opened": False, "independent_test_status": "not_evaluated", "deployment_changed": False,
                 "completed_trials": len(runs), "planned_trials": config["planned_run_count"],
                 "recipe_trial": recipe_winner["name"], "trained_trial": best["name"],
                 "confirmation_trials": [value["name"] for value in confirm], "policies": policies,
                 "serving_key": serving, "three_seed_validation_summary": seed_summary,
                 "validation_bm25": retrieval_summary(validation, sparse)[0], "selection_locked": True}
    write_json(MODELS / "selection.json", selection)
    write_json(REPORTS / "selection.json", selection)
    preservation = json.loads((REPORTS / "previous_experiments_preservation.json").read_text("utf-8"))
    changed = [name for name, expected in preservation.items() if sha(ROOT / name) != expected]
    if changed:
        raise RuntimeError(f"Previous experiment files changed: {changed}")
    write_json(REPORTS / "progress.json", {"status": "completed",
               "selection": selection, "test_opened": False,
               "preserved_previous_files": len(preservation)})
    log(f"COMPLETED selection={selection['key']} step={selection['step']} R100={selection['recall_at_100']:.6f}; test not opened")


if __name__ == "__main__":
    main()
