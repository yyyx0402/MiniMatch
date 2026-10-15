"""Staged retrieval experiments; test is opened only after validation selection is locked."""
import gc
import hashlib
import json
import math
import random
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from transformers import get_linear_schedule_with_warmup

from productsearch.data import ROOT
from productsearch.dense import Encoder, pairwise_loss, product_texts, reciprocal_rank_fusion, top_k
from productsearch.dense_v2 import (RecallSelection, anchor_loss, candidate_summary, contrastive_weights,
                                    dense_rankings, freeze_encoder, mine_ordinal, multi_positive_loss, retrieval_summary)
from productsearch.features import BM25
from productsearch.metrics import ndcg
from productsearch.retrieval_metrics import paired_interval

DATA = ROOT / "data/dense_v2"
MODELS = ROOT / "models/dense_v2"
REPORTS = ROOT / "reports/dense_v2"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), "utf-8")
    temporary.replace(path)


def log(message):
    print(message, flush=True)
    with (REPORTS / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}\n")


def vectors_save(path, value):
    temporary = path.with_suffix(".npy.partial")
    with temporary.open("wb") as stream:
        np.save(stream, value)
    temporary.replace(path)


def query_table(frame):
    return frame.groupby("query_group", sort=True).agg(query=("query", "first")).reset_index()


def sparse_rankings(frame, bm25, config):
    result = {}
    for row in query_table(frame).itertuples():
        scores = bm25.scores(row.query)
        matches = np.flatnonzero(scores > 0)
        result[row.query_group] = matches[top_k(scores[matches], config["retrieval_depth"])]
    return result


def hybrid_rankings(sparse, dense, alpha, config):
    return {key: reciprocal_rank_fusion(sparse[key], dense[key], alpha, config["rrf_k"], config["retrieval_depth"])[0]
            for key in dense}


def validate(encoder, validation, catalog_tokens, config):
    vectors = encoder.encode_tokens(catalog_tokens, config["encode_batch_size"], log)
    queries = query_table(validation)
    qvectors = encoder.encode(queries["query"].tolist(), config["encode_batch_size"])
    rankings = dense_rankings(queries.query_group.tolist(), qvectors, vectors, config["retrieval_depth"])
    summary, rows = retrieval_summary(validation, rankings)
    summary["candidate_ndcg_at_10"], _ = candidate_summary(validation, qvectors, vectors)
    return summary, rows, vectors, qvectors, rankings


def weight_audit(path, bottom_layers, reference=None):
    reference = reference or ROOT / "models/dense_v1/pretrained/model.safetensors"
    changed, total, absolute, count, frozen_changed = 0, 0, 0., 0, []
    with safe_open(str(reference), framework="pt", device="cpu") as old, safe_open(str(path), framework="pt", device="cpu") as new:
        missing = set(old.keys()) - set(new.keys())
        # Transformers omits this non-persistent position buffer on save.
        if missing - {"embeddings.position_ids"}:
            raise RuntimeError(f"Saved checkpoint is missing model tensors: {sorted(missing)}")
        for key in sorted(set(old.keys()) & set(new.keys())):
            delta = (new.get_tensor(key) - old.get_tensor(key)).abs()
            was_changed = bool(torch.any(delta > 0))
            changed += int(was_changed); total += 1
            absolute += float(delta.sum()); count += delta.numel()
            frozen = bottom_layers and (key.startswith("embeddings.") or any(key.startswith(f"encoder.layer.{i}.") for i in range(bottom_layers)))
            if frozen and was_changed:
                frozen_changed.append(key)
    if not changed or frozen_changed:
        raise RuntimeError(f"Invalid trained weights: changed={changed}, frozen changes={frozen_changed}")
    return {"tensors_changed": changed, "tensors_compared": total, "mean_absolute_weight_delta": absolute / count,
            "frozen_tensors_changed": frozen_changed, "omitted_nonpersistent_buffers": sorted(missing)}


def contrastive_batch(keys, groups, judgments, bm25, rng, config):
    docs = []
    for key in keys:
        group = groups[key]
        positives = group.loc[group.label == 3, "product_row"].to_numpy()
        irrelevant = group.loc[group.label == 0, "product_row"].to_numpy()
        docs.extend(map(int, rng.permutation(positives)[:config["contrastive_positives_per_query"]]))
        if len(irrelevant):
            scores = bm25.scores(str(group.iloc[0]["query"]), irrelevant)
            docs.extend(map(int, irrelevant[top_k(scores, config["contrastive_negatives_per_query"])]))
    docs = np.asarray(sorted(set(docs)), dtype=np.int64)
    positives, weights = contrastive_weights(keys, docs, judgments, config["unjudged_negative_weight"])
    return docs, torch.from_numpy(positives), torch.from_numpy(weights)


def train_run(name, recipe, seed, config, frames, pairs_by_pool, products_tokens, catalog_tokens, validation, baseline, mining_bm25):
    directory = MODELS / name
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        value = json.loads((directory / "result.json").read_text("utf-8"))
        if value["recipe"] != recipe or value["seed"] != seed:
            raise ValueError("Completed trial does not match recipe")
        log(f"Using completed trial {name}")
        return value
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.cuda.reset_peak_memory_stats()
    frame = frames[recipe["pool"]]
    queries = query_table(frame)
    encoder = Encoder(ROOT / "models/dense_v1/pretrained", max_length=config["max_length"])
    trainable = freeze_encoder(encoder.model, recipe["frozen_layers"])
    qtokens = encoder.tokenize(queries["query"].tolist())
    key_index = dict(zip(queries.query_group, range(len(queries))))
    groups = {key: group for key, group in frame.groupby("query_group", sort=True)}
    judgments = {key: dict(zip(group.product_row.map(int), group.label.map(int))) for key, group in groups.items()}
    eligible_keys = [key for key, labels in judgments.items() if 3 in labels.values()]
    pairs = pairs_by_pool[recipe["pool"]]
    optimizer = torch.optim.AdamW([p for p in encoder.model.parameters() if p.requires_grad], lr=recipe["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(config["max_optimizer_steps"] * config["warmup_fraction"]), config["max_optimizer_steps"])
    scaler = torch.amp.GradScaler("cuda")
    teacher = Encoder(ROOT / "models/dense_v1/pretrained", max_length=config["max_length"]) if recipe["anchor"] else None
    if teacher:
        teacher.model.eval()
        for p in teacher.model.parameters():
            p.requires_grad_(False)
    state = RecallSelection(baseline["recall_at_100"], config["early_stopping_patience"], config["minimum_improvement"])
    rng = np.random.default_rng(seed)
    history = [{"step": 0, "source": "pretrained", **baseline}]
    best_trained = None
    best_ndcg = None
    observed = set()
    accumulated = []
    started = time.perf_counter()
    order = np.empty(0, dtype=np.int64); cursor = 0; stop = False
    log(f"TRIAL {name}: {recipe}; trainable={trainable:,}")
    for step in range(1, config["max_optimizer_steps"] + 1):
        encoder.model.train()
        if recipe["frozen_layers"]:
            encoder.model.embeddings.eval()
            for layer in encoder.model.encoder.layer[:recipe["frozen_layers"]]:
                layer.eval()
        optimizer.zero_grad(set_to_none=True)
        for _ in range(config["gradient_accumulation"]):
            n = config["batch_size"]
            if recipe["loss"] == "ordinal":
                if cursor + n > len(order):
                    order, cursor = rng.permutation(len(pairs)), 0
                batch = pairs.iloc[order[cursor:cursor + n]]; cursor += len(batch)
                size = len(batch)
                observed.update(batch.query_group)
                inputs = {k: torch.cat([qtokens[k][batch.query_index.to_numpy()], v[batch.positive.to_numpy()], v[batch.negative.to_numpy()]]).to("cuda", dtype=torch.long)
                          for k, v in products_tokens.items()}
                with torch.autocast("cuda", dtype=torch.float16):
                    current = encoder.forward(inputs)
                    primary = pairwise_loss(current[:size], current[size:2 * size], current[2 * size:],
                                            torch.tensor(batch.weight.to_numpy(), device="cuda", dtype=torch.float32), config["temperature"])
            else:
                # Unique queries and unique products per batch; complete training judgments mask false negatives.
                keys = rng.choice(eligible_keys, size=min(n, len(eligible_keys)), replace=False).tolist()
                observed.update(keys)
                docs, positives, weights = contrastive_batch(keys, groups, judgments, mining_bm25, rng, config)
                query_ids = np.asarray([key_index[key] for key in keys])
                size = len(keys)
                inputs = {k: torch.cat([qtokens[k][query_ids], v[docs]]).to("cuda", dtype=torch.long) for k, v in products_tokens.items()}
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
                raise RuntimeError("Non-finite loss")
            scaler.scale(loss / config["gradient_accumulation"]).backward()
            accumulated.append([float(primary.detach()), float(anchor.detach()), float(loss.detach())])
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(encoder.model.parameters(), 1.)
        scaler.step(optimizer); scaler.update(); scheduler.step()
        if step % 15 == 0:
            log(f"{name} step={step} loss={np.mean(accumulated, axis=0)[2]:.4f} observed_queries={len(observed):,}")
        if step in config["evaluation_steps"]:
            log(f"{name} validating full catalog at optimizer step {step}")
            summary, rows, vectors, qvectors, _ = validate(encoder, validation, catalog_tokens, config)
            entry = {"step": step, **summary, "mean_primary_loss": float(np.mean(accumulated, axis=0)[0]),
                     "mean_anchor_loss": float(np.mean(accumulated, axis=0)[1]), "mean_train_loss": float(np.mean(accumulated, axis=0)[2]),
                     "observed_queries": len(observed), "elapsed_seconds": time.perf_counter() - started,
                     "peak_gpu_allocated_mb": torch.cuda.max_memory_allocated() / 2**20}
            history.append(entry); accumulated.clear()
            _, stop = state.update(summary["recall_at_100"], step)
            if best_trained is None or summary["recall_at_100"] > best_trained["recall_at_100"]:
                best_trained = entry
                encoder.save(directory / "encoder")
                vectors_save(directory / "vectors.npy", vectors)
                vectors_save(directory / "validation_queries.npy", qvectors)
                pd.DataFrame(rows).to_csv(directory / "validation_per_query.csv", index=False)
            if best_ndcg is None or summary["candidate_ndcg_at_10"] > best_ndcg["candidate_ndcg_at_10"]:
                best_ndcg = entry
            write_json(directory / "history.json", history)
            log(f"CHECK {name} step={step} R100={summary['recall_at_100']:.6f} NDCG={summary['candidate_ndcg_at_10']:.6f} stop={stop}")
            write_json(REPORTS / "progress.json", {"status": "training", "trial": name, "history": history})
            del vectors, qvectors
            if stop:
                break
    audit = weight_audit(directory / "encoder/model.safetensors", recipe["frozen_layers"])
    result = {"name": name, "recipe": recipe, "seed": seed, "trainable_parameters": trainable,
              "training_pool_queries": int(frame.query_group.nunique()), "eligible_contrastive_queries": len(eligible_keys),
              "training_pairs": len(pairs), "observed_training_queries": len(observed), "steps_completed": step,
              "stop_reason": "validation recall patience" if stop else "optimizer step budget",
              "best_trained": best_trained, "ndcg_selected_checkpoint": best_ndcg,
              "selection_including_pretrained": {"source": "pretrained" if state.step == 0 else "finetuned", "step": state.step, "recall_at_100": state.best},
              "history": history, "weight_updates": audit, "weight_sha256": sha(directory / "encoder/model.safetensors"),
              "vector_sha256": sha(directory / "vectors.npy")}
    write_json(directory / "result.json", result)
    del encoder, optimizer, scheduler, teacher, scaler, qtokens, groups, judgments
    gc.collect(); torch.cuda.empty_cache()
    return result


def best_run(runs):
    # Stable ties prefer the earlier, simpler recipe.
    return max(runs, key=lambda run: run["best_trained"]["recall_at_100"])


def validation_policy(key, vectors, qvectors, validation, sparse, config):
    rankings = dense_rankings(query_table(validation).query_group.tolist(), qvectors, vectors, config["retrieval_depth"])
    dense, _ = retrieval_summary(validation, rankings)
    trials = [{"alpha": alpha, **retrieval_summary(validation, hybrid_rankings(sparse, rankings, alpha, config))[0]} for alpha in config["hybrid_alphas"]]
    best = max(trials, key=lambda row: row["recall_at_100"])
    return {"key": key, "validation_dense": dense, "validation_hybrid_trials": trials, "alpha": best["alpha"], "validation_hybrid_recall_at_100": best["recall_at_100"]}


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Use the authorized Windows CUDA runtime")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    MODELS.mkdir(parents=True, exist_ok=True); REPORTS.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / "configs/dense_v2.json").read_text("utf-8"))
    protocol = json.loads((DATA / "protocol.json").read_text("utf-8"))
    if config != protocol["config"]:
        raise ValueError("Protocol config changed")
    sources = [Path(__file__), ROOT / "scripts/prepare_dense_v2.py", ROOT / "configs/dense_v2.json", DATA / "protocol.json",
               ROOT / "src/productsearch/dense_v2.py", ROOT / "src/productsearch/dense.py", ROOT / "src/productsearch/features.py", ROOT / "src/productsearch/metrics.py"]
    fingerprints = {str(path.relative_to(ROOT)): sha(path) for path in sources}
    fingerprint_path = MODELS / "experiment_fingerprint.json"
    if fingerprint_path.exists() and json.loads(fingerprint_path.read_text("utf-8")) != fingerprints:
        raise ValueError("Frozen experiment source changed; resume refused")
    write_json(fingerprint_path, fingerprints)
    if (REPORTS / "metrics.json").exists():
        log("Final test report already complete; no repeated test-guided tuning")
        return
    # Hash verification reads raw bytes only, never test judgments for tuning.
    for name, expected in protocol["files"].items():
        if sha(DATA / name) != expected:
            raise ValueError(f"Data fingerprint mismatch: {name}")
    catalog = pd.read_parquet(DATA / "catalog.parquet")
    products = pd.read_parquet(DATA / "products.parquet")
    frames = {pool: pd.read_parquet(DATA / f"train_{pool}.parquet") for pool in ("small", "large")}
    validation = pd.read_parquet(DATA / "validation.parquet")
    log(f"Windows GPU={torch.cuda.get_device_name()} eval_catalog={len(catalog):,} training_texts={len(products):,}")
    encoder = Encoder(ROOT / "models/dense_v1/pretrained", max_length=config["max_length"])
    token_path = MODELS / "product_tokens.pt"
    if token_path.exists():
        tokens = torch.load(token_path, map_location="cpu", weights_only=True)
    else:
        log("Tokenizing expanded training text store")
        tokens = encoder.tokenize(product_texts(products)); torch.save(tokens, token_path)
    catalog_tokens = {k: v[catalog.product_row.to_numpy()] for k, v in tokens.items()}
    mining_path, sparse_path = MODELS / "mining_bm25.joblib", MODELS / "bm25.joblib"
    if mining_path.exists():
        mining_bm25 = joblib.load(mining_path)
    else:
        log("Preparing ordinal hard-negative mining index from unlabeled product text")
        mining_bm25 = BM25().fit(product_texts(products)); joblib.dump(mining_bm25, mining_path, compress=3)
    if sparse_path.exists():
        bm25 = joblib.load(sparse_path)
    else:
        bm25 = BM25().fit(product_texts(catalog)); joblib.dump(bm25, sparse_path, compress=3)
    pairs = {}
    for pool in frames:
        path = DATA / f"ordinal_pairs_{pool}.parquet"
        if path.exists():
            pairs[pool] = pd.read_parquet(path)
        else:
            pairs[pool] = mine_ordinal(frames[pool], mining_bm25, config); pairs[pool].to_parquet(path, index=False)
    if (MODELS / "baseline_validation.json").exists():
        baseline = json.loads((MODELS / "baseline_validation.json").read_text("utf-8"))
    else:
        baseline, rows, vectors, qvectors, _ = validate(encoder, validation, catalog_tokens, config)
        vectors_save(MODELS / "pretrained_vectors.npy", vectors)
        vectors_save(MODELS / "pretrained_validation_queries.npy", qvectors)
        pd.DataFrame(rows).to_csv(REPORTS / "validation_pretrained.csv", index=False)
        write_json(MODELS / "baseline_validation.json", baseline)
        del vectors, qvectors
    del encoder; gc.collect(); torch.cuda.empty_cache()
    sparse_val = sparse_rankings(validation, bm25, config)
    runs = []
    def run(name, recipe, seed=42):
        result = train_run(name, recipe, seed, config, frames, pairs, tokens, catalog_tokens, validation, baseline, mining_bm25)
        runs.append(result)
        write_json(REPORTS / "validation_trials.json", {"baseline": baseline, "runs": runs, "test_opened": False})
        return result
    for i, lr in enumerate(config["learning_rates"]):
        run(f"ordinal_lr_{i}_seed42", {"loss": "ordinal", "learning_rate": lr, "frozen_layers": 0, "anchor": False, "pool": "small"})
    lr_winner = best_run(runs)
    current = dict(lr_winner["recipe"])
    run("freeze_bottom4_seed42", {**current, "frozen_layers": config["freeze_bottom_layers"]})
    # Objective comparison keeps the full-parameter LR winner as reference.
    contrastive = run("contrastive_seed42", {**current, "loss": "contrastive"})
    run("contrastive_anchor_seed42", {**contrastive["recipe"], "anchor": True})
    small_winner = best_run(runs)
    run("expanded_pool_seed42", {**small_winner["recipe"], "pool": "large"})
    recipe_winner = best_run(runs)
    confirm = [recipe_winner]
    for seed in config["confirmation_seeds"]:
        if seed != 42:
            confirm.append(run(f"confirmation_seed{seed}", dict(recipe_winner["recipe"]), seed))
    trained_winner = best_run(confirm)
    policies = {}
    paths = {"pretrained": ROOT / "models/dense_v1/pretrained"}
    vector_paths = {"pretrained": MODELS / "pretrained_vectors.npy"}
    validation_vector_paths = {"pretrained": MODELS / "pretrained_validation_queries.npy"}
    for value in confirm:
        key = f"seed_{value['seed']}"
        paths[key] = MODELS / value["name"] / "encoder"
        vector_paths[key] = MODELS / value["name"] / "vectors.npy"
        validation_vector_paths[key] = MODELS / value["name"] / "validation_queries.npy"
    for key in paths:
        policies[key] = validation_policy(key, np.load(vector_paths[key], mmap_mode="r"), np.load(validation_vector_paths[key]), validation, sparse_val, config)
    serving = max(policies, key=lambda key: policies[key]["validation_hybrid_recall_at_100"])
    if policies[serving]["validation_hybrid_recall_at_100"] <= policies["pretrained"]["validation_hybrid_recall_at_100"] + config["minimum_improvement"]:
        serving = "pretrained"
    selection = {"selection_basis": config["selection_metric"], "recipe": recipe_winner["recipe"], "recipe_trial": recipe_winner["name"],
                 "trained_key": f"seed_{trained_winner['seed']}", "trained_trial": trained_winner["name"], "serving_key": serving,
                 "policies": policies, "runs": runs, "confirmation_trials": [value["name"] for value in confirm],
                 "encoder_paths": {key: str(path.relative_to(ROOT)) for key, path in paths.items()},
                 "vector_paths": {key: str(path.relative_to(ROOT)) for key, path in vector_paths.items()},
                 "config": config, "catalog_sha256": protocol["files"]["catalog.parquet"], "locked_before_test": True}
    write_json(MODELS / "selection.json", selection); write_json(REPORTS / "selection.json", selection)
    del mining_bm25, tokens, products, frames, pairs, catalog_tokens
    gc.collect(); torch.cuda.empty_cache()
    log(f"VALIDATION SELECTION LOCKED serving={serving} recipe={recipe_winner['name']}; opening fresh sealed test")
    test = pd.read_parquet(DATA / "test.parquet")
    sparse_test = sparse_rankings(test, bm25, config)
    summary, rows = retrieval_summary(test, sparse_test)
    report = {"experiment": "dense_v2", "status": "complete", "catalog_products": len(catalog), "test_queries": int(test.query_group.nunique()),
              "selection": selection, "validation_baseline": baseline, "retrieval": {"bm25": summary}, "candidate_ranking": {},
              "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name()}, "limits": config["limits"],
              "evaluation": "Known-E Recall and MRR in fixed catalog; NDCG evaluated separately on supplied judgments; unjudged products have no ground-truth label"}
    all_rows = [{**row, "model": "bm25"} for row in rows]
    candidate_rows = []
    test_queries = query_table(test)
    for key in paths:
        encoder = Encoder(paths[key], max_length=config["max_length"])
        vectors = np.load(vector_paths[key], mmap_mode="r")
        qvectors = encoder.encode(test_queries["query"].tolist(), config["encode_batch_size"])
        vectors_save(MODELS / f"{key}_test_queries.npy", qvectors)
        dense = dense_rankings(test_queries.query_group.tolist(), qvectors, vectors, config["retrieval_depth"])
        report["candidate_ranking"][key], per = candidate_summary(test, qvectors, vectors)
        candidate_rows.extend({"model": key, "query_group": q, "ndcg_at_10": value} for q, value in per.items())
        for name, rankings in [(f"dense_{key}", dense), (f"hybrid_{key}", hybrid_rankings(sparse_test, dense, policies[key]["alpha"], config))]:
            value, records = retrieval_summary(test, rankings)
            report["retrieval"][name] = value
            all_rows.extend({**row, "model": name} for row in records)
        log(f"FINAL {key}: {report['retrieval']['dense_' + key]}")
        del encoder; gc.collect(); torch.cuda.empty_cache()
    table = pd.DataFrame(all_rows)
    table.to_csv(REPORTS / "per_query_retrieval.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(REPORTS / "per_query_reranking.csv", index=False)
    pivot = table.pivot(index="query_group", columns="model", values="recall_at_100")
    trained = selection["trained_key"]
    report["paired_bootstrap"] = {"trained_minus_pretrained": paired_interval(pivot[f"dense_{trained}"].to_dict(), pivot.dense_pretrained.to_dict()),
                                  "serving_minus_pretrained_hybrid": paired_interval(pivot[f"hybrid_{serving}"].to_dict(), pivot.hybrid_pretrained.to_dict())}
    report["three_seed_summary"] = {kind: {metric: {"mean": float(np.mean(values := [report["retrieval"][f"{kind}_seed_{s}"][metric] for s in config["confirmation_seeds"]])),
                                                   "std": float(np.std(values, ddof=1))}
                                          for metric in ("recall_at_100", "mrr_at_200")} for kind in ("dense", "hybrid")}
    report["artifact_hashes"] = {str(path.relative_to(ROOT)): sha(path) for path in [MODELS / "selection.json", vector_paths[trained], paths[trained] / "model.safetensors"]}
    write_json(REPORTS / "metrics.json", report)
    write_json(REPORTS / "progress.json", {"status": "complete", "serving_key": serving})
    log("dense_v2 complete. Final test results do not change the locked selection.")


if __name__ == "__main__":
    main()
