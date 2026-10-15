"""Publish a local serving manifest without altering frozen training artifacts."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from .data import ROOT

MANIFEST = Path("models/application/manifest.json")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Application artifacts must stay inside ProductSearch")
    return path


def verify_release(root, manifest):
    for relative, expected in manifest["artifact_hashes"].items():
        if sha256(artifact_path(root, relative)) != expected:
            raise ValueError(f"Application artifact changed: {relative}")


def prepare_release(root=ROOT):
    selection_path = root / "reports/dense_v6/selection.json"
    selection = json.loads(selection_path.read_text("utf-8"))
    if not selection["selection_locked"] or selection["source"] != "finetuned":
        raise ValueError("A locked fine-tuned checkpoint is required")
    trial = selection["trained_trial"]
    config = selection["config"]
    policy = selection["policies"][selection["serving_key"]]
    checkpoint = json.loads((root / f"models/dense_v6/{trial}/result.json").read_text("utf-8"))
    for name, expected in checkpoint["artifact_hashes"].items():
        if sha256(root / f"models/dense_v6/{trial}" / name) != expected:
            raise ValueError(f"Selected checkpoint changed: {name}")
    source_digest = sha256(selection_path)
    target = root / MANIFEST
    if target.exists():
        manifest = json.loads(target.read_text("utf-8"))
        if manifest["selection_sha256"] != source_digest:
            raise ValueError("Application selection is already locked")
        verify_release(root, manifest)
        return manifest
    manifest = {
        "model_name": "微调后的 MiniLM", "default_search": "hybrid",
        "selection_sha256": source_digest,
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "catalog": "data/dense_v2/catalog.parquet", "bm25": "models/dense_v2/bm25.joblib",
        "encoder": f"models/dense_v6/{trial}/encoder",
        "vectors": f"models/dense_v6/{trial}/vectors.npy",
        "pretrained_encoder": "models/dense_v1/pretrained",
        "pretrained_vectors": "models/dense_v2/pretrained_vectors.npy",
        "max_length": config["max_length"], "retrieval_depth": config["retrieval_depth"],
        "rrf_k": config["rrf_k"], "alpha": policy["alpha"],
        "pretrained_alpha": selection["policies"]["pretrained"]["alpha"],
        "source_selection": "reports/dense_v6/selection.json",
        "source_trial": trial, "checkpoint_step": selection["step"],
        "selection_basis": "validation only; user selected the current trained release",
    }
    paths = [manifest[key] for key in ("catalog", "bm25", "vectors", "pretrained_vectors")]
    for key in ("encoder", "pretrained_encoder"):
        paths.extend(str(path.relative_to(root)).replace("\\", "/")
                     for path in (root / manifest[key]).iterdir() if path.is_file())
    manifest["artifact_hashes"] = {name: sha256(root / name) for name in sorted(paths)}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest
