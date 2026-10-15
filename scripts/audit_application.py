"""Audit preservation of frozen model and training inputs after integration."""
import json

from productsearch.data import ROOT
from productsearch.release import MANIFEST, sha256, verify_release


def main():
    manifest = json.loads((ROOT / MANIFEST).read_text("utf-8"))
    verify_release(ROOT, manifest)
    snapshots = {
        "previous_experiments": "reports/dense_v6/previous_experiments_preservation.json",
        "current_training_inputs_and_sources": "models/dense_v6/experiment_fingerprint.json",
    }
    counts = {}
    for label, relative in snapshots.items():
        expected = json.loads((ROOT / relative).read_text("utf-8"))
        changed = [name for name, digest in expected.items() if sha256(ROOT / name) != digest]
        if changed:
            raise ValueError(f"Frozen files changed: {changed}")
        counts[label] = len(expected)
    if sha256(ROOT / manifest["source_selection"]) != manifest["selection_sha256"]:
        raise ValueError("Validation selection changed after release lock")
    report = {"status": "passed", "manifest_sha256": sha256(ROOT / MANIFEST),
              "frozen_files_verified": counts, "application_artifacts_verified": len(manifest["artifact_hashes"])}
    (ROOT / "reports/application/preservation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
