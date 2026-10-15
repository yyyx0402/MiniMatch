"""Rebuild required data and the current model from a clean source checkout."""
import argparse
import json
import subprocess
import sys

from productsearch.data import ROOT, prepare


def main():
    parser = argparse.ArgumentParser(description="Prepare MiniMatch data, train the current model, and register serving assets")
    parser.add_argument("--plan", action="store_true", help="Print steps without downloading, preparing or training")
    args = parser.parse_args()
    steps = ["download_data.py", "download_encoder.py", "prepare baseline query splits",
             "prepare_dense.py", "prepare_dense_v2.py", "prepare_training_cache.py",
             "train_dense_v6.py", "prepare_application.py"]
    if args.plan:
        print("\n".join(steps))
        return
    if (ROOT / "models/application/manifest.json").exists():
        print("MiniMatch serving artifacts already exist. No retraining performed.")
        return
    wrapper = ROOT / "scripts/run_windows.py"
    def run(script):
        subprocess.run([sys.executable, str(wrapper), "scripts/" + script], cwd=ROOT, check=True)
    run("download_data.py")
    run("download_encoder.py")
    if not (ROOT / "data/processed/split_manifest.json").exists():
        prepare(json.loads((ROOT / "configs/baseline.json").read_text("utf-8")))
    run("prepare_dense.py")
    run("prepare_dense_v2.py")
    run("prepare_training_cache.py")
    # A new checkout has no older model experiments to preserve. Never replace
    # a preservation snapshot belonging to an existing local experiment.
    directory = ROOT / "reports/dense_v6"
    directory.mkdir(parents=True, exist_ok=True)
    snapshot = directory / "previous_experiments_preservation.json"
    if not snapshot.exists():
        snapshot.write_text("{}\n", encoding="utf-8")
    run("train_dense_v6.py")
    run("prepare_application.py")


if __name__ == "__main__":
    main()
