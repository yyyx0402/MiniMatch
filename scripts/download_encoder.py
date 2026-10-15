"""Download only the pinned safetensors encoder and tokenizer, no executable repo code."""
import hashlib
import json
import urllib.request

from productsearch.data import ROOT

FILES = ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "vocab.txt", "sentence_bert_config.json", "1_Pooling/config.json", "README.md"]


def main():
    config = json.loads((ROOT / "configs/dense_v1.json").read_text("utf-8"))
    directory = ROOT / "models/dense_v1/pretrained"
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"model_id": config["model_id"], "revision": config["model_revision"], "files": {}}
    for name in FILES:
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/{config['model_id']}/resolve/{config['model_revision']}/{name}"
        if not target.exists():
            request = urllib.request.Request(url, headers={"User-Agent": "ProductSearch/0.2"})
            with urllib.request.urlopen(request, timeout=60) as response:
                partial = target.with_suffix(target.suffix + ".part")
                with partial.open("wb") as stream:
                    while block := response.read(4 * 1024 * 1024):
                        stream.write(block)
                partial.replace(target)
        sha = hashlib.sha256(target.read_bytes()).hexdigest()
        manifest["files"][name] = {"bytes": target.stat().st_size, "sha256": sha, "url": url}
        print(f"Encoder file {name}: {target.stat().st_size:,} bytes", flush=True)
    (directory / "download_manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")


if __name__ == "__main__":
    main()
