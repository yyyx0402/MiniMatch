"""Download the pinned official Amazon ESCI data and verify LFS SHA-256."""
import hashlib
import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REVISION = "7916cdf6ab75a462e77f20ab40428a10923998d5"
FILES = {
    "shopping_queries_dataset_examples.parquet": (51286808, "4a735b693b4a424a6fc67f5be6e4c811495c488bbf66d02a602d308b2744263a"),
    "shopping_queries_dataset_products.parquet": (1108857465, "25124442d064d64b26f74082d6fa09438d679efc0c183cf28d19064a2b65a265"),
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    output = ROOT / "data/raw"
    output.mkdir(parents=True, exist_ok=True)
    manifest = {"repository":"https://github.com/amazon-science/esci-data", "revision":REVISION, "files":{}}
    for name, (size, sha) in FILES.items():
        url = f"https://media.githubusercontent.com/media/amazon-science/esci-data/{REVISION}/shopping_queries_dataset/{name}"
        target = output / name
        partial = output / (name + ".part")
        if not (target.exists() and target.stat().st_size == size and digest(target) == sha):
            for attempt in range(3):
                try:
                    offset = partial.stat().st_size if partial.exists() else 0
                    request = urllib.request.Request(url, headers={"User-Agent":"ProductSearch/0.1", "Range":f"bytes={offset}-"} if offset else {"User-Agent":"ProductSearch/0.1"})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        append = offset > 0 and response.status == 206
                        total = offset if append else 0
                        last_notice = total
                        with partial.open("ab" if append else "wb") as stream:
                            while block := response.read(4 * 1024 * 1024):
                                stream.write(block)
                                total += len(block)
                                if total - last_notice >= 100 * 1024 * 1024:
                                    print(f"{name}: {total / size:.0%}", flush=True)
                                    last_notice = total
                    if partial.stat().st_size != size or digest(partial) != sha:
                        raise ValueError("Downloaded data does not match the official LFS fingerprint")
                    partial.replace(target)
                    break
                except Exception as exc:
                    print(f"Attempt {attempt+1} failed: {type(exc).__name__}: {exc}", flush=True)
                    if attempt == 2:
                        raise
                    time.sleep(2)
        manifest["files"][name] = {"url":url,"bytes":size,"sha256":sha}
        print(f"Verified {name}: {size:,} bytes", flush=True)
    for name in ("LICENSE", "NOTICE"):
        url = f"https://raw.githubusercontent.com/amazon-science/esci-data/{REVISION}/{name}"
        with urllib.request.urlopen(url, timeout=30) as response:
            (output / name).write_bytes(response.read())
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), "utf-8")


if __name__ == "__main__":
    main()
