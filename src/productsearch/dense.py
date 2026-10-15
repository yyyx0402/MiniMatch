"""Local MiniLM shared encoder: masked mean pooling and unit-length embeddings."""
import os
from pathlib import Path

from .data import ROOT

os.environ.setdefault("HF_HOME", str(ROOT / "runtime/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

def product_texts(catalog):
    return (catalog.product_title + " | " + catalog.product_brand + " | " + catalog.product_color
            + " | " + catalog.product_bullet_point + " | " + catalog.product_description).tolist()


class Encoder:
    def __init__(self, path, device=None, max_length=128):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModel.from_pretrained(str(path), local_files_only=True, use_safetensors=True).to(self.device)

    def tokenize(self, texts):
        texts = list(texts)
        output = {}
        for start in range(0,len(texts),1024):
            encoded = self.tokenizer(texts[start:start+1024], max_length=self.max_length, truncation=True,
                                     padding="max_length", return_tensors="pt")
            for key,value in encoded.items():
                if key not in output:
                    output[key]=torch.empty((len(texts),self.max_length),dtype=torch.int32)
                output[key][start:start+len(value)]=value
        return output

    def forward(self, tokens):
        output = self.model(**tokens).last_hidden_state
        mask = tokens["attention_mask"].unsqueeze(-1).to(output.dtype)
        return F.normalize((output * mask).sum(1).float() / mask.sum(1).clamp_min(1).float(), p=2, dim=1)

    def encode_tokens(self, tokens, batch_size=128, progress=None):
        self.model.eval()
        count = len(tokens["input_ids"])
        output = np.empty((count, self.model.config.hidden_size), dtype=np.float32)
        with torch.inference_mode():
            for start in range(0, count, batch_size):
                batch = {k: v[start:start+batch_size].to(self.device, dtype=torch.long) for k, v in tokens.items()}
                with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                    vectors = self.forward(batch)
                output[start:start+len(vectors)] = vectors.cpu().numpy()
                if progress and (start == 0 or (start // batch_size + 1) % 100 == 0):
                    progress(f"Encoded {min(start+batch_size,count):,}/{count:,}")
        return output

    def encode(self, texts, batch_size=128, progress=None):
        return self.encode_tokens(self.tokenize(texts), batch_size, progress)

    def save(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(directory), safe_serialization=True)
        self.tokenizer.save_pretrained(str(directory))


def pairwise_loss(query, positive, negative, weights, temperature=.1):
    delta = (query * negative).sum(-1) - (query * positive).sum(-1)
    return (F.softplus(delta / temperature) * weights).sum() / weights.sum().clamp_min(1e-8)


def top_k(scores, k=200):
    """Scores are in ascending-product-ID catalog order; ties choose smaller IDs."""
    scores = np.asarray(scores)
    k = min(k, len(scores))
    if k == 0:
        return np.empty(0, dtype=np.int64)
    threshold = np.partition(scores, len(scores)-k)[len(scores)-k]
    above = np.flatnonzero(scores > threshold)
    tied = np.flatnonzero(scores == threshold)[:k-len(above)]
    ids = np.concatenate([above, tied])
    return ids[np.lexsort((ids, -scores[ids]))]


def reciprocal_rank_fusion(sparse, dense, alpha=.5, k=60, limit=200):
    """Fuse two unique catalog-row rankings; missing candidates contribute zero."""
    scores = {}
    for weight, ranking in ((1-alpha, sparse), (alpha, dense)):
        if weight == 0:
            continue
        for rank, item in enumerate(ranking, 1):
            scores[int(item)] = scores.get(int(item), 0.0) + weight / (k + rank)
    ordered = sorted(scores, key=lambda item: (-scores[item], item))[:limit]
    return np.asarray(ordered, dtype=np.int64), np.asarray([scores[item] for item in ordered])
