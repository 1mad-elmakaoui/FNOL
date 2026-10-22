"""Local text embeddings.

Backends (``EMBEDDING_BACKEND``):

* ``sentence-transformers`` (default) — ``BAAI/bge-m3`` (1024-d, multilingual:
  fr/en/ar), run locally on CPU or GPU.
* ``hashing`` — a deterministic lexical embedding (hashed word uni/bi-grams and
  character 4-grams, 1024-d). No model download; used by the unit tests and
  handy on very small machines. Retrieval quality is lower but it is a real
  embedding: cosine similarity reflects token overlap.
"""
from __future__ import annotations

import hashlib
import logging
import re
import threading
import unicodedata
from functools import lru_cache
from typing import Protocol, Sequence

import numpy as np

from app.config import get_settings

log = logging.getLogger(__name__)

DIM = 1024


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading embedding model %s on %s", model_name, device)
        self._model = SentenceTransformer(model_name, device=device)
        self._lock = threading.Lock()
        self.name = model_name
        self.dim = int(self._model.get_sentence_embedding_dimension())
        if self.dim != DIM:
            raise RuntimeError(
                f"Embedding model {model_name} produces {self.dim}-d vectors but the chunks.embedding "
                f"column is vector({DIM}). Pick a {DIM}-d model or add a migration."
            )

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        with self._lock:
            vecs = self._model.encode(
                list(texts), batch_size=16, normalize_embeddings=True, convert_to_numpy=True
            )
        return np.asarray(vecs, dtype=np.float32)


_TOKEN = re.compile(r"\w+", re.UNICODE)
_STOPWORDS = frozenset(
    """a an and are as at be but by do for from had has have he her his i if in is it its me my no not of on or
    our she so that the their them they this to was we were what when where which who will with you your
    au aux avec ce ces c ça dans de des du elle en est et il ils je la le les leur lui ma mais me mes mon ne
    nous on ou par pas pour qu que qui sa se ses son sur ta te tu un une vos votre vous y
    في من على الى إلى عن هل و او أو ما""".split()
)


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in text if not unicodedata.combining(c))


class HashingEmbedder:
    name = "hashing"

    def __init__(self, dim: int = DIM):
        self.dim = dim

    def _bucket(self, feature: str) -> tuple[int, float]:
        h = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] & 1 else -1.0
        return idx, sign

    def _features(self, text: str) -> list[tuple[str, float]]:
        tokens = [t for t in _TOKEN.findall(_normalize(text)) if t not in _STOPWORDS]
        feats: list[tuple[str, float]] = [(f"w:{t}", 1.0) for t in tokens]
        feats += [(f"b:{a} {b}", 0.7) for a, b in zip(tokens, tokens[1:])]
        for t in tokens:
            padded = f"#{t}#"
            feats += [(f"c:{padded[i:i + 4]}", 0.25) for i in range(max(1, len(padded) - 3))]
        return feats

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feat, weight in self._features(text):
                idx, sign = self._bucket(feat)
                out[row, idx] += sign * weight
            norm = np.linalg.norm(out[row])
            if norm > 0:
                out[row] /= norm
        return out


@lru_cache(maxsize=2)
def _build(backend: str, model_name: str) -> Embedder:
    if backend == "hashing":
        return HashingEmbedder()
    if backend in {"sentence-transformers", "st", "bge"}:
        return SentenceTransformerEmbedder(model_name)
    raise ValueError(f"Unknown EMBEDDING_BACKEND={backend!r} (use 'sentence-transformers' or 'hashing')")


def get_embedder() -> Embedder:
    s = get_settings()
    return _build(s.embedding_backend, s.embedding_model)


def embed_texts(texts: Sequence[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, DIM), dtype=np.float32)
    return get_embedder().embed(texts)
