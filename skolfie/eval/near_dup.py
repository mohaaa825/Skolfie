"""
skolfie/eval/near_dup.py
------------------------
Near-duplicate clusters of corpus texts, so that two texts that differ by a
few words never end up on different sides of a train / test split.

Exact duplicates are already removed by the corpus builder; this catches the
rest: the same answer with a typo fixed, a WikiHow article copied into a
generation, a question that occurs twice in a source with lightly edited
answers.

Method (MinHash + LSH, all deterministic):
    * Text -> lowercase word tokens -> overlapping word `shingle`-grams.
      Each word gets a stable 64-bit id (blake2b), each shingle a 64-bit
      hash mixed from its words' ids, so nothing depends on PYTHONHASHSEED.
    * MinHash signature of `num_perm` values per text (universal hashing
      modulo the Mersenne prime 2^31 - 1, fixed seeds).
    * LSH with `bands` bands of `num_perm // bands` rows: texts that share a
      band bucket are candidates. Each candidate is compared with the
      bucket's first text (a star, so huge buckets stay linear), and joined
      when the signatures agree on at least `threshold` of their values
      (an estimate of the Jaccard similarity of the shingle sets).
    * Union-find over the joined pairs gives the clusters.

Texts with fewer than `shingle` words get no signature and are never joined
(too short to call near-duplicates). Cost: about a minute for 270k texts on
one CPU core.

`cluster_ids(texts)` returns one int per text: -1 for a text with no near
duplicate, else the cluster's smallest row position (so the labels do not
depend on hashing or bucket order).
"""

from __future__ import annotations

import hashlib
import re
from typing import Sequence

import numpy as np

_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.UNICODE)
_MASK61 = (1 << 61) - 1
_P31 = (1 << 31) - 1

DEFAULT_THRESHOLD = 0.8
DEFAULT_SHINGLE = 5
DEFAULT_NUM_PERM = 64
DEFAULT_BANDS = 16


def _word_id(word: str, cache: dict[str, int]) -> int:
    value = cache.get(word)
    if value is None:
        value = int.from_bytes(hashlib.blake2b(word.encode("utf-8"), digest_size=8).digest(),
                               "little")
        cache[word] = value
    return value


def shingle_hashes(text: str, shingle: int = DEFAULT_SHINGLE,
                   cache: dict[str, int] | None = None) -> np.ndarray:
    """Unique 61-bit hashes of the text's word `shingle`-grams (empty if too short)."""
    cache = {} if cache is None else cache
    words = _WORD.findall(text.lower())
    if len(words) < shingle:
        return np.empty(0, dtype=np.uint64)
    ids = np.fromiter((_word_id(w, cache) for w in words), dtype=np.uint64, count=len(words))
    n = len(ids) - shingle + 1
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for k in range(shingle):
            # rotate-and-xor mixing: order-sensitive, wraps modulo 2^64
            h = (h * np.uint64(0x100000001B3)) ^ ids[k:k + n]
    return np.unique(h & np.uint64(_MASK61))


def _perm_params(num_perm: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Universal-hash parameters modulo the Mersenne prime 2^31 - 1."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, _P31, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, _P31, size=num_perm, dtype=np.uint64)
    return a, b


def signatures(texts: Sequence[str], *, shingle: int = DEFAULT_SHINGLE,
               num_perm: int = DEFAULT_NUM_PERM, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(signature matrix (n, num_perm) uint64, has_signature bool mask)."""
    a, b = _perm_params(num_perm, seed)
    p31 = np.uint64(_P31)
    n = len(texts)
    sig = np.full((n, num_perm), np.iinfo(np.uint64).max, dtype=np.uint64)
    has = np.zeros(n, dtype=bool)
    cache: dict[str, int] = {}
    for i, text in enumerate(texts):
        hs = shingle_hashes(text or "", shingle, cache)
        if hs.size == 0:
            continue
        x = (hs % p31)[:, None]                      # < 2^31
        v = (x * a[None, :] + b[None, :]) % p31  # < 2^62 before the mod: exact
        sig[i] = v.min(axis=0)
        has[i] = True
    return sig, has


class _UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = np.arange(n, dtype=np.int64)

    def find(self, i: int) -> int:
        parent = self.parent
        root = i
        while parent[root] != root:
            root = parent[root]
        while parent[i] != root:
            parent[i], i = root, parent[i]
        return int(root)

    def union(self, i: int, j: int) -> None:
        ri, rj = self.find(i), self.find(j)
        if ri != rj:
            # smaller position becomes the root, so labels are order-free
            if ri < rj:
                self.parent[rj] = ri
            else:
                self.parent[ri] = rj


def cluster_ids(texts: Sequence[str], *, threshold: float = DEFAULT_THRESHOLD,
                shingle: int = DEFAULT_SHINGLE, num_perm: int = DEFAULT_NUM_PERM,
                bands: int = DEFAULT_BANDS, seed: int = 0) -> np.ndarray:
    """Cluster label per text: -1 = no near duplicate, else the smallest position in its cluster."""
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    if bands < 1 or num_perm % bands:
        raise ValueError(f"num_perm ({num_perm}) must be a multiple of bands ({bands})")
    texts = list(texts)
    n = len(texts)
    out = np.full(n, -1, dtype=np.int64)
    if n < 2:
        return out
    sig, has = signatures(texts, shingle=shingle, num_perm=num_perm, seed=seed)
    idx = np.flatnonzero(has)
    if idx.size < 2:
        return out
    uf = _UnionFind(n)
    rows = num_perm // bands
    joined = np.zeros(n, dtype=bool)
    for band in range(bands):
        block = np.ascontiguousarray(sig[idx, band * rows:(band + 1) * rows])
        keys = block.view(np.dtype((np.void, block.dtype.itemsize * rows))).ravel()
        order = np.argsort(keys, kind="stable")
        sorted_keys = keys[order]
        starts = np.flatnonzero(np.r_[True, sorted_keys[1:] != sorted_keys[:-1]])
        ends = np.r_[starts[1:], len(order)]
        for s, e in zip(starts, ends):
            if e - s < 2:
                continue
            members = idx[order[s:e]]
            head = members[0]
            agree = (sig[members[1:]] == sig[head]).mean(axis=1)
            for m in members[1:][agree >= threshold]:
                uf.union(int(head), int(m))
                joined[head] = joined[m] = True
    for i in np.flatnonzero(joined):
        out[i] = uf.find(int(i))
    return out
