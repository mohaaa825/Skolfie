"""
skolfie/features/cache.py
--------------------------
A disk cache for per-text feature vectors, keyed by a hash of the text.

GPT-2 features for the whole corpus take tens of minutes on a GPU, and a
Colab session can drop at any time, so `FeatureCache.compute()` saves its
progress every `flush_every` new rows. Re-running an evaluation then only
computes features for texts it has not seen before.

One cache file holds one feature set:

    <cache_dir>/<name>.parquet      columns: key, then one float32 column per feature

`name` should change whenever the features change (the callers put a
version number and the model/truncation settings in it). A file whose
columns do not match the expected feature names is ignored and rebuilt.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def text_key(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


class FeatureCache:
    def __init__(self, cache_dir: str | os.PathLike[str] | None, name: str,
                 feature_names: Sequence[str], flush_every: int = 20_000,
                 log: Callable[[str], None] = print) -> None:
        self.path = None if cache_dir is None else Path(cache_dir) / f"{name}.parquet"
        self.name = name
        self.feature_names = list(feature_names)
        self.flush_every = flush_every
        self.log = log
        self._index: dict[str, int] = {}
        self._rows: list[np.ndarray] = []
        self._dirty = False
        self._load()

    def __len__(self) -> int:
        return len(self._index)

    def _load(self) -> None:
        if self.path is None or not self.path.is_file():
            return
        import pyarrow.parquet as pq

        table = pq.read_table(self.path)
        if table.column_names != ["key", *self.feature_names]:
            self.log(f"[cache] {self.path}: feature columns changed, ignoring the old file")
            return
        keys = table.column("key").to_pylist()
        values = np.column_stack([table.column(c).to_numpy(zero_copy_only=False)
                                  for c in self.feature_names]).astype(np.float32)
        self._index = {k: i for i, k in enumerate(keys)}
        self._rows = [values]
        self.log(f"[cache] {self.name}: {len(keys):,} cached rows from {self.path}")

    def _matrix(self) -> np.ndarray:
        if len(self._rows) != 1:
            self._rows = [np.concatenate(self._rows, axis=0) if self._rows else
                          np.zeros((0, len(self.feature_names)), dtype=np.float32)]
        return self._rows[0]

    def flush(self) -> None:
        if self.path is None or not self._dirty:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        matrix = self._matrix()
        keys = sorted(self._index, key=self._index.__getitem__)
        columns = {"key": pa.array(keys, type=pa.string())}
        columns.update({name: pa.array(matrix[:, j]) for j, name in enumerate(self.feature_names)})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".parquet.tmp")
        pq.write_table(pa.table(columns), tmp)
        os.replace(tmp, self.path)
        self._dirty = False

    def compute(self, texts: Sequence[str],
                fn: Callable[[list[str]], np.ndarray]) -> np.ndarray:
        """Feature matrix (len(texts), n_features) float32; `fn` runs only on uncached texts."""
        keys = [text_key(t) for t in texts]
        todo: dict[str, str] = {}
        for k, t in zip(keys, texts):
            if k not in self._index and k not in todo:
                todo[k] = t
        if todo:
            self.log(f"[cache] {self.name}: computing {len(todo):,} of {len(keys):,} texts "
                     f"({len(keys) - sum(1 for k in keys if k in todo):,} cached)")
            items = list(todo.items())
            step = self.flush_every if self.path is not None else len(items)
            for start in range(0, len(items), step):
                chunk = items[start:start + step]
                values = np.asarray(fn([t for _, t in chunk]), dtype=np.float32)
                if values.shape != (len(chunk), len(self.feature_names)):
                    raise ValueError(f"{self.name}: feature function returned {values.shape}, "
                                     f"expected {(len(chunk), len(self.feature_names))}")
                offset = len(self._index)
                for i, (k, _) in enumerate(chunk):
                    self._index[k] = offset + i
                self._rows.append(values)
                self._dirty = True
                self.flush()
        matrix = self._matrix()
        return matrix[[self._index[k] for k in keys]]
