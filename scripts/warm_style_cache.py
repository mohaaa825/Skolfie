"""
scripts/warm_style_cache.py
---------------------------
Compute the style features (no language model, CPU only) of every corpus text
into the shared feature cache, so the GPU session of `retrain_modern.ipynb`
only spends GPU time on what needs a GPU (LM / Binoculars features and the
transformers). Texts already in the cache are skipped; progress is saved as it
goes, so an interrupted run resumes.

    python scripts/warm_style_cache.py --corpus data/processed_v3/corpus.parquet \\
        --cache-dir /content/drive/MyDrive/skolfie/cache
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--n-jobs", type=int, default=-1)
    p.add_argument("--batch", type=int, default=20000, help="texts per progress step")
    args = p.parse_args(argv)

    import pandas as pd

    from skolfie.models.feature_model import FeatureModel

    texts = pd.read_parquet(args.corpus, columns=["text"])["text"].tolist()
    model = FeatureModel(use_lm=False, cache_dir=args.cache_dir, n_jobs=args.n_jobs,
                         log=lambda m: print(m, flush=True))
    t0 = time.time()
    for start in range(0, len(texts), args.batch):
        model.base_features(texts[start:start + args.batch])
        for cache in model._caches.values():
            cache.flush()
        print(f"style features: {min(start + args.batch, len(texts)):,}/{len(texts):,} texts "
              f"({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
