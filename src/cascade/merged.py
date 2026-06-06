"""Merged REAL replication corpus: Broad↔Sanger (institute) + BioGRID-ORCS
(cross-study) + Replogle (cross-cell-type), with provenance.

Each source keeps its own task/metric label (fitness KO vs transcriptomic CRISPRi)
via the `source`/`task`/`pair_type` tags — never silently mixed. Cached to
`data/cascade/merged_pairs.pkl` so retraining is fast; the cache records the
contributing data-release provenance hash.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from .corpus import DEFAULT_DATA, build_pairs, load_aligned, provenance_hash
from .types import ReplicationPair

CACHE = DEFAULT_DATA / "cascade" / "merged_pairs.pkl"


@dataclass
class MergedCorpus:
    pairs: list[ReplicationPair]
    stats: dict
    provenance_hash: str


def _stats(pairs: list[ReplicationPair]) -> dict:
    from collections import Counter

    by_type = Counter(p.pair_type for p in pairs)
    by_source = Counter(p.source for p in pairs)
    by_task = Counter(p.task for p in pairs)
    pos = sum(1 for p in pairs if p.label)
    return {
        "n_pairs": len(pairs),
        "by_pair_type": dict(by_type),
        "by_source": dict(by_source),
        "by_task": dict(by_task),
        "base_rate": round(pos / len(pairs), 4) if pairs else 0.0,
        "n_non_hits": sum(1 for p in pairs if not p.label),
    }


def build_merged_corpus(
    seed: int = 0,
    use_cache: bool = True,
    include_orcs: bool = True,
    include_replogle: bool = True,
    orcs_max: int = 30000,
) -> MergedCorpus:
    phash = provenance_hash()
    if use_cache and CACHE.exists():
        obj = pickle.loads(CACHE.read_bytes())
        if obj.get("provenance_hash") == phash:
            return MergedCorpus(pairs=obj["pairs"], stats=obj["stats"], provenance_hash=phash)

    aligned = load_aligned()
    pairs: list[ReplicationPair] = list(build_pairs(aligned, theta=0.5, seed=seed))

    if include_orcs:
        from .orcs import build_orcs_pairs, load_orcs

        orcs = load_orcs(fitness_only=True)
        pairs += build_orcs_pairs(orcs, max_pairs=orcs_max, seed=seed)

    if include_replogle:
        from .replogle import build_replogle_corpus

        rep, _, _ = build_replogle_corpus(seed=seed)
        pairs += rep.pairs

    stats = _stats(pairs)
    if use_cache:
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        CACHE.write_bytes(pickle.dumps({"pairs": pairs, "stats": stats, "provenance_hash": phash}))
    return MergedCorpus(pairs=pairs, stats=stats, provenance_hash=phash)
