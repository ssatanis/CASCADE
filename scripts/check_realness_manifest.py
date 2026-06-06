#!/usr/bin/env python3
"""CI realness check: provenance manifest + shipped artifact are real.

Asserts:
  1. data/PROVENANCE.json exists.
  2. every REQUIRED real dataset is status=ok with url + sha256 + release +
     access_date + license all present and non-empty.
  3. the shipped Oracle artifact exists, its metadata has trained_on_real_data:true,
     and its provenance_hash matches the manifest (the release pin).

Unresolved SECONDARY sources are reported (non-fatal) — never silently
substituted (the realness invariant). Run with the cascade venv python.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MANIFEST = REPO / "data" / "PROVENANCE.json"
ARTIFACT = REPO / "cascade" / "artifacts" / "oracle_v0.pkl"

REQUIRED = [
    "DepMap Public 26Q1/CRISPRGeneEffect.csv",
    "DepMap Public 26Q1/Model.csv",
    "Sanger CRISPR (Project Score, Chronos) v2/gene_effect.csv",
    "Sanger CRISPR (Project Score, Chronos) v2/common_essentials.csv",
    "BIOGRID-ORCS-ALL-homo_sapiens-2.0.18.screens.tar.gz",
    "ReplogleWeissman2022_K562_essential.h5ad",
    "ReplogleWeissman2022_rpe1.h5ad",
]
REQUIRED_FIELDS = ["url", "sha256", "release", "access_date", "license"]


def fail(msg: str) -> None:
    print(f"❌ {msg}")
    sys.exit(1)


def main() -> int:
    if not MANIFEST.exists():
        fail(f"missing provenance manifest: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text())
    entries = manifest.get("entries", {})

    # 1-2. required real datasets complete + ok
    for key in REQUIRED:
        e = entries.get(key)
        if e is None:
            fail(f"required dataset missing from manifest: {key}")
        if e.get("status") != "ok":
            fail(f"required dataset not ok: {key} (status={e.get('status')})")
        for f in REQUIRED_FIELDS:
            if not e.get(f):
                fail(f"required dataset {key} missing field '{f}'")
    print(f"✅ {len(REQUIRED)} required real datasets present + provenance-complete")

    # acceptance: nothing unresolved (every source resolved or honestly absent)
    unresolved = [k for k, v in entries.items() if v.get("status") == "unresolved"]
    if unresolved:
        fail(f"{len(unresolved)} unresolved source(s) (must be resolved or removed, never substituted): {unresolved}")
    print(f"✅ 0 unresolved sources ({len(entries)} entries all resolved)")

    # 3. artifact realness + provenance match
    if not ARTIFACT.exists():
        fail(f"no shipped Oracle artifact: {ARTIFACT} (run `cascade train`)")
    sys.path.insert(0, str(REPO / "cascade" / "src"))
    from cascade.corpus import provenance_hash
    from cascade.oracle import ReplicationOracle

    _, meta = ReplicationOracle.load(ARTIFACT)  # raises unless trained_on_real_data
    if meta.get("trained_on_real_data") is not True:
        fail("artifact metadata is not trained_on_real_data: true")
    ph_manifest = provenance_hash(MANIFEST)
    if meta.get("provenance_hash") != ph_manifest:
        fail(
            f"artifact provenance_hash {meta.get('provenance_hash', '')[:12]} != manifest {ph_manifest[:12]} "
            "(retrain on the current data release)"
        )
    print(f"✅ artifact trained_on_real_data: true; provenance_hash matches manifest ({ph_manifest[:12]})")
    print(f"   releases: {meta.get('data_releases')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
