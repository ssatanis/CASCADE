"""BioGRID-ORCS loader + harmonizer (real cross-study replication labels).

Parses the real `BIOGRID-ORCS-ALL-homo_sapiens-<ver>.screens.tar.gz` (per-screen
.tab.txt files + a SCREEN_INDEX) into harmonized observations. ORCS carries the
authors' own HIT (Yes/No) call per gene per screen — including real NEGATIVES
(HIT=No), which we keep.

Harmonization (no fabrication):
  - gene = OFFICIAL_SYMBOL (HGNC symbol provided directly; no Entrez map needed)
  - modality from LIBRARY_METHODOLOGY (Knockout→KO, Inhibition→CRISPRi, Activation→CRISPRa)
  - phenotype → a small controlled vocabulary
  - cell_line → DepMap Model (StrippedCellLineName) where possible; unmapped rows
    are COUNTED, never silently dropped
  - per-screen significance = within-screen rank of |SCORE.1|, oriented by screen
    type (Negative Selection = depletion → negative). This is the SOURCE-screen
    effect; the replication LABEL comes from the OTHER screen's HIT call (no leak).
"""

from __future__ import annotations

import re
import tarfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .corpus import RAW
from .types import Context, QCBundle, ReplicationPair

NAN = float("nan")


def _qc_carrier(q: float) -> QCBundle:
    return QCBundle(NAN, NAN, float(q), NAN, NAN)


def _modality(methodology: str) -> str:
    m = (methodology or "").lower()
    if "inhib" in m:
        return "CRISPRi"
    if "activ" in m:
        return "CRISPRa"
    return "KO"  # Knockout (Cas9/base-edit) — the dominant ORCS class


def _phenotype(raw: str) -> str:
    r = (raw or "").lower()
    if any(k in r for k in ("prolif", "growth", "fitness", "viab", "essential", "depletion", "survival", "competition")):
        return "fitness"
    if any(k in r for k in ("drug", "chemical", "resist", "sensit", "toxic", "inhibitor")):
        return "drug_response"
    if any(k in r for k in ("differ", "reporter", "marker", "expression", "fluoresc")):
        return "reporter"
    return "other"


def _norm_cl(name: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (name or "").upper())


@dataclass
class ORCSScreen:
    screen_id: str
    pubmed: str
    author: str
    throughput: str
    screen_type: str
    library: str
    modality: str
    cell_line: str
    cell_type: str
    phenotype: str
    condition: str
    model_id: str | None
    lineage: str
    quality: float
    fitness_compatible: bool


@dataclass
class ScreenObs:
    screen_id: str
    pubmed: str
    signed_effect: float  # source-screen effect (sign: neg = depletion)
    hit: bool
    model_id: str | None
    cell_line: str
    lineage: str
    phenotype: str
    modality: str
    quality: float


@dataclass
class ORCSData:
    screens: dict[str, ORCSScreen]
    gene_obs: dict[str, list[ScreenObs]] = field(default_factory=dict)
    n_screens_used: int = 0
    n_obs: int = 0
    n_cell_lines_mapped: int = 0
    n_cell_lines_unmapped: int = 0
    unmapped_cell_lines: list[str] = field(default_factory=list)


def _depmap_cellline_map(model_path: Path) -> dict[str, tuple[str, str]]:
    """norm(cell-line name) -> (ModelID, OncotreeLineage) from DepMap Model.csv."""
    model = pd.read_csv(model_path)
    name_cols = [c for c in ("StrippedCellLineName", "CellLineName", "ModelIDAlias") if c in model.columns]
    lineage_col = "OncotreeLineage" if "OncotreeLineage" in model.columns else None
    out: dict[str, tuple[str, str]] = {}
    for _, r in model.iterrows():
        lin = str(r[lineage_col]) if lineage_col and pd.notna(r[lineage_col]) else "unknown"
        for c in name_cols:
            v = r.get(c)
            if isinstance(v, str) and v:
                out[_norm_cl(v)] = (str(r["ModelID"]), lin)
    return out


def _screen_quality(throughput: str, n_hits: int, scores_size: int) -> float:
    """Real screen-quality proxy in (0,1): high-throughput pooled screens with a
    sane hit fraction score higher (a genome fitness screen calls ~hundreds-thousands)."""
    tp = 0.85 if "high" in (throughput or "").lower() else 0.6
    frac = (n_hits / scores_size) if scores_size else 0.0
    # penalize degenerate hit fractions (0 or absurdly high)
    sane = 1.0 if 0.002 <= frac <= 0.5 else 0.6
    return float(min(0.99, tp * sane))


def load_orcs(
    tar_path: str | Path | None = None,
    model_path: str | Path | None = None,
    fitness_only: bool = True,
    max_obs_per_gene: int = 40,
    max_screens: int | None = None,
) -> ORCSData:
    tar_path = Path(tar_path or RAW / "BIOGRID-ORCS-ALL-homo_sapiens-2.0.18.screens.tar.gz")
    model_path = Path(model_path or RAW / "Model.csv")
    if not tar_path.exists():
        raise FileNotFoundError(f"ORCS archive missing: {tar_path} (run acquire.py --only graphs)")

    cl_map = _depmap_cellline_map(model_path)

    # ---- pass 1: index ----
    screens: dict[str, ORCSScreen] = {}
    mapped, unmapped, unmapped_names = 0, 0, set()
    with tarfile.open(tar_path, "r:gz") as tar:
        index_member = next(m for m in tar.getmembers() if "SCREEN_INDEX" in m.name)
        idx = pd.read_csv(tar.extractfile(index_member), sep="\t", dtype=str).fillna("")
        idx.columns = [c.lstrip("#") for c in idx.columns]
        for _, r in idx.iterrows():
            modality = _modality(r.get("LIBRARY_METHODOLOGY", ""))
            screen_type = r.get("SCREEN_TYPE", "")
            phenotype = _phenotype(r.get("PHENOTYPE", ""))
            fitness_compat = ("Negative Selection" in screen_type) or (phenotype == "fitness")
            if fitness_only and not (fitness_compat and modality == "KO"):
                continue
            cl_raw = r.get("CELL_LINE", "")
            m = cl_map.get(_norm_cl(cl_raw))
            if m:
                model_id, lineage = m
                mapped += 1
            else:
                model_id, lineage = None, "unknown"
                if cl_raw:
                    unmapped += 1
                    unmapped_names.add(cl_raw)
            try:
                n_hits = int(r.get("NUMBER_OF_HITS") or 0)
                scores_size = int(r.get("SCORES_SIZE") or 0)
            except ValueError:
                n_hits, scores_size = 0, 0
            sid = r["SCREEN_ID"]
            screens[sid] = ORCSScreen(
                screen_id=sid, pubmed=r.get("SOURCE_ID", ""), author=r.get("AUTHOR", ""),
                throughput=r.get("THROUGHPUT", ""), screen_type=screen_type,
                library=r.get("LIBRARY", ""), modality=modality,
                cell_line=cl_raw, cell_type=r.get("CELL_TYPE", ""), phenotype=phenotype,
                condition=r.get("CONDITION_NAME", "") or "baseline",
                model_id=model_id, lineage=lineage,
                quality=_screen_quality(r.get("THROUGHPUT", ""), n_hits, scores_size),
                fitness_compatible=fitness_compat,
            )

    if max_screens is not None:
        keep = set(list(screens.keys())[:max_screens])
        screens = {k: v for k, v in screens.items() if k in keep}

    # ---- pass 2: per-screen gene rows ----
    gene_obs: dict[str, list[ScreenObs]] = {}
    n_obs = 0
    screen_re = re.compile(r"SCREEN_(\d+)-")
    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar:
            if "SCREEN_INDEX" in member.name or not member.name.endswith(".screen.tab.txt"):
                continue
            mo = screen_re.search(member.name)
            if not mo:
                continue
            sid = mo.group(1)
            sc = screens.get(sid)
            if sc is None:
                continue
            f = tar.extractfile(member)
            if f is None:
                continue
            df = pd.read_csv(f, sep="\t", dtype=str).fillna("")
            df.columns = [c.lstrip("#") for c in df.columns]
            if "OFFICIAL_SYMBOL" not in df.columns or "HIT" not in df.columns:
                continue
            score = pd.to_numeric(df.get("SCORE.1", pd.Series([], dtype=str)), errors="coerce")
            absscore = score.abs()
            # within-screen significance rank of |SCORE.1| in (0,1]
            rank = absscore.rank(pct=True).fillna(0.0).to_numpy()
            sign = -1.0 if "Negative Selection" in sc.screen_type else 1.0
            symbols = df["OFFICIAL_SYMBOL"].to_numpy()
            hits = (df["HIT"].str.upper() == "YES").to_numpy()
            for i in range(len(df)):
                g = symbols[i]
                if not g or g == "-":
                    continue
                lst = gene_obs.setdefault(g, [])
                if len(lst) >= max_obs_per_gene and not hits[i]:
                    continue  # cap; keep hits preferentially
                lst.append(
                    ScreenObs(
                        screen_id=sid, pubmed=sc.pubmed, signed_effect=float(sign * rank[i]),
                        hit=bool(hits[i]), model_id=sc.model_id, cell_line=sc.cell_line,
                        lineage=sc.lineage, phenotype=sc.phenotype, modality=sc.modality,
                        quality=sc.quality,
                    )
                )
                n_obs += 1

    return ORCSData(
        screens=screens, gene_obs=gene_obs, n_screens_used=len(screens), n_obs=n_obs,
        n_cell_lines_mapped=mapped, n_cell_lines_unmapped=unmapped,
        unmapped_cell_lines=sorted(unmapped_names),
    )


def build_orcs_pairs(
    orcs: ORCSData,
    max_pairs: int = 30000,
    per_source_targets: int = 2,
    max_sources_per_gene: int = 3,
    seed: int = 0,
) -> list[ReplicationPair]:
    """Mint real cross-STUDY replication pairs from ORCS.

    A pair is created from a gene that is a HIT in source screen A (different
    study/pubmed from target B); the LABEL is whether B independently called it a
    hit (both-hit concordance). Features come from A, the label from B → no leak.
    Non-replication negatives (B = No) and same-vs-different cell line are both
    represented. Tagged pair_type='cross_study', source='orcs', task='fitness'.
    """
    rng = np.random.default_rng(seed)
    pairs: list[ReplicationPair] = []
    genes = [g for g, o in orcs.gene_obs.items() if len(o) >= 2]
    rng.shuffle(genes)
    for g in genes:
        obs = orcs.gene_obs[g]
        hit_sources = [x for x in obs if x.hit]
        if not hit_sources:
            continue
        if len(hit_sources) > max_sources_per_gene:
            hit_sources = [hit_sources[i] for i in rng.choice(len(hit_sources), max_sources_per_gene, replace=False)]
        for a in hit_sources:
            others = [b for b in obs if b.pubmed != a.pubmed]
            if not others:
                continue
            rng.shuffle(others)
            for b in others[:per_source_targets]:
                same_cl = (a.model_id is not None and a.model_id == b.model_id) or (a.cell_line == b.cell_line)
                pairs.append(
                    ReplicationPair(
                        gene=g,
                        context_a=Context(cell_line=a.cell_line or "ORCS_A", lineage=a.lineage),
                        context_b=Context(cell_line=b.cell_line or "ORCS_B", lineage=b.lineage),
                        beta_a=a.signed_effect, var_a=max(0.01, (1 - a.quality) * 0.1 + 0.01),
                        beta_b=b.signed_effect, var_b=max(0.01, (1 - b.quality) * 0.1 + 0.01),
                        qc_a=_qc_carrier(a.quality), qc_b=_qc_carrier(b.quality),
                        modality="KO", edist_a=a.quality, label=bool(b.hit),
                        quality_a=a.quality, quality_b=b.quality,
                        pair_type="cross_study_same_cell" if same_cl else "cross_study",
                        source="orcs", task="fitness", study=a.pubmed,
                    )
                )
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs
