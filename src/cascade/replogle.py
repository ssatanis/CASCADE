"""Replogle 2022 genome-wide Perturb-seq → REAL cross-CELL-TYPE replication labels.

The same CRISPRi perturbations were applied in two different cell types (K562,
RPE1). For each perturbed gene we compute, per cell type, from the real
single-cell data:

  - pseudobulk Δ = mean(perturbed cells) − mean(control cells) over the common
    readout genes (the real transcriptomic effect)
  - effect size = ‖Δ‖ (the mean-shift magnitude)
  - E-distance = scPerturb energy distance between the perturbed and control cell
    clouds in PCA space (dispersion-aware effect size)

For genes perturbed in BOTH cell types, the cross-cell-type replication label is
direction concordance of the Δ vectors + both effects being real. This is the
first real "does this hit replicate across cell type" signal — a DISTINCT
transcriptomic task, never pooled with the fitness (KO) metric.

Memory-safe: a PCA basis is fit on a cell subsample, then the data is streamed
once in row chunks (peak well under the file size).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .corpus import DEFAULT_DATA, RAW
from .types import Context, QCBundle, ReplicationPair

NAN = float("nan")
CACHE_DIR = DEFAULT_DATA / "replogle"


def _save_effects(eff: "CellTypeEffects", key: str) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    genes = eff.genes
    delta = np.vstack([eff.delta[g] for g in genes]) if genes else np.zeros((0, eff.n_common_genes))
    np.savez_compressed(
        CACHE_DIR / f"{key}.npz",
        cell_line=eff.cell_line, genes=np.array(genes, dtype=object), delta=delta,
        effect_size=np.array([eff.effects[g].effect_size for g in genes]),
        edistance=np.array([eff.effects[g].edistance for g in genes]),
        n_cells=np.array([eff.effects[g].n_cells for g in genes]),
        n_control=eff.n_control, n_common_genes=eff.n_common_genes,
    )


def _load_effects(key: str) -> "CellTypeEffects | None":
    p = CACHE_DIR / f"{key}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    genes = list(z["genes"])
    # materialize arrays ONCE (npz members are lazy; indexing inside a loop reloads them)
    delta_mat = z["delta"]
    eff_arr = z["effect_size"]
    ed_arr = z["edistance"]
    nc_arr = z["n_cells"]
    delta = {g: delta_mat[i] for i, g in enumerate(genes)}
    effects = {
        g: PertEffect(gene=g, effect_size=float(eff_arr[i]), edistance=float(ed_arr[i]), n_cells=int(nc_arr[i]))
        for i, g in enumerate(genes)
    }
    return CellTypeEffects(
        cell_line=str(z["cell_line"]), genes=genes, delta=delta, effects=effects,
        n_control=int(z["n_control"]), n_common_genes=int(z["n_common_genes"]),
    )


def _qc_carrier(q: float) -> QCBundle:
    return QCBundle(NAN, NAN, float(q), NAN, NAN)


@dataclass
class PertEffect:
    gene: str
    effect_size: float       # ‖pseudobulk Δ‖ over common readout genes
    edistance: float         # energy distance (PCA space), perturbed vs control
    n_cells: int


@dataclass
class CellTypeEffects:
    cell_line: str
    genes: list[str]
    delta: dict[str, np.ndarray]      # gene -> Δ over common readout genes
    effects: dict[str, PertEffect]
    n_control: int
    n_common_genes: int


def _energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """scPerturb energy distance: 2·E‖x−y‖ − E‖x−x'‖ − E‖y−y'‖ (Euclidean)."""
    if len(x) < 2 or len(y) < 2:
        return 0.0
    def _md(a, b):
        d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
        return float(d.mean())
    return max(0.0, 2 * _md(x, y) - _md(x, x) - _md(y, y))


def compute_cell_type_effects(
    h5ad_path: str | Path,
    common_genes: list[str],
    n_pcs: int = 50,
    pca_subsample: int = 8000,
    reservoir: int = 30,
    chunk: int = 10000,
    seed: int = 0,
) -> CellTypeEffects:
    import anndata as ad

    rng = np.random.default_rng(seed)
    A = ad.read_h5ad(h5ad_path, backed="r")
    var_index = {g: i for i, g in enumerate(A.var.index)}
    common = [g for g in common_genes if g in var_index]
    col_idx = np.array([var_index[g] for g in common])
    pert = A.obs["perturbation"].astype(str).to_numpy()
    n = A.n_obs
    cell_line = str(A.obs["cell_line"].iloc[0]) if "cell_line" in A.obs else "?"

    # --- PCA basis from a subsample (standardized common genes) ---
    sub = np.sort(rng.choice(n, size=min(pca_subsample, n), replace=False))
    Xsub = np.asarray(A[sub, :].X[:, col_idx], dtype=np.float32)
    mu = Xsub.mean(0)
    sd = Xsub.std(0)
    sd[sd == 0] = 1.0
    from sklearn.decomposition import PCA

    pca = PCA(n_components=min(n_pcs, Xsub.shape[1] - 1, len(sub) - 1), random_state=seed)
    pca.fit((Xsub - mu) / sd)
    del Xsub

    # --- single chunked pass: Δ accumulation + PCA reservoir per perturbation ---
    sums: dict[str, np.ndarray] = {}
    counts: dict[str, int] = {}
    pc_reservoir: dict[str, list[np.ndarray]] = {}
    ctrl_sum = np.zeros(len(col_idx), dtype=np.float64)
    ctrl_count = 0
    ctrl_pc: list[np.ndarray] = []

    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        Xc = np.asarray(A[start:stop, :].X[:, col_idx], dtype=np.float32)
        pc = pca.transform((Xc - mu) / sd)
        labels = pert[start:stop]
        for i in range(stop - start):
            g = labels[i]
            if g == "control":
                ctrl_sum += Xc[i]
                ctrl_count += 1
                if len(ctrl_pc) < 200:
                    ctrl_pc.append(pc[i])
                continue
            if g not in sums:
                sums[g] = np.zeros(len(col_idx), dtype=np.float64)
                counts[g] = 0
                pc_reservoir[g] = []
            sums[g] += Xc[i]
            counts[g] += 1
            if len(pc_reservoir[g]) < reservoir:
                pc_reservoir[g].append(pc[i])
        del Xc, pc

    ctrl_mean = ctrl_sum / max(ctrl_count, 1)
    ctrl_arr = np.array(ctrl_pc) if ctrl_pc else np.zeros((0, pca.n_components_))
    # precompute control self-distance once (energy distance shares this term)
    ctrl_self = float(np.linalg.norm(ctrl_arr[:, None, :] - ctrl_arr[None, :, :], axis=2).mean()) if len(ctrl_arr) >= 2 else 0.0

    delta: dict[str, np.ndarray] = {}
    effects: dict[str, PertEffect] = {}
    for g, s in sums.items():
        if counts[g] < 5:
            continue
        d = s / counts[g] - ctrl_mean
        delta[g] = d.astype(np.float32)
        res = np.array(pc_reservoir[g])
        if len(res) >= 2 and len(ctrl_arr) >= 2:
            cross = float(np.linalg.norm(res[:, None, :] - ctrl_arr[None, :, :], axis=2).mean())
            self_p = float(np.linalg.norm(res[:, None, :] - res[None, :, :], axis=2).mean())
            ed = max(0.0, 2 * cross - self_p - ctrl_self)
        else:
            ed = 0.0
        effects[g] = PertEffect(gene=g, effect_size=float(np.linalg.norm(d)), edistance=float(ed), n_cells=counts[g])

    return CellTypeEffects(
        cell_line=cell_line, genes=list(effects.keys()), delta=delta, effects=effects,
        n_control=ctrl_count, n_common_genes=len(common),
    )


def common_perturbations(k_path: str | Path, r_path: str | Path) -> tuple[list[str], list[str]]:
    """Return (common perturbations, common readout genes) between two h5ad files."""
    import anndata as ad

    k = ad.read_h5ad(k_path, backed="r")
    r = ad.read_h5ad(r_path, backed="r")
    common_pert = sorted((set(k.obs["perturbation"].astype(str)) & set(r.obs["perturbation"].astype(str))) - {"control"})
    common_genes = sorted(set(k.var.index) & set(r.var.index))
    return common_pert, common_genes


@dataclass
class ReplogleCorpus:
    pairs: list[ReplicationPair]
    cosine_threshold: float
    effect_threshold: float
    n_common_perturbations: int
    base_rate: float


def build_replogle_pairs(
    source: CellTypeEffects,   # K562 (the "hit" cell type)
    target: CellTypeEffects,   # RPE1 (does it replicate here?)
    cos_threshold: float | None = None,
    eff_quantile: float = 0.5,
    seed: int = 0,
) -> ReplogleCorpus:
    """Cross-cell-type replication pairs (source→target). Label = the perturbation
    effect reproduces in the target cell type (concordant Δ direction + real effect)."""
    common = sorted(set(source.delta) & set(target.delta))
    # effect threshold from the source-effect distribution (real, data-driven)
    src_eff = np.array([source.effects[g].effect_size for g in common])
    eff_thr = float(np.quantile(src_eff, eff_quantile)) if len(src_eff) else 0.0
    # cosine threshold default: median cosine across common genes
    coss = []
    for g in common:
        a, b = source.delta[g], target.delta[g]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        coss.append(float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0)
    coss = np.array(coss)
    cos_thr = cos_threshold if cos_threshold is not None else float(np.quantile(coss, 0.5))

    pairs: list[ReplicationPair] = []
    n_pos = 0
    for g, cosine in zip(common, coss):
        se = source.effects[g]
        te = target.effects[g]
        # source must itself be a real effect (a "hit") to ask if it replicates
        if se.effect_size < eff_thr:
            continue
        label = bool(cosine > cos_thr and te.effect_size >= eff_thr)
        n_pos += int(label)
        qa = float(min(0.99, 0.4 + 0.6 * np.tanh(se.n_cells / 200)))
        qb = float(min(0.99, 0.4 + 0.6 * np.tanh(te.n_cells / 200)))
        pairs.append(
            ReplicationPair(
                gene=g,
                context_a=Context(cell_line=source.cell_line, lineage="K562_lineage"),
                context_b=Context(cell_line=target.cell_line, lineage="RPE1_lineage"),
                beta_a=float(se.effect_size), var_a=float(max(0.01, 1.0 / max(se.n_cells, 1))),
                beta_b=float(te.effect_size), var_b=float(max(0.01, 1.0 / max(te.n_cells, 1))),
                qc_a=_qc_carrier(qa), qc_b=_qc_carrier(qb),
                modality="CRISPRi", edist_a=float(np.tanh(se.edistance)), label=label,
                quality_a=qa, quality_b=qb,
                pair_type="cross_cell_type", source="replogle", task="transcriptomic",
            )
        )
    return ReplogleCorpus(
        pairs=pairs, cosine_threshold=cos_thr, effect_threshold=eff_thr,
        n_common_perturbations=len(common), base_rate=(n_pos / len(pairs)) if pairs else 0.0,
    )


def build_replogle_corpus(
    k_path: str | Path | None = None,
    r_path: str | Path | None = None,
    seed: int = 0,
) -> tuple[ReplogleCorpus, CellTypeEffects, CellTypeEffects]:
    k_path = Path(k_path or RAW / "ReplogleWeissman2022_K562_essential.h5ad")
    r_path = Path(r_path or RAW / "ReplogleWeissman2022_rpe1.h5ad")
    for p in (k_path, r_path):
        if not p.exists():
            raise FileNotFoundError(f"Replogle h5ad missing: {p} (run acquire.py --only replogle)")
    k_eff = _load_effects("K562_essential")
    r_eff = _load_effects("rpe1")
    if k_eff is None or r_eff is None:
        _, common_genes = common_perturbations(k_path, r_path)
        if k_eff is None:
            k_eff = compute_cell_type_effects(k_path, common_genes, seed=seed)
            _save_effects(k_eff, "K562_essential")
        if r_eff is None:
            r_eff = compute_cell_type_effects(r_path, common_genes, seed=seed)
            _save_effects(r_eff, "rpe1")
    corpus = build_replogle_pairs(k_eff, r_eff, seed=seed)
    return corpus, k_eff, r_eff
