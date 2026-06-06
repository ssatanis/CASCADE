"""TEST-ONLY synthetic cross-lab CRISPR-screen generator (seeded fixture).

The realness invariant forbids product/runtime/validation code from using
synthetic data. This generator therefore lives under `tests/fixtures/` and is
import-guarded: importing it outside a test session raises. It exists purely to
exercise calibration/coverage properties against KNOWN ground truth (you cannot
verify conformal coverage without ground truth), reproducing the documented
Broad↔Sanger collapse structure.

Product code must NEVER import this module — it trains/validates on real data
only (see `cascade.corpus`, `cascade.train`).
"""

from __future__ import annotations

import os
import sys

# --- import guard: test sessions only --------------------------------------
if "pytest" not in sys.modules and os.environ.get("CASCADE_ALLOW_SYNTHETIC") != "1":
    raise RuntimeError(
        "synthetic_screens is a TEST-ONLY fixture and must not be imported by product "
        "or runtime code. Train/validate the Replication Oracle on real data via "
        "cascade.corpus / cascade.train. (Set CASCADE_ALLOW_SYNTHETIC=1 only inside tests.)"
    )

from dataclasses import dataclass, field

import numpy as np

from cascade.types import Context, GeneEffect, QCBundle, ReplicationPair, ScreenResult

GeneType = str  # "pan" | "selective" | "none"


@dataclass
class SyntheticConfig:
    n_pan: int = 35
    n_selective: int = 50
    n_none: int = 110
    lineages: tuple[str, ...] = ("myeloid", "lung", "neuron", "breast")
    n_labs: int = 8
    hit_threshold: float = 0.5
    pan_effect: float = -1.8
    pan_sd: float = 0.12
    selective_effect: float = -1.2
    selective_off_sd: float = 0.08
    none_sd: float = 0.06
    base_noise_sd: float = 0.2
    seed: int = 0


@dataclass
class SyntheticCohort:
    screens: list[ScreenResult]
    gene_type: dict[str, GeneType]
    gene_pref_lineage: dict[str, str]
    config: SyntheticConfig
    contexts: list[Context] = field(default_factory=list)

    def replication_pairs(self, max_pairs: int = 4000, seed: int | None = None) -> list[ReplicationPair]:
        rng = np.random.default_rng(self.config.seed if seed is None else seed)
        thr = self.config.hit_threshold
        pairs: list[ReplicationPair] = []
        screens = self.screens
        order = rng.permutation(len(screens))
        for ai in order:
            sA = screens[ai]
            for g, geA in sA.effects.items():
                if abs(geA.beta) < thr:
                    continue
                cand = [s for s in screens if s.screen_id != sA.screen_id and g in s.effects]
                if not cand:
                    continue
                rng.shuffle(cand)
                for sB in cand[:3]:
                    geB = sB.effects[g]
                    replicated = (np.sign(geB.beta) == np.sign(geA.beta)) and (abs(geB.beta) >= thr)
                    pairs.append(
                        ReplicationPair(
                            gene=g, context_a=sA.context, context_b=sB.context,
                            beta_a=geA.beta, var_a=geA.variance, beta_b=geB.beta, var_b=geB.variance,
                            qc_a=sA.qc, qc_b=sB.qc, modality=sA.modality,
                            edist_a=sA.pos_control_edistance, label=bool(replicated),
                        )
                    )
                    if len(pairs) >= max_pairs:
                        return pairs
        return pairs

    def raw_vs_dlfc_correlation(self) -> tuple[float, float]:
        labs = sorted({s.lab_id for s in self.screens})
        labA, labB = labs[0], labs[1]
        genes = sorted(self.gene_type.keys())

        def lab_matrix(lab: str) -> dict[str, dict[str, float]]:
            m: dict[str, dict[str, float]] = {}
            for s in self.screens:
                if s.lab_id != lab:
                    continue
                m[s.context.cell_line] = {g: s.effects[g].beta for g in s.genes()}
            return m

        mA, mB = lab_matrix(labA), lab_matrix(labB)
        shared_ctx = sorted(set(mA) & set(mB))
        raw_a, raw_b, dev_a, dev_b = [], [], [], []
        gene_mean_a = {g: np.mean([mA[c][g] for c in shared_ctx]) for g in genes}
        gene_mean_b = {g: np.mean([mB[c][g] for c in shared_ctx]) for g in genes}
        for c in shared_ctx:
            for g in genes:
                raw_a.append(mA[c][g]); raw_b.append(mB[c][g])
                dev_a.append(mA[c][g] - gene_mean_a[g]); dev_b.append(mB[c][g] - gene_mean_b[g])
        return float(np.corrcoef(raw_a, raw_b)[0, 1]), float(np.corrcoef(dev_a, dev_b)[0, 1])


def _make_qc(rng: np.random.Generator, tier: float) -> QCBundle:
    return QCBundle(
        replicate_r=float(np.clip(0.55 + 0.4 * tier + rng.normal(0, 0.03), 0, 0.99)),
        coverage=float(np.clip(80 + 900 * tier + rng.normal(0, 30), 20, 2000)),
        control_separation=float(np.clip(0.6 + 0.38 * tier + rng.normal(0, 0.03), 0, 0.99)),
        library_complexity=float(np.clip(0.5 + 0.45 * tier + rng.normal(0, 0.03), 0.1, 1.0)),
        representation_skew=float(np.clip(0.5 * (1 - tier) + rng.normal(0, 0.02), 0.02, 0.9)),
    )


def generate_synthetic_cohort(config: SyntheticConfig | None = None) -> SyntheticCohort:
    cfg = config or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)

    gene_type: dict[str, GeneType] = {}
    pref: dict[str, str] = {}
    for i in range(cfg.n_pan):
        gene_type[f"PAN{i:03d}"] = "pan"
    for i in range(cfg.n_selective):
        g = f"SEL{i:03d}"
        gene_type[g] = "selective"
        pref[g] = cfg.lineages[i % len(cfg.lineages)]
    for i in range(cfg.n_none):
        gene_type[f"NON{i:03d}"] = "none"
    genes = list(gene_type.keys())

    contexts = [Context(cell_line=f"CL_{ln}", lineage=ln) for ln in cfg.lineages]
    tiers = np.clip(np.linspace(0.35, 1.0, cfg.n_labs) + rng.normal(0, 0.05, cfg.n_labs), 0.2, 1.0)
    lab_shift = rng.normal(0, 0.08, cfg.n_labs)
    lab_scale = np.clip(1.0 + rng.normal(0, 0.08, cfg.n_labs), 0.7, 1.3)

    screens: list[ScreenResult] = []
    for li in range(cfg.n_labs):
        tier = float(tiers[li])
        qc = _make_qc(rng, tier)
        noise_sd = cfg.base_noise_sd / np.sqrt(tier)
        edist = float(np.clip(0.02 + 0.5 * tier + rng.normal(0, 0.02), 0.0, 1.0))
        for ctx in contexts:
            effects: dict[str, GeneEffect] = {}
            for g in genes:
                gt = gene_type[g]
                if gt == "pan":
                    true = cfg.pan_effect + rng.normal(0, cfg.pan_sd)
                elif gt == "selective":
                    true = (cfg.selective_effect + rng.normal(0, cfg.pan_sd)) if pref[g] == ctx.lineage else rng.normal(0, cfg.selective_off_sd)
                else:
                    true = rng.normal(0, cfg.none_sd)
                obs = true * lab_scale[li] + lab_shift[li] + rng.normal(0, noise_sd)
                effects[g] = GeneEffect(gene=g, beta=float(obs), variance=float(noise_sd**2), n_sgrna=4)
            screens.append(
                ScreenResult(
                    screen_id=f"L{li}_{ctx.cell_line}", lab_id=f"lab{li}", context=ctx,
                    modality="KO", qc=qc, effects=effects, pos_control_edistance=edist,
                )
            )

    return SyntheticCohort(screens=screens, gene_type=gene_type, gene_pref_lineage=pref, config=cfg, contexts=contexts)


# --- ground-truth validation helper (test-only) ----------------------------
def meta_analysis_benefit(cohort: SyntheticCohort, qc_params=None) -> dict:
    """Pooled vs single-screen vs uniform MSE against KNOWN synthetic truth.

    Demonstrates the provable C3 claim (inverse-variance + quality pooling beats a
    single screen and uniform averaging). Test-only: it needs ground truth, which
    only the synthetic fixture provides.
    """
    from cascade.metaanalysis import random_effects
    from cascade.provenance import quality_weight

    def _true(gt, pref, lineage, cfg):
        if gt == "pan":
            return cfg.pan_effect
        if gt == "selective":
            return cfg.selective_effect if pref == lineage else 0.0
        return 0.0

    cfg = cohort.config
    by_ctx_gene: dict[tuple[str, str], list] = {}
    for s in cohort.screens:
        for g, ge in s.effects.items():
            by_ctx_gene.setdefault((g, s.context.cell_line), []).append((s, ge))

    pooled_err, single_err, uniform_err = [], [], []
    for (g, _cl), items in by_ctx_gene.items():
        if len(items) < 2:
            continue
        lineage = items[0][0].context.lineage
        truth = _true(cohort.gene_type[g], cohort.gene_pref_lineage.get(g), lineage, cfg)
        betas = np.array([ge.beta for _s, ge in items])
        variances = np.array([ge.variance for _s, ge in items])
        quality = np.array([quality_weight(s.qc, qc_params) for s, _ge in items])
        m = random_effects(betas, variances, quality)
        pooled_err.append((m.effect - truth) ** 2)
        single_err.append(float(np.mean((betas - truth) ** 2)))
        uniform_err.append((float(np.mean(betas)) - truth) ** 2)

    return {
        "pooled_mse": float(np.mean(pooled_err)),
        "single_screen_mse": float(np.mean(single_err)),
        "uniform_mean_mse": float(np.mean(uniform_err)),
        "n_estimands": len(pooled_err),
        "pooled_beats_single": float(np.mean(pooled_err)) < float(np.mean(single_err)),
        "pooled_beats_uniform": float(np.mean(pooled_err)) <= float(np.mean(uniform_err)) + 1e-9,
    }
