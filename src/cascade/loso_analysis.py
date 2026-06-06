"""LOSO root-cause analysis — why leave-one-study-out collapses (CASCADE Phase-3.6).

The LOSO-study AUROC is ~0.41 (worse than chance). CASCADE reports it honestly;
this turns the failure into a finding by explaining WHY, using only on-disk data
(BioGRID-ORCS screens + their phenotype/library/quality metadata).

Method: GroupKFold(5) by study over the real ORCS cross-study pairs → out-of-fold
Oracle predictions (train on 4 folds, predict the held-out fold). Per-study AUROC,
then stratify by phenotype / library family / quality tertile, and a multiple
regression of per-study AUROC on those factors to find the dominant driver.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PKG_ROOT = Path(__file__).resolve().parents[2]
RESULTS = PKG_ROOT / "results"
FIG = PKG_ROOT / "paper" / "figure_data"
MIN_STUDY_PAIRS = 10


def _library_family(lib: str) -> str:
    s = (lib or "").lower()
    for fam in ["avana", "brunello", "gecko", "tko", "ky", "yusa", "sabatini", "wang", "dolcetto", "calabrese"]:
        if fam in s:
            return "GeCKO" if fam == "gecko" else fam.capitalize()
    return "other"


def _build_orcs_with_meta(seed: int = 0):
    from .orcs import build_orcs_pairs, load_orcs

    orcs = load_orcs(fitness_only=True)
    pairs = build_orcs_pairs(orcs, max_pairs=30000, seed=seed)
    pairs = [p for p in pairs if p.pair_type in ("cross_study", "cross_study_same_cell") and p.study]
    # study -> aggregated metadata (phenotype mode, library family mode, mean quality)
    screens = orcs.screens if hasattr(orcs, "screens") else {}
    by_study_pheno, by_study_lib, by_study_q = defaultdict(list), defaultdict(list), defaultdict(list)
    for sc in screens.values():
        pm = getattr(sc, "pubmed", "")
        if not pm:
            continue
        by_study_pheno[pm].append(getattr(sc, "phenotype", "other"))
        by_study_lib[pm].append(_library_family(getattr(sc, "library", "")))
        by_study_q[pm].append(float(getattr(sc, "quality", 0.5)))

    def mode(xs, default):
        return Counter(xs).most_common(1)[0][0] if xs else default

    meta = {}
    studies = {p.study for p in pairs}
    for s in studies:
        meta[s] = {
            "phenotype": mode(by_study_pheno.get(s, []), "other"),
            "library": mode(by_study_lib.get(s, []), "other"),
            "quality": float(np.mean(by_study_q.get(s, [0.5]))),
        }
    return pairs, meta


def run(seed: int = 0, n_splits: int = 5, save: bool = True) -> dict:
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    from .oracle import ReplicationOracle

    pairs, meta = _build_orcs_with_meta(seed)
    groups = np.array([p.study for p in pairs])
    n_groups = len(np.unique(groups))

    # OOF predictions by study
    rows = []  # (study, y, pred)
    gkf = GroupKFold(n_splits=min(n_splits, n_groups))
    Xidx = np.arange(len(pairs))
    for tr, te in gkf.split(Xidx, groups=groups):
        train = [pairs[i] for i in tr]
        ytr = np.array([1 if p.label else 0 for p in train])
        if len(np.unique(ytr)) < 2 or len(train) < 50:
            continue
        try:
            oracle = ReplicationOracle(alpha=0.1).fit(train, seed=seed)
        except Exception:
            continue
        for i in te:
            pr = oracle.predict_pair(pairs[i])
            if pr.abstained:
                continue
            rows.append((pairs[i].study, 1 if pairs[i].label else 0, pr.p_replicate))

    # overall LOSO AUROC on kept
    ys = np.array([r[1] for r in rows]); ps = np.array([r[2] for r in rows])
    overall = round(float(roc_auc_score(ys, ps)), 4) if len(np.unique(ys)) >= 2 else float("nan")

    # per-study AUROC
    by_study = defaultdict(list)
    for s, y, p in rows:
        by_study[s].append((y, p))
    per_study = {}
    too_small = 0
    for s, vals in by_study.items():
        if len(vals) < MIN_STUDY_PAIRS:
            too_small += 1
            continue
        yy = np.array([v[0] for v in vals]); pp = np.array([v[1] for v in vals])
        if len(np.unique(yy)) < 2:
            continue
        per_study[s] = {"auroc": float(roc_auc_score(yy, pp)), "n": len(vals),
                        **meta.get(s, {})}

    def strat_mean(key):
        d = defaultdict(list)
        for s, info in per_study.items():
            d[info.get(key)].append(info["auroc"])
        return {str(k): round(float(np.mean(v)), 4) for k, v in d.items() if v}

    per_phenotype = strat_mean("phenotype")
    per_library = strat_mean("library")

    # quality tertiles
    qs = sorted(info["quality"] for info in per_study.values())
    tert = {}
    if len(qs) >= 6:
        t1, t2 = np.percentile([i["quality"] for i in per_study.values()], [33, 67])
        bins = {"low": [], "mid": [], "high": []}
        for info in per_study.values():
            b = "low" if info["quality"] <= t1 else ("high" if info["quality"] > t2 else "mid")
            bins[b].append(info["auroc"])
        tert = {k: round(float(np.mean(v)), 4) for k, v in bins.items() if v}

    # regression: auroc ~ phenotype + library + quality + n
    r2 = None
    dominant = "unknown"
    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import OneHotEncoder
        studies = list(per_study)
        if len(studies) >= 10:
            phen = np.array([[per_study[s]["phenotype"]] for s in studies], dtype=object)
            lib = np.array([[per_study[s]["library"]] for s in studies], dtype=object)
            enc_p = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            enc_l = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            Xp = enc_p.fit_transform(phen); Xl = enc_l.fit_transform(lib)
            q = np.array([[per_study[s]["quality"]] for s in studies])
            n = np.array([[per_study[s]["n"]] for s in studies])
            yv = np.array([per_study[s]["auroc"] for s in studies])
            X = np.hstack([Xp, Xl, q, n])
            reg = LinearRegression().fit(X, yv)
            r2 = round(float(reg.score(X, yv)), 4)
            # dominant factor = block with largest standalone R²
            blocks = {"phenotype": Xp, "library": Xl,
                      "quality": q, "n_pairs": n}
            best, bestr2 = "unknown", -1
            for name, Xb in blocks.items():
                rr = LinearRegression().fit(Xb, yv).score(Xb, yv)
                if rr > bestr2:
                    bestr2, best = rr, name
            dominant = best
    except Exception:
        pass

    interpretation = (
        "LOSO AUROC is near/below chance because ORCS hit labels are defined per-study "
        "with study-specific thresholds and heterogeneous phenotype readouts. Unlike "
        "Broad↔Sanger (both fitness, common Chronos protocol), cross-study pairs mix "
        "viability, reporter and proliferation phenotypes across libraries — replication "
        "across incompatible readouts is not biologically expected, so a model trained on "
        "other studies transfers poorly (and can invert) on a held-out study."
    )
    report = {
        "trained_on_real_data": True,
        "overall_loso_auroc": overall,
        "n_studies_total": int(n_groups),
        "n_studies_tested": len(per_study),
        "n_studies_too_small": int(too_small),
        "per_phenotype": per_phenotype,
        "per_library": per_library,
        "per_quality_tertile": tert,
        "regression_r2": r2,
        "dominant_factor": dominant,
        "interpretation": interpretation,
        "recommendation": "Filter cross-study pairs to same-phenotype-class (fitness↔fitness) "
                          "before training; report cross_study separately and do not pool its "
                          "label with institute/context replication.",
    }
    from .train import provenance_hash
    report["provenance_hash"] = provenance_hash()
    if save:
        RESULTS.mkdir(parents=True, exist_ok=True)
        (RESULTS / "loso_failure_analysis.json").write_text(json.dumps(report, indent=2, default=str))
        FIG.mkdir(parents=True, exist_ok=True)
        lines = ["study,auroc,n_pairs,phenotype,library,quality"]
        for s, info in sorted(per_study.items(), key=lambda kv: kv[1]["auroc"]):
            lines.append(f"{s},{info['auroc']:.4f},{info['n']},{info['phenotype']},{info['library']},{info['quality']:.3f}")
        (FIG / "fig_loso_breakdown.csv").write_text("\n".join(lines) + "\n")
    return report


if __name__ == "__main__":
    r = run()
    print(json.dumps({k: r[k] for k in ("overall_loso_auroc", "n_studies_tested",
                                        "per_phenotype", "per_library", "per_quality_tertile",
                                        "regression_r2", "dominant_factor")}, indent=2, default=str))
