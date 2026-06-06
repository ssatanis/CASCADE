"""`cascade` CLI — REAL-DATA ONLY surface of CASCADE + the Replication Oracle.

There is no synthetic/demo path here. Every command operates on the real
Broad↔Sanger corpus or the artifact trained on it; if the real data or artifact
is absent, the command declines honestly and exits non-zero — it never fabricates.

Commands:
  collapse              Real Broad↔Sanger concordance collapse (raw vs dLFC).
  train                 Build the real corpus, context-holdout split, train +
                        validate the v0 Oracle, persist the artifact if it passes
                        the beat-baseline + coverage gate.
  benchmark             Print the real held-out benchmark vs baselines (from the
                        last train run); errors if not yet trained.
  replicate '<json>'    P(hit replicates A→B) from the REAL trained artifact, with
                        honest abstention. Declines if no real artifact present.
  meta '<json>'         Inverse-variance random-effects pooling of user studies.
  federated '<gene>'    Plain vs private (secure-agg + DP) pooled effect for a gene
                        across the real Broad screens.
  dp '<json>'           Calibrate Gaussian noise to a target (epsilon, delta).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .corpus import compute_collapse, load_aligned, screens_for_gene
from .federated import FederatedMetaAnalysis, RDPAccountant, analytic_gaussian_sigma, gaussian_delta
from .metaanalysis import random_effects
from .oracle import ReplicationOracle
from .train import ARTIFACT_DIR, RESULTS_DIR, train_v0
from .types import Context, QCBundle, ReplicationPair

ARTIFACT = ARTIFACT_DIR / "oracle_v0.pkl"
NAN = float("nan")


def _emit(obj) -> None:
    print(json.dumps(obj, indent=2, default=str))


def _carrier(quality: float) -> QCBundle:
    return QCBundle(NAN, NAN, float(quality), NAN, NAN)


def cmd_collapse(_args) -> int:
    aligned = load_aligned()
    col = compute_collapse(aligned)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "real_collapse.json").write_text(json.dumps(col, indent=2))
    _emit(col)
    return 0


def cmd_train(args) -> int:
    report = train_v0(theta=args.theta, seed=args.seed, force_save=args.force_save)
    _emit(
        {
            "data_releases": report["data_releases"],
            "provenance_hash": report["provenance_hash"][:16],
            "n_common_cell_lines": report["n_common_cell_lines"],
            "n_common_genes": report["n_common_genes"],
            "split": report["split"],
            "collapse": {
                "r_raw": report["collapse"]["r_raw_fitness"],
                "r_dlfc": report["collapse"]["r_dlfc_deviation"],
            },
            "oracle": report["evaluation"]["oracle"],
            "baselines": report["evaluation"]["baselines"],
            "gate": report["gate"],
            "saved": report["saved"],
            "artifact_path": report["artifact_path"],
        }
    )
    return 0 if report["saved"] else 1


def cmd_benchmark(args) -> int:
    # Frozen Replication Benchmark v1: score a predictions CSV on the leaderboard.
    if getattr(args, "leaderboard", None):
        import csv as _csv

        from .benchmark_v1 import BENCH_DIR, score_predictions

        if not (BENCH_DIR / "test_pairs.csv").exists():
            sys.stderr.write("Benchmark not frozen. Run `cascade freeze-benchmark` first.\n")
            return 2
        # `--leaderboard` with no path → score the committed CASCADE predictions and
        # print the full frozen leaderboard (reproduces the paper table from JSON).
        if args.leaderboard == "__FROZEN__":
            lb_path = BENCH_DIR / "leaderboard.json"
            if lb_path.exists():
                lb = json.loads(lb_path.read_text())
                _emit({"benchmark": "replication_benchmark_v1",
                       "leaderboard": {m: s.get("overall") for m, s in lb.items()}})
                return 0
            preds_path = BENCH_DIR / "predictions_cascade.csv"
        else:
            preds_path = args.leaderboard
        preds = list(_csv.DictReader(open(preds_path)))
        result = score_predictions(BENCH_DIR, preds)
        _emit({"benchmark": "replication_benchmark_v1", "your_model": result})
        return 0

    # Otherwise: the held-out training benchmark from the last `cascade train`.
    path = RESULTS_DIR / "benchmark_real.json"
    if not path.exists():
        sys.stderr.write("No real benchmark found. Run `cascade train` first.\n")
        return 2
    rep = json.loads(path.read_text())
    _emit(
        {
            "evaluation": rep["evaluation"],
            "gate": rep["gate"],
            "collapse": {"r_raw": rep["collapse"]["r_raw_fitness"], "r_dlfc": rep["collapse"]["r_dlfc_deviation"]},
            "data_releases": rep["data_releases"],
            "provenance_hash": rep["provenance_hash"][:16],
        }
    )
    return 0


def cmd_fscp_pilot(args) -> int:
    from .fscp_pilot import federated_beats_alone, run_pilot

    r = run_pilot(epsilon=args.epsilon, delta=args.delta, seed=args.seed)
    r["melloddy_proof_federated_beats_alone"] = federated_beats_alone(r)
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "fscp_pilot.json").write_text(json.dumps(r, indent=2, default=str))
    _emit(r)
    if r["privacy"]["raw_data_crossed_boundary"]:
        sys.stderr.write("PRIVACY VIOLATION: raw data crossed the party boundary.\n")
        return 2
    return 0


def cmd_freeze_benchmark(args) -> int:
    from .benchmark_v1 import BENCH_DIR, freeze_benchmark

    r = freeze_benchmark(seed=args.seed)
    _emit({
        "frozen_to": str(BENCH_DIR),
        "n_test": r["manifest"]["n_test"],
        "n_test_by_stratum": r["manifest"]["n_test_by_stratum"],
        "leakage_check": r["leakage"],
        "leaderboard_overall": {m: s["overall"] for m, s in r["leaderboard"].items()},
    })
    return 0


def cmd_replicate(args) -> int:
    payload = json.loads(args.json)
    # Route by replication TASK: institute/cross-context fitness replication uses
    # the fitness artifact; cross-cell-type (transcriptomic) uses the dedicated
    # Replogle-backed oracle. Only claims cross-cell-type where support exists,
    # else honest abstain.
    task = payload.get("task", "fitness")
    if task == "transcriptomic" or payload.get("replication_kind") == "cross_cell_type":
        task = "transcriptomic"
        artifact = ARTIFACT_DIR / "oracle_crosscelltype_v0.pkl"
        modality, source = payload.get("modality", "CRISPRi"), "replogle"
    else:
        task = "fitness"
        artifact = ARTIFACT
        modality, source = payload.get("modality", "KO"), payload.get("source", "depmap_sanger")

    if not artifact.exists():
        sys.stderr.write(
            f"No real Oracle artifact for task='{task}' ({artifact.name}). "
            "Run `cascade train` on the real data first. Refusing to fabricate a probability.\n"
        )
        return 2
    try:
        oracle, meta = ReplicationOracle.load(artifact)
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"Refusing to load artifact: {e}\n")
        return 2

    qa = float(payload.get("quality_a", 0.7))
    qb = float(payload.get("quality_b", 0.7))
    pair = ReplicationPair(
        gene=payload.get("gene", "GENE"),
        context_a=Context(cell_line=payload.get("cell_line_a", "A"), lineage=payload.get("lineage_a", "unknown")),
        context_b=Context(cell_line=payload.get("cell_line_b", "B"), lineage=payload.get("lineage_b", "unknown")),
        beta_a=float(payload["beta_a"]),
        var_a=float(payload.get("var_a", 0.05)),
        beta_b=0.0,
        var_b=float(payload.get("var_a", 0.05)),
        qc_a=_carrier(qa), qc_b=_carrier(qb),
        modality=modality,
        edist_a=float(payload.get("edist_a", qa)),
        label=False,
        quality_a=qa, quality_b=qb,
        source=source, task=task,
    )
    pred = oracle.predict_pair(pair)
    out = pred.as_dict()
    out["replication_task"] = task
    out["model"] = "CASCADE Replication Oracle v0"
    out["trained_on_real_data"] = meta.get("trained_on_real_data")
    out["data_releases"] = meta.get("data_releases")
    out["provenance_hash"] = meta.get("provenance_hash", "")[:16]
    _emit(out)
    return 0


def cmd_meta(args) -> int:
    payload = json.loads(args.json)
    betas = np.array([float(x["beta"]) for x in payload["studies"]])
    variances = np.array([float(x["variance"]) for x in payload["studies"]])
    quality = (
        np.array([float(x.get("quality", 1.0)) for x in payload["studies"]])
        if any("quality" in x for x in payload["studies"])
        else None
    )
    m = random_effects(betas, variances, quality)
    _emit({
        "effect": round(m.effect, 5), "se": round(m.se, 5),
        "ci95": [round(m.ci_low, 5), round(m.ci_high, 5)],
        "tau2": round(m.tau2, 5), "i2_percent": round(m.i2, 2), "k": m.k, "z": round(m.z, 4),
    })
    return 0


def cmd_federated(args) -> int:
    aligned = load_aligned()
    gene = args.gene
    try:
        screens = screens_for_gene(aligned, gene)
    except KeyError as e:
        sys.stderr.write(f"{e}\n")
        return 2
    fma = FederatedMetaAnalysis()
    plain = fma.aggregate_gene(screens, gene)
    private = fma.private_aggregate_gene(screens, gene, epsilon=args.epsilon, delta=args.delta)
    _emit({
        "gene": gene,
        "n_real_screens": len(screens),
        "plain_effect": None if plain is None else round(plain.effect, 5),
        "plain_se": None if plain is None else round(plain.se, 5),
        "private_effect": None if private["effect"] is None else round(private["effect"], 5),
        "dp": {"epsilon": private.get("epsilon"), "delta": private.get("delta"), "sigma": round(private.get("sigma", 0), 5)},
        "note": "private = clip → secure-aggregate (masks cancel) → Gaussian DP noise → recover; real Broad screens",
    })
    return 0


def cmd_dp(args) -> int:
    payload = json.loads(args.json) if args.json else {}
    eps = float(payload.get("epsilon", 3.0))
    delta = float(payload.get("delta", 1e-6))
    sens = float(payload.get("sensitivity", 1.0))
    steps = int(payload.get("steps", 1))
    sigma = analytic_gaussian_sigma(eps, delta, sens)
    acct = RDPAccountant().add_gaussian(noise_multiplier=sigma / sens, steps=steps)
    rdp_eps, order = acct.get_epsilon(delta)
    _emit({
        "epsilon_target": eps, "delta": delta, "sensitivity": sens, "sigma": round(sigma, 6),
        "achieved_delta_at_eps": round(gaussian_delta(eps, sigma, sens), 9),
        "composition": {"steps": steps, "rdp_epsilon": round(rdp_eps, 5), "optimal_order": order},
    })
    return 0


def cmd_phase3(args) -> int:
    from .phase3 import run_phase3

    rep = run_phase3(seeds=tuple(args.seeds), n_boot=args.n_boot, n_perm=args.n_perm)
    _emit({"overall_auroc": rep["overall"]["delong"]["auroc"],
           "headline": rep["headline_conclusions"],
           "out": "results/phase3_evaluation.json"})
    return 0


def cmd_gate(args) -> int:
    from .gate import run_gate

    rep = run_gate(seeds=tuple(args.seeds))
    _emit({"summary": rep["summary"], "out": "results/scientific_gate.json"})
    return 0


def cmd_mcp_serve(_args) -> int:
    from .mcp_server import serve_stdio

    serve_stdio()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cascade", description="CASCADE + the Replication Oracle (real data only)")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("collapse").set_defaults(func=cmd_collapse)
    bm = sub.add_parser("benchmark")
    bm.add_argument("--leaderboard", nargs="?", const="__FROZEN__",
                    help="score a predictions.csv on the frozen Replication Benchmark v1; "
                         "with no path, print the committed leaderboard (all methods)")
    bm.set_defaults(func=cmd_benchmark)
    fb = sub.add_parser("freeze-benchmark")
    fb.add_argument("--seed", type=int, default=0)
    fb.set_defaults(func=cmd_freeze_benchmark)
    fp = sub.add_parser("fscp-pilot")
    fp.add_argument("--epsilon", type=float, default=4.0)
    fp.add_argument("--delta", type=float, default=1e-6)
    fp.add_argument("--seed", type=int, default=0)
    fp.set_defaults(func=cmd_fscp_pilot)

    tr = sub.add_parser("train")
    tr.add_argument("--theta", type=float, default=0.5)
    tr.add_argument("--seed", type=int, default=0)
    tr.add_argument("--force-save", action="store_true", help="save even if the gate fails (for inspection)")
    tr.set_defaults(func=cmd_train)

    rep = sub.add_parser("replicate")
    rep.add_argument("json", help="hit JSON: {gene, beta_a, var_a, lineage_a, lineage_b, quality_a, quality_b, ...}")
    rep.set_defaults(func=cmd_replicate)

    meta = sub.add_parser("meta")
    meta.add_argument("json", help='{"studies":[{"beta":..,"variance":..,"quality":..}, ...]}')
    meta.set_defaults(func=cmd_meta)

    fed = sub.add_parser("federated")
    fed.add_argument("gene", help="gene symbol+entrez as in the matrix, e.g. 'POLR2A (5430)'")
    fed.add_argument("--epsilon", type=float, default=3.0)
    fed.add_argument("--delta", type=float, default=1e-6)
    fed.set_defaults(func=cmd_federated)

    dp = sub.add_parser("dp")
    dp.add_argument("json", nargs="?", help='{"epsilon":3,"delta":1e-6,"sensitivity":1,"steps":1}')
    dp.set_defaults(func=cmd_dp)

    p3 = sub.add_parser("phase3", help="publication-grade eval on the frozen benchmark (DeLong/BCa/calibration/neg-controls)")
    p3.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p3.add_argument("--n-boot", type=int, default=2000)
    p3.add_argument("--n-perm", type=int, default=2000)
    p3.set_defaults(func=cmd_phase3)

    gt = sub.add_parser("gate", help="the §HONESTY scientific gate: Oracle vs B4/B5 under LOCO/LOSO, ≥5 seeds")
    gt.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    gt.set_defaults(func=cmd_gate)

    sub.add_parser("mcp-serve", help="run the splicr-mcp authority server over stdio").set_defaults(func=cmd_mcp_serve)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])
    return int(args.func(args) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
