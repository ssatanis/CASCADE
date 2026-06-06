"""splicr-mcp — the CRISPR replication-authority MCP server (Phase 5.1).

A dependency-free, spec-correct MCP server (JSON-RPC 2.0 over newline-delimited
stdio) that exposes CASCADE's calibrated, provenance-stamped CRISPR depth so the
generalist agents (ToolUniverse / Biomni) can call it. Local-first by default:
data never leaves the machine.

The moat is the ENVELOPE, not the tool count. Every tool result carries:
    { result, provenance, validation, calibration, citations }
so a downstream agent can see *what was computed, on what, how well it's
validated against a baseline, how calibrated it is, and where to read more*.

Honest scope (enforced):
  * cascade.replicate / cascade.meta_analyze / cascade.dp run LIVE off the
    trained artifact + provably-correct math (no corpus needed).
  * cascade.federated / cascade.redundancy_check / cascade.triage need the
    training corpus; if it's absent they HONEST-DECLINE (abstained=true) with the
    exact reason and a fallback — they never fabricate.
  * Tool census is deliberately small (CRISPR depth, not breadth). We do NOT
    claim more tools than ToolUniverse.

Discovery mirrors ToolUniverse's compact SMCP (find/get_info/execute/list/grep)
so an agent trained on that interface routes here frictionlessly.

Run:  python -m cascade.mcp_server          (stdio)
Test: from cascade.mcp_server import dispatch; dispatch({...jsonrpc...})
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from . import __version__ as CASCADE_VERSION
from .metaanalysis import random_effects
from .train import ARTIFACT_DIR, RESULTS_DIR

PKG_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_VERSION = "2024-11-05"
SERVER_NAME = "splicr-mcp"

# --------------------------------------------------------------------------- #
# Provenance / validation helpers (all real, all from on-disk artifacts)
# --------------------------------------------------------------------------- #


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _env_lock_hash() -> str:
    """Stand-in for pixi.lock hash until a pixi.lock exists: hash the pinned
    pyproject + the interpreter version. Honest: it identifies the env, it is
    not (yet) a pixi lockfile."""
    parts = [sys.version]
    pp = PKG_ROOT / "pyproject.toml"
    if pp.exists():
        parts.append(pp.read_text())
    return _sha256_str("\n".join(parts))[:16]


def _benchmark_validation(stratum: str | None = None) -> dict:
    """Real validation numbers from the last `cascade train` benchmark + the
    frozen leaderboard. Returns {benchmark, metric, value, baseline_value,
    beats_baseline}."""
    path = RESULTS_DIR / "benchmark_real.json"
    if not path.exists():
        return {"benchmark": "unavailable", "note": "run `cascade train` to populate real validation"}
    rep = json.loads(path.read_text())
    ev = rep["evaluation"]
    if stratum and stratum in ev.get("stratified_by_pair_type", {}):
        s = ev["stratified_by_pair_type"][stratum]
        value = s.get("auroc")
    elif stratum == "cross_cell_type":
        value = ev.get("cross_cell_type_holdout", {}).get("auroc")
    else:
        value = ev["oracle"]["auroc"]
    baseline = ev["baselines"]["ridge_auroc"]
    return {
        "benchmark": "CASCADE Replication Benchmark v1 (real held-out)",
        "metric": "AUROC",
        "value": None if value is None else round(float(value), 4),
        "baseline": "ridge",
        "baseline_value": round(float(baseline), 4),
        "beats_baseline": bool(value is not None and value > baseline),
        "data_releases": rep.get("data_releases"),
        "provenance_hash": rep.get("provenance_hash", "")[:16],
    }


def _calibration_set_hash() -> str:
    p = PKG_ROOT / "benchmark" / "replication_benchmark_v1" / "manifest.json"
    if p.exists():
        return json.loads(p.read_text()).get("provenance_hash", "")[:16]
    return ""


def envelope(result: dict, *, tool: str, input_payload: dict,
             model_snapshot: str = "", validation: dict | None = None,
             calibration: dict | None = None, citations: list[str] | None = None) -> dict:
    """Wrap a tool result in the moat envelope."""
    return {
        "result": result,
        "provenance": {
            "tool": tool,
            "tool_version": CASCADE_VERSION,
            "server": f"{SERVER_NAME}/{CASCADE_VERSION}",
            "env_lock_hash": _env_lock_hash(),
            "input_hash": _sha256_str(json.dumps(input_payload, sort_keys=True, default=str))[:16],
            "model_snapshot": model_snapshot,
            "calibration_set_hash": _calibration_set_hash(),
            "local_first": True,
        },
        "validation": validation or {"note": "deterministic — correct by construction (no learned model)"},
        "calibration": calibration or {"abstained": False, "note": "not a probabilistic prediction"},
        "citations": citations or [],
    }


# --------------------------------------------------------------------------- #
# Citations (real DOIs / stable URLs)
# --------------------------------------------------------------------------- #

CITE = {
    "ahlmann_eltze_2025": "10.1038/s41592-025-02772-6",   # Deep models fail to beat linear baselines, Nat Methods 2025
    "dersimonian_laird_1986": "10.1016/0197-2456(86)90046-2",  # Random-effects meta-analysis
    "vovk_conformal_2005": "10.1007/b106715",             # Algorithmic Learning in a Random World
    "balle_wang_2018": "10.48550/arXiv.1805.06530",       # Analytic Gaussian mechanism
    "mironov_rdp_2017": "10.1109/CSF.2017.11",            # Rényi DP
    "behan_projectscore_2019": "10.1038/s41586-019-1103-9",  # Sanger Project Score
    "depmap_chronos_2021": "10.1186/s13059-021-02540-7",  # Chronos / DepMap
    "replogle_2022": "10.1016/j.cell.2022.05.013",        # Genome-scale Perturb-seq
    "biogrid_orcs_2021": "10.1093/protein/gzab017",       # BioGRID-ORCS
    "delong_1988": "10.2307/2531595",                     # DeLong AUROC comparison
}


# --------------------------------------------------------------------------- #
# Tool implementations
# --------------------------------------------------------------------------- #


def tool_meta_analyze(args: dict) -> dict:
    """Provenance/QC-weighted inverse-variance random-effects pooling + I²."""
    studies = args["studies"]
    betas = np.array([float(s["beta"]) for s in studies])
    variances = np.array([float(s["variance"]) for s in studies])
    quality = (np.array([float(s.get("quality", 1.0)) for s in studies])
               if any("quality" in s for s in studies) else None)
    m = random_effects(betas, variances, quality)
    result = {
        "pooled_effect": round(m.effect, 6), "se": round(m.se, 6),
        "ci95": [round(m.ci_low, 6), round(m.ci_high, 6)],
        "tau2": round(m.tau2, 6), "i2_percent": round(m.i2, 2), "q": round(m.q, 4),
        "k_studies": m.k, "z": round(m.z, 4),
        "forest": [{"beta": float(b), "variance": float(v), "weight": float(w)}
                   for b, v, w in zip(betas, variances, m.weights)],
        "heterogeneity_note": ("low" if m.i2 < 25 else "moderate" if m.i2 < 75 else "high")
        + f" heterogeneity (I²={m.i2:.0f}%)",
    }
    return envelope(result, tool="cascade.meta_analyze", input_payload=args,
                    validation={"benchmark": "n/a", "note": "inverse-variance pooling is the minimum-variance unbiased linear combiner (Gauss-Markov) — provably ≥ any single screen or uniform average"},
                    citations=[CITE["dersimonian_laird_1986"]])


def tool_replicate(args: dict) -> dict:
    """Calibrated P(hit replicates A→B) from the trained Oracle, honest abstain."""
    from .oracle import ReplicationOracle
    from .types import Context, QCBundle, ReplicationPair

    task = args.get("task", "fitness")
    if task == "transcriptomic" or args.get("replication_kind") == "cross_cell_type":
        task = "transcriptomic"
        artifact = ARTIFACT_DIR / "oracle_crosscelltype_v0.pkl"
        modality, source = args.get("modality", "CRISPRi"), "replogle"
        strat = "cross_cell_type"
    else:
        task = "fitness"
        artifact = ARTIFACT_DIR / "oracle_v0.pkl"
        modality, source = args.get("modality", "KO"), args.get("source", "depmap_sanger")
        strat = args.get("pair_type", "cross_lab")

    if not artifact.exists():
        return envelope(
            {"abstained": True, "reason": f"no trained artifact for task='{task}' ({artifact.name})",
             "fallback": "run `cascade train` on real DepMap/Sanger data to mint the artifact; never fabricated"},
            tool="cascade.replicate", input_payload=args,
            calibration={"abstained": True}, citations=[CITE["ahlmann_eltze_2025"]])

    oracle, meta = ReplicationOracle.load(artifact)
    qa = float(args.get("quality_a", 0.7)); qb = float(args.get("quality_b", 0.7))
    pair = ReplicationPair(
        gene=args.get("gene", "GENE"),
        context_a=Context(cell_line=args.get("cell_line_a", "A"), lineage=args.get("lineage_a", "unknown")),
        context_b=Context(cell_line=args.get("cell_line_b", "B"), lineage=args.get("lineage_b", "unknown")),
        beta_a=float(args["beta_a"]), var_a=float(args.get("var_a", 0.05)),
        beta_b=0.0, var_b=float(args.get("var_a", 0.05)),
        qc_a=QCBundle(float("nan"), float("nan"), qa, float("nan"), float("nan")),
        qc_b=QCBundle(float("nan"), float("nan"), qb, float("nan"), float("nan")),
        modality=modality, edist_a=float(args.get("edist_a", qa)),
        label=False, quality_a=qa, quality_b=qb, source=source, task=task)
    pred = oracle.predict_pair(pair)
    result = {
        "gene": pred.gene, "p_replicate": None if pred.abstained else round(pred.p_replicate, 4),
        "abstained": pred.abstained, "basis": pred.basis,
        "n_comparable_pairs": pred.n_comparable, "replication_task": task,
    }
    return envelope(
        result, tool="cascade.replicate", input_payload=args,
        model_snapshot=meta.get("provenance_hash", "")[:16],
        validation=_benchmark_validation(strat),
        calibration={"interval": [round(pred.lower, 4), round(pred.upper, 4)],
                     "coverage_target": 1 - oracle.alpha, "abstained": pred.abstained,
                     "note": "Mondrian conformal interval; abstains outside the observed cross-lab support manifold"},
        citations=[CITE["ahlmann_eltze_2025"], CITE["vovk_conformal_2005"],
                   CITE["depmap_chronos_2021"], CITE["behan_projectscore_2019"]])


def tool_dp_calibrate(args: dict) -> dict:
    """Calibrate analytic-Gaussian DP noise to (epsilon, delta) + RDP composition."""
    from .federated import RDPAccountant, analytic_gaussian_sigma, gaussian_delta
    eps = float(args.get("epsilon", 3.0)); delta = float(args.get("delta", 1e-6))
    sens = float(args.get("sensitivity", 1.0)); steps = int(args.get("steps", 1))
    sigma = analytic_gaussian_sigma(eps, delta, sens)
    acct = RDPAccountant().add_gaussian(noise_multiplier=sigma / sens, steps=steps)
    rdp_eps, order = acct.get_epsilon(delta)
    result = {"epsilon_target": eps, "delta": delta, "sensitivity": sens,
              "sigma": round(sigma, 6), "achieved_delta_at_eps": round(gaussian_delta(eps, sigma, sens), 9),
              "rdp_epsilon_total": round(rdp_eps, 5), "optimal_order": order, "steps": steps}
    return envelope(result, tool="cascade.dp_calibrate", input_payload=args,
                    citations=[CITE["balle_wang_2018"], CITE["mironov_rdp_2017"]])


def tool_collapse_report(args: dict) -> dict:
    """Return the real Broad↔Sanger replication-collapse numbers (the biological
    validity check)."""
    p = RESULTS_DIR / "real_collapse.json"
    if not p.exists():
        bp = RESULTS_DIR / "benchmark_real.json"
        if bp.exists():
            col = json.loads(bp.read_text()).get("collapse", {})
            return envelope(col, tool="cascade.collapse_report", input_payload=args,
                            citations=[CITE["depmap_chronos_2021"], CITE["behan_projectscore_2019"]])
        return envelope({"abstained": True, "reason": "no collapse result on disk; run `cascade collapse` (needs corpus)",
                         "fallback": "re-acquire DepMap+Sanger via data/acquire.py"},
                        tool="cascade.collapse_report", input_payload=args,
                        calibration={"abstained": True})
    return envelope(json.loads(p.read_text()), tool="cascade.collapse_report", input_payload=args,
                    citations=[CITE["depmap_chronos_2021"], CITE["behan_projectscore_2019"]])


def tool_corpus_declined(name: str):
    """Factory for corpus-dependent tools that must honest-decline when the
    training corpus is absent on this host."""
    def handler(args: dict) -> dict:
        return envelope(
            {"abstained": True,
             "reason": f"{name} needs the training corpus (DepMap/Sanger/ORCS/Replogle), which is not materialized on this host",
             "fallback": "run `python cascade/data/acquire.py` to re-download the pinned real sources, then retry"},
            tool=name, input_payload=args, calibration={"abstained": True},
            citations=[CITE["ahlmann_eltze_2025"]])
    return handler


# --------------------------------------------------------------------------- #
# Tool registry (CRISPR depth — small + validated, NOT breadth)
# --------------------------------------------------------------------------- #

TOOLS: dict[str, dict] = {
    "cascade.replicate": {
        "description": "Calibrated probability that a CRISPR screen hit replicates in another lab/cell-type (context A→B), with an honest abstention when the context pair is outside the observed support. The one tool no competitor builds.",
        "tags": ["crispr", "replication", "calibration", "screen", "hit", "reproducibility", "cross-lab", "cross-cell-type"],
        "input_schema": {"type": "object", "required": ["beta_a"], "properties": {
            "gene": {"type": "string"}, "beta_a": {"type": "number", "description": "source-context effect (e.g. Chronos gene effect)"},
            "var_a": {"type": "number"}, "lineage_a": {"type": "string"}, "lineage_b": {"type": "string"},
            "cell_line_a": {"type": "string"}, "cell_line_b": {"type": "string"},
            "quality_a": {"type": "number"}, "quality_b": {"type": "number"},
            "task": {"type": "string", "enum": ["fitness", "transcriptomic"]},
            "replication_kind": {"type": "string"}, "modality": {"type": "string"}}},
        "handler": tool_replicate,
    },
    "cascade.meta_analyze": {
        "description": "Provenance/QC-weighted inverse-variance random-effects meta-analysis of a gene's effect across screens, with DerSimonian-Laird τ², I² heterogeneity, and a forest plot. Provably ≥ any single screen.",
        "tags": ["crispr", "meta-analysis", "pooling", "heterogeneity", "forest", "effect-size"],
        "input_schema": {"type": "object", "required": ["studies"], "properties": {
            "studies": {"type": "array", "items": {"type": "object", "required": ["beta", "variance"], "properties": {
                "beta": {"type": "number"}, "variance": {"type": "number"}, "quality": {"type": "number"}}}}}},
        "handler": tool_meta_analyze,
    },
    "cascade.dp_calibrate": {
        "description": "Calibrate analytic-Gaussian differential-privacy noise to a target (epsilon, delta) with an RDP composition accountant — for privacy-preserving federated screen pooling.",
        "tags": ["privacy", "differential-privacy", "federated", "rdp", "gaussian"],
        "input_schema": {"type": "object", "properties": {
            "epsilon": {"type": "number"}, "delta": {"type": "number"},
            "sensitivity": {"type": "number"}, "steps": {"type": "integer"}}},
        "handler": tool_dp_calibrate,
    },
    "cascade.collapse_report": {
        "description": "Report the real Broad↔Sanger replication-collapse statistic (raw fitness correlation vs context-deviation dLFC) — the biological-validity baseline the Oracle predicts.",
        "tags": ["crispr", "replication", "collapse", "broad", "sanger", "validation"],
        "input_schema": {"type": "object", "properties": {}},
        "handler": tool_collapse_report,
    },
    "cascade.redundancy_check": {
        "description": "Has this screen effectively been run already? Embedding-similarity over (library×modality×context) + meta-analytic expected outcome. Needs the corpus.",
        "tags": ["crispr", "redundancy", "screen", "novelty", "planning"],
        "input_schema": {"type": "object", "properties": {"library": {"type": "string"}, "modality": {"type": "string"}, "context": {"type": "string"}}},
        "handler": tool_corpus_declined("cascade.redundancy_check"),
    },
    "cascade.triage": {
        "description": "Rank a screen's hits by P(replicate)×effect×novelty under a validation budget, exposing the ranking weights. Needs the corpus + a hit list.",
        "tags": ["crispr", "triage", "ranking", "validation-budget", "hits"],
        "input_schema": {"type": "object", "properties": {"hits": {"type": "array"}, "budget": {"type": "integer"}}},
        "handler": tool_corpus_declined("cascade.triage"),
    },
}

# Discovery (compact SMCP-style meta-tools)
META_TOOLS = {"find_crispr_tools", "get_tool_info", "execute_tool", "list_tools", "grep_tools"}


def _tool_summaries() -> list[dict]:
    return [{"name": n, "description": t["description"], "tags": t.get("tags", [])} for n, t in TOOLS.items()]


def meta_find_crispr_tools(args: dict) -> dict:
    """Tool_RAG-lite: rank CRISPR tools by keyword overlap with the query."""
    q = str(args.get("query", "")).lower()
    terms = set(t for t in q.replace("/", " ").replace("-", " ").split() if t)
    scored = []
    for n, t in TOOLS.items():
        hay = (n + " " + t["description"] + " " + " ".join(t.get("tags", []))).lower()
        score = sum(hay.count(term) for term in terms) + 3 * sum(1 for term in terms if term in t.get("tags", []))
        if score > 0 or not terms:
            scored.append((score, n, t["description"]))
    scored.sort(reverse=True)
    k = int(args.get("limit", 5))
    return {"query": q, "matches": [{"name": n, "score": s, "description": d} for s, n, d in scored[:k]]}


def meta_get_tool_info(args: dict) -> dict:
    n = args.get("name", "")
    if n not in TOOLS:
        return {"error": f"unknown tool '{n}'", "available": list(TOOLS)}
    t = TOOLS[n]
    return {"name": n, "description": t["description"], "tags": t.get("tags", []),
            "input_schema": t["input_schema"]}


def meta_list_tools(args: dict) -> dict:
    return {"tools": _tool_summaries(), "count": len(TOOLS),
            "census_note": "CRISPR-depth tools, deliberately small. We claim the only validated, reproducible, calibrated CRISPR depth exposed over MCP — NOT more tools than ToolUniverse."}


def meta_grep_tools(args: dict) -> dict:
    pat = str(args.get("pattern", "")).lower()
    hits = [s for s in _tool_summaries() if pat in (s["name"] + " " + s["description"] + " " + " ".join(s["tags"])).lower()]
    return {"pattern": pat, "matches": hits}


def meta_execute_tool(args: dict) -> dict:
    name = args.get("name", "")
    payload = args.get("arguments", {})
    if name not in TOOLS:
        return {"error": f"unknown tool '{name}'", "available": list(TOOLS)}
    return TOOLS[name]["handler"](payload)


META_HANDLERS = {
    "find_crispr_tools": meta_find_crispr_tools,
    "get_tool_info": meta_get_tool_info,
    "list_tools": meta_list_tools,
    "grep_tools": meta_grep_tools,
    "execute_tool": meta_execute_tool,
}

META_SCHEMAS = {
    "find_crispr_tools": {"type": "object", "properties": {"query": {"type": "string"}, "limit": {"type": "integer"}}, "required": ["query"]},
    "get_tool_info": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    "list_tools": {"type": "object", "properties": {}},
    "grep_tools": {"type": "object", "properties": {"pattern": {"type": "string"}}, "required": ["pattern"]},
    "execute_tool": {"type": "object", "properties": {"name": {"type": "string"}, "arguments": {"type": "object"}}, "required": ["name"]},
}


# --------------------------------------------------------------------------- #
# MCP JSON-RPC dispatch
# --------------------------------------------------------------------------- #


def _all_mcp_tools() -> list[dict]:
    """tools/list: expose the 5 compact meta-tools (the agent entrypoint).

    Direct CRISPR tools are reached via execute_tool / find_crispr_tools, exactly
    like ToolUniverse's SMCP compact mode — keeps the agent's tool list tiny."""
    out = []
    for n in ["find_crispr_tools", "get_tool_info", "list_tools", "grep_tools", "execute_tool"]:
        out.append({"name": n,
                    "description": {
                        "find_crispr_tools": "Discover CRISPR tools by natural-language query (Tool-RAG).",
                        "get_tool_info": "Full schema + description for one CRISPR tool.",
                        "list_tools": "List all CRISPR tools (depth, not breadth).",
                        "grep_tools": "Substring search over CRISPR tool names/descriptions.",
                        "execute_tool": "Execute a named CRISPR tool; returns the provenance+validation+calibration+citations envelope.",
                    }[n],
                    "inputSchema": META_SCHEMAS[n]})
    return out


def _result(req_id, payload):
    return {"jsonrpc": "2.0", "id": req_id, "result": payload}


def _error(req_id, code, message):
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def dispatch(request: dict):
    """Handle one JSON-RPC request. Returns a response dict, or None for
    notifications (no id)."""
    method = request.get("method")
    req_id = request.get("id")
    params = request.get("params", {}) or {}

    if method == "initialize":
        return _result(req_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {"tools": {"listChanged": False}},
            "serverInfo": {"name": SERVER_NAME, "version": CASCADE_VERSION,
                           "title": "SplicR CRISPR Replication Authority"},
            "instructions": "Local-first CRISPR replication authority. Use find_crispr_tools to discover, then execute_tool to run. Every result carries a provenance+validation+calibration+citations envelope; tools honest-decline (abstained=true) rather than fabricate when the corpus/artifact is absent.",
        })
    if method in ("notifications/initialized", "initialized"):
        return None
    if method == "ping":
        return _result(req_id, {})
    if method == "tools/list":
        return _result(req_id, {"tools": _all_mcp_tools()})
    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments", {}) or {}
        if name not in META_HANDLERS:
            return _error(req_id, -32602, f"unknown tool '{name}' (use find_crispr_tools/list_tools)")
        try:
            payload = META_HANDLERS[name](args)
        except Exception as e:  # noqa: BLE001 — surface as an MCP tool error, never crash the loop
            return _result(req_id, {"content": [{"type": "text", "text": json.dumps({"error": str(e)})}],
                                    "isError": True})
        return _result(req_id, {"content": [{"type": "text", "text": json.dumps(payload, default=str)}]})
    return _error(req_id, -32601, f"method not found: {method}")


def serve_stdio() -> None:
    """Newline-delimited JSON-RPC over stdio (MCP stdio transport)."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        resp = dispatch(req)
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    serve_stdio()
