"""The Replication Oracle (CASCADE component C5 — the flagship).

Given a hit (gene p, context A) predict the calibrated probability it replicates
in context B, with an HONEST abstention when (A→B) falls outside the support of
observed cross-lab pairs. The mechanism is deliberately shallow (logistic head +
isotonic calibration + conformal interval) — per the spec, the moat is the data
(federated cross-lab labels + provenance), never the architecture.

Pipeline:
  1. featurize pairs and standardize
  2. shallow logistic classifier  → p_raw
  3. isotonic calibration         → p_cal   (reliability)
  4. Mondrian conformal interval  → [lower, upper]   (group-conditional coverage)
  5. k-NN support gate            → in_support / abstain
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from .conformal import MondrianConformalRegressor
from .features import FEATURE_NAMES, context_group, featurize, pair_features
from .provenance import QCWeightParams
from .types import OraclePrediction, ReplicationPair


def _split_indices(n: int, fracs: tuple[float, float, float], seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_train = int(round(fracs[0] * n))
    n_iso = int(round(fracs[1] * n))
    return idx[:n_train], idx[n_train : n_train + n_iso], idx[n_train + n_iso :]


def expected_calibration_error(y_true, p_pred, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p_pred = np.asarray(p_pred, dtype=float)
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p_pred > lo) & (p_pred <= hi) if i > 0 else (p_pred >= lo) & (p_pred <= hi)
        if mask.sum() == 0:
            continue
        conf = p_pred[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / len(y_true)) * abs(conf - acc)
    return float(ece)


class ReplicationOracle:
    def __init__(
        self,
        alpha: float = 0.1,
        k_support: int = 10,
        support_quantile: float = 0.95,
        min_comparable: int = 5,
        c_reg: float = 1.0,
    ):
        self.alpha = alpha
        self.k_support = k_support
        self.support_quantile = support_quantile
        self.min_comparable = min_comparable
        self.c_reg = c_reg
        self._fitted = False

    def fit(
        self,
        pairs: list[ReplicationPair],
        qc_params: QCWeightParams | None = None,
        fracs: tuple[float, float, float] = (0.5, 0.2, 0.3),
        seed: int = 0,
    ) -> "ReplicationOracle":
        from sklearn.isotonic import IsotonicRegression
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors

        if len(pairs) < 20:
            raise ValueError("need >= 20 replication pairs to fit the oracle")
        self.qc_params = qc_params
        X, y = featurize(pairs, qc_params)

        # standardize
        self._mu = X.mean(axis=0)
        self._sd = X.std(axis=0)
        self._sd[self._sd == 0] = 1.0
        Xs = (X - self._mu) / self._sd

        tr, iso, conf = _split_indices(len(pairs), fracs, seed)

        # 1-2. shallow classifier
        self._clf = LogisticRegression(C=self.c_reg, max_iter=2000)
        if len(np.unique(y[tr])) < 2:
            raise ValueError("training split has a single class; need both replicated and not")
        self._clf.fit(Xs[tr], y[tr])

        # 3. isotonic calibration on a disjoint slice
        p_iso = self._clf.predict_proba(Xs[iso])[:, 1]
        if len(np.unique(y[iso])) >= 2:
            self._iso = IsotonicRegression(out_of_bounds="clip")
            self._iso.fit(p_iso, y[iso])
        else:
            self._iso = None

        # 4. Mondrian conformal on the calibration slice (using calibrated preds)
        p_conf = self._calibrate(self._clf.predict_proba(Xs[conf])[:, 1])
        groups_conf = [context_group(pairs[i]) for i in conf]
        self._conf = MondrianConformalRegressor(alpha=self.alpha).fit(y[conf], p_conf, groups_conf)
        self._group_counts: dict[tuple, int] = {}
        for g in groups_conf:
            self._group_counts[g] = self._group_counts.get(g, 0) + 1

        # 5. support model on training features
        self._nn = NearestNeighbors(n_neighbors=min(self.k_support, len(tr)))
        self._nn.fit(Xs[tr])
        d, _ = self._nn.kneighbors(Xs[tr])
        kth = d[:, -1]
        self._support_radius = float(np.quantile(kth, self.support_quantile))

        self._fitted = True
        return self

    def _calibrate(self, p_raw: np.ndarray) -> np.ndarray:
        if getattr(self, "_iso", None) is None:
            return np.asarray(p_raw, dtype=float)
        return self._iso.transform(np.asarray(p_raw, dtype=float))

    def _standardize(self, x: np.ndarray) -> np.ndarray:
        return (x - self._mu) / self._sd

    def predict_pair(self, pair: ReplicationPair) -> OraclePrediction:
        if not self._fitted:
            raise RuntimeError("call fit() first")
        x = self._standardize(pair_features(pair, self.qc_params)).reshape(1, -1)
        p_raw = float(self._clf.predict_proba(x)[0, 1])
        p_cal = float(self._calibrate(np.array([p_raw]))[0])
        group = context_group(pair)
        lower, upper, used_global = self._conf.predict_interval(p_cal, group)
        lower, upper = max(0.0, lower), min(1.0, upper)

        d, _ = self._nn.kneighbors(x)
        kth_dist = float(d[0, -1])
        n_comparable = self._group_counts.get(group, 0)
        in_support = (kth_dist <= self._support_radius) and (n_comparable >= self.min_comparable)
        abstained = (not in_support) or used_global

        if abstained:
            reasons = []
            if kth_dist > self._support_radius:
                reasons.append("(A→B) context pair is outside the observed support manifold")
            if n_comparable < self.min_comparable:
                reasons.append(f"only {n_comparable} comparable cross-lab pairs (< {self.min_comparable})")
            if used_global:
                reasons.append("no group-conditional calibration data for this lineage×modality")
            basis = "Abstained: " + "; ".join(reasons or ["insufficient support"])
        else:
            basis = (
                f"Calibrated on {n_comparable} comparable cross-lab pairs "
                f"(lineage {group[0]}→{group[1]}, {group[2]}); conformal coverage target {1 - self.alpha:.0%}."
            )

        return OraclePrediction(
            gene=pair.gene,
            p_replicate=p_cal,
            abstained=abstained,
            lower=lower,
            upper=upper,
            n_comparable=n_comparable,
            basis=basis,
            in_support=in_support,
        )

    def predict(self, pairs: list[ReplicationPair]) -> list[OraclePrediction]:
        return [self.predict_pair(p) for p in pairs]

    # --- artifact persistence with the realness guard -----------------------

    REALNESS_KEY = "trained_on_real_data"

    def save(self, path: str | Path, metadata: dict) -> None:
        """Persist the fitted Oracle + metadata. Refuses to save unless the
        metadata explicitly flags training on real data."""
        if not self._fitted:
            raise RuntimeError("refusing to save an unfitted Oracle")
        if metadata.get(self.REALNESS_KEY) is not True:
            raise ValueError(
                f"refusing to save artifact without metadata['{self.REALNESS_KEY}'] == True"
            )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"oracle": self, "metadata": metadata, "format": 1}, f)

    @staticmethod
    def load(path: str | Path) -> tuple["ReplicationOracle", dict]:
        """Load a persisted Oracle. Refuses any artifact not flagged real."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no Oracle artifact at {path}")
        with open(path, "rb") as f:
            obj = pickle.load(f)
        meta = obj.get("metadata", {})
        if meta.get(ReplicationOracle.REALNESS_KEY) is not True:
            raise ValueError(
                f"artifact at {path} is not flagged {ReplicationOracle.REALNESS_KEY}: refusing to load"
            )
        return obj["oracle"], meta

    def score(self, pairs: list[ReplicationPair]) -> dict:
        """Evaluate on a held-out set: AUROC, ECE, conformal coverage, abstention."""
        from sklearn.metrics import roc_auc_score

        preds = self.predict(pairs)
        y = np.array([1 if p.label else 0 for p in pairs])
        kept = [(pr, yi, pair) for pr, yi, pair in zip(preds, y, pairs) if not pr.abstained]
        abstain_rate = 1.0 - len(kept) / len(pairs)
        out: dict = {
            "n": len(pairs),
            "n_kept": len(kept),
            "abstention_rate": abstain_rate,
            "feature_names": FEATURE_NAMES,
        }
        if not kept:
            out.update({"auroc": float("nan"), "ece": float("nan"), "coverage": float("nan")})
            return out
        p_hat = np.array([pr.p_replicate for pr, _, _ in kept])
        y_kept = np.array([yi for _, yi, _ in kept])
        covered = np.array([1 if (pr.lower <= yi <= pr.upper) else 0 for pr, yi, _ in kept])
        out["auroc"] = float(roc_auc_score(y_kept, p_hat)) if len(np.unique(y_kept)) >= 2 else float("nan")
        out["ece"] = expected_calibration_error(y_kept, p_hat)
        out["coverage"] = float(np.mean(covered))
        return out
