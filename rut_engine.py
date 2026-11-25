"""
Prediction engine of the Rut-Trigger Predictor.

Loads rut_trigger_model.json (written by 10_rut_trigger_tool.py) and reproduces
the predictions of the offline HTML tool exactly: XGBoost-Cox margin from the
exported trees, Breslow baseline, 90% bands from the 200 bootstrap refits,
risk tier from the training tertiles of P(10 yr), and the kNN data-coverage check.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

f32 = np.float32


def _walk_full(node: dict, x: list[float]) -> float:
    """Walk one tree in XGBoost's JSON dump format (main model)."""
    while "children" in node:
        j = int(node["split"][1:])
        v = x[j]
        if not math.isfinite(v):
            nxt = node["missing"]
        else:
            nxt = node["yes"] if f32(v) < f32(node["split_condition"]) else node["no"]
        node = next(c for c in node["children"] if c["nodeid"] == nxt)
    return node["leaf"]


def _walk_compact(n, x: list[float]) -> float:
    """Walk one tree in the compact [feature, threshold, yes, no] format (bootstrap refits)."""
    while isinstance(n, list):
        n = n[2] if f32(x[n[0]]) < f32(n[1]) else n[3]
    return n


def _step(times: np.ndarray, values: np.ndarray, t: float) -> float:
    """Right-continuous step function: value at the last time <= t (0 before the first)."""
    i = np.searchsorted(times, t, side="right") - 1
    return 0.0 if i < 0 else float(values[i])


@dataclass
class Prediction:
    margin: float
    imputed: list[str]
    out_of_range: list[str]
    p5: float
    p10: float
    p15: float
    t10: float | None
    t25: float | None
    t50: float | None
    ci5: tuple[float, float]
    ci10: tuple[float, float]
    ci15: tuple[float, float]
    tier: str
    tier_band: str            # "" when the band lies within one tier, else "Low–Moderate" etc.
    coverage: str
    novelty: float
    boot_margins: np.ndarray = field(repr=False)


class RutTriggerModel:
    def __init__(self, path: str | Path):
        M = json.loads(Path(path).read_text(encoding="utf-8"))
        self.meta = M["meta"]
        self.features = M["features"]
        self.medians = M["medians"]
        self.offset = M["offset"]
        self.trees = M["trees"]
        self.ut = np.asarray(M["baseline"]["times"], float)
        self.H0 = np.asarray(M["baseline"]["H0"], float)
        self.t_max = float(M["t_max"])
        self.tertiles = M["risk_p10_tertiles"]
        nov = M["novelty"]
        self.k = int(nov["k"])
        self.nov_mean = np.asarray(nov["mean"], float)
        self.nov_sd = np.asarray(nov["sd"], float)
        self.nov_Z = np.asarray(nov["Z"], float)
        self.p90, self.p99 = float(nov["p90"]), float(nov["p99"])
        self.boot = [{"o": b["o"], "tr": b["tr"],
                      "t": np.asarray(b["t"], float), "H": np.asarray(b["H"], float)}
                     for b in M["boot"]]
        self.cv_perf = M.get("cv_perf", "")
        self.threshold = float(self.meta.get("threshold_mm", 10.0))
        self.typical = {f["input"]: f["typical"] for f in self.features}

    # ---------- core ----------
    def to_vector(self, raw: dict) -> tuple[list[float], list[str]]:
        x, imputed = [], []
        for j, f in enumerate(self.features):
            v = raw.get(f["input"], float("nan"))
            v = float("nan") if v is None else float(v)
            if math.isfinite(v) and f["log1p"]:
                v = math.log1p(max(v, 0.0))
            if not math.isfinite(v):
                v = self.medians[j]
                imputed.append(f["label"])
            x.append(v)
        return x, imputed

    def margin(self, x: list[float]) -> float:
        return self.offset + sum(_walk_full(t, x) for t in self.trees)

    def prob(self, m: float, t: float) -> float:
        return 1.0 - math.exp(-_step(self.ut, self.H0, t) * math.exp(m))

    def time_at(self, m: float, p: float) -> float | None:
        for u in self.ut:
            if u > self.t_max:
                break
            if self.prob(m, u) >= p:
                return float(u)
        return None

    def boot_margins(self, x: list[float]) -> np.ndarray:
        return np.array([b["o"] + sum(_walk_compact(t, x) for t in b["tr"]) for b in self.boot])

    def band(self, bm: np.ndarray, t: float) -> tuple[float, float]:
        ps = np.array([1.0 - math.exp(-_step(b["t"], b["H"], t) * math.exp(m))
                       for b, m in zip(self.boot, bm)])
        return float(np.quantile(ps, 0.05)), float(np.quantile(ps, 0.95))

    def tier_of(self, p: float) -> str:
        q = self.tertiles
        return "Low" if p < q[0] else ("Moderate" if p < q[1] else "High")

    def novelty(self, x: list[float]) -> float:
        z = (np.asarray(x) - self.nov_mean) / self.nov_sd
        d = np.sqrt(((self.nov_Z - z) ** 2).sum(axis=1))
        return float(np.sort(d)[: self.k].mean())

    def predict(self, raw: dict) -> Prediction:
        x, imputed = self.to_vector(raw)
        m = self.margin(x)
        bm = self.boot_margins(x)
        ci10 = self.band(bm, 10.0)
        p10 = self.prob(m, 10.0)
        tl, th = self.tier_of(ci10[0]), self.tier_of(ci10[1])
        d = self.novelty(x)
        cov = ("Typical of training data" if d <= self.p90 else
               "Near edge of training data" if d <= self.p99 else "Outside training data")
        oor = []
        for f in self.features:
            v = raw.get(f["input"])
            if v is not None and math.isfinite(float(v)) and not (f["min"] <= float(v) <= f["max"]):
                oor.append(f["label"])
        return Prediction(
            margin=m, imputed=imputed, out_of_range=oor,
            p5=self.prob(m, 5.0), p10=p10, p15=self.prob(m, 15.0),
            t10=self.time_at(m, 0.10), t25=self.time_at(m, 0.25), t50=self.time_at(m, 0.50),
            ci5=self.band(bm, 5.0), ci10=ci10, ci15=self.band(bm, 15.0),
            tier=self.tier_of(p10), tier_band="" if tl == th else f"{tl}–{th}",
            coverage=cov, novelty=d, boot_margins=bm)

    def curve(self, m: float, bm: np.ndarray | None = None, step: float = 0.25):
        """Probability curve over time, with the 90% band if bootstrap margins are given."""
        ts = np.arange(0.0, self.t_max + 1e-9, step)
        p = np.array([self.prob(m, t) for t in ts])
        if bm is None:
            return ts, p, None, None
        lo, hi = zip(*[self.band(bm, t) for t in ts])
        return ts, p, np.array(lo), np.array(hi)
