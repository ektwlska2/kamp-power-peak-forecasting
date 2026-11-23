"""Probability calibration of the peak classifier (rubric: 확률보정).

Selection (July only): classifier trained 01-08..06-30 with copy weights 1/n;
calibrators (none / Platt / isotonic) fitted on its out-of-fold training
scores; the method with the lowest July Brier score is chosen.
Test (once): classifier trained ..07-31, chosen calibrator fitted on its
out-of-fold scores, applied to Aug-Sep. Differences get 7-day block CIs.
Calibration is monotone, so mapping the alert cutoff through it flags the
same hours (checked).

Run: .venv/bin/python experiments/15_calibration.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, load  # noqa: E402
from src.features import build  # noqa: E402
from src.models import fit_detector  # noqa: E402
from src.peak import context_features, peak_label  # noqa: E402
from src.uncertainty import block_bootstrap_ci, brier, ece, fit_calibrator, log_loss  # noqa: E402
from src.weights import copy_weights  # noqa: E402

OUT = ROOT / "outputs" / "exp15"
OUT.mkdir(parents=True, exist_ok=True)
START = pd.Timestamp("2021-01-08")

df = load()
lbl = peak_label(df)
X = pd.concat([build(df, use_weekly=False), context_features(df)], axis=1).assign(hour_next=df.hour)


def run(train_end, test_end):
    tr = (df.ts >= START) & (df.ts < train_end)
    te = (df.ts >= train_end) & (df.ts < test_end)
    model, cut, oof = fit_detector(X, lbl, tr, copy_weights(df, tr), dates=df.date, return_oof=True)
    p = model.predict_proba(X[te])[:, 1]
    return oof, lbl[oof.index].to_numpy(), p, lbl[te].to_numpy(), cut, df.loc[te, "ts"]


# ---- selection on July
oof, yo, pj, yj, cut_j, _ = run(pd.Timestamp("2021-07-01"), pd.Timestamp("2021-08-01"))
sel = []
for kind in ("none", "platt", "isotonic"):
    cal = fit_calibrator(kind, oof, yo)
    q = cal(pj)
    sel.append({"method": kind, "July_brier": brier(q, yj), "July_logloss": log_loss(q, yj), "July_ECE": ece(q, yj)})
sel = pd.DataFrame(sel)
chosen = sel.loc[sel.July_brier.idxmin(), "method"]

# ---- test once
oof, yo, pt, yt, cut, ts = run(pd.Timestamp("2021-08-01"), pd.Timestamp("2021-09-15"))
cal = fit_calibrator(chosen, oof, yo)
qt = cal(pt)
cut_cal = float(cal(np.array([cut]))[0])
same_flags = bool(((pt >= cut) == (qt >= cut_cal)).all())
week = ((ts - pd.Timestamp("2021-08-01")).dt.days // 7).to_numpy()
eps = 1e-6
ll = lambda p, y: -(y * np.log(np.clip(p, eps, 1)) + (1 - y) * np.log(np.clip(1 - p, eps, 1)))
d_ll = ll(qt, yt) - ll(pt, yt)
d_br = (qt - yt) ** 2 - (pt - yt) ** 2
test = {
    "chosen_method": chosen,
    "raw": {"logloss": log_loss(pt, yt), "brier": brier(pt, yt), "ECE": ece(pt, yt)},
    "calibrated": {"logloss": log_loss(qt, yt), "brier": brier(qt, yt), "ECE": ece(qt, yt)},
    "logloss_diff": float(d_ll.mean()), "logloss_diff_CI95": block_bootstrap_ci(d_ll, week),
    "brier_diff": float(d_br.mean()), "brier_diff_CI95": block_bootstrap_ci(d_br, week),
    "cutoff_raw": cut, "cutoff_calibrated": cut_cal, "alerts_unchanged": same_flags,
}
# reliability table (calibrated vs raw)
edges = np.array([0, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0])
rel = []
for name, pv in (("raw", pt), ("calibrated", qt)):
    b = np.clip(np.digitize(pv, edges) - 1, 0, len(edges) - 2)
    g = pd.DataFrame({"b": b, "p": pv, "y": yt}).groupby("b").agg(mean_p=("p", "mean"), rate=("y", "mean"), n=("y", "size"))
    rel.append(g.assign(kind=name).reset_index())
pd.concat(rel).to_csv(OUT / "reliability.csv", index=False)
sel.to_csv(OUT / "selection_july.csv", index=False)
(OUT / "test.json").write_text(json.dumps(test, indent=2))
pd.DataFrame({"ts": ts.to_numpy(), "p_raw": pt, "p_cal": qt, "y": yt}).to_csv(OUT / "test_scores.csv", index=False)
print(sel.round(4).to_string(index=False))
print(json.dumps(test, indent=1))
