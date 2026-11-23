"""Peak-risk detector: does a dedicated classifier with operating-state
probabilities beat reading peaks off the forecaster?

Detectors (cutoffs chosen on training-period out-of-fold scores, max F1):
  A forecast_max   : max of the base forecaster's four quarter forecasts
  B context_rule   : clock 08-11/13-14 & previous-hour mean >= 100 (fixed rule)
  C clf            : LightGBM classifier on base features + past peak context
  D clf+state      : C + state probabilities
  E clf+state+fc   : D + forecaster outputs
Leads: 0 (target hour t, issued at its start) and 1 (target t+1).
Folds: Apr, May, Jun, Jul (selection), Aug-Sep (test); training from 01-08.

Run: .venv/bin/python experiments/04_peak_detector.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, load  # noqa: E402
from src.features import build  # noqa: E402
from src.peak import (CLF_PARAMS, best_f1_cutoff, context_features, detection_scores,  # noqa: E402
                      episode_onset, oof_forecast, peak_label)
from src.state import fit_state_probs  # noqa: E402

OUT = ROOT / "outputs" / "exp04"
OUT.mkdir(parents=True, exist_ok=True)
FOLDS = [("Apr", "2021-04-01", "2021-05-01"), ("May", "2021-05-01", "2021-06-01"),
         ("Jun", "2021-06-01", "2021-07-01"), ("Jul", "2021-07-01", "2021-08-01"),
         ("AugSep", "2021-08-01", "2021-09-15")]
START = "2021-01-08"
RISK_HOURS = [8, 9, 10, 11, 13, 14]

df = load()
lbl, onset = peak_label(df), episode_onset(peak_label(df))
base = build(df, use_weekly=False)
ctx = context_features(df)
# leak check for context features
i = len(df) // 2 + 9
d2 = df.copy(); d2.loc[i, Q + ["mean_load", "peak_q"]] += 1000
assert (context_features(d2).loc[i].fillna(-1) == ctx.loc[i].fillna(-1)).all(), "context leak"

rows, store = [], {}
for name, a, b in FOLDS:
    tr_all = (df.ts >= START) & (df.ts < a)
    te_all = (df.ts >= a) & (df.ts < b)
    P = fit_state_probs(df, tr_all)
    FC = oof_forecast(df, base, tr_all)
    fcf = FC.add_prefix("fc_").assign(fc_max=FC.max(axis=1))
    for lead in (0, 1):
        y = lbl.shift(-lead)
        on = onset.shift(-lead)
        # training rows must have their label inside the training period
        tr = tr_all & y.notna() & (df.ts + pd.Timedelta(hours=lead) < a)
        te = te_all & y.notna() & (df.ts + pd.Timedelta(hours=lead) < b)
        extra = pd.DataFrame({"hour_next": (df.hour + lead) % 24}, index=df.index)
        feats = {"clf": pd.concat([base, ctx, extra], axis=1)}
        feats["clf+state"] = pd.concat([feats["clf"], P], axis=1)
        feats["clf+state+fc"] = pd.concat([feats["clf+state"], fcf], axis=1)
        scores = {}
        # A: forecaster readout (for lead 1: the forecast already issued for hour t)
        scores["forecast_max"] = FC.max(axis=1)
        # B: fixed context rule
        hr = (df.hour + lead) % 24
        scores["context_rule"] = (hr.isin(RISK_HOURS) & (df.mean_load.shift(1) >= 100)).astype(float)
        # C-E: classifiers, OOF on training rows for cutoff selection
        for fname, X in feats.items():
            s = pd.Series(np.nan, index=df.index)
            dates = np.sort(df.loc[tr, "date"].unique())
            for blk in np.array_split(dates, 6):
                hold = tr & df.date.isin(blk)
                clf = lgb.LGBMClassifier(**CLF_PARAMS).fit(X[tr & ~hold], y[tr & ~hold])
                s[hold] = clf.predict_proba(X[hold])[:, 1]
            clf = lgb.LGBMClassifier(**CLF_PARAMS).fit(X[tr], y[tr])
            s[te] = clf.predict_proba(X[te])[:, 1]
            scores[fname] = s
        for det, s in scores.items():
            cut = 0.5 if det == "context_rule" else best_f1_cutoff(s[tr].to_numpy(), y[tr].to_numpy())
            r = {"fold": name, "lead": lead, "detector": det, "cutoff": cut,
                 **detection_scores(s[te].to_numpy(), y[te].to_numpy(), on[te].to_numpy(), cut)}
            rows.append(r)
            store[(name, lead, det)] = s[te]
    print("done", name, flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "results.csv", index=False)
pd.to_pickle(store, OUT / "scores.pkl")
sel = res[res.fold != "AugSep"].groupby(["lead", "detector"]).agg(
    F1=("F1", "mean"), recall=("recall", "mean"), precision=("precision", "mean"),
    PR_AUC=("PR_AUC", "mean"), onsets_caught=("onsets_caught", "sum"), onsets=("onsets", "sum"),
    FP=("FP", "sum"), peak_hours=("peak_hours", "sum"))
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print("\nSelection folds Apr-Jul (F1/recall/precision/PR_AUC = mean of folds; counts = sums):")
    print(sel.round(3))
    print("\nTest Aug-Sep:")
    cols = ["lead", "detector", "cutoff", "peak_hours", "TP", "FP", "FN", "precision", "recall", "F1",
            "onsets", "onsets_caught", "continuing_caught", "continuing", "PR_AUC", "Brier"]
    print(res[res.fold == "AugSep"][cols].round(3).to_string(index=False))
    print("\nPeak hours per fold:", res[(res.lead == 0) & (res.detector == "clf")].set_index("fold").peak_hours.to_dict())
