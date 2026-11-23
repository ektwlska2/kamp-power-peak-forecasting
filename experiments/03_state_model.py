"""Operating-state model vs weekly lags.

Variants (all strict next-hour, same LightGBM settings as exp02):
  base          calendar + recent load
  +weekly       base + same hour last week
  +state        base + state probabilities + state-matched seasonal lag
  +state+weekly both
Selection folds: Apr, May, Jun, Jul (expanding training from 01-08).
Test fold: Aug 1 - Sep 14. Breakdown by day type on every fold.

Run: .venv/bin/python experiments/03_state_model.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, load  # noqa: E402
from src.features import build  # noqa: E402
from src.metrics import score  # noqa: E402
from src.state import (OPER, IDLE, classifier_features, daily_state, fit_state_probs,  # noqa: E402
                       state_features, state_profiles)

OUT = ROOT / "outputs" / "exp03"
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=0, verbose=-1, n_jobs=4)
FOLDS = [("Apr", "2021-04-01", "2021-05-01"), ("May", "2021-05-01", "2021-06-01"),
         ("Jun", "2021-06-01", "2021-07-01"), ("Jul", "2021-07-01", "2021-08-01"),
         ("AugSep", "2021-08-01", "2021-09-15")]
START = "2021-01-08"

df = load()
st = daily_state(df)

# --- leak checks on state inputs: altering hour t must not change hour t's inputs
i = len(df) // 2 + 5
d2 = df.copy(); d2.loc[i, Q + ["mean_load", "peak_q"]] += 1000
a0, a1 = classifier_features(df, st), classifier_features(d2, st)
assert (a0.loc[i].fillna(-1) == a1.loc[i].fillna(-1)).all(), "classifier feature leak"
p0, p1 = state_profiles(df), state_profiles(d2)
assert (p0.loc[i].fillna(-1) == p1.loc[i].fillna(-1)).all(), "profile leak"

# --- day types for breakdown (defined from actual states; descriptive only)
dstate = df["date"].map(st)
prev = df["date"].map(st.shift(1))
daytype = np.select([dstate == IDLE, (dstate != IDLE) & (prev == IDLE), dstate == OPER],
                    ["idle", "restart after idle", "operating"], "partial")
df["daytype"] = daytype

base = build(df, use_weekly=False)
weekly = build(df, use_weekly=True)[[c for c in build(df, use_weekly=True) if "l168" in c]]

rows, clf_rows, keep = [], [], {}
for name, a, b in FOLDS:
    tr = (df.ts >= START) & (df.ts < a)
    te = (df.ts >= a) & (df.ts < b)
    P = fit_state_probs(df, tr)
    S = state_features(df, P)
    pred_state = P.to_numpy().argmax(1)
    acc = (pred_state[te] == df.loc[te, "date"].map(st).to_numpy())
    clf_rows.append({"fold": name, "acc_all_hours": acc.mean(),
                     "acc_hour0": acc[df.loc[te, "hour"].to_numpy() == 0].mean(),
                     "acc_hour>=2": acc[df.loc[te, "hour"].to_numpy() >= 2].mean()})
    variants = {"base": base, "+weekly": pd.concat([base, weekly], axis=1),
                "+state": pd.concat([base, S], axis=1),
                "+state+weekly": pd.concat([base, S, weekly], axis=1)}
    for vname, X in variants.items():
        for tgt in ("direct", "residual"):
            p = {}
            for q in Q:
                off = X[f"{q}_l1"] if tgt == "residual" else 0
                y = df[q] - off
                m = tr & y.notna()
                p[q] = lgb.LGBMRegressor(**PARAMS).fit(X[m], y[m]).predict(X[te]) + (off[te] if tgt == "residual" else 0)
            p = pd.DataFrame(p, index=df.index[te])
            yt = df.loc[te, Q]
            keep[(name, vname, tgt)] = p
            r = {"fold": name, "variant": vname, "target": tgt, **score(yt, p)}
            ae = (p - yt).abs().mean(axis=1)
            for dt, g in ae.groupby(df.loc[te, "daytype"]):
                r[f"MAE[{dt}]"] = g.mean()
            rows.append(r)
    print("done", name, flush=True)

res = pd.DataFrame(rows)
res.to_csv(OUT / "results.csv", index=False)
pd.to_pickle(keep, OUT / "predictions.pkl")
clf = pd.DataFrame(clf_rows).round(3)
print("\nState classifier accuracy:\n", clf.to_string(index=False))

with pd.option_context("display.width", 250, "display.max_columns", 30):
    t = res.pivot_table(index=["target", "variant"], columns="fold", values="MAE").round(3)
    sel = [f for f, *_ in FOLDS if f != "AugSep"]
    t["sel_mean(Apr-Jul)"] = t[sel].mean(axis=1).round(3)
    print("\nMAE by fold:\n", t[sel + ["sel_mean(Apr-Jul)", "AugSep"]])
    t2 = res.pivot_table(index=["target", "variant"], columns="fold", values="peak_MAE").round(2)
    print("\npeak MAE by fold:\n", t2[sel + ["AugSep"]])
    cols = ["target", "variant"] + [c for c in res if c.startswith("MAE[")]
    print("\nAugSep MAE by day type:\n", res[res.fold == "AugSep"][cols].round(2).to_string(index=False))
    print("\nApr-Jul MAE by day type (hour-weighted mean of folds):")
    print(res[res.fold != "AugSep"].groupby(["target", "variant"])[[c for c in res if c.startswith("MAE[")]].mean().round(2))
print("\nday types in AugSep:", df[(df.ts >= "2021-08-01")].groupby("daytype").date.nunique().to_dict())
