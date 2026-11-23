"""Day-ahead forecasting: strict vs production-plan scenario.

Scenarios: strict, strict+weekly (the strict setting is selected on Apr-Jul by
MAE and peak MAE, same rule as next-hour), plan, plan+weather, weather.
The gap strict -> plan is the value of knowing the production plan
("assumes the plan is executed as recorded").

Metrics: MAE, RMSE, peak MAE, daily 15-min maximum error (demand-charge
relevant) and peak-hour F1 from the forecast max, with the cutoff selected on
the previous fold (Jul predictions for Aug-Sep).

Run: .venv/bin/python experiments/06_dayahead_plan.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, load  # noqa: E402
from src.dayahead import assert_strict_no_target_day, build  # noqa: E402
from src.metrics import score  # noqa: E402
from src.peak import best_f1_cutoff, peak_label, detection_scores, episode_onset  # noqa: E402

OUT = ROOT / "outputs" / "exp06"
OUT.mkdir(parents=True, exist_ok=True)
PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=0, verbose=-1, n_jobs=4)
FOLDS = [("Apr", "2021-04-01", "2021-05-01"), ("May", "2021-05-01", "2021-06-01"),
         ("Jun", "2021-06-01", "2021-07-01"), ("Jul", "2021-07-01", "2021-08-01"),
         ("AugSep", "2021-08-01", "2021-09-15")]
START = "2021-01-08"
SCEN = [("strict", False), ("strict+weekly", True), ("opcal", False), ("opcal2", False), ("opcal2+plan", False), ("plan", False), ("plan+weather", False), ("weather", False)]

df = load()
for wk in (False, True):
    assert_strict_no_target_day(df, wk)
from src.state import daily_state  # noqa: E402
DSTATE = daily_state(df).map({0: "idle", 1: "partial", 2: "operating"})
lbl, onset = peak_label(df), episode_onset(peak_label(df))

rows, preds = [], {}
for sname, wk in SCEN:
    scen = sname.replace("strict+weekly", "strict")
    X = build(df, scen, use_weekly=wk)
    for fold, a, b in FOLDS:
        tr = (df.ts >= START) & (df.ts < a)
        te = (df.ts >= a) & (df.ts < b)
        p = pd.DataFrame({q: lgb.LGBMRegressor(**PARAMS).fit(X[tr], df.loc[tr, q]).predict(X[te])
                          for q in Q}, index=df.index[te])
        y = df.loc[te, Q]
        preds[(sname, fold)] = p
        dmax_err = (p.max(axis=1).groupby(df.loc[te, "date"]).max()
                    - y.max(axis=1).groupby(df.loc[te, "date"]).max())
        r = {"scenario": sname, "fold": fold, **score(y, p),
             "daily_max_MAE": dmax_err.abs().mean(), "daily_max_bias": dmax_err.mean()}
        ae = (p - y).abs().mean(axis=1)
        for dt, g in ae.groupby(df.loc[te, "date"].map(DSTATE)):
            r[f"MAE[{dt}]"] = g.mean()
        rows.append(r)
    print("done", sname, flush=True)

res = pd.DataFrame(rows)
# peak-hour detection on Aug-Sep with the cutoff chosen on July predictions
det = []
for sname, _ in SCEN:
    jul, aug = preds[(sname, "Jul")], preds[(sname, "AugSep")]
    cut = best_f1_cutoff(jul.max(axis=1).to_numpy(), lbl[jul.index].to_numpy())
    d = detection_scores(aug.max(axis=1).to_numpy(), lbl[aug.index].to_numpy(),
                         onset[aug.index].to_numpy(), cut)
    det.append({"scenario": sname, "cutoff": cut, **{k: d[k] for k in ("TP", "FP", "FN", "precision", "recall", "F1", "onsets_caught")}})
det = pd.DataFrame(det)
res.to_csv(OUT / "results.csv", index=False)
det.to_csv(OUT / "peak_detection_augsep.csv", index=False)
pd.to_pickle(preds, OUT / "predictions.pkl")

sel = [f for f, *_ in FOLDS if f != "AugSep"]
with pd.option_context("display.width", 250):
    for metric in ("MAE", "peak_MAE", "daily_max_MAE"):
        t = res.pivot_table(index="scenario", columns="fold", values=metric)[sel + ["AugSep"]]
        t.insert(4, "Apr-Jul mean", t[sel].mean(axis=1))
        print(f"\n{metric}:\n", t.round(2))
    bd = res[res.fold == "AugSep"].set_index("scenario")[[c for c in res if c.startswith("MAE[")]]
    print("\nAug-Sep MAE by actual day state:\n", bd.round(2))
    print("\nAug-Sep peak-hour detection (forecast max, cutoff from July):\n", det.round(3).to_string(index=False))
