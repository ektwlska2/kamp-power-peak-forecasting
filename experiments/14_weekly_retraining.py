"""Frozen vs weekly-retrained models on Aug 1 - Sep 14 (deployment test).

Frozen : trained once on 01-08..07-31 (the reported setup).
Weekly : retrained at each 7-day origin (Aug 1, 8, 15, 22, 29, Sep 5, 12) on
         01-08..origin, then used for the next 7 days. Same features, settings
         and copy weights (1/n recomputed inside each training window); the
         classifier cutoff is re-selected out-of-fold at each retrain.
Forecasts always use data through t-1 only. This is a deployment test chosen
after Aug-Sep had been seen; it does not replace the headline score.

Run: .venv/bin/python experiments/14_weekly_retraining.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, load, with_copy_status  # noqa: E402
from src.features import build  # noqa: E402
from src.metrics import score  # noqa: E402
from src.models import fit_detector, fit_quarters, predict_quarters, risk_rule, tier_scores, two_tier  # noqa: E402
from src.peak import context_features, detection_scores, episode_onset, peak_label  # noqa: E402
from src.state import IDLE, daily_state  # noqa: E402
from src.weights import copy_weights  # noqa: E402

OUT = ROOT / "outputs" / "exp14"
OUT.mkdir(parents=True, exist_ok=True)
START = pd.Timestamp("2021-01-08")
TEST_END = pd.Timestamp("2021-09-15")
ORIGINS = pd.date_range("2021-08-01", "2021-09-12", freq="7D")

df = with_copy_status(load())
df["state"] = df.date.map(daily_state(df))
lbl, onset = peak_label(df), episode_onset(peak_label(df))
Xb = build(df, use_weekly=False)
Xc = pd.concat([Xb, context_features(df)], axis=1)
TE = (df.ts >= ORIGINS[0]) & (df.ts < TEST_END)
y = df.loc[TE, Q]

fc = {"frozen": [], "weekly": []}
det = {m: {0: [], 1: []} for m in fc}
cuts = []
frozen_fc, frozen_det = None, {}
for k, origin in enumerate(ORIGINS):
    end = ORIGINS[k + 1] if k + 1 < len(ORIGINS) else TEST_END
    blk = (df.ts >= origin) & (df.ts < end)
    for mode in ("frozen", "weekly"):
        fit_end = ORIGINS[0] if mode == "frozen" else origin
        tr = (df.ts >= START) & (df.ts < fit_end)
        if mode == "frozen" and k > 0:
            fc[mode].append(predict_quarters(frozen_fc, Xb, blk))
        else:
            models = fit_quarters(Xb, df, tr)
            if mode == "frozen":
                frozen_fc = models
            fc[mode].append(predict_quarters(models, Xb, blk))
        for lead in (0, 1):
            yl = lbl.shift(-lead)
            X = Xc.assign(hour_next=(df.hour + lead) % 24)
            trl = (df.ts >= START) & (df.ts + pd.Timedelta(hours=lead) < fit_end)
            if mode == "frozen" and k > 0:
                model, cut = frozen_det[lead]
            else:
                model, cut = fit_detector(X, yl, trl, copy_weights(df, trl), dates=df.date)
                if mode == "frozen":
                    frozen_det[lead] = (model, cut)
                cuts.append({"mode": mode, "origin": origin.date(), "lead": lead, "cutoff": cut})
            bl = blk & (df.ts + pd.Timedelta(hours=lead) < TEST_END)
            det[mode][lead].append(pd.Series(model.predict_proba(X[bl])[:, 1], index=df.index[bl]).to_frame("p")
                                   .assign(cut=cut))
    print("week", origin.date(), "done", flush=True)

rows, weekly_rows = [], []
P = {m: pd.concat(v) for m, v in fc.items()}
for mode in fc:
    s = score(y, P[mode])
    r = {"mode": mode, "MAE": s["MAE"], "RMSE": s["RMSE"], "peak_MAE": s["peak_MAE"], "peak_shortfall": s["peak_shortfall"]}
    for lead in (0, 1):
        d = pd.concat(det[mode][lead])
        yl = lbl.shift(-lead)[d.index].to_numpy(); ol = onset.shift(-lead)[d.index].to_numpy()
        flag = (d.p >= d.cut).to_numpy()
        tp, fp, fn = (flag & (yl == 1)).sum(), (flag & (yl == 0)).sum(), (~flag & (yl == 1)).sum()
        r[f"clf_lead{lead}_TP"], r[f"clf_lead{lead}_FP"] = int(tp), int(fp)
        r[f"clf_lead{lead}_F1"] = 2 * tp / (2 * tp + fp + fn)
        tier = two_tier(d.p, d.cut, risk_rule(df, lead)[d.index])
        ts = tier_scores(tier, pd.Series(yl, index=d.index), pd.Series(ol, index=d.index),
                         df.loc[d.index, "date"], df.loc[d.index, "state"] != IDLE)
        r[f"alert_lead{lead}_caught"] = ts["any_peaks"]
        r[f"alert_lead{lead}_onsets"] = ts["any_onsets"]
        r[f"alert_lead{lead}_missed"] = ts["missed_peaks"]
        r[f"alert_lead{lead}_false_red"] = int(((tier == "RED") & (yl == 0)).sum())
    rows.append(r)
res = pd.DataFrame(rows)
res.round(3).to_csv(OUT / "summary.csv", index=False)

ae = {m: (P[m] - y).abs().mean(axis=1) for m in P}
wk = pd.Series(np.searchsorted(ORIGINS.values, df.loc[TE, "ts"].values, side="right") - 1, index=y.index)
pk = (y >= 190)
for k, origin in enumerate(ORIGINS):
    m = wk == k
    row = {"week_from": origin.date(), "hours": int(m.sum()),
           "idle_days": int((df.loc[y.index[m], "state"] == IDLE).groupby(df.loc[y.index[m], "date"]).first().sum()),
           "peak_readings": int(pk[m].values.sum())}
    for mode in P:
        row[f"MAE_{mode}"] = ae[mode][m].mean()
        e = (P[mode][m] - y[m]).abs().where(pk[m])
        row[f"peakMAE_{mode}"] = np.nanmean(e.values) if pk[m].values.any() else np.nan
    row["MAE_diff(weekly-frozen)"] = row["MAE_weekly"] - row["MAE_frozen"]
    weekly_rows.append(row)
wkt = pd.DataFrame(weekly_rows)
wkt.round(3).to_csv(OUT / "by_week.csv", index=False)

# 7-day block bootstrap of the overall MAE difference
d = (ae["weekly"] - ae["frozen"]).to_numpy(); b = wk.to_numpy()
rng = np.random.default_rng(0)
boots = [d[np.concatenate([np.flatnonzero(b == u) for u in rng.choice(len(ORIGINS), len(ORIGINS))])].mean()
         for _ in range(4000)]
ci = np.percentile(boots, [2.5, 97.5])
pd.DataFrame(cuts).to_csv(OUT / "cutoffs.csv", index=False)

with pd.option_context("display.width", 250, "display.max_columns", 40):
    print("\n", res.round(3).T)
    print(f"\nMAE difference weekly - frozen: {d.mean():+.3f}  (7-day block 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}])")
    print("\nBy week:\n", wkt.round(2).to_string(index=False))
    print("\nClassifier cutoffs:\n", pd.DataFrame(cuts).pivot_table(index="origin", columns=["mode", "lead"], values="cutoff").round(3))
