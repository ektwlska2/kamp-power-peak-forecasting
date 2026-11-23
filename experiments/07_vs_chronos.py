"""Head-to-head vs Chronos-2 (saved zero-shot predictions) on identical hours.

Chronos: 168h context, median forecast of the four readings, issued with data
through t-1 (same contract as ours). Source: data/reference/chronos2_zero_shot_predictions.csv
Ours: base next-hour LightGBM (exp03 predictions, direct target).

Checks that actual values align exactly before scoring.
Run: .venv/bin/python experiments/07_vs_chronos.py   (after 03)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, load  # noqa: E402
from src.metrics import score  # noqa: E402
from src.peak import best_f1_cutoff, detection_scores, episode_onset, peak_label  # noqa: E402
from src.state import daily_state  # noqa: E402

CHRONOS = ROOT / "data" / "reference" / "chronos2_zero_shot_predictions.csv"
OUT = ROOT / "outputs" / "exp07"
OUT.mkdir(parents=True, exist_ok=True)
FOLD_MAP = {"May": "May", "June": "Jun", "July": "Jul", "retrospective": "AugSep"}

df = load()
ts_index = pd.Series(df.index, index=df.ts)
ch = pd.read_csv(CHRONOS, parse_dates=["timestamp"])
print("Chronos folds in file:", ch.fold.unique().tolist())
qmap = {15: "q1", 30: "q2", 45: "q3", 60: "q4"}
ch["q"] = ch.quarter_minute.map(qmap)
chp = ch.pivot(index="timestamp", columns="q", values="prediction")[Q]
cha = ch.pivot(index="timestamp", columns="q", values="actual")[Q]
chp.index = ts_index[chp.index].to_numpy()
cha.index = chp.index
mism = (cha - df.loc[cha.index, Q]).abs().to_numpy().max()
assert mism == 0, f"actuals do not align (max diff {mism})"
print("actual values align exactly on", len(chp), "hours")

ours_all = pd.read_pickle(ROOT / "outputs" / "exp03" / "predictions.pkl")
lbl, onset = peak_label(df), episode_onset(peak_label(df))
dstate = df.date.map(daily_state(df).map({0: "idle", 1: "partial", 2: "operating"}))

rows = []
preds = {}
for fold in ("May", "Jun", "Jul", "AugSep"):
    ours = ours_all[(fold, "base", "direct")]
    idx = ours.index.intersection(chp.index)
    y = df.loc[idx, Q]
    models = {"ours": ours.loc[idx], "chronos": chp.loc[idx], "avg(ours,chronos)": (ours.loc[idx] + chp.loc[idx]) / 2}
    for name, p in models.items():
        preds[(fold, name)] = p
        r = {"fold": fold, "model": name, **score(y, p)}
        ae = (p - y).abs().mean(axis=1)
        for st, g in ae.groupby(dstate[idx]):
            r[f"MAE[{st}]"] = g.mean()
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(OUT / "scores.csv", index=False)

# 7-day block bootstrap of MAE difference ours - chronos on Aug-Sep
idx = preds[("AugSep", "ours")].index
y = df.loc[idx, Q]
d = (preds[("AugSep", "ours")] - y).abs().mean(axis=1) - (preds[("AugSep", "chronos")] - y).abs().mean(axis=1)
dp = ((preds[("AugSep", "ours")] - y).abs().where(y >= 190) - (preds[("AugSep", "chronos")] - y).abs().where(y >= 190)).stack().dropna()
blk = ((df.loc[idx, "ts"] - pd.Timestamp("2021-08-01")).dt.days // 7).to_numpy()
rng = np.random.default_rng(0)
ub = np.unique(blk)
bs, bsp = [], []
blk_p = ((df.loc[dp.index.get_level_values(0), "ts"] - pd.Timestamp("2021-08-01")).dt.days // 7).to_numpy()
for _ in range(4000):
    pick = rng.choice(ub, len(ub))
    m = np.concatenate([np.flatnonzero(blk == b) for b in pick])
    bs.append(d.to_numpy()[m].mean())
    mp = np.concatenate([np.flatnonzero(blk_p == b) for b in pick])
    bsp.append(dp.to_numpy()[mp].mean() if len(mp) else np.nan)
ci = np.percentile(bs, [2.5, 97.5]); cip = np.nanpercentile(bsp, [2.5, 97.5])

# peak-hour detection from each forecast's max, cutoff chosen on July
det = []
for name in ("ours", "chronos", "avg(ours,chronos)"):
    jul, aug = preds[("Jul", name)], preds[("AugSep", name)]
    cut = best_f1_cutoff(jul.max(axis=1).to_numpy(), lbl[jul.index].to_numpy())
    s = detection_scores(aug.max(axis=1).to_numpy(), lbl[aug.index].to_numpy(), onset[aug.index].to_numpy(), cut)
    det.append({"model": name, "cutoff": cut, **{k: s[k] for k in ("TP", "FP", "FN", "precision", "recall", "F1", "onsets_caught")}})
det = pd.DataFrame(det)
det.to_csv(OUT / "peak_detection.csv", index=False)

with pd.option_context("display.width", 250):
    print("\nMAE by fold:\n", res.pivot_table(index="model", columns="fold", values="MAE")[["May", "Jun", "Jul", "AugSep"]].round(3))
    print("\npeak MAE by fold:\n", res.pivot_table(index="model", columns="fold", values="peak_MAE")[["May", "Jun", "Jul", "AugSep"]].round(2))
    cols = ["model", "MAE", "RMSE", "bias", "peak_MAE", "peak_shortfall"] + [c for c in res if c.startswith("MAE[")]
    print("\nAug-Sep detail:\n", res[res.fold == "AugSep"][cols].round(3).to_string(index=False))
    print(f"\nAug-Sep MAE diff ours - chronos: {d.mean():+.3f}  (7-day block 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}])")
    print(f"Aug-Sep peak-reading abs-error diff ours - chronos: {dp.mean():+.3f}  (95% CI [{cip[0]:+.3f}, {cip[1]:+.3f}])")
    print("\nPeak-hour detection from forecast max (cutoff from July):\n", det.round(3).to_string(index=False))
