"""Regime-consistency gate for weekly lags.

Weekly lags fail when the same hour last week belonged to a different operating
regime (shutdown -> restart). Gate signal, available at issuance:
    gap = roll24(t) - roll24(t-168)   (last 24 completed hours vs the same 24h a week earlier)
G_feat : add `gap` as a feature and let the model learn trust
G_mask : set weekly features to NaN when |gap| > tau  (tau fixed a priori = 30, ~1/4 operating level)
Compared with weekly off/on from exp02 (direct and residual, all-days policy).
"""
import sys
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, load, with_copy_status
from src.features import build
from src.metrics import score, peak_hour_f1

PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=0, verbose=-1, n_jobs=4)
FOLDS = {"val_Jul": ("2021-01-08", "2021-07-01", "2021-08-01"),
         "test_AugSep": ("2021-01-08", "2021-08-01", "2021-09-15")}
TAU = 30
df = with_copy_status(load())
base_off, base_on = build(df, use_weekly=False), build(df, use_weekly=True)
gap = base_on["roll24"] - base_on["roll24"].shift(168)
wcols = [c for c in base_on if c.endswith("l168")]
X_feat = base_on.assign(gap=gap)
X_mask = base_on.copy(); X_mask.loc[gap.abs() > TAU, wcols] = np.nan; X_mask["gap"] = gap
variants = {"weekly_off": base_off, "weekly_on": base_on, "G_feat": X_feat, "G_mask": X_mask}
print("share of hours gated |gap|>30:", {f: round(float((gap[(df.ts >= a) & (df.ts < c)].abs() > TAU).mean()), 3)
                                          for f, (a, b, c) in {"train": FOLDS["test_AugSep"]}.items()},
      "test:", round(float((gap[(df.ts >= "2021-08-01")].abs() > TAU).mean()), 3))
rows = []
for name, X in variants.items():
    for target in ("direct", "residual"):
        for fold, (a, b, c) in FOLDS.items():
            tr = (df.ts >= a) & (df.ts < b); te = (df.ts >= b) & (df.ts < c)
            p = {}
            for q in Q:
                off = X[f"{q}_l1"] if target == "residual" else 0
                y = df[q] - off; m = tr & y.notna()
                p[q] = lgb.LGBMRegressor(**PARAMS).fit(X[m], y[m]).predict(X[te]) + (off[te] if target == "residual" else 0)
            p = pd.DataFrame(p, index=df.index[te]); y = df.loc[te, Q]
            r = {"variant": name, "target": target, "fold": fold, **score(y, p)}
            r["pk_F1@190"] = peak_hour_f1(y, p)["F1"]
            rows.append(r)
res = pd.DataFrame(rows).round(3)
res.to_csv(ROOT / "outputs" / "exp02" / "gate_results.csv", index=False)
with pd.option_context("display.width", 200):
    print(res.pivot_table(index=["target", "variant"], columns="fold", values=["MAE", "peak_MAE"]))
