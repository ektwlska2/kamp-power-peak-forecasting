"""Weekly-lag contamination test (part 3) + 인건비 inclusion test.

Grid: weekly lags {off, on} x training policy {all, drop exact day-7 copies,
unique days only, down-weight repeated days 0.3} x target {direct, residual}.
Folds (frozen settings, chosen before looking at results):
  val : train 2021-01-08..06-30, score July
  test: train 2021-01-08..07-31, score 2021-08-01..09-14
Scoring always uses every hour of the scored period (all days are real there
except July's single copy day, kept for comparability).

Run: .venv/bin/python experiments/02_weekly_lag_test.py
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, load, with_copy_status  # noqa: E402
from src.features import assert_no_future_leak, build  # noqa: E402
from src.metrics import peak_hour_f1, score  # noqa: E402

OUT = ROOT / "outputs" / "exp02"
OUT.mkdir(parents=True, exist_ok=True)

PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=20,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=0, verbose=-1, n_jobs=4)
FOLDS = {
    "val_Jul": ("2021-01-08", "2021-07-01", "2021-08-01"),
    "test_AugSep": ("2021-01-08", "2021-08-01", "2021-09-15"),
}
POLICIES = ["all", "drop_exact7", "unique_only", "downweight0.3"]

df = with_copy_status(load())
for kw in ({"use_weekly": True}, {"use_weekly": True, "use_labor": True}):
    assert_no_future_leak(df, **kw)


def weights(policy):
    w = pd.Series(1.0, index=df.index)
    if policy == "drop_exact7":
        w[df.equals_minus7] = 0
    elif policy == "unique_only":
        w[df.copy_status != "unique"] = 0
    elif policy == "downweight0.3":
        w[df.copy_status != "unique"] = 0.3
    return w


def fit_predict(X, target, w, tr, te):
    preds = {}
    for c in Q:
        y = df[c] - (X[f"{c}_l1"] if target == "residual" else 0)
        m = tr & y.notna() & (w > 0)
        model = lgb.LGBMRegressor(**PARAMS).fit(X[m], y[m], sample_weight=w[m])
        base = X.loc[te, f"{c}_l1"] if target == "residual" else 0
        preds[c] = model.predict(X[te]) + base
    return pd.DataFrame(preds, index=df.index[te])


rows, saved = [], {}
configs = [(wk, pol, tgt, False) for wk in (False, True) for pol in POLICIES
           for tgt in ("direct", "residual")]
configs += [(True, "all", tgt, True) for tgt in ("direct", "residual")]
for use_weekly, policy, target, use_labor in configs:
    X = build(df, use_weekly=use_weekly, use_labor=use_labor)
    w = weights(policy)
    for fold, (a, b, c) in FOLDS.items():
        tr = (df.ts >= a) & (df.ts < b)
        te = (df.ts >= b) & (df.ts < c)
        p = fit_predict(X, target, w, tr, te)
        y = df.loc[te, Q]
        r = {"fold": fold, "weekly": use_weekly, "policy": policy, "target": target,
             "labor": use_labor, "train_days": int(df.loc[tr & (w > 0), "date"].nunique())}
        r.update(score(y, p))
        r.update({f"pk_{k}": v for k, v in peak_hour_f1(y, p).items() if k in ("recall", "precision", "F1")})
        rows.append(r)
        saved[(fold, use_weekly, policy, target, use_labor)] = p

res = pd.DataFrame(rows).round(3)
res.to_csv(OUT / "results.csv", index=False)
with pd.option_context("display.width", 250, "display.max_rows", 200):
    for fold in FOLDS:
        print(f"\n--- {fold} ---")
        print(res[res.fold == fold].drop(columns="fold").sort_values(["target", "labor", "policy", "weekly"])
              .to_string(index=False))

# effect of weekly lags per policy (MAE with - without), negative = weekly lags help
eff = res[~res.labor].pivot_table(index=["fold", "target", "policy"], columns="weekly", values="MAE")
eff["weekly_effect"] = (eff[True] - eff[False]).round(3)
print("\nWeekly-lag effect on MAE (on - off); negative means weekly lags help:")
print(eff)
eff.to_csv(OUT / "weekly_effect.csv")

# day-block bootstrap of the key contrast on the test fold
def block_ci(pa, pb, y, days=7, n=2000, seed=0):
    d = (pa - y).abs().mean(axis=1) - (pb - y).abs().mean(axis=1)
    blk = ((d.index.map(lambda i: df.loc[i, "ts"]) - pd.Timestamp("2021-08-01")).days // days)
    g = d.groupby(np.asarray(blk)).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(g), (n, len(g)))
    stats = g["sum"].to_numpy()[idx].sum(1) / g["count"].to_numpy()[idx].sum(1)
    return d.mean(), np.percentile(stats, [2.5, 97.5])

y = df.loc[(df.ts >= "2021-08-01") & (df.ts < "2021-09-15"), Q]
print("\nTest fold, 7-day block bootstrap of MAE difference (weekly on - off):")
for tgt in ("direct", "residual"):
    for pol in POLICIES:
        m, ci = block_ci(saved[("test_AugSep", True, pol, tgt, False)],
                         saved[("test_AugSep", False, pol, tgt, False)], y)
        print(f"  {tgt:8s} {pol:14s} diff {m:+.3f}  95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]")
