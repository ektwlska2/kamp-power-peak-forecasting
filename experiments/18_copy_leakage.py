"""Why copied days must be handled: duplicate-sample leakage in validation.

On the training period (01-08..07-31) we estimate next-hour MAE and peak
classifier quality with common validation schemes, then compare each
estimate with what actually happened on Aug-Sep (no copies there):

  random-hour 5-fold   : rows shuffled (a common default)
  random-day 5-fold    : whole days kept together
  copy-group 5-fold    : whole days kept together AND every copy of a profile in the same fold
  time-ordered         : train <= Jun, validate on July (what we used)

random-day vs copy-group differ ONLY in where copies go, so their gap is the
pure effect of copied days. Also: share of validation days whose exact
96-reading profile is already in the training folds.

Run: .venv/bin/python experiments/18_copy_leakage.py   (after run_all)
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, copy_map, load  # noqa: E402
from src.features import build  # noqa: E402
from src.models import CLF_PARAMS, FC_PARAMS  # noqa: E402
from src.peak import context_features, peak_label  # noqa: E402
from src.uncertainty import brier  # noqa: E402

OUT = ROOT / "outputs" / "exp18"
OUT.mkdir(parents=True, exist_ok=True)
df = load()
cm = copy_map(df)
grp = df.date.map(cm.group)
lbl = peak_label(df)
Xb = build(df, use_weekly=False)
Xc = pd.concat([Xb, context_features(df)], axis=1).assign(hour_next=df.hour)
TR = (df.ts >= "2021-01-08") & (df.ts < "2021-08-01")
idx = df.index[TR]
rng = np.random.default_rng(0)
K = 5


def folds(scheme):
    if scheme == "random-hour":
        f = rng.integers(0, K, len(idx))
        return pd.Series(f, index=idx)
    if scheme == "random-day":
        days = df.loc[idx, "date"].unique()
        m = dict(zip(days, rng.permutation(np.arange(len(days)) % K)))
        return df.loc[idx, "date"].map(m)
    if scheme == "copy-group":
        g = grp[idx].unique()
        m = dict(zip(g, rng.permutation(np.arange(len(g)) % K)))
        return grp[idx].map(m)
    raise ValueError(scheme)


def contamination(fold_of):
    """Share of validation days whose exact profile also appears in the training folds."""
    d = pd.DataFrame({"date": df.loc[idx, "date"], "fold": fold_of, "grp": grp[idx]}).drop_duplicates("date")
    shares = []
    for k in range(K):
        val, trn = d[d.fold == k], d[d.fold != k]
        if len(val) == 0:
            continue
        shares.append(val.grp.isin(set(trn.grp)).mean())
    return float(np.mean(shares))


def cv(fold_of):
    pred = pd.DataFrame(index=idx, columns=Q, dtype=float)
    prob = pd.Series(np.nan, index=idx)
    for k in range(K):
        va = idx[(fold_of == k).to_numpy()]
        trn = idx[(fold_of != k).to_numpy()]
        for q in Q:
            pred.loc[va, q] = lgb.LGBMRegressor(**FC_PARAMS).fit(Xb.loc[trn], df.loc[trn, q]).predict(Xb.loc[va])
        prob[va] = lgb.LGBMClassifier(**CLF_PARAMS).fit(Xc.loc[trn], lbl[trn]).predict_proba(Xc.loc[va])[:, 1]
    y = df.loc[idx, Q]
    pk = y.to_numpy() >= 190
    e = (pred - y).abs().to_numpy()
    return {"MAE": float(e.mean()), "peak_MAE": float(e[pk].mean()),
            "PR_AUC": float(average_precision_score(lbl[idx], prob)), "Brier": brier(prob, lbl[idx])}


rows = []
for scheme in ("random-hour", "random-day", "copy-group"):
    f = folds(scheme)
    r = {"scheme": scheme, "val_days_with_copy_in_training": contamination(f), **cv(f)}
    rows.append(r)
    print(r, flush=True)

# time-ordered (July), as used in the project
trj = (df.ts >= "2021-01-08") & (df.ts < "2021-07-01")
tej = (df.ts >= "2021-07-01") & (df.ts < "2021-08-01")
pj = pd.DataFrame({q: lgb.LGBMRegressor(**FC_PARAMS).fit(Xb[trj], df.loc[trj, q]).predict(Xb[tej]) for q in Q}, index=df.index[tej])
yj = df.loc[tej, Q]
ej = (pj - yj).abs().to_numpy()
prj = lgb.LGBMClassifier(**CLF_PARAMS).fit(Xc[trj], lbl[trj]).predict_proba(Xc[tej])[:, 1]
jul_days = df.loc[tej, "date"].drop_duplicates()
train_grps = set(grp[trj])
rows.append({"scheme": "time-ordered (July)", "val_days_with_copy_in_training": float(jul_days.map(cm.group).isin(train_grps).mean()),
             "MAE": float(ej.mean()), "peak_MAE": float(ej[yj.to_numpy() >= 190].mean()),
             "PR_AUC": float(average_precision_score(lbl[tej], prj)), "Brier": brier(prj, lbl[tej])})

# what actually happened on the test period (from run_all / exp15)
s = json.loads((ROOT / "outputs" / "final" / "summary.json").read_text())
t15 = pd.read_csv(ROOT / "outputs" / "exp15" / "test_scores.csv")
rows.append({"scheme": "ACTUAL Aug-Sep test", "val_days_with_copy_in_training": 0.0,
             "MAE": s["next_hour"]["test_MAE"], "peak_MAE": s["next_hour"]["test_peak_MAE"],
             "PR_AUC": float(average_precision_score(t15.y, t15.p_raw)), "Brier": brier(t15.p_raw, t15.y)})
res = pd.DataFrame(rows)
res["MAE_gap_vs_actual_pct"] = 100 * (res.MAE / res.MAE.iloc[-1] - 1)
res.round(4).to_csv(OUT / "cv_vs_actual.csv", index=False)
eff = {"days": int(cm.shape[0]), "distinct_profiles": int(cm.group.nunique()),
       "train_days": int(df.loc[idx, "date"].nunique()), "train_distinct_profiles": int(grp[idx].nunique())}
(OUT / "summary.json").write_text(json.dumps(eff, indent=2))
with pd.option_context("display.width", 200):
    print(res.round(3).to_string(index=False))
print(eff)
