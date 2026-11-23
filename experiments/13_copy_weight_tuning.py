"""Copy weight as a hyperparameter (pre-declared protocol).

Grid: w in {0, 0.1, 0.3, 0.5, 1.0} for repeated-profile days (copy + first),
plus the rule-based 1/group_size (group size counted inside the training
window only). Unique days always weight 1.

Selection on July only (train 01-08..06-30; July has 0 contaminated days):
  peak classifier     : PR-AUC (threshold-free), F1 secondary
  load-response model : MAE + peak MAE rank sum (day-ahead, plan+calendar+temp)
Day-block bootstrap (2000) of the difference best - 1/group_size on July.
Rule: adopt the best grid value only if its 95% CI excludes zero in its
favour; otherwise use 1/group_size.
Aug-Sep is scored once for the chosen weight; the full Aug-Sep grid is saved
as information only (not used for selection).

Run: .venv/bin/python experiments/13_copy_weight_tuning.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, copy_map, load, with_copy_status  # noqa: E402
from src.dayahead import opcal, strict  # noqa: E402
from src.features import build  # noqa: E402
from src.metrics import score  # noqa: E402
from src.peak import (CLF_PARAMS, FC_PARAMS, best_f1_cutoff, context_features,  # noqa: E402
                      detection_scores, episode_onset, peak_label)

OUT = ROOT / "outputs" / "exp13"
OUT.mkdir(parents=True, exist_ok=True)
GRID = ["0", "0.1", "0.3", "0.5", "1.0", "1/n"]
START = pd.Timestamp("2021-01-08")
FOLDS = {"Jul": ("2021-07-01", "2021-08-01"), "AugSep": ("2021-08-01", "2021-09-15")}
N_BOOT = 2000

df = with_copy_status(load())
cm = copy_map(df)
grp = df.date.map(cm.group)
lbl, onset = peak_label(df), episode_onset(peak_label(df))


def weights(wname, train_mask):
    w = pd.Series(1.0, index=df.index)
    rep = df.copy_status != "unique"
    if wname == "1/n":
        # group size counted among training-window dates only
        tr_dates = df.loc[train_mask, ["date"]].drop_duplicates()
        n = tr_dates.date.map(cm.group).value_counts()
        w[rep] = 1.0 / grp[rep].map(n).fillna(1)
    else:
        w[rep] = float(wname)
    return w


def day_bootstrap(fn_by_rows, test_idx, seed=0):
    """Resample test days with replacement; fn_by_rows(idx) -> dict of metric per weight."""
    days = df.loc[test_idx, "date"]
    ud = days.unique()
    pos = {d: np.flatnonzero(days.to_numpy() == d) for d in ud}
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(N_BOOT):
        pick = rng.choice(ud, len(ud))
        rows = np.concatenate([pos[d] for d in pick])
        out.append(fn_by_rows(rows))
    return pd.DataFrame(out)


decision = {}

# ------------------------------------------------------------ peak classifier
Xc = pd.concat([build(df, use_weekly=False), context_features(df),
                pd.DataFrame({"hour_next": df.hour}, index=df.index)], axis=1)
clf_rows, clf_scores = [], {}
for fold, (a, b) in FOLDS.items():
    tr = (df.ts >= START) & (df.ts < a)
    te = (df.ts >= a) & (df.ts < b)
    for wname in GRID:
        w = weights(wname, tr)
        trm = tr & (w > 0)
        s = pd.Series(np.nan, index=df.index)
        for blk in np.array_split(np.sort(df.loc[trm, "date"].unique()), 6):
            hold = trm & df.date.isin(blk)
            s[hold] = lgb.LGBMClassifier(**CLF_PARAMS).fit(
                Xc[trm & ~hold], lbl[trm & ~hold], sample_weight=w[trm & ~hold]).predict_proba(Xc[hold])[:, 1]
        s[te] = lgb.LGBMClassifier(**CLF_PARAMS).fit(Xc[trm], lbl[trm], sample_weight=w[trm]).predict_proba(Xc[te])[:, 1]
        cut = best_f1_cutoff(s[trm].to_numpy(), lbl[trm].to_numpy())
        d = detection_scores(s[te].to_numpy(), lbl[te].to_numpy(), onset[te].to_numpy(), cut)
        clf_rows.append({"fold": fold, "weight": wname, "cutoff": cut,
                         **{k: d[k] for k in ("PR_AUC", "F1", "TP", "FP", "FN", "onsets_caught", "Brier")}})
        clf_scores[(fold, wname)] = s[te]
clf = pd.DataFrame(clf_rows)
clf.to_csv(OUT / "classifier_grid.csv", index=False)

jul_idx = clf_scores[("Jul", "1/n")].index
yj = lbl[jul_idx].to_numpy()


def clf_boot(rows):
    y = yj[rows]
    return {w: average_precision_score(y, clf_scores[("Jul", w)].to_numpy()[rows]) if y.any() else np.nan
            for w in GRID}


bc = day_bootstrap(clf_boot, jul_idx)
jul = clf[clf.fold == "Jul"].set_index("weight")
best = jul.PR_AUC.idxmax()
diff = (bc[best] - bc["1/n"]).dropna()
ci = np.percentile(diff, [2.5, 97.5])
chosen_clf = best if (best != "1/n" and ci[0] > 0) else "1/n"
decision["classifier"] = {"best_on_July": best, "July_PR_AUC_best": jul.PR_AUC[best],
                          "July_PR_AUC_1/n": jul.PR_AUC["1/n"], "diff_CI95": ci.tolist(), "chosen": chosen_clf}

# ------------------------------------------------------------ load-response model
PROD_FEATS = ["prod_h", "prod_prev", "prod_next"]


def plan_feats(prod, date):
    g = prod.groupby(date)
    return pd.DataFrame({"prod_h": prod, "prod_prev": g.shift(1).fillna(0),
                         "prod_next": g.shift(-1).fillna(0), "prod_day": g.transform("sum")})


Xr = pd.concat([strict(df), opcal(df, levels=2), df[["temp"]], plan_feats(df.production, df.date)], axis=1)
mono = [1 if c in PROD_FEATS else 0 for c in Xr.columns]
resp_rows, resp_preds = [], {}
for fold, (a, b) in FOLDS.items():
    tr = (df.ts >= START) & (df.ts < a)
    te = (df.ts >= a) & (df.ts < b)
    for wname in GRID:
        w = weights(wname, tr)
        trm = tr & (w > 0)
        p = pd.DataFrame({q: lgb.LGBMRegressor(**FC_PARAMS, monotone_constraints=mono)
                          .fit(Xr[trm], df.loc[trm, q], sample_weight=w[trm]).predict(Xr[te]) for q in Q},
                         index=df.index[te])
        resp_preds[(fold, wname)] = p
        resp_rows.append({"fold": fold, "weight": wname, **score(df.loc[te, Q], p)})
resp = pd.DataFrame(resp_rows)
resp.to_csv(OUT / "response_grid.csv", index=False)
rj = resp[resp.fold == "Jul"].set_index("weight")
rank = rj.MAE.rank() + rj.peak_MAE.rank()
best_r = rank.idxmin()
yv = df.loc[resp_preds[("Jul", "1/n")].index, Q].to_numpy()


def resp_boot(rows):
    return {w: np.abs(resp_preds[("Jul", w)].to_numpy()[rows] - yv[rows]).mean() for w in GRID}


br = day_bootstrap(resp_boot, resp_preds[("Jul", "1/n")].index)
diff_r = br["1/n"] - br[best_r]           # positive = best is better
ci_r = np.percentile(diff_r, [2.5, 97.5])
chosen_r = best_r if (best_r != "1/n" and ci_r[0] > 0) else "1/n"
decision["response_model"] = {"best_on_July": best_r, "July_MAE_best": rj.MAE[best_r],
                              "July_MAE_1/n": rj.MAE["1/n"], "MAE_gain_CI95": ci_r.tolist(), "chosen": chosen_r}
(OUT / "decision.json").write_text(json.dumps(decision, indent=2, default=float))

with pd.option_context("display.width", 250):
    print("Peak classifier, JULY (selection):\n", jul[["PR_AUC", "F1", "TP", "FP", "onsets_caught", "cutoff"]].round(3))
    print(f"  best {best}; PR-AUC diff vs 1/n 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}] -> chosen: {chosen_clf}")
    print("\nLoad-response model, JULY (selection):\n", rj[["MAE", "peak_MAE", "bias"]].round(3), "\n rank sum:", rank.to_dict())
    print(f"  best {best_r}; MAE gain vs 1/n 95% CI [{ci_r[0]:+.3f}, {ci_r[1]:+.3f}] -> chosen: {chosen_r}")
    print("\n=== Aug-Sep, chosen weights (scored once) ===")
    print(" classifier:", clf.query("fold == 'AugSep' and weight == @chosen_clf")
          [["weight", "PR_AUC", "F1", "TP", "FP", "FN", "onsets_caught"]].round(3).to_string(index=False))
    print(" response  :", resp.query("fold == 'AugSep' and weight == @chosen_r")
          [["weight", "MAE", "peak_MAE", "bias"]].round(3).to_string(index=False))
    print("\n(information only, not used for selection) Aug-Sep full grid:")
    print(clf[clf.fold == "AugSep"].set_index("weight")[["PR_AUC", "F1", "TP", "FP"]].round(3))
    print(resp[resp.fold == "AugSep"].set_index("weight")[["MAE", "peak_MAE"]].round(3))
