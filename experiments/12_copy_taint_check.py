"""Does the copy augmentation taint (a) our validation folds or (b) the trained models?

(a) Evaluation leakage: a test day is 'contaminated' if its exact 96-reading
    profile also occurs on a training-period date. Re-score exp03 predictions on
    clean test days only.
(b) Training taint: already tested for the forecaster in exp02 (copy weighting /
    removal). Here: the peak classifier with (i) copy-group-aware OOF blocks for
    cutoff selection, (ii) copies down-weighted 0.3, (iii) unique days only.
"""
import sys
from pathlib import Path
import lightgbm as lgb, numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, copy_map, load, with_copy_status
from src.features import build
from src.metrics import score
from src.peak import CLF_PARAMS, best_f1_cutoff, context_features, detection_scores, episode_onset, peak_label

OUT = ROOT / "outputs" / "exp12"; OUT.mkdir(parents=True, exist_ok=True)
df = with_copy_status(load()); cm = copy_map(df)
grp = df.date.map(cm.group)
FOLDS = [("Apr", "2021-04-01", "2021-05-01"), ("May", "2021-05-01", "2021-06-01"),
         ("Jun", "2021-06-01", "2021-07-01"), ("Jul", "2021-07-01", "2021-08-01"),
         ("AugSep", "2021-08-01", "2021-09-15")]
first_date = cm.reset_index().groupby("group").date.min()

print("=== (a) test days whose exact profile already appears in training ===")
preds = pd.read_pickle(ROOT / "outputs" / "exp03" / "predictions.pkl")
rows = []
for f, a, b in FOLDS:
    te = (df.ts >= a) & (df.ts < b)
    contaminated = te & (grp.map(first_date) < pd.Timestamp(a))
    nd, nc = df.loc[te, "date"].nunique(), df.loc[contaminated, "date"].nunique()
    clean = te & ~contaminated
    r = {"fold": f, "test_days": nd, "contaminated_days": nc}
    for v in ("base", "+weekly", "+state"):
        p = preds[(f, v, "direct")]
        for name, m in (("all", te), ("clean", clean), ("contam", contaminated)):
            idx = df.index[m]
            if len(idx):
                s = score(df.loc[idx, Q], p.loc[idx])
                r[f"{v}|{name}|MAE"] = s["MAE"]; r[f"{v}|{name}|pkMAE"] = s["peak_MAE"]
                r[f"{name}_peak_readings"] = int((df.loc[idx, Q] >= 190).values.sum())
    rows.append(r)
t = pd.DataFrame(rows).set_index("fold")
t.to_csv(OUT / "clean_vs_contaminated.csv")
with pd.option_context("display.width", 250, "display.max_columns", 40):
    print(t[["test_days", "contaminated_days", "clean_peak_readings", "contam_peak_readings"]])
    for v in ("base", "+weekly", "+state"):
        print(f"\n{v}:"); print(t[[c for c in t if c.startswith(v + "|")]].round(2))
    cleanfolds = ["Apr", "May", "Jun", "Jul"]
    print("\nSelection on CLEAN hours only (mean over folds that have clean hours):")
    for v in ("base", "+weekly", "+state"):
        print(f"  {v:8s} MAE {t[f'{v}|clean|MAE'][cleanfolds].mean():.2f}   peak MAE {t[f'{v}|clean|pkMAE'][cleanfolds].mean():.2f}"
              f"   (peak MAE folds: {t[f'{v}|clean|pkMAE'][cleanfolds].round(1).to_dict()})")

print("\n=== (b) peak classifier: copy-aware training / cutoff selection ===")
lbl, onset = peak_label(df), episode_onset(peak_label(df))
X = pd.concat([build(df, use_weekly=False), context_features(df), pd.DataFrame({"hour_next": df.hour}, index=df.index)], axis=1)
res = []
for f, a, b in FOLDS[3:]:
    tr = (df.ts >= "2021-01-08") & (df.ts < a); te = (df.ts >= a) & (df.ts < b)
    for variant in ("current", "group_oof", "copies0.3", "unique_only"):
        w = pd.Series(1.0, index=df.index)
        if variant == "copies0.3": w[df.copy_status != "unique"] = 0.3
        if variant == "unique_only": w[df.copy_status != "unique"] = 0.0
        trm = tr & (w > 0)
        # OOF blocks: by date (current) or by copy group so a day's copies never sit on both sides
        key = grp if variant == "group_oof" else df.date
        keys = np.sort(key[trm].unique()); rng = np.random.default_rng(0)
        blocks = np.array_split(keys if variant != "group_oof" else rng.permutation(keys), 6)
        s = pd.Series(np.nan, index=df.index)
        for blk in blocks:
            hold = trm & key.isin(blk)
            s[hold] = lgb.LGBMClassifier(**CLF_PARAMS).fit(X[trm & ~hold], lbl[trm & ~hold], sample_weight=w[trm & ~hold]).predict_proba(X[hold])[:, 1]
        s[te] = lgb.LGBMClassifier(**CLF_PARAMS).fit(X[trm], lbl[trm], sample_weight=w[trm]).predict_proba(X[te])[:, 1]
        cut = best_f1_cutoff(s[trm].to_numpy(), lbl[trm].to_numpy())
        d = detection_scores(s[te].to_numpy(), lbl[te].to_numpy(), onset[te].to_numpy(), cut)
        res.append({"fold": f, "variant": variant, "cutoff": cut, **{k: d[k] for k in ("TP", "FP", "FN", "F1", "onsets_caught", "PR_AUC")}})
res = pd.DataFrame(res); res.to_csv(OUT / "classifier_copy_variants.csv", index=False)
print(res.round(3).to_string(index=False))
