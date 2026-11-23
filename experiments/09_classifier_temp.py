"""Does past temperature improve the peak classifier? (Chapter 3 -> Chapter 2 loop)

clf       : exp04 classifier features (base + peak context)
clf+temp  : + temp at t-1, 24h max temp, temp change over 3h (all past-only)
Same training as the final pipeline (src.models.fit_detector, copy weights 1/n, OOF max-F1 cutoff).
Selection folds: Jun, Jul (Apr/May have 1-2 peak hours). Test: Aug-Sep.

Run: .venv/bin/python experiments/09_classifier_temp.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, load  # noqa: E402
from src.features import build  # noqa: E402
from src.models import fit_detector  # noqa: E402
from src.peak import context_features, detection_scores, episode_onset, peak_label  # noqa: E402
from src.weights import copy_weights  # noqa: E402

OUT = ROOT / "outputs" / "exp09"
OUT.mkdir(parents=True, exist_ok=True)
FOLDS = [("Jun", "2021-06-01", "2021-07-01"), ("Jul", "2021-07-01", "2021-08-01"),
         ("AugSep", "2021-08-01", "2021-09-15")]
START = "2021-01-08"


def temp_features(df):
    t = df["temp"].shift(1)
    return pd.DataFrame({"temp_l1": t, "temp_max24": t.rolling(24).max(),
                         "temp_d3": t - t.shift(3)}, index=df.index)


df = load()
lbl, onset = peak_label(df), episode_onset(peak_label(df))
common = pd.concat([build(df, use_weekly=False), context_features(df)], axis=1)
T = temp_features(df)
i = len(df) // 2
d2 = df.copy(); d2.loc[i, "temp"] += 50
assert (temp_features(d2).loc[i].fillna(-1) == T.loc[i].fillna(-1)).all(), "temp leak"

rows, store = [], {}
for name, a, b in FOLDS:
    for lead in (0, 1):
        y = lbl.shift(-lead); on = onset.shift(-lead)
        tr = (df.ts >= START) & (df.ts < a) & y.notna() & (df.ts + pd.Timedelta(hours=lead) < a)
        te = (df.ts >= a) & (df.ts < b) & y.notna() & (df.ts + pd.Timedelta(hours=lead) < b)
        extra = pd.DataFrame({"hour_next": (df.hour + lead) % 24}, index=df.index)
        for fname, X in {"clf": pd.concat([common, extra], axis=1),
                         "clf+temp": pd.concat([common, extra, T], axis=1)}.items():
            # same training as the final pipeline: copy weights 1/n, out-of-fold F1 cutoff
            model, cut = fit_detector(X, y, tr, copy_weights(df, tr), dates=df.date)
            s = pd.Series(np.nan, index=df.index)
            s[te] = model.predict_proba(X[te])[:, 1]
            rows.append({"fold": name, "lead": lead, "model": fname, "cutoff": cut,
                         **detection_scores(s[te].to_numpy(), y[te].to_numpy(), on[te].to_numpy(), cut)})
            store[(name, lead, fname)] = s[te]
res = pd.DataFrame(rows)
res.to_csv(OUT / "results.csv", index=False)
pd.to_pickle(store, OUT / "scores.pkl")
cols = ["fold", "lead", "model", "peak_hours", "TP", "FP", "FN", "precision", "recall", "F1", "onsets_caught", "PR_AUC", "Brier"]
print(res[cols].round(3).to_string(index=False))
