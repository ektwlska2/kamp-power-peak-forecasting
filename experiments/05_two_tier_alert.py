"""Two-tier peak alert built from exp04 scores (no refitting).

  RED    : classifier score >= its training-selected cutoff  -> act (defer loads)
  YELLOW : context rule (clock 08-11/13-14 & previous-hour mean >= 100), not red -> watch
Scored on Jul (selection) and Aug-Sep (test), leads 0 and 1.

Run: .venv/bin/python experiments/05_two_tier_alert.py   (after 04)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import ROOT, load  # noqa: E402
from src.peak import episode_onset, peak_label  # noqa: E402

E4 = ROOT / "outputs" / "exp04"
OUT = ROOT / "outputs" / "exp05"
OUT.mkdir(parents=True, exist_ok=True)
store = pd.read_pickle(E4 / "scores.pkl")
res = pd.read_csv(E4 / "results.csv")
df = load()
lbl = peak_label(df)
onset = episode_onset(lbl)

rows = []
for fold in ("Jul", "AugSep"):
    for lead in (0, 1):
        clf = store[(fold, lead, "clf")]
        cut = res.query("fold == @fold and lead == @lead and detector == 'clf'").cutoff.iloc[0]
        rule = store[(fold, lead, "context_rule")] >= 0.5
        idx = clf.index
        y = lbl.shift(-lead)[idx].astype(bool)
        on = onset.shift(-lead)[idx].astype(bool)
        red = clf >= cut
        yellow = rule & ~red
        anyal = red | yellow
        days = df.loc[idx, "date"].nunique()
        opdays = df.loc[idx].groupby("date").mean_load.mean().ge(35).sum()
        r = {"fold": fold, "lead": lead, "peak_hours": int(y.sum()), "onsets": int(on.sum())}
        for name, m in (("red", red), ("yellow", yellow), ("any", anyal)):
            r[f"{name}_hours"] = int(m.sum())
            r[f"{name}_peaks"] = int((m & y).sum())
            r[f"{name}_precision"] = (m & y).sum() / m.sum() if m.sum() else np.nan
            r[f"{name}_onsets"] = int((m & on).sum())
        r["missed_peaks"] = int((y & ~anyal).sum())
        r["recall_any"] = r["any_peaks"] / r["peak_hours"]
        r["F1_red"] = 2 * r["red_peaks"] / (r["red_hours"] + r["peak_hours"])
        r["F1_any"] = 2 * r["any_peaks"] / (r["any_hours"] + r["peak_hours"])
        r["red_per_opday"] = r["red_hours"] / opdays
        r["yellow_per_opday"] = r["yellow_hours"] / opdays
        r["false_red_per_opday"] = (red & ~y).sum() / opdays
        rows.append(r)
        # missed peak hours for later error analysis
        miss = df.loc[idx[(y & ~anyal).to_numpy()], ["ts", "hour", "mean_load", "production"]]
        miss.to_csv(OUT / f"missed_{fold}_lead{lead}.csv", index=False)

t = pd.DataFrame(rows)
t.to_csv(OUT / "two_tier.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 40):
    print(t.round(3).T)
