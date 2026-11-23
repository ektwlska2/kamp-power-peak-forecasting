"""Peak-error difference LightGBM - Chronos-2 on Aug-Sep with a 7-day block
bootstrap 95% CI, and a week-by-week breakdown (honest statistics).

Run: .venv/bin/python experiments/17_peak_error_ci.py   (after run_all)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, load  # noqa: E402
from src.uncertainty import block_bootstrap_ci  # noqa: E402

OUT = ROOT / "outputs" / "exp17"
OUT.mkdir(parents=True, exist_ok=True)
CAP = 190
df = load()
tp = pd.read_csv(ROOT / "outputs" / "final" / "test_predictions.csv", parse_dates=["ts"])
ch = pd.read_csv(ROOT / "data" / "reference" / "chronos2_zero_shot_predictions.csv", parse_dates=["timestamp"])
ch["q"] = ch.quarter_minute.map({15: "15분", 30: "30분", 45: "45분", 60: "60분"})
chp = ch.pivot(index="timestamp", columns="q", values="prediction")
rows = []
for _, r in tp.iterrows():
    for c in ("15분", "30분", "45분", "60분"):
        a = r[f"actual_{c}"]
        if a >= CAP:
            rows.append({"ts": r.ts, "actual": a, "ours": abs(r[f"forecast_{c}"] - a),
                         "weekly": abs(r[f"forecast_weekly_{c}"] - a), "chronos": abs(chp.loc[r.ts, c] - a)})
d = pd.DataFrame(rows)
d["week"] = ((d.ts - pd.Timestamp("2021-08-01")).dt.days // 7)
out = {"peak_readings": len(d)}
for name in ("ours", "weekly"):
    diff = (d[name] - d.chronos).to_numpy()
    out[f"{name}_minus_chronos"] = float(diff.mean())
    out[f"{name}_minus_chronos_CI95"] = block_bootstrap_ci(diff, d.week.to_numpy())
    out[f"{name}_peak_MAE"] = float(d[name].mean())
out["chronos_peak_MAE"] = float(d.chronos.mean())
wk = d.groupby("week").agg(start=("ts", lambda s: s.min().strftime("%m-%d")), readings=("actual", "size"),
                           ours=("ours", "mean"), chronos=("chronos", "mean"))
wk["ours_better"] = wk.ours < wk.chronos
out["weeks_ours_better"] = f"{int(wk.ours_better.sum())} / {len(wk)}"
# leave-one-week-out: is the advantage driven by a single week?
loo = {int(w): float((d[d.week != w].ours - d[d.week != w].chronos).mean()) for w in wk.index}
out["leave_one_week_out_diff"] = loo
wk.round(2).to_csv(OUT / "by_week.csv")
(OUT / "summary.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=1))
print(wk.round(2))
