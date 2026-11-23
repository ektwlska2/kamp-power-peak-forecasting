"""Peak-cap (load-deferral) simulation on recorded 15-min readings (Chapter 4).

For a cap C, any demand above C is deferred to later 15-min intervals with
headroom (energy conserving). Reports, per cap:
  - days needing deferral, energy deferred (recorded unit x hours),
    share of that day's energy, longest delay, days where the backlog is not
    cleared by the end of the day
  - how much of the excess falls in hours the two-tier alert flagged
    (lead 0 = at hour start; lead 1 = one hour ahead)
  - monthly maximum 15-min demand before/after (what a demand charge sees)
This is an arithmetic bound on required flexibility, not a demonstrated
reduction: which loads can be deferred is unknown in this dataset.

Run: .venv/bin/python experiments/11_peak_cap_sim.py   (after 04)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import Q, ROOT, load  # noqa: E402

OUT = ROOT / "outputs" / "exp11"
OUT.mkdir(parents=True, exist_ok=True)
CAPS = [200, 190, 180, 170]

df = load()
store = pd.read_pickle(ROOT / "outputs" / "exp04" / "scores.pkl")
r4 = pd.read_csv(ROOT / "outputs" / "exp04" / "results.csv")


def alert_flags(lead):
    cut = r4.query("fold == 'AugSep' and lead == @lead and detector == 'clf'").cutoff.iloc[0]
    s = store[("AugSep", lead, "clf")]
    rule = store[("AugSep", lead, "context_rule")] >= 0.5
    red = s >= cut
    # flags issued at row t target hour t+lead -> re-index to the target hour
    red.index = red.index + lead
    anyal = (red | pd.Series(rule.to_numpy(), index=red.index))
    return red, anyal


def simulate(day_rows, cap):
    d = day_rows[Q].to_numpy().ravel().astype(float)       # 96 intervals
    out, B, delay_start, max_delay = [], 0.0, None, 0
    for i, x in enumerate(d):
        want = x + B
        o = min(want, cap)
        newB = want - o
        if newB > 0 and delay_start is None:
            delay_start = i
        if newB == 0 and delay_start is not None:
            max_delay = max(max_delay, i - delay_start + 1); delay_start = None
        out.append(o); B = newB
    if delay_start is not None:
        max_delay = max(max_delay, len(d) - delay_start)
    excess = np.clip(d - cap, 0, None).sum()                  # energy that exceeded the cap at source
    return np.array(out), excess / 4, B / 4, max_delay * 15, d


periods = {"Jun-Jul": ("2021-06-01", "2021-08-01"), "Aug-Sep": ("2021-08-01", "2021-09-15")}
red0, any0 = alert_flags(0)
red1, any1 = alert_flags(1)
rows, monthly = [], []
for pname, (a, b) in periods.items():
    P = df[(df.ts >= a) & (df.ts < b)]
    for cap in CAPS:
        day_stats = []
        exc_by_hour = pd.Series(0.0, index=P.index)
        for day, g in P.groupby("date"):
            out, exc, left, delay, raw = simulate(g, cap)
            energy = raw.sum() / 4
            exc_by_hour[g.index] = np.clip(g[Q].to_numpy() - cap, 0, None).sum(axis=1) / 4
            day_stats.append({"date": day, "excess": exc, "left_over": left, "max_delay_min": delay,
                              "energy": energy, "raw_max": raw.max(), "new_max": out.max()})
            monthly.append({"period": pname, "cap": cap, "month": day.month, "raw_max": raw.max(), "new_max": out.max()})
        ds = pd.DataFrame(day_stats)
        hit = ds[ds.excess > 0]
        r = {"period": pname, "cap": cap, "days": len(ds), "days_needing_deferral": len(hit),
             "mean_deferred_per_day": hit.excess.mean() if len(hit) else 0,
             "max_deferred_per_day": hit.excess.max() if len(hit) else 0,
             "deferred_share_of_day_energy_%": 100 * (hit.excess / hit.energy).mean() if len(hit) else 0,
             "longest_delay_min": ds.max_delay_min.max(),
             "days_backlog_not_cleared": int((ds.left_over > 1e-9).sum())}
        if pname == "Aug-Sep":
            tot = exc_by_hour.sum()
            for nm, fl in (("red0", red0), ("any0", any0), ("red1", red1), ("any1", any1)):
                f = fl.reindex(exc_by_hour.index).fillna(False).astype(bool)
                r[f"excess_in_{nm}_%"] = 100 * exc_by_hour[f].sum() / tot if tot else np.nan
        rows.append(r)
res = pd.DataFrame(rows)
res.to_csv(OUT / "cap_simulation.csv", index=False)
mon = pd.DataFrame(monthly).groupby(["cap", "month"]).agg(raw_max=("raw_max", "max"), new_max=("new_max", "max"))
mon.to_csv(OUT / "monthly_max.csv")
with pd.option_context("display.width", 250, "display.max_columns", 30):
    print(res.round(2).T)
    print("\nMonthly max 15-min demand (recorded units), raw vs capped:")
    print(mon.unstack("cap").round(0))
print("\nAnnual (Jan-Sep) max 15-min reading:", int(df[Q].max().max()),
      "| Jan-May max:", int(df[df.ts < "2021-06-01"][Q].max().max()))
