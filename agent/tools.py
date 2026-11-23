"""Deterministic tools the agent calls. Every tool takes `now`, the start of
hour t (the issuance time), and only uses:
  * model outputs issued at t from the cache (built with data through t-1)
  * recorded rows with ts < now
  * statistics from before the test period (cache 'stats', 'rate_table')
Each tool returns a plain dict. Numbers the agent may quote are listed under
'evidence' so the grounding check can verify every number in a message.
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import Q
from src.models import RISK_HOURS
from src.state import IDLE

ROOT = Path(__file__).resolve().parents[1]
FEATURE_KO = {
    "risk_window": "위험 시간대 규칙 (08–11·13–14시, 직전 평균 100 이상)",
    "intra_slope_l1": "직전 시간 안의 상승폭 (60분값 − 15분값)",
    "hours_since_peak": "마지막 피크 이후 경과 시간",
    "roll24": "최근 24시간 평균", "roll6": "최근 6시간 평균", "roll3": "최근 3시간 평균",
    "max24": "최근 24시간 최대값", "max_l1": "직전 시간 최대값", "mean_l1": "직전 시간 평균",
    "d_l1": "직전 1시간 변화량", "d_l24": "어제 같은 시각의 변화량", "mean_l24": "어제 같은 시각 평균",
    "yday_max": "어제 최대값", "sofar_max": "오늘 지금까지 최대값", "peaks_last24": "최근 24시간 피크 횟수",
    "peak_l1": "직전 시간 피크 여부", "peak_l24": "어제 같은 시각 피크 여부",
    "hour": "시각", "hour_next": "시각", "dow": "요일", "weekend": "주말 여부", "holiday": "공휴일 여부",
    "offday": "휴일 여부", "d1_win3": "어제 같은 시각 부근 평균",
}


def feature_ko(name):
    if name in FEATURE_KO:
        return FEATURE_KO[name]
    if name[:1] == "q" and "_l" in name:           # q4_l1 -> "1시간 전 60분값"
        q, lag = name.split("_l")
        return f"{lag}시간 전 {int(q[1]) * 15}분값"
    return name


class PeakTools:
    def __init__(self, df: pd.DataFrame, cache: dict, sop: dict):
        self.df = df
        self.h = cache["hours"]
        self.stats = cache["stats"]
        self.rate = cache["rate_table"]
        self.sop = sop
        self.cap = sop["cap"]
        self.pos = pd.Series(df.index, index=df.ts)
        self.gap_hist = self._gap_history(cache.get("gap_july", pd.Series(dtype=float)))

    GAP_WINDOW_DAYS, GAP_QUANTILE = 28, 0.99     # fixed before looking at the test flags

    def _gap_history(self, gap_july):
        """|actual hourly mean - next-hour forecast mean| per completed hour, used for the
        rolling unusual-load threshold. Hours whose own readings or forecast inputs
        (previous 3 hours) contain a data fault are left out. July (forecaster trained
        <= Jun) is the warm start; each value is only used once its hour is in the past."""
        df, h = self.df, self.h
        if h.empty:
            return gap_july
        qs = df[Q].to_numpy(float)
        flat = qs.ravel()
        jump = np.abs(np.diff(flat, prepend=flat[0])).reshape(-1, 4).max(axis=1) > self.stats["jump_p999"]
        fault = pd.Series((qs == 0).any(axis=1) | jump, index=df.index)
        affected = fault | fault.shift(1, fill_value=False) | fault.shift(2, fill_value=False) | fault.shift(3, fill_value=False)
        rows = h.index
        gap = (df.loc[rows, Q].mean(axis=1) - h[[f"fc_{q}" for q in Q]].mean(axis=1)).abs()
        keep = ~affected.loc[rows]
        test_hist = pd.Series(gap[keep].to_numpy(), index=df.loc[rows[keep.to_numpy()], "ts"].to_numpy())
        return pd.concat([gap_july, test_hist]).sort_index()

    def gap_threshold(self, hour_ts):
        """99th percentile of the gap over the 28 days before `hour_ts` (completed hours only)."""
        hist = self.gap_hist
        w = hist[(hist.index < hour_ts) & (hist.index >= hour_ts - pd.Timedelta(days=self.GAP_WINDOW_DAYS))]
        if len(w) < 24 * 7:
            return float(self.stats["resid_q99"])
        return round(float(w.quantile(self.GAP_QUANTILE)), 1)

    @classmethod
    def load(cls, df):
        cache = pd.read_pickle(ROOT / "agent" / "cache" / "cache.pkl")
        sop = json.loads((ROOT / "agent" / "sop.json").read_text())
        return cls(df, cache, sop)

    # ------------------------------------------------------------ helpers
    def _row(self, now):
        return int(self.pos[pd.Timestamp(now)])

    def _past(self, now):
        return self.df[self.df.ts < pd.Timestamp(now)]

    def _day_state(self, date):
        # shutdown calendar: whether the plant runs on `date` is known a day ahead
        s = self.df.loc[self.df.date == date, "state"]
        return int(s.iloc[0]) if len(s) else None

    # ------------------------------------------------------------ tools
    def status(self, now):
        i = self._row(now)
        r = self.h.loc[i]
        fc = [float(r[f"fc_{q}"]) for q in Q]
        p_show = float(r.p0_cal) if "p0_cal" in r.index and pd.notna(r.p0_cal) else float(r.p0)
        out = {"tool": "status", "ts": str(now), "hour": int(pd.Timestamp(now).hour),
               "forecast": [round(x, 1) for x in fc], "forecast_max": round(max(fc), 1),
               "p_peak": round(p_show, 3), "p_peak_raw": round(float(r.p0), 3), "tier": r.tier0,
               "yellow_reason": r.get("yellow_reason0", "rule") or "rule",
               "pi_upper_max": round(float(r.pi_upper_max), 1) if "pi_upper_max" in r.index else None,
               "pi_q": round(float(r.pi_q), 1) if "pi_q" in r.index else None}
        nxt = self.h.loc[i]  # lead-1 fields on row t describe hour t+1
        if pd.notna(nxt.tier1):
            p1 = float(nxt.p1_cal) if "p1_cal" in nxt.index and pd.notna(nxt.p1_cal) else float(nxt.p1)
            out.update({"next_hour": int((pd.Timestamp(now) + pd.Timedelta(hours=1)).hour),
                        "next_tier": nxt.tier1, "next_p_peak": round(p1, 3)})
        out["evidence"] = {"hour": out["hour"], "forecast": out["forecast"], "forecast_max": out["forecast_max"],
                           "pi_upper_max": out["pi_upper_max"], "pi_q": out["pi_q"], "pi_level": 90,
                           "p_peak": out["p_peak"], "p_peak_pct": round(out["p_peak"] * 100),
                           "next_hour": out.get("next_hour"), "next_p_peak_pct": round(out.get("next_p_peak", 0) * 100),
                           "cap": self.cap}
        return out

    def condition_rate(self, now):
        """Historical peak rate (Jun-Jul) for this clock band and previous-hour load band."""
        past = self._past(now)
        prev = float(past.mean_load.iloc[-1])
        h = pd.Timestamp(now).hour
        hg = "00-07" if h <= 7 else "08-11" if h <= 11 else "12" if h == 12 else "13-14" if h <= 14 else "15-17" if h <= 17 else "18-23"
        lb = "<80" if prev < 80 else "80-100" if prev < 100 else "100-120" if prev < 120 else "120-140" if prev < 140 else ">=140"
        try:
            m, n = self.rate.loc[(hg, lb)]
        except KeyError:
            m, n = np.nan, 0
        return {"tool": "condition_rate", "clock_band": hg, "prev_mean": round(prev), "load_band": lb,
                "rate": None if n == 0 else round(float(m) * 100), "hours": int(n),
                "evidence": {"prev_mean": round(prev), "rate_pct": None if n == 0 else round(float(m) * 100),
                             "hours": int(n), "rate_period_months": [6, 7]}}

    def analog_days(self, now, k=5):
        """Most similar earlier running days by today's profile so far (hours 0..h-1).
        Copy groups count once (most recent member kept)."""
        now = pd.Timestamp(now)
        h = now.hour
        today = now.normalize()
        if h < 3:
            return {"tool": "analog_days", "available": False, "reason": "오늘 데이터가 3시간 미만", "evidence": {}}
        df = self.df
        cur = df[(df.date == today) & (df.hour < h)].mean_load.to_numpy()
        cand = df[(df.date < today) & (df.state != IDLE)]
        prof = cand.pivot(index="date", columns="hour", values="mean_load")
        pk = cand.assign(pk=cand[Q].max(axis=1) >= self.cap).pivot(index="date", columns="hour", values="pk")
        grp = cand.groupby("date").group.first()
        d = np.sqrt(((prof.iloc[:, :h].to_numpy() - cur) ** 2).mean(axis=1))
        rank = pd.DataFrame({"dist": d, "group": grp.reindex(prof.index).to_numpy()}, index=prof.index)
        rank = rank.sort_values("dist").reset_index().sort_values(["group", "date"]).drop_duplicates("group", keep="last")
        rank = rank.sort_values("dist").head(k).set_index("date")
        peaked_now = pk.loc[rank.index, h].astype(bool)
        peaked_rest = pk.loc[rank.index, list(range(h, 24))].any(axis=1)
        return {"tool": "analog_days", "available": True, "k": len(rank),
                "dates": [str(x.date()) for x in rank.index], "distance": rank.dist.round(1).tolist(),
                "peaked_this_hour": int(peaked_now.sum()), "peaked_rest_of_day": int(peaked_rest.sum()),
                "evidence": {"k": len(rank), "peaked_this_hour": int(peaked_now.sum()),
                             "peaked_rest_of_day": int(peaked_rest.sum()), "hour": h}}

    def explain(self, now):
        i = self._row(now)
        factors = [{"feature": f, "name_ko": feature_ko(f), "value": round(v, 1), "contribution": round(c, 2)}
                   for f, v, c in self.h.loc[i, "top_factors"]]
        st, cr, an = self.status(now), self.condition_rate(now), self.analog_days(now)
        label_nums = [float(x) for f in factors for x in re.findall(r"\d+", f["name_ko"])]
        ev = {"factor_values": [f["value"] for f in factors], "label_numbers": label_nums,
              **st["evidence"], **cr["evidence"],
              **{f"analog_{k}": v for k, v in an["evidence"].items()}}
        return {"tool": "explain", "status": st, "factors": factors, "condition": cr, "analog": an, "evidence": ev}

    def day_briefing(self, date):
        """Issued at 00:00 of `date`: day-ahead forecast (with shutdown calendar),
        restart-day and hot-day flags, watch hours."""
        date = pd.Timestamp(date).normalize()
        st = self._day_state(date)
        prev_st = self._day_state(date - pd.Timedelta(days=1))
        rows = self.df.index[self.df.date == date]
        s = self.stats
        if st == IDLE:
            return {"tool": "day_briefing", "date": str(date.date()), "running": False,
                    "evidence": {"month": date.month, "day": date.day}}
        da = self.h.loc[rows, [f"da_{q}" for q in Q]].max(axis=1)
        exp = da + s["margin_dayahead"]
        watch = [int(self.df.loc[r, "hour"]) for r in rows if exp[r] >= self.cap]
        tmax = float(self.df.loc[rows, "temp"].max())   # recorded value used as a forecast proxy
        restart = prev_st == IDLE
        hot = tmax >= 28
        out = {"tool": "day_briefing", "date": str(date.date()), "running": True, "restart_day": bool(restart),
               "hot_day": bool(hot), "temp_max": round(tmax, 1), "dayahead_max": round(float(da.max()), 1),
               "margin": s["margin_dayahead"], "watch_hours": watch,
               "rate_restart": round(s["peak_rate_restart_day"] * 100, 1), "rate_running": round(s["peak_rate_running_day"] * 100, 1),
               "rate_hot": round(s["peak_rate_hot"] * 100, 1), "rate_not_hot": round(s["peak_rate_not_hot"] * 100, 1)}
        out["evidence"] = {k: out[k] for k in ("temp_max", "dayahead_max", "margin", "rate_restart", "rate_running",
                                                "rate_hot", "rate_not_hot")}
        out["evidence"].update({"watch_hours": watch, "month": date.month, "day": date.day, "cap": self.cap})
        return out

    def propose(self, now):
        """SOP action for this hour. RED: hold back the amount by which the
        expected load (forecast + July peak shortfall margin) exceeds the cap."""
        st = self.status(now)
        tier = st["tier"]
        s = self.stats
        if tier == "GREEN":
            return {"tool": "propose", "tier": tier, "type": "none", "evidence": st["evidence"]}
        if tier == "YELLOW":
            return {"tool": "propose", "tier": tier, "type": "watch", "actions": self.sop["tiers"]["YELLOW"]["actions_ko"],
                    "evidence": st["evidence"]}
        expected = [x + s["margin_peak"] for x in st["forecast"]]
        red_q = [max(e - self.cap, 0.0) for e in expected]
        hold_units = round(max(red_q), 1)
        hold_energy = round(sum(red_q) / 4, 1)
        out = {"tool": "propose", "tier": tier, "type": "defer" if hold_units > 0 else "precaution",
               "expected_max": round(max(expected), 1), "margin": s["margin_peak"],
               "hold_units": hold_units, "hold_energy": hold_energy, "reduce_by_quarter": [round(x, 1) for x in red_q]}
        out["evidence"] = {**st["evidence"], "expected_max": out["expected_max"], "margin": out["margin"],
                           "hold_units": hold_units, "hold_energy": hold_energy}
        return out

    def verify(self, now, proposal, committed=None, max_delay_hours=None):
        """Energy-conserving check on the EXPECTED profile. Hour t is cut to the
        cap and the held energy is released into later hours' headroom:
          * later hours: day-ahead forecast + margin
          * the next hour has no headroom if it is itself forecast RED (lead 1)
          * headroom already promised to earlier actions today (`committed`,
            {(row, quarter): quarter-units}) is not available again
        Release must finish within max_delay hours and before the deadline hour.
        On success, `allocation` lists the headroom this action uses."""
        if proposal.get("type") != "defer":
            return {"tool": "verify", "applicable": False, "evidence": {}}
        committed = committed if committed is not None else {}
        now = pd.Timestamp(now)
        i = self._row(now)
        d = self.sop["deferral"]
        s = self.stats
        backlog = proposal["hold_energy"] * 4          # in quarter-units
        delay = d["max_delay_hours"] if max_delay_hours is None else max_delay_hours
        last_hour = min(now.hour + delay, d["release_must_finish_by_hour"] - 1)
        released_by, alloc = None, {}
        next_red = self.h.loc[i, "tier1"] == "RED"
        for k, hr in enumerate(range(now.hour + 1, last_hour + 1), start=1):
            j = i + k
            if j not in self.h.index:
                break
            if (k == 1 and next_red) or hr in RISK_HOURS:
                continue   # never release deferred load into a RED hour or the SOP risk window
            exp_q = [float(self.h.loc[j, f"da_{q}"]) + s["margin_dayahead"] for q in Q]
            for qi, e in enumerate(exp_q):
                room = max(self.cap - e - committed.get((j, qi), 0.0), 0.0)
                take = min(room, backlog)
                if take > 0:
                    alloc[(j, qi)] = take
                backlog -= take
                if backlog <= 1e-9 and released_by is None:
                    released_by = f"{hr:02d}:{qi * 15:02d}"
            if backlog <= 1e-9:
                break
        ok = backlog <= 1e-9
        out = {"tool": "verify", "applicable": True, "passed": ok, "released_by": released_by,
               "allocation": alloc if ok else {}, "next_hour_red": bool(next_red),
               "left_over": round(backlog / 4, 1), "max_delay_hours": delay,
               "deadline_hour": d["release_must_finish_by_hour"]}
        out["evidence"] = {"left_over": out["left_over"], "max_delay_hours": delay,
                           "deadline_hour": d["release_must_finish_by_hour"],
                           "released_hour": int(released_by[:2]) if released_by else None,
                           "released_min": int(released_by[3:]) if released_by else None}
        if not ok:
            out["fallback"] = self.sop["fallback_ko"]
        return out

    def outcome(self, now):
        """After hour t has completed (available from t+1): what actually happened."""
        i = self._row(now)
        a = self.df.loc[i, Q].to_numpy(float)
        tier = self.h.loc[i, "tier0"]
        peak = bool(a.max() >= self.cap)
        verdict = ("적중" if peak and tier == "RED" else "주의 단계에서 발생" if peak and tier == "YELLOW"
                   else "미탐지" if peak else "오경보" if tier == "RED" else "정상")
        excess = float(np.clip(a - self.cap, 0, None).sum() / 4)
        return {"tool": "outcome", "actual_max": int(a.max()), "peak": peak, "tier": tier, "verdict": verdict,
                "excess_energy": round(excess, 2),
                "evidence": {"actual_max": int(a.max()), "excess_energy": round(excess, 2), "cap": self.cap,
                             "hour": int(pd.Timestamp(now).hour)}}

    # ------------------------------------------------------------ data quality and unusual load
    def _quarters_until(self, end_row):
        """15-min readings of the last 24 completed hours up to and including end_row."""
        return self.df.loc[max(self.df.index[0], end_row - 23):end_row, Q].to_numpy(float).ravel()

    def data_check(self, now):
        """Checks the hour that has just completed (t-1) at issuance time t.
        Rules and thresholds come from pre-test data: zeros never occur before
        August, operating-level stuck runs never occur, jumps above the pre-test
        99.9th percentile are rare."""
        now = pd.Timestamp(now)
        i = self._row(now) - 1
        s = self.stats
        q = self._quarters_until(i)
        last4 = q[-4:]
        n_zero = int((last4 == 0).sum())
        # stuck: run of identical values at operating level reaching into hour t-1
        run = 1
        for k in range(len(q) - 1, 0, -1):
            if q[k] == q[k - 1]:
                run += 1
            else:
                break
        stuck = run >= s["stuck_min_quarters"] and q[-1] >= s["stuck_min_value"]
        jumps = np.abs(np.diff(q[-5:])) if len(q) >= 5 else np.array([0.0])
        max_jump = float(jumps.max())
        flags = []
        if n_zero:
            flags.append("zero")
        if stuck:
            flags.append("stuck")
        if max_jump > s["jump_p999"]:
            flags.append("jump")
        return {"tool": "data_check", "hour": int((now - pd.Timedelta(hours=1)).hour), "flags": flags,
                "n_zero": n_zero, "stuck_quarters": int(run) if stuck else 0, "max_jump": round(max_jump),
                "evidence": {"hour": int((now - pd.Timedelta(hours=1)).hour), "n_zero": n_zero,
                             "stuck_quarters": int(run) if stuck else 0, "max_jump": round(max_jump),
                             "jump_p999": s["jump_p999"], "stuck_min_quarters": s["stuck_min_quarters"],
                             "stuck_min_value": s["stuck_min_value"], "input_hours": 3}}

    def inputs_affected(self, now, lookback=3):
        """True if any of the last `lookback` completed hours failed the data check
        (the forecaster and detector use up to 3 hours of recent readings)."""
        now = pd.Timestamp(now)
        return any(self.data_check(now - pd.Timedelta(hours=k))["flags"] for k in range(lookback))

    def load_check(self, now):
        """Unusual load in hour t-1: actual hourly mean vs the next-hour forecast
        issued for it, compared with July's 99th percentile of that gap."""
        now = pd.Timestamp(now)
        i = self._row(now) - 1
        if i not in self.h.index:
            return {"tool": "load_check", "available": False, "evidence": {}}
        actual = float(self.df.loc[i, Q].mean())
        fc = float(self.h.loc[i, [f"fc_{q}" for q in Q]].mean())
        gap = actual - fc
        th = self.gap_threshold(pd.Timestamp(self.df.loc[i, "ts"]))
        flag = abs(gap) > th
        return {"tool": "load_check", "available": True, "hour": int(self.df.loc[i, "hour"]),
                "actual_mean": round(actual, 1), "forecast_mean": round(fc, 1), "gap": round(gap, 1),
                "threshold": th, "unusual": bool(flag), "direction": "higher" if gap > 0 else "lower",
                "evidence": {"hour": int(self.df.loc[i, "hour"]), "actual_mean": round(actual, 1),
                             "forecast_mean": round(fc, 1), "gap": round(gap, 1), "gap_abs": round(abs(gap), 1),
                             "threshold": th, "pct": 1, "window_days": self.GAP_WINDOW_DAYS}}
