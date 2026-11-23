"""Rule-based peak agent --> no LLM or API key required. (demo)

Follows the SOP on a schedule and answers operator questions by routing
them to the deterministic tools, writing messages from templates.
Every message records the numbers it was built from ('evidence') and passes
the grounding check before it is shown.
"""
from dataclasses import dataclass, field

import pandas as pd

from src.data import load, with_copy_status, copy_map
from src.state import daily_state

from .grounding import check
from .tools import PeakTools

TIER_KO = {"RED": "경고", "YELLOW": "주의", "GREEN": "정상"}


@dataclass
class Message:
    issued: str              # timestamp when the message is issued
    kind: str                # briefing | pre_alert | watch | action | outcome | answer
    tier: str
    text: str
    evidence: dict = field(default_factory=dict)
    check: dict = field(default_factory=dict)
    source: str = "template"
    grounded: bool = True
    unsupported: list = field(default_factory=list)


def prepare_df():
    df = with_copy_status(load())
    st = daily_state(df)
    df["state"] = df.date.map(st)
    df["group"] = df.date.map(copy_map(df).group)
    return df


def _hh(h):
    return f"{int(h):02d}:00"


class RuleAgent:
    def __init__(self, tools: PeakTools):
        self.t = tools
        self.sop = tools.sop
        self.ledger = {}          # headroom promised to earlier deferrals today: {(row, quarter): units}

    @classmethod
    def create(cls):
        return cls(PeakTools.load(prepare_df()))

    def _msg(self, issued, kind, tier, text, evidence, chk=None, source="template"):
        ok, bad = check(text, evidence, self.sop)
        return Message(str(issued), kind, tier, text, evidence, chk or {}, source, ok, bad)

    # ------------------------------------------------------------ scheduled messages
    def day_briefing(self, date):
        b = self.t.day_briefing(date)
        d = pd.Timestamp(date)
        head = f"[일일 사전 브리핑] {d.month}월 {d.day}일"
        if not b["running"]:
            return self._msg(d, "briefing", "GREEN", f"{head}은 휴지일(가동 없음)입니다. 피크 경보 대상이 아닙니다.", b["evidence"])
        lines = [f"{head} 가동일입니다."]
        acts = self.sop["horizons"]["day_ahead"]["actions_ko"]
        if b["restart_day"]:
            lines.append(f"휴지 후 첫 가동일입니다. 과거 재가동일 피크 발생률은 {b['rate_restart']}%로 일반 가동일 {b['rate_running']}%보다 높습니다.")
            lines.append("권고: " + acts["restart_day"])
        if b["hot_day"]:
            lines.append(f"예상 최고기온 {b['temp_max']} °C. 더운 날(28 °C 이상) 피크 발생률 {b['rate_hot']}%, 그 외 {b['rate_not_hot']}%입니다.")
            lines.append("권고: " + acts["hot_day"])
        if b["watch_hours"]:
            hrs = ", ".join(f"{h}시" for h in b["watch_hours"])
            lines.append(f"하루 전 예측 최대 {b['dayahead_max']}에 오차 여유 {b['margin']}를 더하면 상한 {self.t.cap}에 닿을 수 있는 시간: {hrs}.")
            lines.append("권고: " + acts["high_dayahead_max"])
        else:
            lines.append(f"하루 전 예측 최대 {b['dayahead_max']}. 오차 여유 {b['margin']}를 더해도 상한 {self.t.cap} 아래로 예상됩니다.")
        tier = "YELLOW" if (b["restart_day"] or b["hot_day"] or b["watch_hours"]) else "GREEN"
        return self._msg(d, "briefing", tier, "\n".join(lines), b["evidence"])

    def hour_messages(self, now):
        """Alert messages issued at the start of hour `now` (running days)."""
        now = pd.Timestamp(now)
        out = []
        st = self.t.status(now)
        caution = ""
        if self.t.inputs_affected(now):
            caution = ("\n주의: 최근 3시간 안에 측정 이상이 있어 이번 시간 예측과 경보를 신뢰하기 어렵습니다. "
                       "현장 계측값을 먼저 확인하십시오.")
        # pre-alert for the next hour (lead 1): same day, and only when the level escalates
        rank = {"GREEN": 0, "YELLOW": 1, "RED": 2}
        if (st.get("next_tier") in ("RED", "YELLOW") and st["next_hour"] > now.hour
                and rank[st["next_tier"]] > rank[st["tier"]]):
            act = self.sop["horizons"]["hour_ahead"]["actions_ko"][st["next_tier"]]
            txt = (f"[사전 경보 {_hh(st['next_hour'])} 대상] 다음 시간 {TIER_KO[st['next_tier']]} 예상 "
                   f"(피크 확률 {st['evidence']['next_p_peak_pct']}%). 조치: {act}")
            out.append(self._msg(now, "pre_alert", st["next_tier"], txt, st["evidence"]))
        if st["tier"] == "YELLOW" and st.get("yellow_reason") == "interval":
            txt = (f"[주의 {_hh(now.hour)}] 90% 예측 구간의 위쪽 끝이 {st['pi_upper_max']}로 기준 {self.t.cap} 이상입니다 "
                   f"(예측 최대 {st['forecast_max']} + 구간 폭 {st['pi_q']}). 위험 시간대 규칙과 별개로 피크 가능성이 있습니다. "
                   "조치: " + " / ".join(self.sop["tiers"]["YELLOW"]["actions_ko"]) + caution)
            out.append(self._msg(now, "watch", "YELLOW", txt, {**st["evidence"], "input_hours": 3}))
        elif st["tier"] == "YELLOW":
            cr = self.t.condition_rate(now)
            rate = f"과거 같은 조건 피크율 {cr['rate']}% ({cr['hours']}시간)" if cr["rate"] is not None else "과거 같은 조건 사례 없음"
            txt = (f"[주의 {_hh(now.hour)}] 위험 시간대 규칙 발동 (직전 시간 평균 {cr['prev_mean']}). {rate}. "
                   f"조치: " + " / ".join(self.sop["tiers"]["YELLOW"]["actions_ko"]) + caution)
            out.append(self._msg(now, "watch", "YELLOW", txt, {**st["evidence"], **cr["evidence"], "input_hours": 3}))
        elif st["tier"] == "RED":
            m = self.action_message(now)
            if caution:
                m.text += caution
                m.evidence["input_hours"] = 3
                m.grounded, m.unsupported = check(m.text, m.evidence, self.sop)
            out.append(m)
        return out

    def monitor_messages(self, now):
        """Data-quality and unusual-load messages about the hour that just ended
        (issued at `now`, every day). Data faults are reported when they start
        and when they clear, not every hour."""
        now = pd.Timestamp(now)
        out = []
        dc = self.t.data_check(now)
        prev = self.t.data_check(now - pd.Timedelta(hours=1))
        names = {"zero": f"0값 {dc['n_zero']}개", "stuck": f"같은 값 {dc['stuck_quarters']}구간 연속",
                 "jump": f"15분 사이 급변 {dc['max_jump']} (기준 {dc['evidence']['jump_p999']})"}
        if dc["flags"] and not prev["flags"]:
            what = ", ".join(names[f] for f in dc["flags"])
            txt = (f"[데이터 점검 {_hh(dc['hour'])}] 측정 이상 의심: {what}. "
                   f"계측기와 통신 상태를 확인하십시오. 이후 {dc['evidence']['input_hours']}시간 동안 예측과 경보의 신뢰도가 낮습니다.")
            m = self._msg(now, "data", "YELLOW", txt, dc["evidence"])
            m.check = {"event": "start", "flags": dc["flags"]}
            out.append(m)
        elif prev["flags"] and not dc["flags"]:
            txt = f"[데이터 점검 {_hh(dc['hour'])}] 측정값이 정상 범위로 돌아왔습니다."
            m = self._msg(now, "data", "GREEN", txt, dc["evidence"])
            m.check = {"event": "clear"}
            out.append(m)
        lc = self.t.load_check(now)
        # skip when the forecast for hour t-1 was itself built from faulty readings
        forecast_inputs_bad = self.t.inputs_affected(now - pd.Timedelta(hours=1)) or bool(dc["flags"])
        if lc.get("available") and lc["unusual"] and not forecast_inputs_bad:
            kind = "계획 외 가동 또는 설비 추가 기동" if lc["direction"] == "higher" else "설비 정지 또는 생산 중단"
            txt = (f"[이상 부하 {_hh(lc['hour'])}] 실제 평균 {lc['actual_mean']}, 예측 {lc['forecast_mean']} "
                   f"(차이 {lc['gap']}, 최근 {lc['evidence']['window_days']}일 오차 상위 {lc['evidence']['pct']}% 기준 {lc['threshold']} 초과). "
                   f"{kind} 여부를 확인하십시오.")
            m = self._msg(now, "unusual", "YELLOW", txt, lc["evidence"])
            m.check = {"gap": lc["gap"], "direction": lc["direction"]}
            out.append(m)
        return out

    def action_message(self, now):
        now = pd.Timestamp(now)
        p = self.t.propose(now)
        ev = dict(p["evidence"])
        head = f"[경고 {_hh(now.hour)}] 피크 가능성 높음 (피크 확률 {ev['p_peak_pct']}%)."
        if p["type"] == "precaution":
            txt = (f"{head} 예측 최대 {ev['forecast_max']}에 여유 {p['margin']}를 더한 {p['expected_max']}가 상한 {self.t.cap} 아래이므로 "
                   "이연 없이 새 재량 부하 기동만 보류하십시오.")
            return self._msg(now, "action", "RED", txt, ev)
        v = self.t.verify(now, p, self.ledger)
        for key, units in v.get("allocation", {}).items():
            self.ledger[key] = self.ledger.get(key, 0.0) + units
        ev.update(v["evidence"])
        txt = (f"{head} 예상 최대 {p['expected_max']} (예측 {ev['forecast_max']} + 과거 피크 과소예측 여유 {p['margin']}). "
               f"제안: 이번 시간 부하를 최대 {p['hold_units']} 낮추고 이연 에너지 {p['hold_energy']}를 이후 시간에 재투입.")
        if v["passed"]:
            txt += (f"\n검증 통과: {v['released_by']}까지 재투입 완료 예상 "
                    f"(최대 {v['max_delay_hours']}시간, {v['deadline_hour']}시 이전 조건 충족"
                    + (", 다음 시간도 경고라 그 다음 시간부터 재투입" if v["next_hour_red"] else "") + ").")
        else:
            txt += (f"\n검증 실패: {v['max_delay_hours']}시간 안에 재투입할 여유가 부족합니다 (남는 이연량 {v['left_over']}"
                    + (", 다음 시간도 경고" if v["next_hour_red"] else "") + "). "
                    f"이연 대신 대체 조치: {v['fallback'][1]} / {v['fallback'][2]}")
        return self._msg(now, "action", "RED", txt, ev, chk={"passed": v["passed"], "left_over": v["left_over"],
                                                               "hold_units": p["hold_units"], "hold_energy": p["hold_energy"]})

    def outcome_message(self, now):
        """Issued at now + 1h, when hour `now` has completed."""
        now = pd.Timestamp(now)
        o = self.t.outcome(now)
        txt = f"[결과 기록 {_hh(now.hour)}] 실제 최대 {o['actual_max']}, 경보 {TIER_KO[o['tier']]} → {o['verdict']}"
        if o["peak"]:
            txt += f" (상한 초과 에너지 {o['excess_energy']})"
        m = self._msg(now + pd.Timedelta(hours=1), "outcome", o["tier"], txt, o["evidence"])
        m.check = {"verdict": o["verdict"], "peak": o["peak"], "excess_energy": o["excess_energy"]}
        return m

    def run_day(self, date):
        """All messages for one day, in issue order."""
        date = pd.Timestamp(date).normalize()
        self.ledger = {}
        msgs = [self.day_briefing(date)]
        running = self.t.day_briefing(date)["running"]
        for h in range(24):
            now = date + pd.Timedelta(hours=h)
            msgs += self.monitor_messages(now)          # data checks run every day, idle days included
            if not running:
                continue
            msgs += self.hour_messages(now)
            o = self.t.outcome(now)
            if o["peak"] or o["tier"] != "GREEN":
                msgs.append(self.outcome_message(now))
        return sorted(msgs, key=lambda m: m.issued)

    # ------------------------------------------------------------ questions
    INTENTS = [
        ("data", ("측정", "데이터", "센서", "계측", "이상 부하", "data", "sensor")),
        ("explain", ("왜", "이유", "why", "근거")),
        ("action", ("얼마", "조치", "무엇을", "뭘 해", "어떻게", "what should", "action", "how much")),
        ("briefing", ("브리핑", "내일", "오늘 하루", "briefing", "tomorrow")),
        ("analog", ("비슷", "과거", "similar", "past")),
        ("status", ("다음", "지금", "상태", "next", "now", "status")),
    ]

    def route(self, question):
        q = question.lower()
        for intent, keys in self.INTENTS:
            if any(k in q for k in keys):
                return intent
        return "help"

    def tool_results(self, question, now):
        """The tool outputs an answer to this question is allowed to use."""
        intent = self.route(question)
        now = pd.Timestamp(now)
        if intent == "data":
            return intent, [self.t.data_check(now), self.t.load_check(now)]
        if intent == "explain":
            return intent, [self.t.explain(now)]
        if intent == "action":
            p = self.t.propose(now)
            return intent, [p, self.t.verify(now, p)]
        if intent == "briefing":
            return intent, [self.t.day_briefing(now.normalize())]
        if intent == "analog":
            return intent, [self.t.analog_days(now)]
        if intent == "status":
            return intent, [self.t.status(now)]
        return intent, []

    def answer(self, question, now):
        intent, res = self.tool_results(question, now)
        now = pd.Timestamp(now)
        if intent == "data":
            dc, lc = res
            ev = {**dc["evidence"], **lc.get("evidence", {})}
            if dc["flags"]:
                parts = {"zero": f"0값 {dc['n_zero']}개", "stuck": f"같은 값 {dc['stuck_quarters']}구간 연속",
                         "jump": f"15분 사이 급변 {dc['max_jump']}"}
                txt = f"{_hh(dc['hour'])} 측정값 이상 의심: " + ", ".join(parts[f] for f in dc["flags"]) + ". 계측기를 확인하십시오."
                return self._msg(now, "answer", "YELLOW", txt, ev)
            txt = f"{_hh(dc['hour'])} 측정값은 정상입니다 (0값, 멈춤, 급변 없음)."
            fc_bad = self.t.inputs_affected(now - pd.Timedelta(hours=1))
            if self.t.inputs_affected(now) or fc_bad:
                txt += f" 다만 최근 {dc['evidence']['input_hours']}시간 안에 측정 이상이 있었습니다."
            if lc.get("available"):
                txt += f" 실제 평균 {lc['actual_mean']}, 예측 {lc['forecast_mean']} (차이 {lc['gap']}): "
                if lc["unusual"] and fc_bad:
                    txt += "차이가 크지만, 이 시간의 예측이 이상 측정값을 입력으로 썼기 때문일 수 있습니다."
                elif lc["unusual"]:
                    txt += "평소보다 큰 차이입니다. 계획 외 가동이나 정지 여부를 확인하십시오."
                else:
                    txt += f"평소 범위(기준 {lc['threshold']}) 안입니다."
            return self._msg(now, "answer", "GREEN", txt, ev)
        if intent == "explain":
            e = res[0]
            fs = []
            for f in e["factors"]:
                val = ("발동" if f["value"] >= 1 else "미발동") if f["feature"] == "risk_window" else f["value"]
                fs.append(f"{f['name_ko']} = {val}")
            st, cr, an = e["status"], e["condition"], e["analog"]
            lines = [f"{_hh(now.hour)} 판단: {TIER_KO[st['tier']]} (피크 확률 {st['evidence']['p_peak_pct']}%).",
                     "위험을 가장 크게 올린 요인: " + "; ".join(fs) + "."]
            if cr["rate"] is not None:
                lines.append(f"과거(6–7월) 같은 시간대·직전 부하 조건에서 피크율 {cr['rate']}% ({cr['hours']}시간).")
            if an["available"]:
                lines.append(f"오늘과 비슷한 과거 가동일 {an['k']}일 중 {an['peaked_this_hour']}일이 이 시각에 피크, "
                             f"{an['peaked_rest_of_day']}일이 오늘 남은 시간 중 피크.")
            return self._msg(now, "answer", st["tier"], "\n".join(lines), e["evidence"])
        if intent == "action":
            st = self.t.status(now)
            if st["tier"] == "RED":
                m = self.action_message(now)
                m.kind = "answer"
                return m
            if st["tier"] == "YELLOW":
                txt = f"{_hh(now.hour)} 주의 단계입니다. 조치: " + " / ".join(self.sop["tiers"]["YELLOW"]["actions_ko"])
                return self._msg(now, "answer", "YELLOW", txt, st["evidence"])
            return self._msg(now, "answer", "GREEN", f"{_hh(now.hour)} 정상입니다. 별도 조치가 필요 없습니다.", st["evidence"])
        if intent == "briefing":
            m = self.day_briefing(now.normalize())
            m.kind = "answer"
            return m
        if intent == "analog":
            an = res[0]
            if not an["available"]:
                return self._msg(now, "answer", "GREEN", "오늘 데이터가 아직 3시간 미만이라 비슷한 날을 찾을 수 없습니다.",
                                 {"min_hours": 3})
            txt = (f"오늘과 비슷한 과거 가동일 {an['k']}일 (" + ", ".join(an["dates"]) + f"): {an['peaked_this_hour']}일이 "
                   f"{_hh(now.hour)}에 피크, {an['peaked_rest_of_day']}일이 남은 시간 중 피크. 복제된 날짜는 한 번만 셉니다.")
            ev = dict(an["evidence"])
            ev["dates"] = [float(x) for d in an["dates"] for x in d.split("-")]
            return self._msg(now, "answer", "GREEN", txt, ev)
        if intent == "status":
            st = res[0]
            txt = (f"{_hh(st['hour'])} 예측 최대 {st['forecast_max']}, 경보 {TIER_KO[st['tier']]} (피크 확률 {st['evidence']['p_peak_pct']}%).")
            if "next_tier" in st:
                txt += f" 다음 시간 {_hh(st['next_hour'])}: {TIER_KO[st['next_tier']]} (피크 확률 {st['evidence']['next_p_peak_pct']}%)."
            return self._msg(now, "answer", st["tier"], txt, st["evidence"])
        return self._msg(now, "answer", "GREEN",
                         "질문 예시: '왜 경고야?', '지금 뭘 해야 해?', '오늘 브리핑', '비슷한 과거 날은?', '다음 시간 상태는?'", {})
