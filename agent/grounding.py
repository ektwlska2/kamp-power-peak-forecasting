"""Grounding check: every number in a message must come from tool evidence
(or from the fixed SOP constants). Used on template messages as a self-test and
on any LLM-written text before it is shown; failures fall back to the template.
"""
import re

TIME = re.compile(r"(\d{1,2}):(\d{2})")
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?")


def _flatten(ev):
    out = []
    for v in ev.values() if isinstance(ev, dict) else ev:
        if isinstance(v, dict):
            out += _flatten(v)
        elif isinstance(v, (list, tuple)):
            out += _flatten(list(v))
        elif isinstance(v, bool) or v is None:
            continue
        elif isinstance(v, (int, float)):
            out.append(float(v))
    return out


def allowed_values(evidence, sop):
    vals = set(_flatten(evidence))
    d = sop["deferral"]
    # fixed SOP constants that may appear in any message
    vals |= {float(sop["cap"]), float(d["max_delay_hours"]), float(d["release_must_finish_by_hour"]),
             8.0, 11.0, 12.0, 13.0, 14.0, 15.0, 28.0, 100.0, 7.0, 0.0}
    for t in sop["tiers"].values():
        if "response_minutes" in t:
            vals.add(float(t["response_minutes"]))
    return vals


def check(text, evidence, sop):
    """Returns (ok, unsupported_numbers)."""
    vals = allowed_values(evidence, sop)
    bad = []
    stripped = text
    for m in TIME.finditer(text):
        h, mm = float(m.group(1)), float(m.group(2))
        if h not in vals and h != 0:
            bad.append(m.group(0))
        if mm not in (0.0, 15.0, 30.0, 45.0) and mm not in vals:
            bad.append(m.group(0))
    stripped = TIME.sub(" ", stripped)
    for m in NUM.finditer(stripped):
        s = m.group(0)
        x = float(s)
        dec = len(s.split(".")[1]) if "." in s else 0
        tol = 0.5 * 10 ** (-dec) + 1e-9
        if not any(abs(x - v) <= tol for v in vals):
            bad.append(s)
    return (len(bad) == 0, bad)
