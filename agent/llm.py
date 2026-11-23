"""Optional LLM layer (off by default; nothing here needs an API key).

Pipeline for an operator question:
  1. the rule router picks the tools and runs them (deterministic numbers)
  2. an LLM client writes the answer from the tool results only
  3. the grounding check verifies every number in the LLM text
  4. if any number is unsupported, the template answer is shown instead and the
     rejected numbers are logged

`LLMClient` is the only thing a real model has to implement. A production
client (for example Claude with tool use) would receive the same tool results
as JSON and return text; it is not included because this project runs without
an API key. `ScriptedClient` replays fixed texts so the safeguard can be tested.
"""
import json
from typing import Protocol

from .agent import Message, RuleAgent
from .grounding import check

SYSTEM_PROMPT_KO = (
    "당신은 공장 전력 피크 관리 보조원입니다. 아래 도구 결과(JSON)에 있는 숫자만 사용하여 한국어로 "
    "짧게 답하십시오. 도구 결과에 없는 숫자, 추정, 절감 금액은 쓰지 마십시오. 조치는 제안만 하고 "
    "최종 결정은 작업자에게 맡기십시오."
)


class LLMClient(Protocol):
    def write(self, system: str, question: str, tool_results_json: str) -> str: ...


class ScriptedClient:
    """Returns pre-written texts in order (for tests and offline demos)."""

    def __init__(self, texts):
        self.texts = list(texts)

    def write(self, system, question, tool_results_json):
        return self.texts.pop(0)


def _evidence(results):
    ev = {}
    for i, r in enumerate(results):
        ev[f"r{i}"] = r.get("evidence", {})
    return ev


def llm_answer(agent: RuleAgent, question: str, now, client: LLMClient) -> Message:
    intent, results = agent.tool_results(question, now)
    payload = json.dumps(results, ensure_ascii=False, default=str)
    text = client.write(SYSTEM_PROMPT_KO, question, payload)
    ev = _evidence(results)
    ok, bad = check(text, ev, agent.sop)
    if ok:
        return Message(str(now), "answer", results[0].get("tier", "GREEN") if results else "GREEN",
                       text, ev, {"intent": intent}, source="llm", grounded=True)
    fallback = agent.answer(question, now)
    fallback.source = "template (LLM answer rejected)"
    fallback.check = {"intent": intent, "rejected_llm_text": text, "unsupported_numbers": bad}
    return fallback
