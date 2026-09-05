"""AI 자칭: 자신을 AI·모델·프로그램이라고 말하거나 내부 구현을 언급하면 거절."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from persona_sft_data.core.gates import GateSettings, Verdict
from persona_sft_data.core.persona import Persona
from persona_sft_data.core.registry import RULES
from persona_sft_data.rules.base import assistant_texts, bad_value

AI_WORDS = ("AI", "A.I.", "인공지능", "언어모델", "언어 모델", "챗봇", "챗 봇", "프로그램",
            "컴퓨터", "모델이야", "시스템 프롬프트", "학습 데이터", "토큰")


@dataclass(frozen=True)
class AiClaimRule:
    name: str = "ai_claim"

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None:
        if any(w in t for t in assistant_texts(turns) for w in AI_WORDS):
            verdict.fail("claims_to_be_ai")


@RULES.register("ai_claim", origin="builtin")
class AiClaimFactory:
    name = "ai_claim"
    constraint_key = "AI 자칭"

    def build(self, persona: Persona, value: str, settings: GateSettings):
        if value == "금지":
            return AiClaimRule()
        raise bad_value(self.constraint_key, value, "금지")
