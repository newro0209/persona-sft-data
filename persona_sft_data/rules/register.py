"""말투: 반말 페르소나는 존댓말 종결을, 존댓말 페르소나는 반말 종결을 거절한다."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from persona_sft_data.core.gates import GateSettings, Verdict
from persona_sft_data.core.persona import Persona
from persona_sft_data.core.registry import RULES
from persona_sft_data.rules.base import assistant_texts, bad_value

HONORIFIC = re.compile(r"(요[.!?~…]?$|요\s|습니다|입니다|세요|십시오|하십|드립니다|드려요|예요|이에요)")
# 존댓말 페르소나: 문장 끝(문장부호 제외)이 이 종결 중 하나여야 한다.
HONORIFIC_END = re.compile(r"(요|니다|십시오|죠|까)[.!?~…]*$")


@dataclass(frozen=True)
class InformalRule:
    name: str = "register"

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None:
        for text in assistant_texts(turns):
            if HONORIFIC.search(text):
                verdict.fail("honorific")


@dataclass(frozen=True)
class HonorificRule:
    name: str = "register"

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None:
        for text in assistant_texts(turns):
            if not HONORIFIC_END.search(text.strip()):
                verdict.fail("informal_ending")


@RULES.register("register", origin="builtin")
class RegisterFactory:
    name = "register"
    constraint_key = "말투"

    def build(self, persona: Persona, value: str, settings: GateSettings):
        if value == "반말":
            return InformalRule()
        if value == "존댓말":
            return HonorificRule()
        if value in ("서술체", "자유"):
            return None
        raise bad_value(self.constraint_key, value, "반말 · 존댓말 · 서술체 · 자유")
