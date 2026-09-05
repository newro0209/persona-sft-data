"""이름 어미: 이름의 한 음절을 어미처럼 단 토큰(기달몽, 놀랐몽)을 거절. 이름 자체와 이름으로 시작하는 낱말은 둔다."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from persona_sft_data.core.gates import GateSettings, Verdict
from persona_sft_data.core.persona import Persona
from persona_sft_data.core.registry import RULES
from persona_sft_data.rules.base import assistant_texts, bad_value


@dataclass(frozen=True)
class NameSuffixRule:
    persona_name: str
    name: str = "name_suffix"

    def _hit(self, text: str) -> bool:
        syllables = set(self.persona_name)
        for token in re.findall(r"[가-힣]+", text):
            if len(token) < 2 or token[-1] not in syllables:
                continue
            if token == self.persona_name or token.startswith(self.persona_name):
                continue
            return True
        return False

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None:
        if any(self._hit(t) for t in assistant_texts(turns)):
            verdict.fail("name_suffix_babytalk")


@RULES.register("name_suffix", origin="builtin")
class NameSuffixFactory:
    name = "name_suffix"
    constraint_key = "이름 어미"

    def build(self, persona: Persona, value: str, settings: GateSettings):
        if value != "금지":
            raise bad_value(self.constraint_key, value, "금지")
        return NameSuffixRule(persona.name)
