"""3인칭 자칭: ``<이름>이도 잘 잤어``처럼 자기 이름을 주어로 쓰면 거절. 이름은 문서에서 온다."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from persona_sft_data.core.gates import GateSettings, Verdict
from persona_sft_data.core.persona import Persona
from persona_sft_data.core.registry import RULES
from persona_sft_data.rules.base import assistant_texts, bad_value


@dataclass(frozen=True)
class ThirdPersonRule:
    pattern: re.Pattern
    name: str = "third_person_self"

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None:
        if any(self.pattern.search(t) for t in assistant_texts(turns)):
            verdict.fail("third_person_self")


@RULES.register("third_person_self", origin="builtin")
class ThirdPersonFactory:
    name = "third_person_self"
    constraint_key = "3인칭 자칭"

    def build(self, persona: Persona, value: str, settings: GateSettings):
        if value != "금지":
            raise bad_value(self.constraint_key, value, "금지")
        return ThirdPersonRule(re.compile(rf"^{re.escape(persona.name)}(이|이가|은|는|도|이도)\b"))
