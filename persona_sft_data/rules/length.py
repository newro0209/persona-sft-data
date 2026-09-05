"""발화 길이: ``N~M글자`` 또는 ``N~M문장``."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from persona_sft_data.core.gates import GateSettings, Verdict
from persona_sft_data.core.persona import Persona
from persona_sft_data.core.registry import RULES
from persona_sft_data.rules.base import assistant_texts, bad_value

_VALUE = re.compile(r"^(\d+)\s*~\s*(\d+)\s*(글자|문장)$")
_SENTENCE_END = re.compile(r"[.?!…]+")


def sentence_count(text: str) -> int:
    return len([s for s in _SENTENCE_END.split(text) if s.strip()])


@dataclass(frozen=True)
class LengthRule:
    lo: int
    hi: int
    unit: str
    name: str = "length"

    def measure(self, text: str) -> int:
        return len(text) if self.unit == "글자" else sentence_count(text)

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None:
        for text in assistant_texts(turns):
            n = self.measure(text)
            if n < self.lo:
                verdict.fail("assistant_too_short")
            if n > self.hi:
                verdict.fail("assistant_too_long")


@RULES.register("length", origin="builtin")
class LengthFactory:
    name = "length"
    constraint_key = "발화 길이"

    def build(self, persona: Persona, value: str, settings: GateSettings):
        m = _VALUE.match(value.strip())
        if not m or int(m.group(1)) > int(m.group(2)):
            raise bad_value(self.constraint_key, value, "N~M글자 · N~M문장")
        return LengthRule(int(m.group(1)), int(m.group(2)), m.group(3))
