"""반복: 같은 단어를 연달아, 또는 같은 구절을 붙여 두 번."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from persona_sft_data.core.gates import GateSettings, Verdict
from persona_sft_data.core.persona import Persona
from persona_sft_data.core.registry import RULES
from persona_sft_data.rules.base import assistant_texts, bad_value


def repeats_phrase(text: str, *, min_len: int = 3) -> bool:
    words = text.split()
    for a, b in zip(words, words[1:]):
        if a == b and len(a) >= 2:
            return True
    for n in range(min_len, max(min_len, len(text) // 2) + 1):
        for i in range(len(text) - 2 * n + 1):
            chunk = text[i:i + n]
            if chunk.strip() and chunk == text[i + n:i + 2 * n]:
                return True
    return False


@dataclass(frozen=True)
class RepeatRule:
    name: str = "repeat"

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None:
        if any(repeats_phrase(t) for t in assistant_texts(turns)):
            verdict.fail("repeated_phrase")


@RULES.register("repeat", origin="builtin")
class RepeatFactory:
    name = "repeat"
    constraint_key = "반복"

    def build(self, persona: Persona, value: str, settings: GateSettings):
        if value == "금지":
            return RepeatRule()
        raise bad_value(self.constraint_key, value, "금지")
