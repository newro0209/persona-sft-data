"""이모지: 기호 범주(So·Sk)와 이모지 블록의 문자를 거절한다. 문장부호와 ``…``은 허용."""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from persona_sft_data.core.gates import GateSettings, Verdict
from persona_sft_data.core.persona import Persona
from persona_sft_data.core.registry import RULES
from persona_sft_data.rules.base import assistant_texts, bad_value

ALLOWED_PUNCT = set(" .,!?~…'\"()·-\n")


def has_emoji(text: str) -> bool:
    for ch in text:
        if ch in ALLOWED_PUNCT:
            continue
        if unicodedata.category(ch) in {"So", "Sk"}:
            return True
        if 0x1F000 <= ord(ch) <= 0x1FAFF or 0x2600 <= ord(ch) <= 0x27BF:
            return True
    return False


@dataclass(frozen=True)
class EmojiRule:
    name: str = "emoji"

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None:
        if any(has_emoji(t) for t in assistant_texts(turns)):
            verdict.fail("emoji")


@RULES.register("emoji", origin="builtin")
class EmojiFactory:
    name = "emoji"
    constraint_key = "이모지"

    def build(self, persona: Persona, value: str, settings: GateSettings):
        if value == "금지":
            return EmojiRule()
        if value == "허용":
            return None
        raise bad_value(self.constraint_key, value, "금지 · 허용")
