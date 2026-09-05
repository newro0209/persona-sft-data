"""마크다운: 제목·강조·목록·코드·링크."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from persona_sft_data.core.gates import GateSettings, Verdict
from persona_sft_data.core.persona import Persona
from persona_sft_data.core.registry import RULES
from persona_sft_data.rules.base import assistant_texts, bad_value

MARKDOWN = re.compile(r"(^#{1,6}\s|\*\*|^\s*[-*+]\s|^\s*\d+\.\s|```|\[.+\]\(.+\))", re.MULTILINE)


@dataclass(frozen=True)
class MarkdownRule:
    name: str = "markdown"

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None:
        if any(MARKDOWN.search(t) for t in assistant_texts(turns)):
            verdict.fail("markdown")


@RULES.register("markdown", origin="builtin")
class MarkdownFactory:
    name = "markdown"
    constraint_key = "마크다운"

    def build(self, persona: Persona, value: str, settings: GateSettings):
        if value == "금지":
            return MarkdownRule()
        if value == "허용":
            return None
        raise bad_value(self.constraint_key, value, "금지 · 허용")
