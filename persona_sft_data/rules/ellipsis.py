"""말줄임표: ``…`` 개수 상한. 값은 ``최대 N개``."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from persona_sft_data.core.gates import GateSettings, Verdict
from persona_sft_data.core.persona import Persona
from persona_sft_data.core.registry import RULES
from persona_sft_data.rules.base import assistant_texts, bad_value

_VALUE = re.compile(r"^최대\s*(\d+)\s*개$")


@dataclass(frozen=True)
class EllipsisRule:
    limit: int
    name: str = "ellipsis"

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None:
        if any(t.count("…") > self.limit for t in assistant_texts(turns)):
            verdict.fail("multiple_ellipsis")


@RULES.register("ellipsis", origin="builtin")
class EllipsisFactory:
    name = "ellipsis"
    constraint_key = "말줄임표"

    def build(self, persona: Persona, value: str, settings: GateSettings):
        m = _VALUE.match(value.strip())
        if not m:
            raise bad_value(self.constraint_key, value, "최대 N개")
        return EllipsisRule(int(m.group(1)))
