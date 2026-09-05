"""역할 표기: 발화 어디든 ``U:`` ``A:`` ``사용자:`` 같은 표기가 있으면 거절. 끝에 붙은 것이 실제로 코퍼스에 들어간 적이 있다."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from persona_sft_data.core.gates import GateSettings, Verdict
from persona_sft_data.core.persona import Persona
from persona_sft_data.core.registry import RULES
from persona_sft_data.rules.base import assistant_texts, bad_value

ROLE_LABEL_ANYWHERE = re.compile(r"(^|\s)(U|A|P|사용자|유저|user|assistant|pet)\s*[:：]", re.IGNORECASE)


@dataclass(frozen=True)
class RoleLabelRule:
    name: str = "role_label"

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None:
        if any(ROLE_LABEL_ANYWHERE.search(t) for t in assistant_texts(turns)):
            verdict.fail("role_label_in_text")


@RULES.register("role_label", origin="builtin")
class RoleLabelFactory:
    name = "role_label"
    constraint_key = "역할 표기"

    def build(self, persona: Persona, value: str, settings: GateSettings):
        if value == "금지":
            return RoleLabelRule()
        raise bad_value(self.constraint_key, value, "금지")
