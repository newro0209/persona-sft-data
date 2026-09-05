"""게이트: 구조 규칙 하나와 제약 표에서 만든 규칙 체인.

규칙은 코드에 기본값이 없다. 페르소나 문서 ``## 제약`` 표에 행이 있으면 그 값으로
규칙이 만들어지고, 없으면 꺼진 것이다. 그래서 존댓말을 쓰는 NPC와 반말을 쓰는 펫이
같은 코드로 서로 다른 검열을 받는다.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from persona_sft_data.core.persona import Persona, PersonaError
from persona_sft_data.core.registry import RULES


@dataclass
class Verdict:
    ok: bool = True
    reasons: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> "Verdict":
        self.ok = False
        if reason not in self.reasons:
            self.reasons.append(reason)
        return self


@dataclass(frozen=True)
class GateSettings:
    """문서가 말하지 않는 것만 설정이다."""

    min_turns: int = 2
    max_turns: int = 16


@dataclass(frozen=True)
class Gate:
    rules: tuple[Any, ...]

    def check(self, record: Mapping[str, Any]) -> Verdict:
        verdict = Verdict()
        turns = record.get("turns") or []
        for rule in self.rules:
            rule.check(turns, verdict)
            if not turns:
                break
        return verdict


def build_gate(persona: Persona, settings: GateSettings) -> Gate:
    """구조 규칙 + 제약 표의 행마다 규칙 하나. 모르는 키는 문서 오류다."""
    from persona_sft_data.rules.structure import StructureRule

    factories = {f.constraint_key: f for f in RULES.items().values()}
    rules: list[Any] = [StructureRule(settings.min_turns, settings.max_turns)]
    for key, value in persona.constraints.items():
        if key not in factories:
            raise PersonaError(
                f"'## 제약'의 규칙 키 {key!r}를 아는 규칙 플러그인이 없다 (아는 키: {sorted(factories)})"
            )
        rule = factories[key].build(persona, value, settings)
        if rule is not None:
            rules.append(rule)
    return Gate(tuple(rules))
