"""항상 켜지는 구조 규칙. 플러그인이 아니다."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from persona_sft_data.core.gates import Verdict


@dataclass(frozen=True)
class StructureRule:
    min_turns: int
    max_turns: int
    name: str = "structure"

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None:
        if not turns:
            verdict.fail("empty")
            return
        if len(turns) < self.min_turns:
            verdict.fail("too_few_turns")
        if len(turns) > self.max_turns:
            verdict.fail("too_many_turns")
        if turns[0].get("role") != "user":
            verdict.fail("does_not_start_with_user")
        if turns[-1].get("role") != "assistant":
            verdict.fail("does_not_end_with_assistant")
        for a, b in zip(turns, turns[1:]):
            if a.get("role") == b.get("role"):
                verdict.fail("roles_not_alternating")
                break
        if any(not str(t.get("text", "")).strip() for t in turns):
            verdict.fail("utterance_empty")
