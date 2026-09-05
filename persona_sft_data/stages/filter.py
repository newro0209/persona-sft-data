"""filter: raw 세션 파일마다 한 번씩. 러너가 레코드별 게이트를 다시 걸고, 이 단계는
파일 전체를 봐야 하는 것만 한다 — 같은 assistant 발화의 과다 반복. 교사는 한 시드의
변주 여럿에 같은 답을 돌려주기 쉽고, 한 문장이 천 개의 질문에 답하는 코퍼스는
모델에게 늘 그 문장을 말하라고 가르친다.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from persona_sft_data.core.registry import STAGES
from persona_sft_data.core.runner import StageContext, reject_record


@dataclass(frozen=True)
class FilterSettings:
    max_identical_assistant_turns: int = 20
    min_turns: int = 2
    max_turns: int = 16


@STAGES.register("filter", origin="builtin")
class FilterStage:
    config_name = "filter"
    mode = "records"
    record_kind = "session"
    produces = "filtered"
    settings_type = FilterSettings

    def __init__(self, source: str = "filter") -> None:
        self.name = source  # 인스턴스 이름 = 읽을 raw 파일 = 쓸 filtered 파일

    def requires(self, config: Any) -> tuple[str, ...]:
        return config.session_stages()

    def instances(self, config: Any) -> list["FilterStage"]:
        names = [n for n in config.session_stages() if config.raw(n).exists()]
        if not names:
            raise FileNotFoundError(
                f"filter가 읽을 raw 세션 파일이 없다. 먼저 {config.session_stages()} 중 하나를 돌려라."
            )
        return [FilterStage(n) for n in names]

    def preflight(self, ctx: StageContext) -> None:
        return None

    def run(self, ctx: StageContext) -> Iterator[dict[str, Any]]:
        limit = int(ctx.settings.max_identical_assistant_turns)
        counts: dict[str, int] = {}
        dropped = 0
        for record in ctx.read(self.name):
            overused = False
            for turn in record.get("turns") or []:
                if turn.get("role") != "assistant":
                    continue
                text = turn.get("text", "")
                counts[text] = counts.get(text, 0) + 1
                if counts[text] > limit:
                    overused = True
            if overused:
                # 거절 레코드도 파일에 남아야 하므로 센티널로 넘긴다. 러너가 세니
                # metric(rejected=...)으로 또 세지 않는다 — 이중 계수가 된다.
                dropped += 1
                yield reject_record(record, ["assistant_line_overused"])
                continue
            yield record
        ctx.log(f"[{self.name}] distinct assistant utterances: {len(counts):,}; dropped for overuse (> {limit}x): {dropped:,}")
