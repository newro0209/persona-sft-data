"""respond: ingest가 모은 사람의 발화에 교사가 페르소나로 한 줄 답한다.

사용자 쪽이 사람이 쓴(또는 사람이 쓴 것을 번역한) 문장이라는 점이 dialogue와
다르다. 출처 필드는 발화 레코드에서 세션 레코드로 그대로 옮긴다.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from persona_sft_data.core.registry import STAGES, TEACHERS
from persona_sft_data.core.runner import StageContext, metric
from persona_sft_data.teacher import prompts
from persona_sft_data.teacher.base import Request, batched


@dataclass(frozen=True)
class RespondSettings:
    teacher: str
    limit: int = 4000


@STAGES.register("respond", origin="builtin")
class RespondStage:
    name = "respond"
    config_name = "respond"
    mode = "records"
    record_kind = "session"
    produces = "raw"
    settings_type = RespondSettings

    def __init__(self, teacher: Any = None) -> None:
        self._teacher = teacher

    def requires(self, config: Any) -> tuple[str, ...]:
        return ("ingest",)

    def instances(self, config: Any) -> list[Any]:
        return [self]

    def _teacher_for(self, ctx: StageContext) -> Any:
        if self._teacher is not None:
            return self._teacher
        cfg = ctx.config.teacher_for(ctx.name)
        return TEACHERS.get(cfg.kind).build(cfg)

    def preflight(self, ctx: StageContext) -> None:
        self._teacher_for(ctx).check()

    def run(self, ctx: StageContext) -> Iterator[dict[str, Any]]:
        utterances = list(ctx.read("ingest"))
        ctx.rng.shuffle(utterances)
        limit = int(ctx.settings.limit)
        if limit and len(utterances) > limit:
            ctx.log(f"[{ctx.name}] limit {limit:,}: {len(utterances) - limit:,} unused")
            utterances = utterances[:limit]
        if not utterances:
            ctx.log(f"[{ctx.name}] ingest 출력이 비어 있어 만들 것이 없다")
            return

        cfg = ctx.config.teacher_for(ctx.name)
        teacher = self._teacher_for(ctx)
        teacher.check()
        batch_size = max(1, int(cfg.concurrency))
        started = time.time()
        done = 0
        for batch in batched(utterances, batch_size):
            keyed = {u["id"]: u for u in batch}
            requests = [
                Request(key=u["id"], system=prompts.respond_system(ctx.persona, ctx.profile, ctx.rng),
                        user=prompts.respond_user(u["text"]))
                for u in batch
            ]
            results = {r.key: r for r in teacher.generate(requests)}
            failures = tokens = 0
            reasons: dict[str, int] = {}
            for request in requests:
                result = results.get(request.key)
                if result is None or not result.ok:
                    failures += 1
                    reasons["teacher_error"] = reasons.get("teacher_error", 0) + 1
                    continue
                tokens += result.completion_tokens
                reply = prompts.reply_text(result.text)
                if not reply:
                    reasons["empty_reply"] = reasons.get("empty_reply", 0) + 1
                    continue
                u = keyed[request.key]
                yield {
                    "id": f"respond-{u['id']}", "source": "respond", "scenario": f"source:{u['source']}",
                    "utterance_id": u["id"], "source_dataset": u["source"], "source_url": u.get("url"),
                    "original_language": u.get("original_language"), "translator": u.get("translator"),
                    "license": u["license"],
                    "generator": [cfg.model],
                    "turns": [{"role": "user", "text": u["text"]}, {"role": "assistant", "text": reply}],
                }
            yield metric(calls=len(requests), failures=failures, completion_tokens=tokens,
                         rejected=sum(reasons.values()), reject_reasons=reasons)
            done += len(batch)
            ctx.log(f"[{ctx.name}] {done:,}/{len(utterances):,} answered | {time.time() - started:.0f}s")
