"""The seed stage: the reasoning teacher writes one skeleton per beat.

This is the narrow end of the funnel. It produces comparatively few dialogues
with the expensive teacher, and ``expand`` multiplies them with the cheap one.
Everything that decides *what* a dialogue is about is decided here, which is why
the beat list is walked exhaustively rather than sampled: a situation the
persona document names but the corpus never covers is a hole nothing downstream
can fill.

Nothing about the persona is written here. The beats come from the document via
``Persona.beats``, the prompt text comes from ``prompts``, and the model id comes
from the config. This module contains no Korean prose for that reason.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

from persona_sft_data import backend, prompts, runner
from persona_sft_data.backend import Request, Teacher
from persona_sft_data.runner import StageContext


class SeedStage:
    """Generate ``per_situation`` scenario skeletons for every persona beat."""

    name = "seed"
    produces = "raw"

    def __init__(self, teacher: Teacher | None = None) -> None:
        # Injectable only so the stage tests can run against a FakeTeacher;
        # the pipeline never passes one and builds from config below.
        self._teacher = teacher

    def run(self, ctx: StageContext) -> Iterator[dict]:
        cfg = ctx.config.teacher_for(ctx.name)
        teacher = self._teacher if self._teacher is not None else backend.build(cfg)
        # Fail before generating anything if the server is down or is serving
        # the other teacher: eight million tokens from the wrong model look
        # entirely normal until you read them.
        teacher.check()

        beats = ctx.persona.beats
        per_situation = int(ctx.settings["per_situation"])
        turn_choices = list(ctx.settings["turns"])
        if not beats:
            raise ValueError("persona document yielded no beats to seed from")
        if not turn_choices:
            raise ValueError(f"stage {ctx.name!r} has an empty 'turns' list")

        # One batch per round trip, sized to the teacher's concurrency: vLLM
        # wants the whole batch at once (2,517 tok/s at 400 concurrent against
        # 275 at 20), and the batch is also the only thing held in memory.
        batch_size = max(1, int(cfg.concurrency))
        total = len(beats) * per_situation

        index = 0
        issued = 0
        started = time.time()
        requests = self._requests(ctx, per_situation, turn_choices)
        for batch in backend.batched(requests, batch_size):
            results = {r.key: r for r in teacher.generate([req for req, _ in batch])}
            failures = 0
            tokens = 0
            reasons: dict[str, int] = {}

            for request, beat_index in batch:
                result = results.get(request.key)
                if result is None or not result.ok:
                    # A dropped call is a lost dialogue, so count it as a reject
                    # too. teacher_failures alone would leave the yield rate
                    # silently flattering.
                    failures += 1
                    reasons["teacher_error"] = reasons.get("teacher_error", 0) + 1
                    continue
                tokens += result.completion_tokens
                turns = prompts.repair_dialogue(prompts.parse_dialogue(result.text or ""))
                if not turns:
                    reasons["unparseable"] = reasons.get("unparseable", 0) + 1
                    continue
                yield _record(index, beats[beat_index], turns, cfg.model)
                index += 1

            yield runner.metric(
                calls=len(batch),
                failures=failures,
                completion_tokens=tokens,
                rejected=sum(reasons.values()),
                reject_reasons=reasons,
            )

            issued += len(batch)
            done = batch[-1][1] + 1  # beats are issued in order
            ctx.log(
                f"[{ctx.name}] beats {done}/{len(beats)} | "
                f"{index:,} records | {issued:,}/{total:,} calls | "
                f"{time.time() - started:.0f}s"
            )

    def _requests(
        self, ctx: StageContext, per_situation: int, turn_choices: list[Any]
    ) -> Iterator[tuple[Request, int]]:
        """Lazily build every request, paired with the beat it came from.

        A generator rather than a list: 59 beats x 400 dialogues is 23,600
        prompts of a few kilobytes each, and only one batch of them needs to
        exist at a time. The beat index travels alongside instead of being
        parsed back out of the key, so the key stays an opaque handle.
        """
        for beat_index, beat in enumerate(ctx.persona.beats):
            for n in range(per_situation):
                turns = ctx.rng.choice(turn_choices)
                # A fresh system prompt per dialogue: persona_block rotates a
                # different vocabulary sample each call, which is what keeps the
                # corpus off the same handful of phrases.
                yield (
                    Request(
                        key=f"{beat_index}:{n}",
                        system=prompts.seed_system(ctx.persona, ctx.rng),
                        user=prompts.seed_user(ctx.persona, beat, turns, ctx.rng),
                    ),
                    beat_index,
                )


def _record(
    index: int, scenario: str, turns: list[dict[str, str]], model: str
) -> dict[str, Any]:
    """One session record. The runner validates and writes it."""
    return {
        "id": f"seed-{index:06d}",
        "source": "teacher_seed",
        "scenario": scenario,
        "generator": [model],
        "license": "synthetic",
        "turns": turns,
    }


__all__ = ["SeedStage"]
