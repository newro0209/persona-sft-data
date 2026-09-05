"""The expand stage: the bulk teacher varies every seed dialogue.

``seed`` decides what a dialogue is about; this stage decides how many ways it
can be said. It is the volume stage, so it runs against the small teacher and
does one thing the seed stage does not: it compares each variant with the
dialogue it came from and throws away the ones that came back unchanged.

That check has to live here. The runner's fingerprint dedupe only sees this
stage's own output, and the seed it copied is in a different file — without this
the corpus would carry the same dialogue twice under two ids and count it as
two meanings.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

from persona_sft_data import backend, prompts, runner, schema
from persona_sft_data.backend import Request, Teacher
from persona_sft_data.runner import StageContext


class ExpandStage:
    """Rewrite each seed dialogue ``variants_per_seed`` times."""

    name = "expand"
    produces = "raw"

    def __init__(self, teacher: Teacher | None = None) -> None:
        # Injectable only so the stage tests can run against a FakeTeacher;
        # the pipeline never passes one and builds from config below.
        self._teacher = teacher

    def run(self, ctx: StageContext) -> Iterator[dict]:
        cfg = ctx.config.teacher_for(ctx.name)
        teacher = self._teacher if self._teacher is not None else backend.build(cfg)
        # Fail before generating anything if the server is down or is serving
        # the other teacher: the two teachers share a port and are loaded one
        # at a time, so pointing at the wrong one is a live possibility.
        teacher.check()

        variants = int(ctx.settings["variants_per_seed"])
        if variants < 1:
            raise ValueError(f"stage {ctx.name!r} needs variants_per_seed >= 1")

        # One batch per round trip, sized to the teacher's concurrency: vLLM
        # wants the whole batch at once, and the batch is also the only thing
        # held in memory — the seed file is streamed, never loaded.
        batch_size = max(1, int(cfg.concurrency))

        index = 0
        issued = 0
        started = time.time()
        requests = self._requests(ctx, variants)
        for batch in backend.batched(requests, batch_size):
            results = {
                r.key: r for r in teacher.generate([req for req, _, _ in batch])
            }
            failures = 0
            tokens = 0
            reasons: dict[str, int] = {}

            for request, _seed_index, seed in batch:
                result = results.get(request.key)
                if result is None or not result.ok:
                    # A dropped call is a lost variant, so count it as a reject
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
                if _same_dialogue(turns, seed.get("turns") or []):
                    reasons["identical_to_seed"] = (
                        reasons.get("identical_to_seed", 0) + 1
                    )
                    continue
                yield _record(index, seed, turns, cfg.model)
                index += 1

            yield runner.metric(
                calls=len(batch),
                failures=failures,
                completion_tokens=tokens,
                rejected=sum(reasons.values()),
                reject_reasons=reasons,
            )

            issued += len(batch)
            seeds_done = batch[-1][1] + 1  # seeds are streamed in order
            ctx.log(
                f"[{ctx.name}] seeds {seeds_done:,} | {index:,} records | "
                f"{issued:,} calls | {time.time() - started:.0f}s"
            )

    def _requests(
        self, ctx: StageContext, variants: int
    ) -> Iterator[tuple[Request, int, dict[str, Any]]]:
        """Lazily build every request, paired with the seed it varies.

        The seed record travels alongside the request rather than being looked
        up later, so the whole seed file never has to be resident: only the
        seeds referenced by the batch in flight are.
        """
        for seed_index, seed in enumerate(ctx.read("seed")):
            # Rendered once per seed, not once per variant — the source
            # dialogue is the same for all of them.
            rendered = prompts.render_dialogue(seed.get("turns") or [])
            user = prompts.expand_user(rendered)
            for v in range(variants):
                # A fresh system prompt per variant: persona_block rotates a
                # different vocabulary sample each call, which is what stops
                # every variant of a seed from converging on the same words.
                yield (
                    Request(
                        key=f"{seed_index}:{v}",
                        system=prompts.expand_system(ctx.persona, ctx.rng),
                        user=user,
                    ),
                    seed_index,
                    seed,
                )


def _same_dialogue(
    turns: Sequence[Mapping[str, str]], seed_turns: Sequence[Mapping[str, str]]
) -> bool:
    """Did the teacher hand back the dialogue it was given?

    Compared after the corpus text normalisation, so a variant that differs
    only in spacing counts as identical — it is not a variation, and calling it
    one would overstate the corpus's distinct-meaning count.
    """
    if len(turns) != len(seed_turns):
        return False
    return all(
        a.get("role") == b.get("role")
        and schema.normalize_text(str(a.get("text", "")))
        == schema.normalize_text(str(b.get("text", "")))
        for a, b in zip(turns, seed_turns)
    )


def _record(
    index: int, seed: Mapping[str, Any], turns: list[dict[str, str]], model: str
) -> dict[str, Any]:
    """One session record. The runner validates and writes it.

    ``scenario`` is inherited rather than re-derived: the variant is about
    whatever its seed was about, and that is what makes the beat coverage of
    the seed stage survive into the bulk corpus.
    """
    return {
        "id": f"expand-{index:06d}",
        "source": "teacher_expand",
        "scenario": str(seed.get("scenario", "unknown")),
        "seed_id": seed.get("id"),
        "generator": [model],
        "license": "synthetic",
        "turns": turns,
    }


__all__ = ["ExpandStage"]
