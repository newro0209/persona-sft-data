"""The quality gate, applied to every raw source in one place.

The previous pipeline had this spread across filter.py, quality.py,
pair_quality.py, policy.py, text_rules.py and four validate_* modules, each
grown for one source. Reading them did not tell you what a record had to
satisfy. Here there is one gate (``persona_sft_data.gates``), applied identically to
everything, and the rules come from the persona document rather than from
whichever module happened to be edited last.

The runner already gates and deduplicates whatever a stage yields, so this
stage's own job is small: read a raw file, hand every record through
unchanged, and let the runner do the counting. What it adds is cross-record
work the runner cannot do per-record — near-duplicate detection across the
whole file, and a per-source breakdown in the stats.
"""

from __future__ import annotations

from collections.abc import Iterator

from persona_sft_data import schema
from persona_sft_data.runner import StageContext, metric


class FilterStage:
    """Gate one raw source into ``filtered/``.

    ``name`` is set per instance because this stage runs once per source; the
    config's ``filter`` block holds the thresholds shared by all of them.
    """

    produces = "filtered"

    def __init__(self, source: str) -> None:
        self.name = source

    def run(self, ctx: StageContext) -> Iterator[dict]:
        settings = ctx.config.stages.get("filter", {})
        max_repeat = int(settings.get("max_identical_pet_turns", 40))

        # Near-duplicate control the runner's exact fingerprint cannot do. A
        # teacher asked for 24 variants of one seed will happily return the
        # same reply for many of them, and a corpus where one sentence answers
        # a thousand different prompts teaches the model to always say it.
        pet_line_counts: dict[str, int] = {}
        by_source: dict[str, int] = {}
        dropped_overused = 0

        for record in ctx.read(self.name, stage="raw"):
            turns = record.get("turns") or []
            overused = False
            for turn in turns:
                if turn.get("role") != "pet":
                    continue
                text = turn.get("text", "")
                count = pet_line_counts.get(text, 0) + 1
                pet_line_counts[text] = count
                if count > max_repeat:
                    overused = True
            if overused:
                dropped_overused += 1
                continue

            by_source[record.get("source", "unknown")] = (
                by_source.get(record.get("source", "unknown"), 0) + 1
            )
            yield record

        yield metric(
            rejected=dropped_overused,
            reject_reasons={"pet_line_overused": dropped_overused} if dropped_overused else {},
        )
        ctx.log(
            f"[{self.name}] distinct pet utterances: {len(pet_line_counts):,}; "
            f"dropped for overuse (> {max_repeat}x): {dropped_overused:,}"
        )


def stages_for(ctx_config) -> list[FilterStage]:
    """One filter stage per raw file that exists."""
    raw_dir = ctx_config.data_root / "raw"
    if not raw_dir.exists():
        return []
    names = sorted(
        p.name[: -len(".jsonl")]
        for p in raw_dir.glob("*.jsonl")
        if not p.name.endswith((".rejected.jsonl", ".sample.jsonl"))
    )
    return [FilterStage(name) for name in names]


__all__ = ["FilterStage", "stages_for", "schema"]
