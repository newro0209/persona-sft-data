"""The stage contract, and the bookkeeping every stage shares.

Each stage reads jsonl and writes three files: the output, a stats file, and
the records it rejected with the reason. The rejects are the point — unless you
can count what was thrown away you cannot say anything about quality, and the
previous pipeline could not.

A stage implements ``run(ctx) -> Iterator[dict]`` and yields records. The runner
validates, deduplicates, gates, counts and writes. Stages do not open files.
"""

from __future__ import annotations

import json
import platform
import random
import sys
import time
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from persona_sft_data import schema
from persona_sft_data.config import PipelineConfig
from persona_sft_data.gates import Gate, Verdict
from persona_sft_data.persona import Persona, load_cached


@dataclass
class StageContext:
    """What a stage is handed. Everything it needs, nothing it must construct."""

    name: str
    config: PipelineConfig
    persona: Persona
    settings: dict[str, Any]
    rng: random.Random
    output: Path
    gate: Gate | None = None
    log: Callable[[str], None] = lambda msg: print(msg, flush=True)

    def input_path(self, stage_name: str, *, stage: str = "raw") -> Path:
        """Where an upstream stage left its output."""
        return getattr(self.config, stage)(stage_name)

    def read(self, stage_name: str, *, stage: str = "raw") -> Iterator[dict]:
        path = self.input_path(stage_name, stage=stage)
        if not path.exists():
            raise FileNotFoundError(
                f"stage {self.name!r} needs {path}, which does not exist.\n"
                f"  Run the {stage_name!r} stage first."
            )
        yield from schema.read_jsonl(path)


class Stage(Protocol):
    name: str
    produces: str  # "raw" | "filtered" | "final"

    def run(self, ctx: StageContext) -> Iterator[dict]: ...


@dataclass
class StageStats:
    """What every stage reports. Written next to the output."""

    stage: str
    output: str
    started: str
    seconds: float = 0.0
    produced: int = 0
    rejected: int = 0
    duplicates: int = 0
    # Source material a stage declined to use, kept apart from rejected: a
    # corpus we chose not to draw from is not a quality failure, and folding
    # the two together made the real stage report a 0.2% yield when 80% of
    # what it actually generated passed.
    source_filtered: int = 0
    source_filter_reasons: dict[str, int] = field(default_factory=dict)
    reject_reasons: dict[str, int] = field(default_factory=dict)
    teacher_model: str | None = None
    teacher_calls: int = 0
    teacher_failures: int = 0
    completion_tokens: int = 0
    distinct_meanings: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d["yield_rate"] = (
            round(self.produced / (self.produced + self.rejected), 4)
            if (self.produced + self.rejected)
            else None
        )
        d.update(self.extra)
        d["environment"] = {
            "python": platform.python_version(),
            "platform": platform.platform(),
        }
        return d


def execute(stage: Stage, config: PipelineConfig, *, log=print) -> StageStats:
    """Run one stage: validate, dedupe, gate, count, write."""
    settings = config.stage(stage.name)
    persona = load_cached(config.persona_doc)
    output = getattr(config, stage.produces)(stage.name)
    output.parent.mkdir(parents=True, exist_ok=True)

    ctx = StageContext(
        name=stage.name,
        config=config,
        persona=persona,
        settings=settings,
        rng=random.Random(config.stage_seed(stage.name)),
        output=output,
        gate=Gate.from_config(persona, config.stages.get("filter", {})),
        log=log,
    )

    stats = StageStats(
        stage=stage.name,
        output=str(output.relative_to(config.root)),
        started=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    if "teacher" in settings:
        stats.teacher_model = config.teacher_for(stage.name).model

    rejected_path = config.rejected_path(output)
    sample_path = config.sample_path(output)
    seen: set[str] = set()
    verdicts: list[Verdict] = []
    kept: list[dict] = []

    t0 = time.time()
    with output.open("w", encoding="utf-8") as out, \
         rejected_path.open("w", encoding="utf-8") as rej:

        def reject(record: Mapping[str, Any], reasons: list[str]) -> None:
            stats.rejected += 1
            payload = dict(record)
            payload["_reject_reasons"] = reasons
            rej.write(json.dumps(payload, ensure_ascii=False) + "\n")

        for record in stage.run(ctx):
            # Stages report teacher usage through a sentinel record so they
            # never have to touch the stats object directly.
            if record.get("_metric"):
                stats.teacher_calls += record.get("calls", 0)
                stats.teacher_failures += record.get("failures", 0)
                stats.completion_tokens += record.get("completion_tokens", 0)
                for reason, n in (record.get("reject_reasons") or {}).items():
                    stats.reject_reasons[reason] = stats.reject_reasons.get(reason, 0) + n
                stats.rejected += record.get("rejected", 0)
                stats.source_filtered += record.get("source_filtered", 0)
                for reason, n in (record.get("source_filter_reasons") or {}).items():
                    stats.source_filter_reasons[reason] = (
                        stats.source_filter_reasons.get(reason, 0) + n
                    )
                continue

            try:
                normalized = schema.normalize_session(record)
            except schema.SessionError as exc:
                reject(record, [f"schema:{exc}"])
                continue

            fingerprint = schema.content_fingerprint(normalized)
            if fingerprint in seen:
                stats.duplicates += 1
                reject(normalized, ["duplicate"])
                continue
            seen.add(fingerprint)

            if ctx.gate is not None:
                verdict = ctx.gate.check(normalized)
                if not verdict.ok:
                    verdicts.append(verdict)
                    reject(normalized, verdict.reasons)
                    continue

            out.write(json.dumps(normalized, ensure_ascii=False) + "\n")
            stats.produced += 1
            if len(kept) < 200:
                kept.append(normalized)
            elif ctx.rng.random() < 0.001:
                kept[ctx.rng.randrange(200)] = normalized

    stats.seconds = round(time.time() - t0, 2)
    for reason, n in _tally(verdicts).items():
        stats.reject_reasons[reason] = stats.reject_reasons.get(reason, 0) + n
    stats.reject_reasons = dict(
        sorted(stats.reject_reasons.items(), key=lambda kv: -kv[1])
    )
    stats.distinct_meanings = len(seen) or None

    # A sample a person can actually read, rather than a review workflow that
    # was built once and never used twice.
    schema.write_jsonl(sample_path, kept)
    config.stats_path(output).write_text(
        json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    total = stats.produced + stats.rejected
    rate = f"{stats.produced / total:.1%}" if total else "n/a"
    log(
        f"[{stage.name}] {stats.produced:,} kept / {stats.rejected:,} rejected "
        f"({rate}) in {stats.seconds:.1f}s -> {output.name}"
    )
    if stats.reject_reasons:
        top = ", ".join(f"{k}={v}" for k, v in list(stats.reject_reasons.items())[:5])
        log(f"[{stage.name}] top rejects: {top}")
    if stats.source_filtered:
        top = ", ".join(f"{k}={v:,}" for k, v in stats.source_filter_reasons.items())
        log(f"[{stage.name}] source material not used: {stats.source_filtered:,} ({top})")
    if stats.teacher_failures:
        log(f"[{stage.name}] teacher failures: {stats.teacher_failures:,}")
    return stats


def _tally(verdicts) -> dict[str, int]:
    counts: dict[str, int] = {}
    for v in verdicts:
        for reason in v.reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return counts


def metric(**kwargs) -> dict:
    """A sentinel a stage yields to report teacher usage or its own rejects."""
    return {"_metric": True, **kwargs}
