"""Mix the filtered sources to the target ratios, split, and write a manifest.

This is the last gate, and the only stage that can answer "where did this
corpus come from". The old manifest recorded counts and hashes but not the
models, ratios or seed that produced them, so the corpus could be described
and not reproduced. This one embeds the config that made it.

Token accounting has a circularity: the corpus is measured in tokens, but the
tokenizer is trained on the corpus. Rather than bootstrap a throwaway
tokenizer, this stage uses a real one when the config points at an existing
model and a characters-per-token estimate otherwise — and records which, so no
one later reads an estimate as a measurement.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from persona_sft_data import schema
from persona_sft_data.runner import StageContext, metric

# Which ratio bucket a record's `source` belongs to. Sources are named for the
# stage that made them; ratios are named for the spec's three-way split.
BUCKETS = {
    "teacher_seed": "teacher",
    "teacher_expand": "teacher",
    "real": "real",
    "template": "template",
}


def bucket_of(source: str) -> str:
    if source in BUCKETS:
        return BUCKETS[source]
    # A new generator should land somewhere deliberate, not silently in
    # whichever bucket sorts first.
    if source.startswith("teacher"):
        return "teacher"
    raise ValueError(
        f"record source {source!r} maps to no ratio bucket; "
        f"add it to persona_sft_data/stages/assemble.py:BUCKETS"
    )


@dataclass
class TokenCounter:
    """Exact when a tokenizer is available, estimated otherwise."""

    method: str
    chars_per_token: float | None = None
    _sp: Any = None

    @classmethod
    def build(cls, tokenizer_path: Path | None, fallback_ratio: float) -> "TokenCounter":
        if tokenizer_path and tokenizer_path.exists():
            try:
                import sentencepiece as spm
            except ImportError:
                pass
            else:
                sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
                return cls(method=f"sentencepiece:{tokenizer_path.name}", _sp=sp)
        return cls(method="estimated_from_characters", chars_per_token=fallback_ratio)

    def count(self, text: str) -> int:
        if self._sp is not None:
            return len(self._sp.encode(text, out_type=int))
        return max(1, round(len(text) / (self.chars_per_token or 1.88)))


class AssembleStage:
    """Ratio-mix the filtered sources, split, and write final/ + manifest."""

    name = "assemble"
    produces = "final"

    def run(self, ctx: StageContext) -> Iterator[dict]:
        settings = ctx.settings
        target = int(settings["target_tokens"])
        ratios: dict[str, float] = dict(settings["ratios"])
        split: dict[str, float] = dict(settings["split"])

        total_ratio = sum(ratios.values())
        if abs(total_ratio - 1.0) > 1e-6:
            raise ValueError(f"ratios sum to {total_ratio}, not 1.0")
        total_split = sum(split.values())
        if abs(total_split - 1.0) > 1e-6:
            raise ValueError(f"split sums to {total_split}, not 1.0")

        tok_cfg = settings.get("tokenizer")
        counter = TokenCounter.build(
            (ctx.config.root / tok_cfg) if tok_cfg else None,
            float(settings.get("chars_per_token_estimate", 1.88)),
        )
        ctx.log(f"[assemble] token accounting: {counter.method}")

        # Load every filtered source, bucketed.
        pools: dict[str, list[dict]] = {b: [] for b in set(ratios)}
        filtered_dir = ctx.config.data_root / "filtered"
        for path in sorted(filtered_dir.glob("*.jsonl")):
            if path.name.endswith((".rejected.jsonl", ".sample.jsonl")):
                continue
            for record in schema.read_jsonl(path):
                try:
                    bucket = bucket_of(record.get("source", ""))
                except ValueError as exc:
                    ctx.log(f"[assemble] {exc}")
                    raise
                if bucket in pools:
                    record["_tokens"] = counter.count(schema.session_text(record))
                    pools[bucket].append(record)

        for bucket, pool in pools.items():
            available = sum(r["_tokens"] for r in pool)
            wanted = int(target * ratios[bucket])
            ctx.log(
                f"[assemble] {bucket}: {len(pool):,} sessions, {available:,} tokens "
                f"available, {wanted:,} wanted "
                f"({'OK' if available >= wanted else 'SHORT'})"
            )

        # Draw each bucket to its token budget. Shuffle first so a short pool
        # is not systematically the earliest-generated records.
        selected: list[dict] = []
        shortfalls: dict[str, int] = {}
        for bucket, pool in pools.items():
            ctx.rng.shuffle(pool)
            budget = int(target * ratios[bucket])
            used = 0
            for record in pool:
                if used >= budget:
                    break
                used += record["_tokens"]
                selected.append(record)
            if used < budget:
                shortfalls[bucket] = budget - used

        if shortfalls:
            # Report rather than silently producing a corpus with different
            # ratios than the config asked for.
            ctx.log(f"[assemble] SHORTFALL (tokens): {shortfalls}")

        ctx.rng.shuffle(selected)

        # Split by session, never mid-session: a session split across train and
        # val would leak.
        n = len(selected)
        n_val = int(n * split["val"])
        n_test = int(n * split["test"])
        for i, record in enumerate(selected):
            if i < n_val:
                record["split"] = "val"
            elif i < n_val + n_test:
                record["split"] = "test"
            else:
                record["split"] = "train"
            record.pop("_tokens", None)
            yield record

        self._write_splits(ctx, selected, counter, ratios, split, target, shortfalls)
        yield metric()

    # ---- manifest --------------------------------------------------------

    def _write_splits(self, ctx: StageContext, selected, counter, ratios,
                      split, target, shortfalls) -> None:
        """Write train/val/test beside corpus.jsonl, then the manifest.

        The runner writes corpus.jsonl (this stage's output). The per-split
        files are derived from the same records so they cannot disagree.
        """
        final_dir = ctx.config.data_root / "final"
        by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
        for record in selected:
            by_split[record["split"]].append(record)

        files: dict[str, dict[str, str]] = {}
        for name, records in by_split.items():
            path = final_dir / f"{name}.jsonl"
            schema.write_jsonl(path, records)
            files[name] = {
                "path": str(path.relative_to(ctx.config.root)),
                "sha256": _sha256(path),
                "sessions": len(records),
            }

        corpus_path = ctx.config.final("assemble")
        source_tokens: dict[str, int] = {}
        source_sessions: dict[str, int] = {}
        for record in selected:
            b = bucket_of(record["source"])
            source_sessions[b] = source_sessions.get(b, 0) + 1
            source_tokens[b] = source_tokens.get(b, 0) + counter.count(
                schema.session_text(record)
            )
        total_tokens = sum(source_tokens.values())

        manifest = {
            "generated_by": "persona_sft_data",
            "config_path": str(ctx.config.path.relative_to(ctx.config.root)),
            "config": json.loads(ctx.config.path.read_text(encoding="utf-8")),
            "seed": ctx.config.seed,
            "persona_doc": str(ctx.config.persona_doc.relative_to(ctx.config.root)),
            "persona_sha256": _sha256(ctx.config.persona_doc),
            "token_accounting": {
                "method": counter.method,
                "chars_per_token_estimate": counter.chars_per_token,
            },
            "target_tokens": target,
            "actual_tokens": total_tokens,
            "source_tokens": source_tokens,
            "source_ratios": {
                k: round(v / total_tokens, 6) if total_tokens else 0.0
                for k, v in source_tokens.items()
            },
            "requested_ratios": ratios,
            "shortfall_tokens": shortfalls,
            "source_sessions": source_sessions,
            "split_sessions": {k: len(v) for k, v in by_split.items()},
            "requested_split": split,
            "files": files,
            "stages": _stage_stats(ctx.config),
        }
        # corpus.jsonl is written by the runner after this returns, so its hash
        # is recorded by the CLI rather than guessed at here.
        manifest["files"]["corpus"] = {
            "path": str(corpus_path.relative_to(ctx.config.root)),
            "sha256": None,
            "sessions": len(selected),
        }

        path = final_dir / "manifest.json"
        path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        ctx.log(
            f"[assemble] {len(selected):,} sessions, {total_tokens:,} tokens; "
            f"manifest -> {path.name}"
        )


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _stage_stats(config) -> dict[str, Any]:
    """Fold every stage's stats file into the manifest, so one file answers
    what produced this corpus."""
    out: dict[str, Any] = {}
    for sub in ("raw", "filtered"):
        for stats_file in sorted((config.data_root / sub).glob("*.stats.json")):
            key = f"{sub}/{stats_file.name[: -len('.jsonl.stats.json')]}"
            try:
                data = json.loads(stats_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            out[key] = {
                k: data.get(k)
                for k in ("produced", "rejected", "yield_rate", "seconds",
                          "teacher_model", "teacher_calls", "teacher_failures",
                          "reject_reasons")
            }
    return out
