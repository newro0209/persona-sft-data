"""ingest: 외부 소스 → 발화 레코드.

소스마다 가져오기 → 포맷·추출 → 싼 필터(정규화·길이·중복) → 표집 → (다른 언어면)
번역 → 주제·안전 필터. 번역이 비싸므로 표집을 번역 앞에 둔다. 번역 결과는 발화
레코드에 원문과 함께 남아 ``raw/ingest.jsonl``에 캐시되고, 데이터 카드가 출처를
말할 수 있다.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import islice
from typing import Any

from persona_sft_data.core.config import ConfigError, require_int, require_number
from persona_sft_data.core.registry import STAGES, TEACHERS, TRANSLATORS
from persona_sft_data.core.runner import StageContext, metric
from persona_sft_data.core.schema import normalize_text
from persona_sft_data.sources import topic
from persona_sft_data.sources.base import fetch_source, read_utterances
from persona_sft_data.sources.safety import DEFAULT_STEMS, is_unsafe

STAT_KEYS = ("raw", "distinct", "sampled", "translated", "translation_failed", "in_scope", "unsafe", "off_topic")


@dataclass(frozen=True)
class IngestSettings:
    teacher: str
    sources: tuple[str, ...]
    translator: str = "teacher"
    limit_per_source: int = 3000
    min_chars: int = 2
    max_chars: int = 60
    topic_min_hits: int = 1
    blocked_stems: tuple[str, ...] | None = None
    download_timeout: float = 60.0

    def __post_init__(self) -> None:
        """값도 로드 시점에 본다. 실행 중에 터지면 소스를 이미 다 내려받은 뒤다.

        ``check``가 통과한 설정이 ``run``에서 조용히 아무것도 내지 않는 일이 없게
        한다 — ``min_chars > max_chars``면 모든 발화가 길이 필터에 걸려 출력이 빈다.
        """
        if not self.sources:
            raise ConfigError("stages.ingest.sources가 비어 있다 (읽을 소스 이름을 하나 이상 적는다)")
        require_int("stages.ingest.limit_per_source", self.limit_per_source, minimum=1)
        min_chars = require_int("stages.ingest.min_chars", self.min_chars, minimum=1)
        max_chars = require_int("stages.ingest.max_chars", self.max_chars, minimum=1)
        if min_chars > max_chars:
            raise ConfigError(
                f"stages.ingest.min_chars({min_chars})가 max_chars({max_chars})보다 크다 — 통과할 발화가 없다"
            )
        require_int("stages.ingest.topic_min_hits", self.topic_min_hits, minimum=0)
        require_number("stages.ingest.download_timeout", self.download_timeout, minimum=0, exclusive=True)


@STAGES.register("ingest", origin="builtin")
class IngestStage:
    name = "ingest"
    config_name = "ingest"
    mode = "records"
    record_kind = "utterance"
    produces = "raw"
    settings_type = IngestSettings

    def __init__(self, teacher: Any = None) -> None:
        self._teacher = teacher

    def requires(self, config: Any) -> tuple[str, ...]:
        return ()

    def instances(self, config: Any) -> list[Any]:
        return [self]

    def _needs_translation(self, ctx: StageContext) -> bool:
        return any(ctx.config.source(n).language != ctx.config.language for n in ctx.settings.sources)

    def _teacher_for(self, ctx: StageContext) -> Any:
        if self._teacher is not None:
            return self._teacher
        cfg = ctx.config.teacher_for(ctx.name)
        return TEACHERS.get(cfg.kind).build(cfg)

    def preflight(self, ctx: StageContext) -> None:
        cache = ctx.config.data_root / "cache"
        for name in ctx.settings.sources:
            cfg = ctx.config.source(name)
            data = fetch_source(cfg, cache, timeout=ctx.settings.download_timeout, log=ctx.log)
            if data is None:
                continue
            # run과 같은 처리: 읽을 수 없는 소스 하나는 그 소스만 건너뛴다. check가
            # 깨진 jsonl이나 pyarrow 없는 parquet에서 트레이스백으로 죽으면 나머지
            # 소스·단계를 아예 점검하지 못한다.
            try:
                sample = list(islice(read_utterances(cfg, data), 3))
            except Exception as exc:  # noqa: BLE001 - 소스 하나가 점검을 죽이지 않는다
                ctx.log(f"[{ctx.name}] {name}: 읽을 수 없다 ({type(exc).__name__}: {exc}); 건너뛴다")
                continue
            ctx.log(f"[{ctx.name}] {name} ({cfg.language}): {sample}")
        if self._needs_translation(ctx):
            self._teacher_for(ctx).check()

    def run(self, ctx: StageContext) -> Iterator[dict[str, Any]]:
        s = ctx.settings
        cache = ctx.config.data_root / "cache"
        signal = topic.signal(ctx.persona)
        stems = tuple(s.blocked_stems or DEFAULT_STEMS)
        teacher = translator = None
        teacher_model = ctx.config.teacher_for(ctx.name).model
        per_source: dict[str, dict[str, int]] = {}
        calls = failures = 0

        for name in s.sources:
            cfg = ctx.config.source(name)
            st = dict.fromkeys(STAT_KEYS, 0)
            per_source[name] = st
            data = fetch_source(cfg, cache, timeout=s.download_timeout, log=ctx.log)
            if data is None:
                continue
            try:
                texts = list(read_utterances(cfg, data))
            except Exception as exc:  # noqa: BLE001 - 소스 하나가 실행을 죽이지 않는다
                ctx.log(f"[{ctx.name}] {name}: 읽을 수 없다 ({type(exc).__name__}: {exc}); 건너뛴다")
                continue
            st["raw"] = len(texts)

            # 번역이 필요한 소스는 원문 기준으로 두 배까지 허용하고, 번역 뒤 다시 max_chars로 잰다.
            translate = cfg.language != ctx.config.language
            ceiling = s.max_chars * 2 if translate else s.max_chars
            seen: set[str] = set()
            pool: list[str] = []
            for text in texts:
                text = normalize_text(text)
                if not s.min_chars <= len(text) <= ceiling:
                    continue
                key = text.casefold()
                if key in seen:
                    continue
                seen.add(key)
                pool.append(text)
            st["distinct"] = len(pool)
            # 번역 전에 표집해 비용을 묶는다.
            ctx.rng.shuffle(pool)
            pool = pool[: s.limit_per_source]
            st["sampled"] = len(pool)

            if translate:
                if translator is None:
                    teacher = self._teacher_for(ctx)
                    teacher.check()
                    translator = TRANSLATORS.get(s.translator).build(ctx, teacher)
                translated = translator.translate(pool, cfg.language)
                pairs = [(o, t) for o, t in zip(pool, translated) if t]
                st["translated"] = len(pairs)
                st["translation_failed"] = len(pool) - len(pairs)
                calls += len(pool)
                failures += st["translation_failed"]
            else:
                pairs = [(None, t) for t in pool]

            index = 0
            for original, text in pairs:
                if is_unsafe(text, stems):
                    st["unsafe"] += 1
                    continue
                if not topic.in_scope(text, signal, min_hits=s.topic_min_hits, min_chars=s.min_chars, max_chars=s.max_chars):
                    st["off_topic"] += 1
                    continue
                st["in_scope"] += 1
                record: dict[str, Any] = {
                    "id": f"{name}-{index:06d}", "text": text, "source": name,
                    "language": ctx.config.language, "license": cfg.license, "url": cfg.url,
                }
                if original is not None:
                    record.update(original_text=original, original_language=cfg.language, translator=teacher_model)
                yield record
                index += 1
            ctx.log(f"[{ctx.name}] {name}: raw {st['raw']:,} → distinct {st['distinct']:,} → sampled {st['sampled']:,}"
                    f"{' → translated ' + format(st['translated'], ',') if translate else ''} → in scope {st['in_scope']:,}")

        yield metric(
            calls=calls, failures=failures,
            rejected=sum(st["translation_failed"] for st in per_source.values()),
            reject_reasons={"translation_failed": failures} if failures else {},
            source_filtered=sum(st["unsafe"] + st["off_topic"] for st in per_source.values()),
            source_filter_reasons={k: v for k, v in (
                ("off_topic", sum(st["off_topic"] for st in per_source.values())),
                ("unsafe_source", sum(st["unsafe"] for st in per_source.values())),
            ) if v},
            extra={"sources": per_source},
        )
