"""단계 실행 계약과 모든 단계가 공유하는 장부.

레코드를 내는 단계는 ``run(ctx)``에서 dict를 yield 할 뿐이다. 러너가 레코드 종류에
맞게 정규화하고, 지문으로 중복을 걸러 내고, 세션이면 게이트를 통과시키고, 세고,
쓴다. 거절은 버리지 않고 사유와 함께 ``.rejected.jsonl``에 남긴다 — 버려진 것을
셀 수 없으면 품질을 말할 수 없다. 사람이 읽을 표본 200개도 같이 떨군다.

단계가 스스로 거절한 레코드도 파일에 남아야 하므로 ``reject_record()`` 센티널로
러너에 넘긴다. 예외는 남길 레코드 자체가 없는 거절 — 교사 호출 실패
(``teacher_error``), 응답 파싱 실패(``unparseable``), 빈 응답(``empty_reply``)은
쓸 것이 없어 ``metric(rejected=...)``으로 개수만 센다.

출력·거절 파일은 ``.tmp``에 쓰고 단계가 끝까지 성공한 뒤에만 제자리로 옮긴다.
교사 서버가 죽은 재실행이 전날 산출물을 0바이트로 지우는 일이 없어야 한다.

파일을 직접 쓰는 단계(``mode="artifact"``)는 컨텍스트만 받고 자기 통계를 돌려준다.
러너가 통과시킨 것을 근거로 파생 파일을 써야 하는 단계는 ``finalize(ctx, stats)``를
선언한다 — 러너가 출력을 제자리에 옮기고 통계·표본까지 쓴 뒤에 부른다.
"""

from __future__ import annotations

import json
import os
import platform
import random
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from persona_sft_data.core import schema
from persona_sft_data.core.config import PipelineConfig
from persona_sft_data.core.gates import Gate, GateSettings, build_gate
from persona_sft_data.core.persona import Persona, load_cached
from persona_sft_data.core.registry import PROFILES

SAMPLE_SIZE = 200


@dataclass
class StageContext:
    """단계에 건네는 것. 필요한 전부이고, 단계가 직접 만들 것은 없다."""

    name: str
    config: PipelineConfig
    persona: Persona
    profile: Any
    settings: Any
    rng: random.Random
    output: Path | None
    gate: Gate | None
    log: Callable[[str], None] = lambda msg: print(msg, flush=True)

    def input_path(self, stage_name: str, *, area: str = "raw") -> Path:
        return getattr(self.config, area)(stage_name)

    def read(self, stage_name: str, *, area: str = "raw") -> Iterator[dict[str, Any]]:
        path = self.input_path(stage_name, area=area)
        if not path.exists():
            raise FileNotFoundError(
                f"stage {self.name!r}은(는) {path}가 필요한데 없다.\n  먼저 {stage_name!r} 단계를 돌려라."
            )
        yield from schema.read_jsonl(path)


@dataclass
class StageStats:
    """모든 단계가 보고하는 것. 출력 옆에 ``.stats.json``으로 쓴다."""

    stage: str
    output: str
    started: str
    seconds: float = 0.0
    produced: int = 0
    # 품질 때문에 떨어진 것: 스키마 위반, 게이트 위반, 단계가 스스로 보고한 거절.
    rejected: int = 0
    # 지문이 같아 한 번만 남긴 것. 거절과 따로 센다 — 같은 대화가 두 번 나온 것은
    # 품질 문제가 아니라서 수율 계산에 넣지 않는다. 다만 사유와 함께 rejected 파일에는 남긴다.
    duplicates: int = 0
    # 쓰지 않기로 한 소스 재료. 거절과 분리한다 — 페르소나 범위 밖 문장을 거절로 세면
    # 생성물의 80%가 통과한 단계가 수율 0.2%로 보인다.
    source_filtered: int = 0
    source_filter_reasons: dict[str, int] = field(default_factory=dict)
    reject_reasons: dict[str, int] = field(default_factory=dict)
    teacher_model: str | None = None
    teacher_calls: int = 0
    teacher_failures: int = 0
    completion_tokens: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        total = self.produced + self.rejected
        d["yield_rate"] = round(self.produced / total, 4) if total else None
        d.update(self.extra)
        d["environment"] = {"python": platform.python_version(), "platform": platform.platform()}
        return d


def metric(**kwargs: Any) -> dict[str, Any]:
    """단계가 교사 사용량이나 남길 레코드가 없는 거절을 보고할 때 yield 하는 센티널."""
    return {"_metric": True, **kwargs}


def reject_record(record: Mapping[str, Any], reasons: Sequence[str]) -> dict[str, Any]:
    """단계가 스스로 거절한 레코드를 러너에 넘길 때 yield 하는 센티널.

    러너가 사유를 세고 ``.rejected.jsonl``에 ``_reject_reasons``와 함께 남긴다.
    같은 거절을 ``metric(rejected=...)``으로 또 세면 이중 계수가 되니 둘 중 하나만 쓴다.
    """
    return {"_reject": True, "record": dict(record), "reasons": [str(r) for r in reasons]}


def gate_settings_for(config: PipelineConfig) -> GateSettings:
    """턴 수 범위는 filter 단계 설정에서, 없으면 기본값."""
    if config.has_stage("filter"):
        fs = config.stage_settings("filter")
        return GateSettings(min_turns=int(getattr(fs, "min_turns", 2)), max_turns=int(getattr(fs, "max_turns", 16)))
    return GateSettings()


def build_context(stage: Any, config: PipelineConfig, *, log: Callable[[str], None] = print) -> StageContext:
    profile = PROFILES.get(config.profile)
    persona = load_cached(config.persona_doc, profile.required_sections)
    kind = schema.RECORD_KINDS[stage.record_kind] if stage.record_kind else None
    output = getattr(config, stage.produces)(stage.name) if stage.produces else None
    gate = build_gate(persona, gate_settings_for(config)) if kind is not None and kind.gated else None
    return StageContext(
        name=stage.name, config=config, persona=persona, profile=profile,
        settings=config.stage_settings(stage.config_name),
        rng=random.Random(config.stage_seed(stage.name)), output=output, gate=gate, log=log,
    )


def _absorb_metric(stats: StageStats, record: Mapping[str, Any]) -> None:
    stats.teacher_calls += int(record.get("calls", 0))
    stats.teacher_failures += int(record.get("failures", 0))
    stats.completion_tokens += int(record.get("completion_tokens", 0))
    stats.rejected += int(record.get("rejected", 0))
    for reason, n in (record.get("reject_reasons") or {}).items():
        stats.reject_reasons[reason] = stats.reject_reasons.get(reason, 0) + int(n)
    stats.source_filtered += int(record.get("source_filtered", 0))
    for reason, n in (record.get("source_filter_reasons") or {}).items():
        stats.source_filter_reasons[reason] = stats.source_filter_reasons.get(reason, 0) + int(n)
    for key, value in (record.get("extra") or {}).items():
        stats.extra[key] = value


def _display_path(path: Path, root: Path) -> str:
    """통계에 적는 출력 경로. 저장소 루트 기준 상대 경로, 밖이면 절대 경로."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _tmp_path(path: Path) -> Path:
    """제자리 교체 전에 쓰는 임시 경로. 같은 디렉터리여야 ``os.replace``가 원자적이다."""
    return path.with_name(path.name + ".tmp")


def execute(stage: Any, config: PipelineConfig, *, log: Callable[[str], None] = print) -> StageStats:
    """단계 하나를 돌린다.

    출력과 거절 파일은 ``.tmp``에 쓰고 단계가 끝까지 성공한 뒤에만 제자리로 옮긴다.
    첫 반복에서 터지는 예외(교사 죽음, 입력 없음, 단계의 ValueError)에도 지난
    산출물·통계·표본은 그대로 남는다. 예외는 그대로 올려 호출자가 메시지를 낸다.
    """
    ctx = build_context(stage, config, log=log)
    if stage.mode == "artifact":
        return stage.run(ctx)

    kind = schema.RECORD_KINDS[stage.record_kind]
    output = ctx.output
    assert output is not None
    output.parent.mkdir(parents=True, exist_ok=True)
    stats = StageStats(stage=stage.name, output=_display_path(output, config.root),
                       started=time.strftime("%Y-%m-%dT%H:%M:%S"))
    teacher = getattr(ctx.settings, "teacher", None)
    if teacher is not None:
        stats.teacher_model = config.teachers[teacher].model

    seen: set[str] = set()
    kept: list[dict[str, Any]] = []
    rejected_path = config.rejected_path(output)
    out_tmp, rej_tmp = _tmp_path(output), _tmp_path(rejected_path)
    t0 = time.time()
    try:
        with out_tmp.open("w", encoding="utf-8", newline="\n") as out, \
             rej_tmp.open("w", encoding="utf-8", newline="\n") as rej:

            def drop(record: Mapping[str, Any], reasons: list[str]) -> None:
                """사유를 세고 rejected 파일에 남긴다. ``rejected`` 카운트는 호출자가 정한다."""
                for reason in reasons:
                    stats.reject_reasons[reason] = stats.reject_reasons.get(reason, 0) + 1
                schema.append_jsonl(rej, {**record, "_reject_reasons": reasons})

            def reject(record: Mapping[str, Any], reasons: list[str]) -> None:
                stats.rejected += 1
                drop(record, reasons)

            for record in stage.run(ctx):
                if record.get("_metric"):
                    _absorb_metric(stats, record)
                    continue
                if record.get("_reject"):
                    # 단계가 스스로 거절한 레코드. 정규화·게이트를 다시 걸 필요가 없다.
                    reject(record.get("record") or {}, list(record.get("reasons") or ["rejected"]))
                    continue
                try:
                    normalized = kind.normalize(record)
                except schema.SchemaError as exc:
                    reject(record, [f"schema:{exc}"])
                    continue
                fingerprint = kind.fingerprint(normalized)
                if fingerprint in seen:
                    stats.duplicates += 1
                    drop(normalized, ["duplicate"])
                    continue
                seen.add(fingerprint)
                if ctx.gate is not None:
                    verdict = ctx.gate.check(normalized)
                    if not verdict.ok:
                        reject(normalized, verdict.reasons)
                        continue
                schema.append_jsonl(out, normalized)
                stats.produced += 1
                if len(kept) < SAMPLE_SIZE:
                    kept.append(normalized)
                elif ctx.rng.random() < 0.001:
                    kept[ctx.rng.randrange(SAMPLE_SIZE)] = normalized
    except BaseException:
        # 반쪽 산출물을 남기지 않는다. 지난 출력·통계·표본은 손대지 않은 채 예외를 올린다.
        out_tmp.unlink(missing_ok=True)
        rej_tmp.unlink(missing_ok=True)
        raise

    os.replace(out_tmp, output)
    os.replace(rej_tmp, rejected_path)

    stats.seconds = round(time.time() - t0, 2)
    stats.reject_reasons = dict(sorted(stats.reject_reasons.items(), key=lambda kv: -kv[1]))
    schema.write_jsonl(config.sample_path(output), kept)
    config.stats_path(output).write_text(
        json.dumps(stats.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
    )

    finalize = getattr(stage, "finalize", None)
    if callable(finalize):
        finalize(ctx, stats)

    total = stats.produced + stats.rejected
    rate = f"{stats.produced / total:.1%}" if total else "n/a"
    log(
        f"[{stage.name}] {stats.produced:,} 통과 / {stats.rejected:,} 거절 / {stats.duplicates:,} 중복 "
        f"({rate}) {stats.seconds:.1f}s -> {output.name}"
    )
    if stats.reject_reasons:
        top = ", ".join(f"{k}={v}" for k, v in list(stats.reject_reasons.items())[:5])
        log(f"[{stage.name}] 주요 거절 사유: {top}")
    if stats.source_filtered:
        top = ", ".join(f"{k}={v:,}" for k, v in stats.source_filter_reasons.items())
        log(f"[{stage.name}] 쓰지 않은 소스 재료: {stats.source_filtered:,} ({top})")
    if stats.teacher_failures:
        log(f"[{stage.name}] 교사 호출 실패: {stats.teacher_failures:,}")
    return stats
