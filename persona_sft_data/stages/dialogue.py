"""dialogue: 추론 교사가 다룰 상황의 beat마다 대화를 쓴다.

beat 목록은 표집하지 않고 전부 돈다 — 문서가 이름 붙인 상황이 코퍼스에 없으면
아래 어느 단계도 메울 수 없다. 흐름은 문서의 대화 흐름(없으면 프로필 기본값)에서,
턴 수는 설정(없으면 프로필 기본값)에서 뽑는다. 이 모듈에는 한국어 산문이 없다.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from persona_sft_data.core.config import ConfigError
from persona_sft_data.core.registry import STAGES, TEACHERS
from persona_sft_data.core.runner import StageContext, metric
from persona_sft_data.teacher import prompts
from persona_sft_data.teacher.base import Request, batched


def _positive(where: str, value: Any) -> int:
    """설정 값을 1 이상의 정수로. 어긋나면 ``ConfigError``다 — 맨 ``ValueError``는
    CLI가 잡지 않아 사용자가 트레이스백을 본다."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ConfigError(f"{where}는 정수여야 한다: {value!r}") from None
    if number < 1:
        raise ConfigError(f"{where}는 1 이상이어야 한다: {value!r}")
    return number


@dataclass(frozen=True)
class DialogueSettings:
    teacher: str
    per_situation: int = 40
    turns: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        """값도 로드 시점에 본다. 실행 중에 터지면 교사 예산을 이미 다 쓴 뒤다.

        frozen dataclass라 고쳐 담지는 않고 검증만 한다.
        """
        _positive("stages.dialogue.per_situation", self.per_situation)
        if self.turns is None:
            return
        if isinstance(self.turns, (str, bytes)) or not isinstance(self.turns, Sequence):
            raise ConfigError(f"stages.dialogue.turns는 정수 목록이어야 한다: {self.turns!r}")
        if not self.turns:
            raise ConfigError("stages.dialogue.turns가 비어 있다 (프로필 기본값을 쓰려면 키를 빼라)")
        for turn in self.turns:
            _positive("stages.dialogue.turns의 각 항목", turn)


@STAGES.register("dialogue", origin="builtin")
class DialogueStage:
    name = "dialogue"
    config_name = "dialogue"
    mode = "records"
    record_kind = "session"
    produces = "raw"
    settings_type = DialogueSettings

    def __init__(self, teacher: Any = None) -> None:
        self._teacher = teacher

    def requires(self, config: Any) -> tuple[str, ...]:
        return ()

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
        cfg = ctx.config.teacher_for(ctx.name)
        teacher = self._teacher_for(ctx)
        teacher.check()  # 다른 모델이 떠 있으면 생성 전에 멈춘다

        beats = ctx.persona.beats
        per_situation = int(ctx.settings.per_situation)
        turn_choices = list(ctx.settings.turns or ctx.profile.default_turns)
        flows = list(ctx.persona.flows or ctx.profile.default_flows)
        # per_situation·turns는 설정이 이미 검증했다. 남은 것은 문서·프로필에서 오는 것들이다.
        if not beats or per_situation < 1 or not turn_choices or not flows:
            raise ConfigError(f"stage {ctx.name!r}: beat·per_situation·turns·flows 중 빈 것이 있다")

        batch_size = max(1, int(cfg.concurrency))
        total = len(beats) * per_situation
        index = issued = 0
        started = time.time()
        for batch in batched(self._requests(ctx, per_situation, turn_choices, flows), batch_size):
            results = {r.key: r for r in teacher.generate([req for req, _ in batch])}
            failures = tokens = 0
            reasons: dict[str, int] = {}
            for request, beat_index in batch:
                result = results.get(request.key)
                if result is None or not result.ok:
                    failures += 1
                    reasons["teacher_error"] = reasons.get("teacher_error", 0) + 1
                    continue
                tokens += result.completion_tokens
                turns = prompts.repair_dialogue(prompts.parse_dialogue(result.text or ""))
                if not turns:
                    reasons["unparseable"] = reasons.get("unparseable", 0) + 1
                    continue
                yield {
                    "id": f"dialogue-{index:06d}", "source": "dialogue", "scenario": beats[beat_index],
                    "generator": [cfg.model], "license": "synthetic", "turns": turns,
                }
                index += 1
            yield metric(calls=len(batch), failures=failures, completion_tokens=tokens,
                         rejected=sum(reasons.values()), reject_reasons=reasons)
            issued += len(batch)
            ctx.log(f"[{ctx.name}] beats {batch[-1][1] + 1}/{len(beats)} | {index:,} records | "
                    f"{issued:,}/{total:,} calls | {time.time() - started:.0f}s")

    def _requests(self, ctx: StageContext, per_situation: int, turn_choices: list[int],
                  flows: list[str]) -> Iterator[tuple[Request, int]]:
        """요청은 지연 생성. 한 번에 한 배치만 메모리에 있다."""
        for beat_index, beat in enumerate(ctx.persona.beats):
            for n in range(per_situation):
                turns = ctx.rng.choice(turn_choices)
                flow = ctx.rng.choice(flows)
                yield (
                    Request(
                        key=f"{beat_index}:{n}",
                        system=prompts.dialogue_system(ctx.persona, ctx.profile, ctx.rng),
                        user=prompts.dialogue_user(ctx.persona, ctx.profile, beat, flow, turns),
                    ),
                    beat_index,
                )
