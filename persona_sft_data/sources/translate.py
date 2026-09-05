"""번역기. 내장은 교사에게 배치로 묻는 ``teacher`` 하나다.

언어 코드 → 이름 표는 ``teacher/prompts.py``의 ``LANGUAGE_NAMES``에 있고, 프롬프트도
거기서 만든다. 여기는 배치 나누기와 실패 자리 표시만 맡는다.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from persona_sft_data.core.registry import TRANSLATORS
from persona_sft_data.teacher import prompts
from persona_sft_data.teacher.base import Request, batched


class TeacherTranslator:
    name = "teacher"

    def __init__(self, teacher: Any, target_language: str, *, log: Callable[[str], None] = print,
                 batch_size: int = 64) -> None:
        self.teacher = teacher
        self.target_language = target_language
        self.log = log
        self.batch_size = max(1, batch_size)

    def translate(self, texts: Sequence[str], source_language: str) -> list[str | None]:
        """입력 순서대로. 실패한 자리는 ``None``."""
        out: list[str | None] = []
        system = prompts.translate_system(source_language, self.target_language)
        for batch in batched(list(enumerate(texts)), self.batch_size):
            requests = [Request(key=str(i), system=system, user=prompts.translate_user(t)) for i, t in batch]
            results = {r.key: r for r in self.teacher.generate(requests)}
            for i, _ in batch:
                r = results.get(str(i))
                text = prompts.reply_text(r.text) if r is not None and r.ok else ""
                out.append(text or None)
        return out


@TRANSLATORS.register("teacher", origin="builtin")
class TeacherTranslatorFactory:
    name = "teacher"

    def build(self, ctx: Any, teacher: Any) -> TeacherTranslator:
        cfg = ctx.config.teacher_for(ctx.name)
        return TeacherTranslator(teacher, ctx.config.language, log=ctx.log, batch_size=cfg.concurrency)
