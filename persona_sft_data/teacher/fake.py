"""GPU 없는 교사 둘.

``FakeTeacher``는 테스트가 키별 답을 주입한다. ``EchoTeacher``는 스모크 설정용으로,
프롬프트 모양만 보고 형식에 맞는 대화·한 줄 답·한글 "번역"을 돌려준다. 품질은
없고 형식만 있다 — 파이프라인 전체가 교사 없이 끝까지 도는지 보는 용도다.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

from persona_sft_data.core.config import TeacherConfig
from persona_sft_data.core.registry import TEACHERS
from persona_sft_data.teacher.base import Request, Result

_HANGUL = re.compile(r"[가-힣]")
_LINES = re.compile(r"길이:.*?(\d+)번")


class FakeTeacher:
    def __init__(self, replies: dict[str, str] | None = None, *, default: str = "",
                 reply_fn: Callable[[Request], str] | None = None) -> None:
        self.name = "fake"
        self.replies = replies or {}
        self.default = default
        self.reply_fn = reply_fn
        self.seen: list[Request] = []
        self.checked = False  # 단계가 생성 전에 check()를 불렀는지 테스트가 본다

    def check(self) -> None:
        self.checked = True

    def generate(self, requests: Sequence[Request]) -> list[Result]:
        out = []
        for req in requests:
            self.seen.append(req)
            text = self.reply_fn(req) if self.reply_fn else self.replies.get(req.key, self.default)
            out.append(Result(req.key, text, completion_tokens=len(text)))
        return out


class EchoTeacher:
    """프롬프트 모양으로 응답 종류를 고른다: 대화 요청·번역 요청·그 밖의 한 줄 답."""

    def __init__(self, name: str) -> None:
        self.name = name

    def check(self) -> None:
        return None

    @staticmethod
    def _reply(req: Request) -> str:
        first = req.user.strip().splitlines()[0] if req.user.strip() else ""
        m = _LINES.search(req.user)
        if first.startswith("상황:") and m:
            situation = first.split(":", 1)[1].strip()
            lines = []
            for i in range(int(m.group(1))):
                lines.append(f"U: {situation} 어때?" if i == 0 else f"U: 그리고 {situation}은 어때?")
                lines.append(f"A: 응, {situation} 좋아." if i == 0 else "A: 응, 조금 더 하고 싶어.")
            return "\n".join(lines)
        text = req.user.strip().splitlines()[-1] if req.user.strip() else ""
        if not _HANGUL.search(text):
            return f"같이 놀자, {len(text)}번째 말이야."
        return f"응, {text[:12]} 좋아."

    def generate(self, requests: Sequence[Request]) -> list[Result]:
        return [Result(r.key, self._reply(r), completion_tokens=8) for r in requests]


@TEACHERS.register("fake", origin="builtin")
class FakeFactory:
    name = "fake"

    def build(self, cfg: TeacherConfig) -> EchoTeacher:
        return EchoTeacher(cfg.name)
