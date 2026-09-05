"""교사 인터페이스와 요청·결과 타입.

파이프라인과 모델 사이의 경계는 하나다: OpenAI 호환 chat-completions HTTP. 어떤
백엔드든 ``TeacherFactory``로 등록하면 설정의 ``kind`` 한 줄로 바꿔 끼운다.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass


class TeacherError(RuntimeError):
    """서버가 설정이 요구한 것을 서빙하지 못한다."""


@dataclass(frozen=True)
class Request:
    """생성 하나. ``key``가 따라다녀서 결과를 순서가 아니라 키로 맞춘다."""

    key: str
    system: str
    user: str
    max_tokens: int | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class Result:
    key: str
    text: str | None
    completion_tokens: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.text is not None


def batched(items: Iterable, size: int) -> Iterator[list]:
    """최대 ``size``개씩. 메모리를 묶는 것이지 동시성을 제한하는 것이 아니다."""
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
