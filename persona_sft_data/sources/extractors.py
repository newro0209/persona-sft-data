"""행에서 발화를 뽑는 전략. 설정 dataclass가 각자의 옵션이다."""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from persona_sft_data.core.config import ConfigError
from persona_sft_data.core.registry import EXTRACTORS


def _texts(row: Mapping[str, Any], fields: Sequence[str]) -> Iterator[Any]:
    """선택한 열의 값을 순서대로. 없는 열과 ``None``은 건너뛴다."""
    for f in fields:
        value = row.get(f)
        if value is not None:
            yield value


@dataclass(frozen=True)
class FieldSettings:
    """옵션 없음. 모르는 키가 오면 ``build_settings``가 잡는다."""


@EXTRACTORS.register("field", origin="builtin")
class FieldExtractor:
    """선택한 열 각각의 문자열이 발화 하나."""

    name = "field"
    settings_type = FieldSettings

    def extract(self, row: Mapping[str, Any], fields: Sequence[str], settings: FieldSettings) -> Iterator[str]:
        for value in _texts(row, fields):
            if isinstance(value, str) and value.strip():
                yield value.strip()


@dataclass(frozen=True)
class RegexSettings:
    pattern: str
    group: int = 1


@lru_cache(maxsize=32)
def _compiled(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.DOTALL)


@EXTRACTORS.register("regex", origin="builtin")
class RegexExtractor:
    """열의 문자열에서 패턴의 그룹을 전부."""

    name = "regex"
    settings_type = RegexSettings

    def extract(self, row: Mapping[str, Any], fields: Sequence[str], settings: RegexSettings) -> Iterator[str]:
        for value in _texts(row, fields):
            for m in _compiled(settings.pattern).finditer(str(value)):
                text = m.group(settings.group).strip()
                if text:
                    yield text


@dataclass(frozen=True)
class ConversationSettings:
    role_key: str = "role"
    content_key: str = "content"
    include_roles: tuple[str, ...] = ()
    exclude_roles: tuple[str, ...] = ()


@EXTRACTORS.register("conversation", origin="builtin")
class ConversationExtractor:
    """열이 ``[{role, content}]`` 목록. include가 있으면 그 역할만, 아니면 exclude를 뺀 전부."""

    name = "conversation"
    settings_type = ConversationSettings

    def extract(self, row: Mapping[str, Any], fields: Sequence[str], settings: ConversationSettings) -> Iterator[str]:
        include = tuple(r.lower() for r in settings.include_roles)
        exclude = tuple(r.lower() for r in settings.exclude_roles)
        for turns in _texts(row, fields):
            if not isinstance(turns, list):
                continue
            for turn in turns:
                if not isinstance(turn, Mapping):
                    continue
                role = str(turn.get(settings.role_key, "")).lower()
                if include and role not in include:
                    continue
                if role in exclude:
                    continue
                text = str(turn.get(settings.content_key) or "").strip()
                if text:
                    yield text


@dataclass(frozen=True)
class ListSettings:
    keep: str = "even"


@EXTRACTORS.register("list", origin="builtin")
class ListExtractor:
    """열이 교대 화자의 문자열 목록. 짝수·홀수 인덱스 또는 전부."""

    name = "list"
    settings_type = ListSettings

    def extract(self, row: Mapping[str, Any], fields: Sequence[str], settings: ListSettings) -> Iterator[str]:
        if settings.keep not in ("even", "odd", "all"):
            raise ConfigError(f"list 추출기의 keep은 even·odd·all 중 하나다: {settings.keep!r}")
        for items in _texts(row, fields):
            if not isinstance(items, list):
                continue
            for i, item in enumerate(items):
                if settings.keep == "even" and i % 2 or settings.keep == "odd" and not i % 2:
                    continue
                text = str(item or "").strip()
                if text:
                    yield text
