"""규칙들이 같이 쓰는 문자 클래스와 도우미."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from persona_sft_data.core.persona import PersonaError

HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
CJK = re.compile(r"[一-鿿㐀-䶿]")
KANA = re.compile(r"[぀-ヿ]")
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")


def assistant_texts(turns: Sequence[Mapping[str, str]]) -> list[str]:
    """페르소나 규칙은 assistant 발화에만 묶인다. 사용자는 어떤 말투든 쓴다."""
    return [t.get("text", "") for t in turns if t.get("role") == "assistant"]


def bad_value(key: str, value: str, allowed: str) -> PersonaError:
    return PersonaError(f"'## 제약'의 {key!r} 값 {value!r}은(는) 허용되지 않는다 (허용: {allowed})")
