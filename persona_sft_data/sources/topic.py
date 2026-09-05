"""주제 필터. 페르소나 문서(다룰 상황·어휘·배경)의 한글 바이그램이 신호다.

낱말이 아니라 바이그램인 것은 한국어가 조사·어미를 어간에 붙이기 때문이다: 배고파,
배고픈데, 배고프다는 배고픔과 낱말을 공유하지 않지만 배고는 공유한다. 거친
필터이고, 답이 페르소나 안에 머무는지는 게이트가 본다.
"""

from __future__ import annotations

import re

from persona_sft_data.core.persona import Persona

_HANGUL = re.compile(r"[가-힣]")


def bigrams(text: str) -> frozenset[str]:
    """한글 연속 구간 안의 두 글자 조각 전부."""
    return frozenset(
        run[i:i + 2]
        for run in re.findall(r"[가-힣]+", text)
        for i in range(len(run) - 1)
    )


def signal(persona: Persona) -> frozenset[str]:
    """다룰 상황의 낱개 순간, 어휘 표의 라벨과 예시, 배경의 바이그램 합집합."""
    words: list[str] = list(persona.beats)
    for label, examples in persona.vocabulary.items():
        words.append(label)
        words.extend(examples)
    if persona.background:
        words.append(persona.background)
    out: set[str] = set()
    for w in words:
        out |= bigrams(w)
    return frozenset(out)


def in_scope(text: str, signal_set: frozenset[str], *, min_hits: int = 1,
             min_chars: int = 2, max_chars: int = 60) -> bool:
    """길이가 범위 안이고 한글이 있고 신호 바이그램이 ``min_hits``개 이상이면 안."""
    if not min_chars <= len(text) <= max_chars:
        return False
    if not _HANGUL.search(text):
        return False
    return len(bigrams(text) & signal_set) >= min_hits
