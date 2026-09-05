"""금칙어. 사람 쪽 발화는 게이트에 묶이지 않으므로 여기서 바닥만 깐다.

실측으로 고른 목록이다: 두 소스 12,409문장에서 후보 43건 중 대부분이 어설픈 어간이
평범한 낱말(새끼곰, 가시발새우)을 잡은 것이었고, 남은 열 건 남짓이 욕설과 성적
표현이었다. 그래서 목록은 짧고, 무해한 해석이 없는 어간만 있으며, 토큰 앞부분에서만
맞춘다. 분류기가 아니라 바닥이다. 설정의 ``blocked_stems``가 통째로 대체한다.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

DEFAULT_STEMS: tuple[str, ...] = (
    "씨발", "시발", "개새", "병신", "좆", "존나", "지랄", "썅", "니미",
    "미친년", "미친놈", "야동", "섹스", "강간",
)


def is_unsafe(text: str, stems: Sequence[str] = DEFAULT_STEMS) -> bool:
    """한글 토큰의 앞부분이 어간 하나와 맞으면 위험."""
    for token in re.split(r"[^가-힣]+", text):
        if token and any(token.startswith(stem) for stem in stems):
            return True
    return False
