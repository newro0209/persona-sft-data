"""말투: 반말 페르소나는 존댓말 종결을, 존댓말 페르소나는 반말 종결을 거절한다."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from persona_sft_data.core.gates import GateSettings, Verdict
from persona_sft_data.core.persona import Persona
from persona_sft_data.core.registry import RULES
from persona_sft_data.rules.base import assistant_texts, bad_value

HONORIFIC = re.compile(r"(요[.!?~…]?$|요\s|습니다|입니다|세요|십시오|하십|드립니다|드려요|예요|이에요)")

# 존댓말 페르소나의 판정은 화이트리스트다: 마지막 문장이 아래 종결 중 하나로 끝나야 한다.
# 한국어 존댓말 종결은 크게 세 갈래이고, 셋을 다 담지 않으면 '네.'·'아니오.' 같은 정상
# 존댓말이 거절되어 npc·lore 프로필(제약 표 기본 말투=존댓말)의 수율이 구조적으로 깎인다.
#  (1) 하십시오체 — '-습니다/-ㅂ니다', 의문 '-습니까/-ㅂ니까', 명령 '-십시오/-(으)시오'
#  (2) 해요체 — '-요/-죠'. '-세요·-셔요·-네요·-군요·-나요·-가요·-까요·-데요·-지요·-시죠'가
#      모두 여기로 수렴하지만, 왜 통과하는지 읽는 사람이 알도록 대표형을 함께 적는다.
#  (3) 단독 응답 '네/예/아니오/아니요' — 종결어미가 없어 (1)·(2)에 걸리지 않는다.
#      다만 '재밌네'처럼 '네'로 끝나는 반말을 통과시키면 안 되므로, 이 갈래는 발화
#      전체가 그 응답(과 감탄사)일 때만 인정한다. 아래 INTERJECTION_ONLY가 그 조건이다.
POLITE_ENDINGS = (
    "습니다", "ㅂ니다", "니다",                                  # (1) 서술
    "습니까", "ㅂ니까", "니까",                                  # (1) 의문
    "십시오", "시오",                                            # (1) 명령
    "세요", "셔요", "시죠", "군요", "네요", "나요", "가요",       # (2) 대표형
    "까요", "데요", "지요", "요", "죠",                          # (2) 해요체 일반
)
# 끝의 문장부호·닫는 따옴표는 종결 판정에서 뺀다.
TRAILING = r"""[.!?~…\s"'”’)\]]*$"""
HONORIFIC_END = re.compile("(?:" + "|".join(POLITE_ENDINGS) + ")" + TRAILING)

# 감탄사와 단독 응답만으로 된 짧은 발화. 종결어미가 없어 위 화이트리스트에 걸리지
# 않지만 반말도 아니다 — '네.', '예!', '아니오.', '음…', '어머!'는 존댓말 페르소나에서
# 정상이다. 반말의 대답('응', '어')과 반말 종결('좋아', '그래', '했어')은 넣지 않는다.
INTERJECTION = r"(?:아니오|아니요|네|예|아|음|흠|어머|아이고|아하|오호|저런|앗|하+|허+|호+|후+|오+)"
INTERJECTION_ONLY = re.compile(rf"^{INTERJECTION}(?:[\s,·…]*{INTERJECTION})*" + TRAILING)


def _is_honorific(text: str) -> bool:
    """존댓말 종결로 끝나거나, 감탄사·단독 응답만으로 된 짧은 발화인가."""
    text = text.strip()
    return bool(HONORIFIC_END.search(text) or INTERJECTION_ONLY.match(text))


@dataclass(frozen=True)
class InformalRule:
    name: str = "register"

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None:
        for text in assistant_texts(turns):
            if HONORIFIC.search(text):
                verdict.fail("honorific")


@dataclass(frozen=True)
class HonorificRule:
    name: str = "register"

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None:
        for text in assistant_texts(turns):
            if not _is_honorific(text):
                verdict.fail("informal_ending")


@RULES.register("register", origin="builtin")
class RegisterFactory:
    name = "register"
    constraint_key = "말투"

    def build(self, persona: Persona, value: str, settings: GateSettings):
        if value == "반말":
            return InformalRule()
        if value == "존댓말":
            return HonorificRule()
        if value in ("서술체", "자유"):
            return None
        raise bad_value(self.constraint_key, value, "반말 · 존댓말 · 서술체 · 자유")
