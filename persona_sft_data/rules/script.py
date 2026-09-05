"""문자: 한글 페르소나는 한자·가나·영단어를, 영문 페르소나는 한글·한자·가나를 거절한다."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from persona_sft_data.core.gates import GateSettings, Verdict
from persona_sft_data.core.persona import Persona
from persona_sft_data.core.registry import RULES
from persona_sft_data.rules.base import CJK, HANGUL, KANA, LATIN_WORD, assistant_texts, bad_value


@dataclass(frozen=True)
class HangulRule:
    name: str = "script"

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None:
        for text in assistant_texts(turns):
            if CJK.search(text):
                verdict.fail("cjk_characters")
            if KANA.search(text):
                verdict.fail("kana_characters")
            if LATIN_WORD.search(text):
                verdict.fail("latin_words")
            if not HANGUL.search(text):
                verdict.fail("no_hangul")


@dataclass(frozen=True)
class LatinRule:
    name: str = "script"

    def check(self, turns: Sequence[Mapping[str, str]], verdict: Verdict) -> None:
        for text in assistant_texts(turns):
            if HANGUL.search(text):
                verdict.fail("hangul_characters")
            if CJK.search(text):
                verdict.fail("cjk_characters")
            if KANA.search(text):
                verdict.fail("kana_characters")


@RULES.register("script", origin="builtin")
class ScriptFactory:
    name = "script"
    constraint_key = "문자"

    def build(self, persona: Persona, value: str, settings: GateSettings):
        if value == "한글":
            return HangulRule()
        if value == "영문":
            return LatinRule()
        if value == "혼용":
            return None
        raise bad_value(self.constraint_key, value, "한글 · 영문 · 혼용")
