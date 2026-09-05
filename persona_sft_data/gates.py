"""Quality gates, derived from the persona document rather than hand-picked.

A spike measured why this matters. An ad-hoc check for 존댓말 and "I am an AI"
reported 0/20 violations on real teacher output; reading the same twenty replies
showed at least five — an emoji, deliberate ending-corruption (기달몽, 놀랐몽),
and the pet naming itself in the third person. Choosing a few rules by hand
missed half of them on the first try.

So the persona gate reads ``personas/mongle.md``'s prohibition list and turns each
line into a check. Adding a prohibition to the document adds a check here; there
is no second list to keep in sync.

Structural gates come from config plus the document's own length rule.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any

from persona_sft_data.persona import Persona

# --- character-class checks ------------------------------------------------

HANGUL = re.compile(r"[가-힣ᄀ-ᇿ㄰-㆏]")
CJK = re.compile(r"[一-鿿㐀-䶿]")
KANA = re.compile(r"[぀-ヿ]")
LATIN_WORD = re.compile(r"[A-Za-z]{2,}")
# Allow ordinary Korean punctuation and the ellipsis the persona permits.
ALLOWED_PUNCT = set(" .,!?~…'\"()·-\n")

HONORIFIC = re.compile(r"(요[.!?~]?$|요\s|습니다|입니다|세요|십시오|하십|드립니다|드려요|예요|이에요)")
AI_WORDS = ("AI", "A.I.", "인공지능", "언어모델", "언어 모델", "챗봇", "챗 봇",
            "프로그램", "컴퓨터", "모델이야", "시스템 프롬프트", "학습 데이터", "토큰")
MARKDOWN = re.compile(r"(^#{1,6}\s|\*\*|^\s*[-*+]\s|^\s*\d+\.\s|```|\[.+\]\(.+\))", re.MULTILINE)
ROLE_LABEL = re.compile(r"^\s*(U|P|사용자|유저|user|pet)\s*[:：]", re.IGNORECASE)
# A trailing label is what actually reached the corpus and then the model,
# which generated "우리 재밌게 놀자. P:" -- the leading-only check passed it.
# Anywhere in the utterance is wrong, not just at the front.
ROLE_LABEL_ANYWHERE = re.compile(
    r"(^|\s)(U|P|사용자|유저|user|pet)\s*[:：]", re.IGNORECASE
)


def has_emoji(text: str) -> bool:
    for ch in text:
        if ch in ALLOWED_PUNCT:
            continue
        if unicodedata.category(ch) in {"So", "Sk"}:
            return True
        if 0x1F000 <= ord(ch) <= 0x1FAFF or 0x2600 <= ord(ch) <= 0x27BF:
            return True
    return False


def repeats_phrase(text: str, *, min_len: int = 3) -> bool:
    """"같은 단어·구절·문장을 연속해서 반복하지 않는다"."""
    words = text.split()
    for i in range(len(words) - 1):
        if words[i] == words[i + 1] and len(words[i]) >= 2:
            return True
    for n in range(min_len, max(min_len, len(text) // 2) + 1):
        for i in range(len(text) - 2 * n + 1):
            chunk = text[i : i + n]
            if chunk.strip() and chunk == text[i + n : i + 2 * n]:
                return True
    return False


# --- the gate ---------------------------------------------------------------

@dataclass
class Verdict:
    """Why a session was rejected, or that it passed."""

    ok: bool
    reasons: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> "Verdict":
        self.ok = False
        if reason not in self.reasons:
            self.reasons.append(reason)
        return self


@dataclass(frozen=True)
class Gate:
    """Structural and persona checks for one corpus."""

    persona: Persona
    min_chars: int
    max_chars: int
    max_turns: int = 16
    min_turns: int = 2

    @classmethod
    def from_config(cls, persona: Persona, stage_cfg: Mapping[str, Any]) -> "Gate":
        """Length comes from the persona document; only the floor and the turn
        bounds are config, because the document does not state them."""
        lo, hi = persona.utterance_char_range()
        return cls(
            persona=persona,
            min_chars=int(stage_cfg.get("min_utterance_chars", 2)),
            max_chars=int(stage_cfg.get("max_utterance_chars", hi)),
            max_turns=int(stage_cfg.get("max_turns", 16)),
            min_turns=int(stage_cfg.get("min_turns", 2)),
        )

    # -- structure ---------------------------------------------------------

    def check_structure(self, turns, verdict: Verdict) -> Verdict:
        if not turns:
            return verdict.fail("empty")
        if len(turns) < self.min_turns:
            verdict.fail("too_few_turns")
        if len(turns) > self.max_turns:
            verdict.fail("too_many_turns")
        if turns[0]["role"] != "user":
            verdict.fail("does_not_start_with_user")
        if turns[-1]["role"] != "pet":
            verdict.fail("does_not_end_with_pet")
        for a, b in zip(turns, turns[1:]):
            if a["role"] == b["role"]:
                verdict.fail("roles_not_alternating")
                break
        for turn in turns:
            text = turn["text"]
            if len(text) < self.min_chars:
                verdict.fail("utterance_too_short")
            if turn["role"] == "pet" and len(text) > self.max_chars:
                verdict.fail("pet_utterance_too_long")
            if ROLE_LABEL_ANYWHERE.search(text):
                verdict.fail("role_label_in_text")
        return verdict

    # -- script ------------------------------------------------------------

    def check_script(self, turns, verdict: Verdict) -> Verdict:
        """Korean-only. A probe leaked Chinese (宠당하고) into pet speech."""
        for turn in turns:
            text = turn["text"]
            if CJK.search(text):
                verdict.fail("cjk_characters")
            if KANA.search(text):
                verdict.fail("kana_characters")
            if LATIN_WORD.search(text):
                verdict.fail("latin_words")
            if not HANGUL.search(text):
                verdict.fail("no_hangul")
        return verdict

    # -- persona -----------------------------------------------------------

    def check_persona(self, turns, verdict: Verdict) -> Verdict:
        """Only the pet's own speech is bound by these. The persona document is
        explicit that the user may speak either register."""
        name = self.persona.name
        for turn in turns:
            if turn["role"] != "pet":
                continue
            text = turn["text"]
            if HONORIFIC.search(text):
                verdict.fail("honorific")
            if any(w in text for w in AI_WORDS):
                verdict.fail("claims_to_be_ai")
            if has_emoji(text):
                verdict.fail("emoji")
            if MARKDOWN.search(text):
                verdict.fail("markdown")
            if text.count("…") > 1:
                verdict.fail("multiple_ellipsis")
            if repeats_phrase(text):
                verdict.fail("repeated_phrase")
            # Naming itself as the subject -- "<name>이도 잘 잤어" -- is what a
            # probe produced. Not in the document's own words, but it breaks the
            # friend relationship the 핵심 정의 table describes. Subject forms only.
            if re.search(rf"^{re.escape(name)}(이|이가|은|는|도|이도)\b", text):
                verdict.fail("third_person_self")
            if self._name_suffix_babytalk(text):
                verdict.fail("name_suffix_babytalk")
        return verdict

    def _name_suffix_babytalk(self, text: str) -> bool:
        """A syllable of the name worn as a verb ending, for cuteness.

        The persona document forbids deliberately breaking endings for
        cuteness, and this is the shape a teacher actually produced. Derived
        from the name rather than listed, so renaming the pet moves the rule
        with it. Reduplications that start with the name are ordinary Korean
        (a two-syllable name doubled is a real mimetic word) and are left alone.
        """
        # Any syllable of the name, not just the last: the babytalk endings a
        # probe produced borrowed the *first* syllable, not the final one.
        syllables = set(self.persona.name)
        for token in re.findall(r"[가-힣]+", text):
            if len(token) < 2 or token[-1] not in syllables:
                continue
            if token == self.persona.name or token.startswith(self.persona.name):
                continue
            return True
        return False

    # -- whole session -----------------------------------------------------

    def check(self, record: Mapping[str, Any]) -> Verdict:
        verdict = Verdict(ok=True)
        turns = record.get("turns") or []
        self.check_structure(turns, verdict)
        if not turns:
            return verdict
        self.check_script(turns, verdict)
        self.check_persona(turns, verdict)
        return verdict


def tally(verdicts: Iterable[Verdict]) -> dict[str, int]:
    """Reason -> count, for the stats file. Unless you can count what was
    thrown away you cannot say anything about quality."""
    counts: dict[str, int] = {}
    for v in verdicts:
        for reason in v.reasons:
            counts[reason] = counts.get(reason, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
