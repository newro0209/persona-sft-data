"""프롬프트 조립 — Persona와 Profile을 교사 텍스트로 바꾸는 유일한 모듈.

금지만 나열한 프롬프트는 모델을 거절로 수렴시킨다는 것이 측정됐다(싫어 3/20).
그래서 모든 프롬프트가 발화 원칙과 선호 어휘를 금지와 함께 싣는다. 프로필 종류에
따른 분기문은 없다 — 라벨·프레이밍·추가 규칙은 전부 프로필 객체의 속성이다.
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
from typing import Any

from persona_sft_data.core.persona import SECTION_BACKGROUND, Persona

USER_TAG = "U:"
ASSISTANT_TAG = "A:"

LANGUAGE_NAMES = {"ko": "한국어", "en": "영어", "ja": "일본어", "zh": "중국어", "es": "스페인어",
                  "fr": "프랑스어", "de": "독일어"}


def language_name(code: str) -> str:
    return LANGUAGE_NAMES.get(code, code)


def _numbered(items: Sequence[str]) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))


def _bulleted(items: Sequence[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def persona_block(persona: Persona, profile: Any, *, vocabulary_sample: int = 0,
                  rng: random.Random | None = None) -> str:
    """캐릭터 설명. ``vocabulary_sample``만큼만 어휘 행을 보여 줘 코퍼스가 몇 구절로 수렴하지 않게 한다."""
    parts = ["[캐릭터]"]
    parts += [f"{k}: {v}" for k, v in persona.core.items()]
    if persona.background:
        parts += ["", f"[{SECTION_BACKGROUND}]", persona.background]
    parts += ["", "[말하는 방식]", _numbered(persona.principles)]
    if persona.vocabulary:
        rows = list(persona.vocabulary.items())
        if vocabulary_sample and vocabulary_sample < len(rows):
            rows = (rng or random).sample(rows, vocabulary_sample)
        vocab = "\n".join(f"- {emotion}: {', '.join(words)}" for emotion, words in rows)
        parts += ["", "[자주 쓰는 표현]", vocab,
                  "위 표현은 말투를 보여 주는 예시일 뿐이다. 그대로 베끼지 말고 같은 감정을 다른 말로 표현해라."]
    if persona.prohibitions:
        parts += ["", "[절대 하지 않는 것]", _bulleted(persona.prohibitions)]
    if persona.examples:
        parts += ["", "[예시 대화]", render_dialogue(persona.examples[0])]
    return "\n".join(parts)


def hard_rules(persona: Persona, profile: Any) -> str:
    """제약 표를 문장으로. 표에 없는 규칙은 말하지 않는다."""
    lines = []
    for key, value in persona.constraints.items():
        if key == "말투" and value in ("반말", "존댓말"):
            lines.append(f"- {profile.assistant_label}의 말은 항상 {value}이다.")
        elif key == "발화 길이":
            lines.append(f"- 한 발화는 {value}다. 넘기지 마라.")
        elif key == "문자" and value == "한글":
            lines.append("- 한글과 기본 문장부호만 쓴다. 한자, 영어, 이모지를 섞지 않는다.")
        elif key == "문자" and value == "영문":
            lines.append("- 영문과 기본 문장부호만 쓴다.")
        elif value == "금지":
            lines.append(f"- {key}: 쓰지 않는다.")
        elif key == "말줄임표":
            lines.append(f"- 말줄임표는 {value}.")
    lines += [f"- {rule}" for rule in profile.extra_rules]
    return "[반드시 지킬 것]\n" + "\n".join(lines)


def _output_format(persona: Persona, profile: Any) -> str:
    return f"""[출력 형식]
- 한 줄에 한 발화. {profile.user_label} 발화는 `{USER_TAG}`, {profile.assistant_label}({persona.name}) 발화는 `{ASSISTANT_TAG}`로 시작한다.
- **첫 줄은 반드시 `{USER_TAG}`다.** 마지막 줄은 반드시 `{ASSISTANT_TAG}`다. 두 역할이 정확히 번갈아 나온다.
- 설명, 번호, 제목, 따옴표를 붙이지 않는다. 대화만 쓴다."""


def dialogue_system(persona: Persona, profile: Any, rng: random.Random) -> str:
    return "\n\n".join([
        profile.writer_framing,
        persona_block(persona, profile, vocabulary_sample=4, rng=rng),
        hard_rules(persona, profile),
        _output_format(persona, profile),
    ])


def dialogue_user(persona: Persona, profile: Any, situation: str, flow: str, turns: int) -> str:
    return (f"상황: {situation}\n흐름: {flow}\n"
            f"길이: {profile.user_label} {turns}번, {profile.assistant_label} {turns}번 (총 {turns * 2}줄)\n\n"
            "이 상황의 대화를 하나 써라.")


def respond_system(persona: Persona, profile: Any, rng: random.Random) -> str:
    return "\n\n".join([
        f"{profile.user_label}가 한 말에 아래 캐릭터로서 한 번 답한다.",
        persona_block(persona, profile, vocabulary_sample=5, rng=rng),
        "[중요]\n- 상대의 말투가 어떻든 캐릭터의 말투를 유지한다.\n"
        "- 상대의 말이 캐릭터의 범위 밖이면 짧게 모른다고 말하고 캐릭터의 화제로 돌아온다. 아는 척하지 않는다.",
        hard_rules(persona, profile),
        f"[출력 형식]\n- 답변 한 줄만 쓴다. `{ASSISTANT_TAG}` 같은 표시도, 설명도 붙이지 않는다.",
    ])


def respond_user(text: str) -> str:
    return text


def translate_system(source_language: str, target_language: str) -> str:
    return (f"다음 {language_name(source_language)} 문장을 자연스러운 {language_name(target_language)} 구어체로 옮겨라.\n"
            "뜻만 옮긴다. 설명, 따옴표, 역할 표기 없이 한 줄만 쓴다.")


def translate_user(text: str) -> str:
    return text


# -- 교사 출력 파싱 -------------------------------------------------------------

def parse_dialogue(text: str) -> list[dict[str, str]]:
    """``U:``/``A:`` 줄을 turns로. 모양이 틀리면 추측하지 않고 ``[]``."""
    turns: list[dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line[:2].upper() == USER_TAG:
            role = "user"
        elif line[:2].upper() == ASSISTANT_TAG:
            role = "assistant"
        else:
            continue
        body = line[2:].strip().strip('"').strip("'")
        if not body:
            return []
        turns.append({"role": role, "text": body})
    return turns


def repair_dialogue(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    """같은 역할이 연달아 오면 한 발화로 합치고, 앞의 assistant·뒤의 user를 잘라 낸다. 만들어 내지는 않는다."""
    merged: list[dict[str, str]] = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1] = {"role": turn["role"], "text": f"{merged[-1]['text']} {turn['text']}".strip()}
        else:
            merged.append(dict(turn))
    start = 0
    while start < len(merged) and merged[start]["role"] != "user":
        start += 1
    end = len(merged)
    while end > start and merged[end - 1]["role"] != "assistant":
        end -= 1
    trimmed = merged[start:end]
    return trimmed if len(trimmed) >= 2 else []


def render_dialogue(turns: Sequence[dict[str, str]]) -> str:
    tag = {"user": USER_TAG, "assistant": ASSISTANT_TAG}
    return "\n".join(f"{tag[t['role']]} {t['text']}" for t in turns)


_LEADING_ROLE = re.compile(r"^\s*(U|A|P|사용자|유저|user|assistant|pet)\s*[:：]\s*", re.IGNORECASE)


def reply_text(raw: str | None) -> str:
    """한 줄만 달라고 했으니 첫 줄만. 역할 표기와 따옴표는 벗겨 낸다."""
    if not raw:
        return ""
    for line in raw.splitlines():
        line = line.strip()
        if line:
            return _LEADING_ROLE.sub("", line).strip().strip('"').strip("'").strip()
    return ""
