"""Prompt assembly — the only module that turns a Persona into teacher text.

Prompts live here rather than in each stage so that a per-model branch
(``if prompt_profile == "exaone35"`` appeared at three places in the previous
pipeline) has nowhere to grow. Stages ask for a prompt; they do not write one.

One measured lesson shapes this file. A first spike used a system prompt that
listed only prohibitions — no 존댓말, no emoji, no third person. Violations fell
from 5/20 to 1-2/20, but the replies collapsed into refusals: 싫어 three times
in twenty. Listing what is forbidden without what is wanted teaches the model to
say as little as possible. So every prompt built here carries the persona's
speech principles and preferred vocabulary alongside its bans.
"""

from __future__ import annotations

import random

from persona_sft_data.persona import Persona
from persona_sft_data.schema import PET_TAG, USER_TAG


def _numbered(items) -> str:
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, 1))


def _bulleted(items) -> str:
    return "\n".join(f"- {item}" for item in items)


def persona_block(persona: Persona, *, vocabulary_sample: int = 0,
                  rng: random.Random | None = None) -> str:
    """The shared description of who the pet is.

    ``vocabulary_sample`` shows only that many emotion rows. Showing all ten
    every time pushes the model toward the same handful of phrases; rotating a
    subset keeps the corpus from converging on them.
    """
    rows = list(persona.vocabulary.items())
    if vocabulary_sample and vocabulary_sample < len(rows):
        rows = (rng or random).sample(rows, vocabulary_sample)
    vocab = "\n".join(f"- {emotion}: {', '.join(words)}" for emotion, words in rows)
    # A first probe copied these verbatim: "네 옆에 있을게" and "불러 줘서 좋아"
    # came back word for word, and 히히 appeared in three of six dialogues. The
    # persona document warns against exactly this convergence, so say so.
    vocab += (
        "\n\n위 표현은 말투를 보여 주는 예시일 뿐이다. 그대로 베끼지 말고 "
        "같은 감정을 다른 말로 표현해라."
    )

    return f"""[캐릭터]
이름: {persona.name}
정체성: {persona.identity}
관계: {persona.core['사용자와의 관계']}
말투: {persona.speech}
성격: {persona.personality}
응답 길이: {persona.length_rule}
지식 범위: {persona.core['지식 범위']}

[말하는 방식]
{_numbered(persona.principles)}

[자주 쓰는 표현]
{vocab}

[절대 하지 않는 것]
{_bulleted(persona.prohibitions)}"""


def _hard_rules(persona: Persona) -> str:
    """The two things a probe showed the teacher gets wrong most expensively.

    Length: the persona's 4~35 character rule was present but buried in the
    character block, and replies ran to 40+ characters with stage directions
    ("잠깐 고개를 갸우뚱하더니...") that the persona forbids outright.

    Script: one probe emitted Chinese — 宠당하고, 宠받고 — into a Korean-only
    corpus. The filter catches it either way, but generating rejects burns GPU
    time, so say it up front.
    """
    lo, hi = persona.utterance_char_range()
    return f"""[반드시 지킬 것]
- 한 발화는 {lo}~{hi}글자다. 넘기지 마라. 짧을수록 좋다.
- 한글과 기본 문장부호만 쓴다. 한자, 영어, 이모지를 절대 섞지 않는다.
- 대사만 쓴다. 행동이나 표정 묘사를 넣지 않는다."""


# --- seed: the reasoning teacher writes scenario skeletons -----------------

def seed_system(persona: Persona, rng: random.Random) -> str:
    return f"""너는 한국어 대화 데이터를 만드는 작가다. 아래 캐릭터가 사용자와
주고받는 짧은 대화를 쓴다.

{persona_block(persona, vocabulary_sample=4, rng=rng)}

{_hard_rules(persona)}

[출력 형식]
- 한 줄에 한 발화. 사용자 발화는 `U:`, {persona.name}의 발화는 `P:`로 시작한다.
- **첫 줄은 반드시 `U:`다.** 사용자가 먼저 말을 걸고 {persona.name}가 답한다.
- 마지막 줄은 반드시 `P:`다. 두 역할이 정확히 번갈아 나온다.
- 설명, 번호, 제목, 따옴표를 붙이지 않는다. 대화만 쓴다."""


def seed_user(persona: Persona, situation: str, turns: int,
              rng: random.Random) -> str:
    tone = rng.choice([
        "사용자가 다정하게 말을 거는 흐름",
        "사용자가 무심하게 툭 던지는 흐름",
        "사용자가 걱정하며 묻는 흐름",
        "사용자가 장난스럽게 구는 흐름",
        f"{persona.name}가 먼저 원하는 것을 말하고 사용자가 반응하는 흐름",
        f"{persona.name}가 부탁을 거절하고 이유를 짧게 말하는 흐름",
    ])
    return f"""상황: {situation}
흐름: {tone}
길이: 사용자 {turns}번, {persona.name} {turns}번 (총 {turns * 2}줄)

이 상황의 대화를 하나 써라."""


# --- expand: the bulk teacher varies a skeleton ----------------------------

def expand_system(persona: Persona, rng: random.Random) -> str:
    return f"""너는 한국어 대화를 자연스럽게 바꿔 쓰는 작가다. 주어진 대화와
같은 뜻, 같은 감정을 유지하면서 표현만 다르게 바꾼다.

{persona_block(persona, vocabulary_sample=4, rng=rng)}

[바꿀 것]
- 어휘와 문장 구조를 바꾼다. 같은 문장을 그대로 두지 않는다.

[바꾸지 않을 것]
- 상황, 감정, 누가 무엇을 원하는지
- 발화 수와 순서

{_hard_rules(persona)}

[출력 형식]
- 한 줄에 한 발화. 사용자 발화는 `U:`, {persona.name}의 발화는 `P:`로 시작한다.
- **첫 줄은 반드시 `U:`다.** 마지막 줄은 반드시 `P:`다.
- 원본과 같은 줄 수를 유지한다. 줄을 더하거나 빼지 않는다.
- 설명, 번호, 제목, 따옴표를 붙이지 않는다. 대화만 쓴다."""


def expand_user(dialogue: str) -> str:
    """Show the source and demand both sides back.

    Asking only "위 대화를 다르게 표현해라" made the model rewrite the pet's
    lines and drop the user's entirely — a quarter of the replies came back as
    P: lines only, which is not a dialogue.
    """
    lines = dialogue.count("\n") + 1
    return (
        f"{dialogue}\n\n"
        f"위 대화를 다르게 표현해라. **사용자 발화(`U:`)와 펫 발화(`P:`) 양쪽을 모두** "
        f"다시 써야 한다. 정확히 {lines}줄, `U:`로 시작해 `P:`로 끝난다."
    )


# --- real: give a human-written utterance a persona-consistent reply -------

def real_system(persona: Persona, rng: random.Random) -> str:
    return f"""사용자가 한 말에 아래 캐릭터로서 한 번 답한다.

{persona_block(persona, vocabulary_sample=5, rng=rng)}

[중요]
- 사용자의 말이 존댓말이어도 너는 반말로 답한다.
- 사용자의 말이 펫의 생활 범위 밖이면, 짧게 모른다고 말하고 가까운 일상 화제로
  돌아온다. 아는 척하지 않는다.

{_hard_rules(persona)}

[출력 형식]
- 답변 한 줄만 쓴다. `P:` 같은 표시도, 설명도 붙이지 않는다."""


def real_user(utterance: str) -> str:
    return utterance


# --- parsing the teacher's output -----------------------------------------

def parse_dialogue(text: str) -> list[dict[str, str]]:
    """Turn ``U:``/``P:`` lines into schema turns.

    Returns ``[]`` when the shape is wrong rather than guessing, so the caller
    records a reject with a reason instead of writing a malformed session.
    """
    turns: list[dict[str, str]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("U:", "u:")):
            role = "user"
        elif line.startswith(("P:", "p:")):
            role = "pet"
        else:
            continue
        body = line[2:].strip().strip('"').strip("'")
        if not body:
            return []
        turns.append({"role": role, "text": body})
    return turns


def repair_dialogue(turns: list[dict[str, str]]) -> list[dict[str, str]]:
    """Trim a dialogue back to a well-formed user→…→pet exchange.

    Even with the format spelled out, a third of expand's replies came back
    with a leading pet turn or an unpaired trailing one. The content of those
    is fine; only the framing is off, and discarding a generated dialogue over
    its first line wastes the expensive part.

    So: drop pet turns before the first user turn, drop user turns after the
    last pet turn. Nothing is invented and no turn is reordered — if what
    remains still does not alternate, the caller rejects it.
    """
    # Consecutive same-role lines are one utterance the model split across two.
    # Trimming alone cannot fix them, and rejecting on them cost more than the
    # leading/trailing faults did -- so join them and let the gate rule on the
    # result's length.
    merged: list[dict[str, str]] = []
    for turn in turns:
        if merged and merged[-1]["role"] == turn["role"]:
            merged[-1] = {
                "role": turn["role"],
                "text": f"{merged[-1]['text']} {turn['text']}".strip(),
            }
        else:
            merged.append(dict(turn))

    start = 0
    while start < len(merged) and merged[start]["role"] != "user":
        start += 1
    end = len(merged)
    while end > start and merged[end - 1]["role"] != "pet":
        end -= 1
    trimmed = merged[start:end]
    return trimmed if len(trimmed) >= 2 else []


def render_dialogue(turns) -> str:
    """The inverse, for showing a teacher an existing dialogue."""
    tag = {"user": "U:", "pet": "P:"}
    return "\n".join(f"{tag[t['role']]} {t['text']}" for t in turns)


__all__ = [
    "persona_block",
    "seed_system",
    "seed_user",
    "expand_system",
    "expand_user",
    "real_system",
    "real_user",
    "parse_dialogue",
    "repair_dialogue",
    "render_dialogue",
    "PET_TAG",
    "USER_TAG",
]
