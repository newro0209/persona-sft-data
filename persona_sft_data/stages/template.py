"""The template stage: rule-based coverage of the persona's situations.

No teacher runs here. This slice exists so that every situation the persona
document lists is present in the corpus in text that is clean by construction —
if a beat is under-represented after the teacher stages, this is the one place
that can be told to cover it without another eight million tokens of GPU time.

**Everything the pet says is assembled out of the persona document.** The 감정
표현과 어휘 table supplies the phrases and 다룰 상황 supplies the beats, so
there is not one line of pet speech typed into this file. That is the whole
point: the previous pipeline's ``templates.py`` held 17 persona strings and 200
hand-written pet sentences, and editing the frozen document left them behind.

How a record is built:

1. Pick one line of 다룰 상황 and walk one to three of its beats in document
   order. Those lines are written as a progression — 배고픔, 밥 요청, 먹는 중,
   배부름 — so keeping the order gives a session that moves the way the document
   says the situation moves.
2. For each beat, find the phrase the persona would reach for. The beat's
   character bigrams are matched against the table's example phrases first
   (심심함 finds 심심해), and against the emotion labels second — a label names
   two poles, 배고픔·배부름, and the document lists its phrases in that order, so
   a beat matching the second pole is answered from the second half of the row.
   A beat with no emotional signal at all — 첫 만남, 자기소개 — falls back to the
   table's opening row, which is the calm one and contradicts nothing.
3. Compose the reply from that phrase alone, or pair it with a phrase from the
   calm row: 응, 배고파. Nothing else is ever paired with it, because the only
   thing known to be compatible with a topical phrase is a neutral
   acknowledgement — 이제 배불러, 밥 줘 would be assembled from one row and mean
   the opposite of itself. Phrases sharing a bigram are never joined either,
   which keeps the gate's repetition check happy.
4. The user's side is a neutral frame with the beat dropped into it as the
   topic. The frames carry no persona voice — they are the plainest way to raise
   a subject — and the particle in each is computed from the beat's final
   syllable rather than picked, because a wrong 은/는 is the loudest kind of
   unnatural Korean. The user turns here are deliberately plain; naturalness on
   the human side is what the real and teacher slices are for.

Long beats are skipped. A line like 즉각적인 위험이나 심각한 고통에 대한 짧은
도움 요청 권고 describes a whole scene rather than naming a topic, and no
slot-filled frame can raise it without producing nonsense — worse, the safety
answer it needs is not in the vocabulary table. The teacher stages cover those.
"""

from __future__ import annotations

import collections
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Callable

from persona_sft_data.persona import Persona
from persona_sft_data.runner import StageContext

# A beat longer than this is a description of a scene, not a topic a one-line
# frame can raise. See the module docstring.
MAX_BEAT_CHARS = 12

# How many exchanges a session gets, sampled from this bag. Weighted short: the
# persona's 응답 길이 rule is one sentence, and a long slot-filled session reads
# like a form.
EXCHANGE_SHAPE = (1, 1, 1, 2, 2, 3)

# Consecutive duplicates that mean the combination space is used up. The runner
# would reject the duplicates anyway; stopping here keeps them out of the reject
# file, where they would drown the reasons that matter.
STALL_LIMIT = 2000

_TERMINAL = ".?!~…"


# --- Korean helpers --------------------------------------------------------

def _bigrams(text: str) -> frozenset[str]:
    """Character bigrams inside each run of Hangul in ``text``.

    Bigrams rather than words because Korean glues endings onto the stem that
    carries the meaning: the beat 심심함 and the phrase 심심해 share no word and
    do share 심심.

    Runs rather than the whole string because gluing across a space invents
    bigrams that were never written. It is not hypothetical — 놀이 제안 spans
    into 이제, which is how a first version answered a request to play with
    이제 배불러.
    """
    return frozenset(
        run[i : i + 2]
        for run in re.findall(r"[가-힣]+", text)
        for i in range(len(run) - 1)
    )


def _shares_bigram(a: str, b: str) -> bool:
    return bool(_bigrams(a) & _bigrams(b))


def _josa(word: str, with_final: str, without_final: str) -> str:
    """Choose 은/는, 이/가 or 을/를 by the last syllable's 받침.

    Computed rather than tabulated: the beats come out of a document that can be
    edited, so a particle chosen by hand would be wrong the first time a beat
    changes. The modulo is the Hangul syllable block's own arithmetic — every
    syllable is (initial, vowel, final) packed from U+AC00, and a remainder of
    zero means no final consonant.
    """
    for ch in reversed(word):
        if "가" <= ch <= "힣":
            return with_final if (ord(ch) - 0xAC00) % 28 else without_final
    return without_final


# The user side. Neutral frames only — no persona voice, no emotion, nothing
# that presumes what the pet will answer. The topic is always the beat.
USER_FRAMES: tuple[Callable[[str], str], ...] = (
    lambda beat: f"{beat}{_josa(beat, '은', '는')} 어때?",
    lambda beat: f"지금 {beat}{_josa(beat, '은', '는')} 어때?",
    lambda beat: f"{beat}{_josa(beat, '이', '가')} 궁금해.",
    lambda beat: f"{beat} 얘기해 줘.",
    lambda beat: f"우리 {beat} 얘기하자.",
    lambda beat: f"오늘 {beat}{_josa(beat, '은', '는')} 어땠어?",
    lambda beat: f"{beat}{_josa(beat, '을', '를')} 말해 줄래?",
    lambda beat: f"{beat}, 지금 어떤 기분이야?",
    lambda beat: f"{beat} 생각하고 있었어.",
    lambda beat: f"{beat} 얘기 좀 하자.",
)


# --- persona -> slots ------------------------------------------------------

def beat_groups(persona: Persona) -> tuple[tuple[str, ...], ...]:
    """The beats of 다룰 상황, grouped by the line they came from.

    ``Persona.beats`` flattens the same split; the grouping is what lets one
    session walk consecutive moments of a single situation instead of jumping
    between unrelated ones. Anything the property does not recognise as a beat
    is dropped rather than invented here, so the two views cannot disagree.
    """
    known = set(persona.beats)
    groups: list[tuple[str, ...]] = []
    for line in persona.situations:
        beats = tuple(
            beat
            for beat in (part.strip() for part in line.split(","))
            if beat in known and len(beat) <= MAX_BEAT_CHARS
        )
        if beats:
            groups.append(beats)
    return tuple(groups)


def _share(phrases: Sequence[str], index: int, parts: int) -> tuple[str, ...]:
    """The slice of a row that belongs to one part of its label.

    A label names its poles in the order the row lists their phrases —
    배고픔·배부름 over 배고파, 꼬르륵, 밥 줘, 이제 배불러 — so the second pole is
    answered from the second half. The mapping is approximate and the caller
    narrows it further; what it buys is that 배부름 is never answered from the
    half of the row that means the opposite.
    """
    count = len(phrases)
    share = tuple(phrases[index * count // parts : (index + 1) * count // parts])
    return share or tuple(phrases)


def _narrow(beat: str, phrases: Sequence[str]) -> tuple[str, ...]:
    """Keep the phrases that share the most characters with the beat.

    Single characters, where the bigram match already failed. It is a weak
    signal used only to order phrases that are all plausible anyway, and it is
    what separates 이제 배불러 from 밥 줘 for the beat 배부름.
    """
    letters = set(re.sub(r"[^가-힣]", "", beat))
    scored = [(len(letters & set(phrase)), phrase) for phrase in phrases]
    best = max((score for score, _ in scored), default=0)
    if not best:
        return tuple(phrases)
    return tuple(phrase for score, phrase in scored if score == best)


def heads_for(beat: str, vocabulary: dict[str, tuple[str, ...]]) -> tuple[str, ...]:
    """The phrases of 감정 표현과 어휘 that may lead this beat's reply.

    Example phrases are matched before labels because they are the specific
    signal: 궁금해하기 finds 궁금해 in the 놀람·호기심 row even though the label
    says nothing about it. A matched phrase leads on its own — that is what
    keeps 다시 만남, which reaches the 삐짐·화해 row only through 다시 같이 놀자,
    from answering with 조금 삐졌어.
    """
    grams = _bigrams(beat)

    best_score = 0
    anchor: str | None = None
    for phrases in vocabulary.values():
        for phrase in phrases:
            score = len(grams & _bigrams(phrase))
            if score > best_score:
                best_score, anchor = score, phrase
    if anchor is not None:
        return (anchor,)

    best: tuple[str, ...] = ()
    for label, phrases in vocabulary.items():
        parts = [part.strip() for part in label.split("·") if part.strip()]
        for index, part in enumerate(parts):
            score = len(grams & _bigrams(part))
            if score > best_score:
                best_score = score
                best = _narrow(beat, _share(phrases, index, len(parts)))
    if best:
        return best

    # No emotional signal at all — 첫 만남, 자기소개, 관계 확인. The 감정 표현과
    # 어휘 table opens with the calm row, so that is the safe default; a random
    # row would answer a greeting with 아파.
    return next(iter(vocabulary.values()))


def _can_lead(text: str) -> bool:
    """A phrase that already ends a sentence cannot be the first half of one."""
    return bool(text) and text[-1] not in _TERMINAL


def _finish(text: str) -> str:
    return text if text and text[-1] in _TERMINAL else f"{text}."


def replies_for(
    beat: str,
    vocabulary: dict[str, tuple[str, ...]],
    *,
    min_chars: int,
    max_chars: int,
) -> tuple[str, ...]:
    """Every pet utterance this beat can be answered with.

    One topical phrase, or that phrase joined by a comma to an acknowledgement
    from the calm row. Only the calm row: it is the one part of the table known
    not to contradict whatever the beat is about, and a reply assembled from two
    halves of a two-pole row can mean the opposite of itself.

    Two phrases sharing a bigram are never joined either — the gate rejects
    repeated fragments, and 같이 놀자, 다시 같이 놀자 is the shape it looks for.
    The length bounds are the persona document's own 4~35 rule, handed down by
    the caller.
    """
    heads = heads_for(beat, vocabulary)
    calm = next(iter(vocabulary.values()))
    partners = tuple(dict.fromkeys((*calm, *heads)))

    out: set[str] = set()
    for head in heads:
        out.add(_finish(head))
        for other in partners:
            if other == head or _shares_bigram(other, head):
                continue
            # 응, 배고파 — a short acknowledgement in front of the state.
            if len(other) <= len(head) and _can_lead(other):
                out.add(_finish(f"{other}, {head}"))
            # 배고파, 밥 줘 — the state first and the short request after, which
            # is the order 발화 원칙 1 asks for.
            if _can_lead(head):
                out.add(_finish(f"{head}, {other}"))
    return tuple(sorted(r for r in out if min_chars <= len(r) <= max_chars))


def beat_weights(replies: dict[str, tuple[str, ...]]) -> dict[str, float]:
    """How much of the slice each beat is worth, which is not one each.

    Most of the document's beats have no phrase of their own and are answered
    from the calm row, so they all say the same three or four things. Sampling
    beats uniformly measured 응, 네 옆에 있을게 and two rearrangements of it in
    26,000 of 87,000 pet turns — a corpus that teaches one answer to everything.

    A beat is worth what it can say that no other beat can, so every reply
    contributes ``1 / (number of beats that can produce it)``. The beats that
    share the calm row keep their coverage; they stop taking the slice with them.
    """
    owners = collections.Counter(
        reply for beat_replies in replies.values() for reply in beat_replies
    )
    return {
        beat: sum(1.0 / owners[reply] for reply in beat_replies)
        for beat, beat_replies in replies.items()
    }


# --- the plan --------------------------------------------------------------

@dataclass(frozen=True)
class Plan:
    """What the persona document decided, resolved once before generating.

    Nothing in here is configuration or chance: two runs against the same
    document build the same plan, and every session is a draw from it.
    """

    groups: tuple[tuple[str, ...], ...]
    replies: dict[str, tuple[str, ...]]
    weight: dict[str, float]

    @property
    def beats(self) -> tuple[tuple[int, str], ...]:
        """Every beat, carrying the index of the line it belongs to."""
        return tuple(
            (index, beat)
            for index, group in enumerate(self.groups)
            for beat in group
        )

    @property
    def distinct_replies(self) -> int:
        return len({reply for replies in self.replies.values() for reply in replies})


def plan_from(persona: Persona, *, min_chars: int, max_chars: int) -> Plan:
    """Read the document into beats, the replies they allow, and their weights."""
    replies = {
        beat: replies_for(
            beat, persona.vocabulary, min_chars=min_chars, max_chars=max_chars
        )
        for group in beat_groups(persona)
        for beat in group
    }
    groups = tuple(
        answerable
        for group in beat_groups(persona)
        if (answerable := tuple(beat for beat in group if replies[beat]))
    )
    if not groups:
        raise ValueError(
            "no beat of 다룰 상황 could be answered from 감정 표현과 어휘 — "
            "has the persona document changed shape?"
        )
    kept = {beat: replies[beat] for group in groups for beat in group}
    return Plan(groups=groups, replies=kept, weight=beat_weights(kept))


# --- the stage -------------------------------------------------------------

class TemplateStage:
    """Slot-filled sessions, every phrase of them read from the persona."""

    name = "template"
    produces = "raw"

    def run(self, ctx: StageContext) -> Iterator[dict]:
        limit = int(ctx.settings["limit"])
        if limit < 1:
            raise ValueError(f"stage {ctx.name!r} needs limit >= 1")

        low, high = ctx.persona.utterance_char_range()
        # The document says 4~35 글자; the gate's floor can be looser, and a
        # record this stage emits should satisfy both.
        plan = plan_from(
            ctx.persona,
            min_chars=max(low, ctx.gate.min_chars if ctx.gate is not None else low),
            max_chars=min(high, ctx.gate.max_chars if ctx.gate is not None else high),
        )
        ctx.log(
            f"[{ctx.name}] {len(plan.beats)} beats, {plan.distinct_replies} distinct "
            f"pet utterances, {len(USER_FRAMES)} user frames"
        )

        seen: set[tuple[str, ...]] = set()
        stall = 0
        produced = 0
        while produced < limit and stall < STALL_LIMIT:
            turns, beats = self._session(ctx, plan)
            fingerprint = tuple(turn["text"] for turn in turns)
            if fingerprint in seen:
                # Sampled, not enumerated, so collisions turn up long before
                # the combination space is actually used up.
                stall += 1
                continue
            seen.add(fingerprint)
            stall = 0
            yield _record(produced, beats, turns)
            produced += 1

        if produced < limit:
            ctx.log(
                f"[{ctx.name}] the persona's own vocabulary yields {produced:,} "
                f"distinct sessions; the configured limit of {limit:,} is above "
                "what this slice can cover without repeating itself"
            )

    def _session(
        self, ctx: StageContext, plan: Plan
    ) -> tuple[list[dict[str, str]], list[str]]:
        """One session: a walk through one situation line's beats.

        Both the beat that leads the session and the moments that follow it are
        drawn against :func:`beat_weights` — a follower is as much of the corpus
        as a leader, so weighting only the leader would let the beats that all
        say the same thing back in through the side door.
        """
        population = plan.beats
        index, lead = ctx.rng.choices(
            population, [plan.weight[beat] for _, beat in population]
        )[0]
        group = plan.groups[index]
        count = min(ctx.rng.choice(EXCHANGE_SHAPE), len(group), len(USER_FRAMES))

        others = [beat for beat in group if beat != lead]
        followers = (
            ctx.rng.choices(others, [plan.weight[b] for b in others], k=count - 1)
            if count > 1 and others
            else []
        )
        # Sorted back into document order: 다룰 상황 lists each line's moments as
        # a progression, and walking them in order is what makes a
        # multi-exchange session read as one situation rather than three. The
        # set is what makes a repeated draw shorten the walk instead of asking
        # the same thing twice.
        beats = sorted({lead, *followers}, key=group.index)
        frames = ctx.rng.sample(USER_FRAMES, len(beats))

        turns: list[dict[str, str]] = []
        for beat, frame in zip(beats, frames):
            turns.append({"role": "user", "text": frame(beat)})
            turns.append({"role": "pet", "text": ctx.rng.choice(plan.replies[beat])})
        return turns, beats


def _record(
    index: int, beats: Sequence[str], turns: list[dict[str, str]]
) -> dict[str, Any]:
    """One session record. The runner validates, gates and writes it.

    ``scenario`` is the beat the session opens on, which is what the mix report
    counts by; the full walk is kept alongside it so a session covering three
    moments is not filed as covering one.
    """
    return {
        "id": f"template-{index:06d}",
        "source": "template",
        "scenario": beats[0],
        "beats": list(beats),
        "generator": ["template"],
        "license": "synthetic",
        "turns": turns,
    }


__all__ = [
    "Plan",
    "TemplateStage",
    "beat_groups",
    "beat_weights",
    "heads_for",
    "plan_from",
    "replies_for",
]
