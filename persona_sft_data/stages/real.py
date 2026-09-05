"""The real stage: human Korean on the user side, a generated pet reply.

The specification asks for 10~20% real dialogue, where *real* means "not
synthesised". A record qualifies when the user's utterance is something a person
actually wrote, so this stage collects human Korean and asks the bulk teacher for
one persona-consistent reply to it. That is what replaces the previous
pipeline's ``NEUTRAL_REPLIES``/``QUESTION_REPLIES`` tuples — fourteen hardcoded
pet sentences that made every real session end one of fourteen ways.

Two sources, both downloaded once and cached:

``smilestyle``
    Smilegate AI's ``korean_smile_style_dataset``. A TSV of 17 style columns, one
    meaning per row rendered in every register. Only the casual columns are read:
    the pet is spoken to by someone it lives with, not addressed in a press
    release. Section 4.4 of the design doc measured the ceiling and it is worth
    repeating — 3,470 usable rows means 3,470 *distinct meanings* however many
    columns are taken, so this source cannot be scaled up. ``stats.json`` reports
    the count rather than hiding it.

``korean_safe_conversation``
    ``jojo0217/korean_safe_conversation``, read through its parquet export.
    **Only the ``instruction`` column is ever touched.** The answer column of
    that dataset is AI-assistant prose — a measured value is "저는 인공지능
    챗봇이기 때문에 여행을 떠나지는 못했습니다." — which is the exact sentence
    ``personas/mongle.md`` forbids. Pairing instruction with answer would inject
    banned phrasing straight into the corpus, so the parquet reader projects the
    instruction column by name and the answers are never decoded at all.

Neither download may take the stage down with it. A source that cannot be
fetched or parsed is logged and skipped, and whatever worked still runs.

Measured on 2026-09-04, both sources downloaded and read by this module:

===========================  =======  ========  =========  =======
source                           raw  distinct   in scope   unsafe
===========================  =======  ========  =========  =======
smilestyle (2 columns)         6,940     6,834      3,638        0
korean_safe_conversation      26,979    26,782      8,765       13
===========================  =======  ========  =========  =======

Two things that table does not say. The topic filter is a bigram overlap and a
coarse instrument: much of what survives from the second source is still an
open-domain question (여행, 상식) that the persona answers by saying it does not
know and returning to daily life — which is 다룰 상황's own situation 14, and is
useful in small quantities rather than at 8,765. And 12,403 utterances is the
ceiling of the entire slice against a config limit of 60,000, so the design
doc's warning that the real sources are nearly exhausted is not theoretical.

One note on reading this stage's ``stats.json``. Utterances the filters threw
out are counted as rejects, so ``yield_rate`` here measures the funnel from
downloaded source to record — 1.4% in the run above — and not how often the
teacher answered. That number is ``teacher_calls`` against
``teacher_failures``. Material dropped before a record existed is still
material dropped, and the pipeline counts what it discards.
"""

from __future__ import annotations

import csv
import io
import re
import time
import urllib.request
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from persona_sft_data import backend, gates, prompts, runner, schema
from persona_sft_data.backend import Request, Teacher
from persona_sft_data.persona import Persona
from persona_sft_data.runner import StageContext

# --- sources ---------------------------------------------------------------

SMILESTYLE_URL = (
    "https://raw.githubusercontent.com/smilegate-ai/"
    "korean_smile_style_dataset/main/smilestyle_dataset.tsv"
)
SMILESTYLE_NAME = "smilestyle"
SMILESTYLE_LICENSE = "smilestyle"
SMILESTYLE_CACHE = "smilestyle_dataset.tsv"
# Of the 17 registers these two are how a person talks to something they live
# with. The rest — formal, choding, joongding and so on — are the same meaning
# wearing a costume, and nobody addresses a pet in any of them.
SMILESTYLE_COLUMNS = ("informal", "chat")

SAFE_CONVERSATION_URL = (
    "https://huggingface.co/datasets/jojo0217/korean_safe_conversation/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
SAFE_CONVERSATION_NAME = "korean_safe_conversation"
SAFE_CONVERSATION_LICENSE = "apache-2.0"
SAFE_CONVERSATION_CACHE = "korean_safe_conversation-train-0000.parquet"
INSTRUCTION_COLUMN = "instruction"
# The projection handed to parquet: one column, by name. The answers in this
# dataset are AI-assistant text and are never materialised — see the module
# docstring for the sentence that would otherwise reach the corpus.
SAFE_CONVERSATION_COLUMNS = (INSTRUCTION_COLUMN,)

# Korean role-play dialogue: personas talking in character, which is closer to
# this corpus's shape than instruction data is. Only the human turns are taken
# -- the character replies belong to someone else's persona, not this one.
ROLEPLAY_URL = (
    "https://huggingface.co/datasets/huggingface-KREW/korean-role-playing/"
    "resolve/refs%2Fconvert%2Fparquet/general-roleplay-data/train/0000.parquet"
)
ROLEPLAY_NAME = "korean_role_playing"
ROLEPLAY_LICENSE = "apache-2.0"
ROLEPLAY_CACHE = "korean_role_playing-general-0000.parquet"
ROLEPLAY_COLUMNS = ("text",)

# KoAlpaca, ShareGPT-ko, OIG-ko and Korquad-Chat, merged and tagged with
# <usr>/<bot>. Same rule: the <usr> turns are human-written, the <bot> turns
# are assistant text this persona must never produce.
INSTRUCTIONS_URL = (
    "https://huggingface.co/datasets/heegyu/open-korean-instructions/"
    "resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet"
)
INSTRUCTIONS_NAME = "open_korean_instructions"
INSTRUCTIONS_LICENSE = "mit"
INSTRUCTIONS_CACHE = "open_korean_instructions-0000.parquet"
INSTRUCTIONS_COLUMNS = ("text",)
USER_TURN = re.compile(r"<usr>\s*(.*?)\s*(?=<bot>|<usr>|<sys>|$)", re.DOTALL)

SCENARIO = "실제 한국어 구어"

# A user turn longer than this is a paragraph, not something said to a pet. The
# gate takes the pet's length rule from the persona document but says nothing
# about the human side, so that ceiling lives here.
MAX_USER_CHARS = 60

# The gate binds the pet's speech to the persona's prohibitions; nothing binds
# the human half, which is copied out of an open dataset. These are the stems
# that make an utterance unusable as something said to a pet.
#
# Measured rather than imagined. A scan of both sources found 43 candidate
# matches in 12,409 in-scope utterances, and most were a careless stem catching
# an ordinary word — 새끼곰, 가시발새우, 소나무가 죽어서. What was left was
# slurs and sexual content, about ten of them. So the list is short, it holds
# only stems with no innocent reading, and it matches at the start of a token
# so that 가시발새우 is a shrimp rather than an insult. It is a floor on what
# reaches the corpus, not a classifier; ``blocked_stems`` in the stage config
# replaces it.
UNSAFE_STEMS = (
    "씨발", "시발", "개새", "병신", "좆", "존나", "지랄", "썅", "니미",
    "미친년", "미친놈", "야동", "섹스", "강간",
)


@dataclass(frozen=True)
class Utterance:
    """One human-written line, with the provenance its record has to carry."""

    text: str
    dataset: str
    license: str
    url: str


# --- download and cache ----------------------------------------------------

def _fetch(url: str, timeout: float) -> bytes:
    """The only network call in this module. Tests replace it."""
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read()


def cached_bytes(
    cache_dir: Path,
    filename: str,
    url: str,
    *,
    timeout: float,
    log: Callable[[str], None],
    label: str,
) -> bytes | None:
    """Read the cached copy, downloading it once if it is not there yet.

    Returns ``None`` instead of raising. A source that cannot be reached makes a
    smaller corpus, not a failed run, and the caller carries on with the other
    one — which of them contributed how much is in ``stats.json`` either way.
    """
    path = cache_dir / filename
    if path.exists():
        return path.read_bytes()
    try:
        data = _fetch(url, timeout)
    except Exception as exc:  # noqa: BLE001 - every fetch fault is the same here
        log(
            f"[{label}] could not download {url}: {type(exc).__name__}: {exc}. "
            f"Continuing without this source; put the file at {path} to use it."
        )
        return None
    cache_dir.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    log(f"[{label}] downloaded {len(data):,} bytes -> {path}")
    return data


# --- parsing ---------------------------------------------------------------

def smilestyle_utterances(
    data: bytes, *, columns: Sequence[str] = SMILESTYLE_COLUMNS
) -> list[Utterance]:
    """Read the casual columns of the SmileStyle TSV.

    Empty cells are normal — not every meaning is rendered in every register —
    and are skipped rather than counted as a fault.
    """
    reader = csv.DictReader(
        io.StringIO(data.decode("utf-8-sig", errors="replace")), delimiter="\t"
    )
    present = [c for c in columns if c in (reader.fieldnames or ())]
    if not present:
        return []
    out: list[Utterance] = []
    for row in reader:
        for column in present:
            text = (row.get(column) or "").strip()
            if text:
                out.append(
                    Utterance(
                        text=text,
                        dataset=SMILESTYLE_NAME,
                        license=SMILESTYLE_LICENSE,
                        url=SMILESTYLE_URL,
                    )
                )
    return out


def read_parquet_rows(
    data: bytes, *, columns: Sequence[str] = SAFE_CONVERSATION_COLUMNS
) -> list[dict[str, Any]]:
    """Materialise the named columns of a parquet file, and nothing else.

    pyarrow is imported here rather than at module scope on purpose: it is a
    training-side dependency, and neither the other source nor this module's
    tests should need it installed. An ImportError reaches the caller as "this
    source could not be read", which is the handling a failed download gets.
    """
    import pyarrow.parquet as pq  # noqa: PLC0415 - optional dependency, see above

    return pq.read_table(io.BytesIO(data), columns=list(columns)).to_pylist()


def safe_conversation_utterances(
    rows: Iterable[Mapping[str, Any]],
) -> list[Utterance]:
    """Take the human instruction out of each row.

    The answer column is not read here, and :func:`read_parquet_rows` does not
    even decode it. Both halves of that are load-bearing: this is the dataset
    whose answers say the model is a chatbot.
    """
    out: list[Utterance] = []
    for row in rows:
        text = str(row.get(INSTRUCTION_COLUMN) or "").strip()
        if text:
            out.append(
                Utterance(
                    text=text,
                    dataset=SAFE_CONVERSATION_NAME,
                    license=SAFE_CONVERSATION_LICENSE,
                    url=SAFE_CONVERSATION_URL,
                )
            )
    return out


def roleplay_utterances(rows: Iterable[Mapping[str, Any]]) -> list[Utterance]:
    """Human turns from role-play transcripts.

    Each row's ``text`` is a list of {role, content}. Roles vary across the
    subsets, so anything that is not the assistant/character side is treated as
    human -- erring toward dropping material rather than admitting another
    persona's voice.
    """
    out: list[Utterance] = []
    for row in rows:
        turns = row.get("text") or []
        if not isinstance(turns, list):
            continue
        for turn in turns:
            if not isinstance(turn, Mapping):
                continue
            role = str(turn.get("role", "")).lower()
            if role in {"assistant", "bot", "character", "system"}:
                continue
            text = str(turn.get("content") or "").strip()
            if text:
                out.append(
                    Utterance(
                        text=text,
                        dataset=ROLEPLAY_NAME,
                        license=ROLEPLAY_LICENSE,
                        url=ROLEPLAY_URL,
                    )
                )
    return out


def instruction_utterances(rows: Iterable[Mapping[str, Any]]) -> list[Utterance]:
    """The <usr> spans of the merged instruction corpus, and nothing else."""
    out: list[Utterance] = []
    for row in rows:
        blob = str(row.get("text") or "")
        for match in USER_TURN.finditer(blob):
            text = match.group(1).strip()
            if text:
                out.append(
                    Utterance(
                        text=text,
                        dataset=INSTRUCTIONS_NAME,
                        license=INSTRUCTIONS_LICENSE,
                        url=INSTRUCTIONS_URL,
                    )
                )
    return out


# --- topic filter ----------------------------------------------------------

def _bigrams(text: str) -> frozenset[str]:
    """Character bigrams of the Hangul in ``text``.

    Bigrams rather than words because Korean glues particles and endings onto
    the noun that carries the meaning: 배고파, 배고픈데 and 배고프다 share no
    *word* with the persona's 배고픔, and all three share 배고 with it.
    """
    squashed = re.sub(r"[^가-힣]", "", text)
    return frozenset(squashed[i : i + 2] for i in range(len(squashed) - 1))


def topic_signal(persona: Persona) -> frozenset[str]:
    """The persona's world, as a bag of bigrams.

    Built from 다룰 상황 (what the pet's life contains) and 감정 표현과 어휘 (how
    it talks about it). Nothing here is typed out — edit the document and the
    filter moves with it.
    """
    words: list[str] = list(persona.beats)
    for label, examples in persona.vocabulary.items():
        words.append(label)
        words.extend(examples)
    grams: set[str] = set()
    for word in words:
        grams |= _bigrams(word)
    return frozenset(grams)


def in_scope(
    text: str,
    signal: frozenset[str],
    *,
    min_hits: int = 1,
    min_chars: int = 2,
    max_chars: int = MAX_USER_CHARS,
) -> bool:
    """Is this something the pet could plausibly be spoken to about?

    Deliberately coarse, with two jobs. Keep out what the pet has no business
    answering — the safe-conversation instructions run to 여행 후기 and 방송
    편성 — and keep the script clean so a teacher call is not spent on a record
    the gate would reject anyway. Whether the *reply* stays in persona is the
    gate's question, not this one's, so a single shared bigram lets an utterance
    through; ``topic_min_hits`` in the stage config tightens it.
    """
    if not min_chars <= len(text) <= max_chars:
        return False
    if not gates.HANGUL.search(text):
        return False
    if gates.CJK.search(text) or gates.KANA.search(text):
        return False
    if gates.LATIN_WORD.search(text):
        return False
    return len(_bigrams(text) & signal) >= min_hits


def is_unsafe(text: str, stems: Sequence[str] = UNSAFE_STEMS) -> bool:
    """Does this carry a slur or sexual content the corpus should not repeat?

    Token-initial, because the substring is not the word: 가시발새우 is a shrimp.
    """
    for token in re.split(r"[^가-힣]+", text):
        if token and any(token.startswith(stem) for stem in stems):
            return True
    return False


# --- reply parsing ---------------------------------------------------------

_LEADING_ROLE = re.compile(r"^\s*(U|P|사용자|유저|user|pet)\s*[:：]\s*", re.IGNORECASE)


def reply_text(raw: str | None) -> str:
    """The teacher is asked for one bare line; take the first one it gives.

    A role tag and surrounding quotes are stripped rather than rejected. The
    prompt forbids both, but a model that added one has still answered the
    question, and throwing that away costs another generation.
    """
    if not raw:
        return ""
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        line = _LEADING_ROLE.sub("", line)
        return line.strip().strip('"').strip("'").strip()
    return ""


# --- the stage -------------------------------------------------------------

class RealStage:
    """Human user turns from two Korean corpora, pet turns from the teacher."""

    name = "real"
    produces = "raw"

    def __init__(self, teacher: Teacher | None = None) -> None:
        # Injectable only so the stage tests can run against a FakeTeacher;
        # the pipeline never passes one and builds from config below.
        self._teacher = teacher

    # -- sources -----------------------------------------------------------

    def _collect(
        self, ctx: StageContext, cache: Path, timeout: float
    ) -> list[Utterance]:
        """Both sources, in whatever state the network left them."""
        pool: list[Utterance] = []

        data = cached_bytes(
            cache,
            SMILESTYLE_CACHE,
            SMILESTYLE_URL,
            timeout=timeout,
            log=ctx.log,
            label=ctx.name,
        )
        if data is not None:
            found = smilestyle_utterances(data)
            if found:
                ctx.log(f"[{ctx.name}] {SMILESTYLE_NAME}: {len(found):,} utterances")
            else:
                ctx.log(
                    f"[{ctx.name}] {SMILESTYLE_NAME}: none of {SMILESTYLE_COLUMNS} "
                    "is a column of the cached TSV; skipping this source"
                )
            pool.extend(found)

        data = cached_bytes(
            cache,
            SAFE_CONVERSATION_CACHE,
            SAFE_CONVERSATION_URL,
            timeout=timeout,
            log=ctx.log,
            label=ctx.name,
        )
        if data is not None:
            try:
                rows: list[Mapping[str, Any]] = read_parquet_rows(data)
            except Exception as exc:  # noqa: BLE001 - no pyarrow, or a bad file
                ctx.log(
                    f"[{ctx.name}] {SAFE_CONVERSATION_NAME}: cannot read parquet "
                    f"({type(exc).__name__}: {exc}); continuing without this source"
                )
                rows = []
            found = safe_conversation_utterances(rows)
            if found:
                ctx.log(
                    f"[{ctx.name}] {SAFE_CONVERSATION_NAME}: {len(found):,} "
                    "instructions (the answer column is never read)"
                )
            pool.extend(found)

        for cache_name, url, columns, reader, label in (
            (ROLEPLAY_CACHE, ROLEPLAY_URL, ROLEPLAY_COLUMNS,
             roleplay_utterances, ROLEPLAY_NAME),
            (INSTRUCTIONS_CACHE, INSTRUCTIONS_URL, INSTRUCTIONS_COLUMNS,
             instruction_utterances, INSTRUCTIONS_NAME),
        ):
            data = cached_bytes(
                cache, cache_name, url, timeout=timeout, log=ctx.log, label=ctx.name
            )
            if data is None:
                continue
            try:
                rows = read_parquet_rows(data, columns=columns)
            except Exception as exc:  # noqa: BLE001 - one bad source must not stop the rest
                ctx.log(f"[{ctx.name}] {label}: cannot read parquet ({exc}); skipping")
                continue
            found = reader(rows)
            ctx.log(f"[{ctx.name}] {label}: {len(found):,} human turns")
            pool.extend(found)

        return pool

    def _select(
        self, ctx: StageContext, pool: Sequence[Utterance]
    ) -> tuple[list[Utterance], dict[str, int]]:
        """Distinct, usable, on-topic utterances, shuffled and capped.

        The second element is why the rest were dropped. They never became
        records, but they are still material thrown away, and the pipeline's
        rule is that thrown-away material gets counted.
        """
        signal = topic_signal(ctx.persona)
        min_chars = ctx.gate.min_chars if ctx.gate is not None else 2
        min_hits = int(ctx.settings.get("topic_min_hits", 1))
        stems = tuple(ctx.settings.get("blocked_stems", UNSAFE_STEMS))

        selected: list[Utterance] = []
        seen: set[str] = set()
        dropped = {"off_topic": 0, "unsafe_source": 0}
        for utterance in pool:
            # The two SmileStyle columns of one row are often the same sentence
            # twice, and the parquet has repeats of its own. Identical text is
            # one teacher call, not two — compared after the corpus's own
            # normalisation, so a difference in spacing does not buy a second.
            key = schema.normalize_text(utterance.text)
            if key in seen:
                continue
            seen.add(key)
            if is_unsafe(utterance.text, stems):
                dropped["unsafe_source"] += 1
            elif in_scope(
                utterance.text, signal, min_hits=min_hits, min_chars=min_chars
            ):
                selected.append(utterance)
            else:
                dropped["off_topic"] += 1

        ctx.log(
            f"[{ctx.name}] {len(selected):,} of {len(seen):,} distinct utterances "
            f"are inside the persona's world ({dropped['unsafe_source']:,} dropped "
            "for slurs or sexual content)"
        )

        ctx.rng.shuffle(selected)
        limit = int(ctx.settings.get("limit", 0))
        if limit and len(selected) > limit:
            ctx.log(f"[{ctx.name}] limit {limit:,}: {len(selected) - limit:,} unused")
            selected = selected[:limit]
        return selected, dropped

    # -- run ---------------------------------------------------------------

    def run(self, ctx: StageContext) -> Iterator[dict]:
        cache = ctx.config.data_root / "cache"
        timeout = float(ctx.settings.get("download_timeout", 60.0))

        pool = self._collect(ctx, cache, timeout)
        if not pool:
            ctx.log(f"[{ctx.name}] no source could be read; produced nothing")
            return

        selected, dropped = self._select(ctx, pool)
        # Not rejects: this is source material outside the persona's world,
        # plus a few unsafe utterances. Counting it as rejected would report a
        # 0.2% yield for a stage whose generated records pass at ~80%.
        yield runner.metric(
            source_filtered=sum(dropped.values()),
            source_filter_reasons={k: v for k, v in dropped.items() if v},
        )
        if not selected:
            return

        cfg = ctx.config.teacher_for(ctx.name)
        teacher = self._teacher if self._teacher is not None else backend.build(cfg)
        # Fail before generating anything if the server is down or is serving
        # the other teacher: the two share a port and are loaded one at a time.
        teacher.check()

        # One batch per round trip, sized to the teacher's concurrency — vLLM
        # wants the whole batch at once, and the batch is the only thing this
        # stage holds beyond the utterance list itself.
        batch_size = max(1, int(cfg.concurrency))

        index = 0
        started = time.time()
        for batch in backend.batched(selected, batch_size):
            requests: list[Request] = []
            keyed: dict[str, Utterance] = {}
            for utterance in batch:
                key = f"{index:06d}"
                keyed[key] = utterance
                requests.append(
                    Request(
                        key=key,
                        # Built per request rather than per batch: persona_block
                        # rotates a different vocabulary sample each call, which
                        # is what stops every reply converging on five phrases.
                        system=prompts.real_system(ctx.persona, ctx.rng),
                        user=prompts.real_user(utterance.text),
                    )
                )
                index += 1

            failures = 0
            tokens = 0
            reasons: dict[str, int] = {}
            results = {r.key: r for r in teacher.generate(requests)}
            for request in requests:
                result = results.get(request.key)
                if result is None or not result.ok:
                    # A dropped call is a lost record, so it is counted as a
                    # reject as well: teacher_failures alone would leave the
                    # yield rate silently flattering.
                    failures += 1
                    reasons["teacher_error"] = reasons.get("teacher_error", 0) + 1
                    continue
                tokens += result.completion_tokens
                reply = reply_text(result.text)
                if not reply:
                    reasons["empty_reply"] = reasons.get("empty_reply", 0) + 1
                    continue
                yield _record(request.key, keyed[request.key], reply, cfg.model)

            yield runner.metric(
                calls=len(requests),
                failures=failures,
                completion_tokens=tokens,
                rejected=sum(reasons.values()),
                reject_reasons=reasons,
            )
            ctx.log(
                f"[{ctx.name}] {index:,} of {len(selected):,} utterances answered "
                f"| {time.time() - started:.0f}s"
            )


def _record(key: str, utterance: Utterance, reply: str, model: str) -> dict[str, Any]:
    """One session record. The runner validates, gates and writes it.

    ``key`` is the utterance's index, already zero-padded, so a failed
    generation leaves a gap in the ids rather than renumbering everything after
    it — the id points at the utterance it was made from.

    ``license`` and ``source_url`` travel per record rather than per file
    because the two sources are mixed here and are not under the same terms.
    """
    return {
        "id": f"real-{key}",
        "source": "real",
        "real_source": utterance.dataset,
        "scenario": SCENARIO,
        "license": utterance.license,
        "source_url": utterance.url,
        "generator": [model],
        "turns": [
            {"role": "user", "text": utterance.text},
            {"role": "pet", "text": reply},
        ],
    }


__all__ = [
    "RealStage",
    "Utterance",
    "cached_bytes",
    "in_scope",
    "is_unsafe",
    "read_parquet_rows",
    "reply_text",
    "safe_conversation_utterances",
    "smilestyle_utterances",
    "topic_signal",
]
