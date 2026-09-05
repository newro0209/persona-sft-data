"""The persona, parsed from ``personas/mongle.md``.

That document declares itself frozen, yet the previous pipeline had it copied
into ten Python files — the name alone appeared 32 times in this module. Editing
one line of the document left the code disagreeing with it, silently.

So there are no persona strings here. Everything is read at runtime, and a test
fails the build if the pet's name appears literally in any file under
``persona_sft_data``.

The parser is deliberately strict. A frozen document that changes shape should
raise, not quietly yield an empty prohibition list that lets banned phrasing
into the corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


class PersonaError(ValueError):
    """Raised when the persona document is missing or shaped unexpectedly."""


@dataclass(frozen=True)
class Persona:
    """Everything the prompts and the quality gates need, from one document."""

    name: str
    core: dict[str, str]
    principles: tuple[str, ...]
    vocabulary: dict[str, tuple[str, ...]]
    prohibitions: tuple[str, ...]
    situations: tuple[str, ...]
    preamble: str
    source: Path

    @property
    def identity(self) -> str:
        return self.core["정체성"]

    @property
    def speech(self) -> str:
        return self.core["말투"]

    @property
    def personality(self) -> str:
        return self.core["성격"]

    @property
    def length_rule(self) -> str:
        return self.core["응답 길이"]

    @property
    def beats(self) -> tuple[str, ...]:
        """The situation list, split into individual beats.

        Each numbered line in 다룰 상황 names four or five related moments —
        "첫 만남, 아침·낮·밤 인사, 다시 만남, 작별". Handing a whole line to a
        teacher makes it cram all of them into one dialogue, which is what a
        first probe produced: a six-turn exchange that jumped morning to noon
        to night. Split on commas and each prompt gets one moment.
        """
        out: list[str] = []
        for line in self.situations:
            for beat in line.split(","):
                beat = beat.strip()
                if beat:
                    out.append(beat)
        return tuple(out)

    def utterance_char_range(self) -> tuple[int, int]:
        """The 4~35 in "한 발화는 대체로 4~35글자로 제한한다", read rather than
        re-typed into the filter config."""
        match = re.search(r"(\d+)\s*~\s*(\d+)\s*글자", self.length_rule)
        if not match:
            raise PersonaError(
                f"no '<min>~<max>글자' range in 응답 길이: {self.length_rule!r}"
            )
        return int(match.group(1)), int(match.group(2))

    def system_prompt(self) -> str:
        """The persona as a system prompt for fine-tuning.

        Nothing here is written for the prompt: it is the document's core
        table, speech principles and prohibitions, rendered as plain text
        under the document's own headings. That keeps this module free of
        persona prose, and it means the prompt a fine-tuned model is trained
        against is the same definition the corpus was generated and gated
        against -- there is no second wording to drift.

        The parsers already strip markdown emphasis, so what comes out is
        plain text: a model does not need to be told "항상 반말" was bold.
        """
        lines = [f"{key}: {value}" for key, value in self.core.items()]
        lines.append("")
        lines.append(f"{PRINCIPLES_SECTION}:")
        lines.extend(f"{i}. {p}" for i, p in enumerate(self.principles, 1))
        lines.append("")
        lines.append(f"{PROHIBITIONS_SECTION}:")
        lines.extend(f"- {p}" for p in self.prohibitions)
        return "\n".join(lines)


# Section headings, named once so the loader and the system prompt agree on
# what the document calls things.
PRINCIPLES_SECTION = "발화 원칙"
PROHIBITIONS_SECTION = "하지 않는 말과 행동"


_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def _sections(text: str) -> dict[str, str]:
    """Split the document into ``## heading`` -> body."""
    marks = list(_SECTION.finditer(text))
    if not marks:
        raise PersonaError("no '## ' sections found")
    out: dict[str, str] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[m.group(1)] = text[m.end() : end]
    return out


def _require(sections: dict[str, str], name: str) -> str:
    if name not in sections:
        raise PersonaError(f"missing section '## {name}' (have: {sorted(sections)})")
    return sections[name]


def _table(body: str, section: str) -> dict[str, str]:
    """Read a two-column markdown table, skipping the header separator."""
    rows: dict[str, str] = {}
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != 2:
            continue
        key, value = cells
        if not key or set(key) <= set("- :"):
            continue
        rows[_strip_emphasis(key)] = _strip_emphasis(value)
    if not rows:
        raise PersonaError(f"section '## {section}' has no two-column table rows")
    # The header row survives the loop; drop it by its known label position.
    first = next(iter(rows))
    if first in {"항목", "감정·상태"}:
        rows.pop(first)
    return rows


def _strip_emphasis(cell: str) -> str:
    return cell.replace("**", "").replace("`", "").strip()


def _numbered(body: str, section: str) -> tuple[str, ...]:
    """Read a numbered list whose items may wrap onto indented continuations."""
    items: list[str] = []
    for line in body.splitlines():
        if re.match(r"^\s*\d+\.\s", line):
            items.append(re.sub(r"^\s*\d+\.\s*", "", line).strip())
        elif items and line.startswith((" ", "\t")) and line.strip():
            items[-1] += " " + line.strip()
        elif not line.strip():
            continue
        elif items:
            break
    if not items:
        raise PersonaError(f"section '## {section}' has no numbered list")
    return tuple(_strip_emphasis(i) for i in items)


def _bullets(body: str, section: str) -> tuple[str, ...]:
    """Read a ``- `` list whose items may wrap onto indented continuations."""
    items: list[str] = []
    for line in body.splitlines():
        if re.match(r"^-\s+", line):
            items.append(re.sub(r"^-\s+", "", line).strip())
        elif items and line.startswith((" ", "\t")) and line.strip():
            items[-1] += " " + line.strip()
    if not items:
        raise PersonaError(f"section '## {section}' has no bullet list")
    return tuple(_strip_emphasis(i) for i in items)


def _fenced(body: str, section: str, lang: str = "text") -> str:
    match = re.search(rf"```{lang}\n(.*?)\n```", body, re.DOTALL)
    if not match:
        raise PersonaError(f"section '## {section}' has no ```{lang} block")
    block = match.group(1).strip()
    if not block:
        raise PersonaError(f"section '## {section}' has an empty ```{lang} block")
    return block


def _split_examples(cell: str) -> tuple[str, ...]:
    """`응`, `그래`, `네 옆에 있을게` -> three entries."""
    return tuple(p.strip() for p in cell.split(",") if p.strip())


CORE_KEYS = ("이름", "정체성", "사용자와의 관계", "말투", "성격", "응답 길이", "지식 범위")


def load(path: str | Path) -> Persona:
    """Parse the persona document. Raises rather than returning partial data."""
    path = Path(path)
    if not path.exists():
        raise PersonaError(f"persona document not found: {path}")
    text = path.read_text(encoding="utf-8")
    sections = _sections(text)

    core = _table(_require(sections, "핵심 정의"), "핵심 정의")
    missing = [k for k in CORE_KEYS if k not in core]
    if missing:
        raise PersonaError(f"핵심 정의 table is missing rows: {missing}")

    vocab_rows = _table(_require(sections, "감정 표현과 어휘"), "감정 표현과 어휘")
    vocabulary = {k: _split_examples(v) for k, v in vocab_rows.items()}

    persona = Persona(
        name=core["이름"],
        core=core,
        principles=_numbered(_require(sections, PRINCIPLES_SECTION), PRINCIPLES_SECTION),
        vocabulary=vocabulary,
        prohibitions=_bullets(_require(sections, PROHIBITIONS_SECTION), PROHIBITIONS_SECTION),
        situations=_numbered(_require(sections, "다룰 상황"), "다룰 상황"),
        preamble=_fenced(_require(sections, "고정 프리앰블 대화"), "고정 프리앰블 대화"),
        source=path,
    )
    if not persona.name:
        raise PersonaError("핵심 정의 gives an empty 이름")
    persona.utterance_char_range()  # fail here, not deep inside the filter
    return persona


@lru_cache(maxsize=4)
def _cached(path: str) -> Persona:
    return load(path)


def load_cached(path: str | Path) -> Persona:
    """Same as :func:`load`, memoised — stages call this once per record."""
    return _cached(str(Path(path).resolve()))
