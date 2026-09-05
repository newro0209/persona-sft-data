"""페르소나 문서 파서.

``personas/<이름>.md`` 하나가 페르소나의 단일 출처다. 코드에는 페르소나 문자열이
없고, 테스트가 그것을 강제한다. 파서는 일부러 엄격하다 — 필수 절이 빠지거나 표
모양이 다르면 빈 값을 돌려주지 않고 예외를 던진다. 조용히 비어 버린 규칙 목록은
금지 표현을 그대로 통과시킨다.

절 구성은 스펙 §6이다. ``## 제약`` 표는 게이트 규칙의 유일한 출처이고, 여기서는
행을 그대로 읽어 둘 뿐 해석은 ``rules/``가 한다.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

SECTION_CORE = "핵심 정의"
SECTION_CONSTRAINTS = "제약"
SECTION_PRINCIPLES = "발화 원칙"
SECTION_SITUATIONS = "다룰 상황"
SECTION_BACKGROUND = "배경"
SECTION_PROHIBITIONS = "하지 않는 말과 행동"
SECTION_VOCABULARY = "어휘와 표현"
SECTION_FLOWS = "대화 흐름"
SECTION_EXAMPLES = "예시 대화"

CORE_KEYS = ("이름", "정체성", "사용자와의 관계", "말투", "성격", "응답 길이", "지식 범위")

_SECTION = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
_EXAMPLE_LINE = re.compile(r"^(U|A):\s*(.*)$")


class PersonaError(ValueError):
    """문서가 없거나 모양이 스키마와 다르다."""


@dataclass(frozen=True)
class Persona:
    """프롬프트·게이트·시스템 프롬프트가 필요로 하는 전부."""

    name: str
    core: dict[str, str]
    constraints: dict[str, str]
    principles: tuple[str, ...]
    situations: tuple[str, ...]
    background: str | None
    prohibitions: tuple[str, ...]
    vocabulary: dict[str, tuple[str, ...]]
    flows: tuple[str, ...]
    examples: tuple[tuple[dict[str, str], ...], ...]
    source: Path

    @property
    def beats(self) -> tuple[str, ...]:
        """다룰 상황의 각 줄을 쉼표로 쪼갠 낱개 순간. 교사에게는 한 번에 하나만 준다."""
        return tuple(
            beat.strip() for line in self.situations for beat in line.split(",") if beat.strip()
        )

    def system_prompt(self) -> str:
        """핵심 정의·배경·발화 원칙·하지 않는 것을 평문으로. 프롬프트용 문장은 따로 쓰지 않는다."""
        lines = [f"{key}: {value}" for key, value in self.core.items()]
        if self.background:
            lines += ["", f"{SECTION_BACKGROUND}:", self.background]
        lines += ["", f"{SECTION_PRINCIPLES}:"]
        lines += [f"{i}. {p}" for i, p in enumerate(self.principles, 1)]
        if self.prohibitions:
            lines += ["", f"{SECTION_PROHIBITIONS}:"]
            lines += [f"- {p}" for p in self.prohibitions]
        return "\n".join(lines)


# -- 절 단위 파서 -------------------------------------------------------------

def _sections(text: str) -> dict[str, str]:
    marks = list(_SECTION.finditer(text))
    if not marks:
        raise PersonaError("'## ' 절이 하나도 없다")
    out: dict[str, str] = {}
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        out[m.group(1)] = text[m.end():end]
    return out


def _strip_emphasis(cell: str) -> str:
    return cell.replace("**", "").replace("`", "").strip()


def _table(body: str, section: str) -> dict[str, str]:
    """2열 표. 구분선 바로 위의 행이 머리글이며 버린다. 열이 둘이 아닌 행은 오류다."""
    rows: list[tuple[str, str]] = []
    header_index: int | None = None
    for line in body.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells):
            header_index = len(rows) - 1
            continue
        if len(cells) != 2:
            raise PersonaError(f"'## {section}' 표의 행이 2열이 아니다: {line!r}")
        rows.append((_strip_emphasis(cells[0]), _strip_emphasis(cells[1])))
    if header_index is not None and 0 <= header_index < len(rows):
        rows.pop(header_index)
    if not rows:
        raise PersonaError(f"'## {section}' 절에 2열 표가 없다")
    return dict(rows)


def _numbered(body: str, section: str) -> tuple[str, ...]:
    items: list[str] = []
    for line in body.splitlines():
        if re.match(r"^\s*\d+\.\s", line):
            items.append(re.sub(r"^\s*\d+\.\s*", "", line).strip())
        elif items and line.startswith((" ", "\t")) and line.strip():
            items[-1] += " " + line.strip()
        elif items and not line.strip():
            continue
        elif items:
            break
    if not items:
        raise PersonaError(f"'## {section}' 절에 번호 목록이 없다")
    return tuple(_strip_emphasis(i) for i in items)


def _bullets(body: str, section: str) -> tuple[str, ...]:
    items: list[str] = []
    for line in body.splitlines():
        if re.match(r"^-\s+", line):
            items.append(re.sub(r"^-\s+", "", line).strip())
        elif items and line.startswith((" ", "\t")) and line.strip():
            items[-1] += " " + line.strip()
    if not items:
        raise PersonaError(f"'## {section}' 절에 불릿 목록이 없다")
    return tuple(_strip_emphasis(i) for i in items)


def _fenced_blocks(body: str, lang: str = "text") -> list[str]:
    return [m.group(1).strip() for m in re.finditer(rf"```{lang}\n(.*?)\n```", body, re.DOTALL)]


def parse_example_block(text: str) -> tuple[dict[str, str], ...]:
    """``U:``/``A:`` 줄을 turns로. user로 시작해 assistant로 끝나며 번갈아야 한다."""
    turns: list[dict[str, str]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _EXAMPLE_LINE.match(line)
        if not m:
            raise PersonaError(f"예시 대화 줄은 'U:' 또는 'A:'로 시작해야 한다: {line!r}")
        role = "user" if m.group(1) == "U" else "assistant"
        turns.append({"role": role, "text": m.group(2).strip()})
    if len(turns) < 2 or len(turns) % 2:
        raise PersonaError("예시 대화는 2개 이상 짝수 줄이어야 한다")
    for i, turn in enumerate(turns):
        if turn["role"] != ("user", "assistant")[i % 2] or not turn["text"]:
            raise PersonaError(f"예시 대화 {i + 1}번째 줄의 역할 또는 내용이 잘못됐다")
    return tuple(turns)


def _split_examples(cell: str) -> tuple[str, ...]:
    return tuple(p.strip() for p in cell.split(",") if p.strip())


# -- 로드 ---------------------------------------------------------------------

def load(path: str | Path, *, required_sections: Iterable[str] = ()) -> Persona:
    """문서를 파싱한다. 부분 결과를 돌려주지 않고 예외를 던진다."""
    path = Path(path)
    if not path.exists():
        raise PersonaError(f"페르소나 문서가 없다: {path}")
    sections = _sections(path.read_text(encoding="utf-8"))

    def require(name: str) -> str:
        if name not in sections:
            raise PersonaError(f"'## {name}' 절이 없다 (있는 절: {sorted(sections)})")
        return sections[name]

    for name in required_sections:
        require(name)

    core = _table(require(SECTION_CORE), SECTION_CORE)
    missing = [k for k in CORE_KEYS if k not in core]
    if missing:
        raise PersonaError(f"'## {SECTION_CORE}' 표에 행이 빠졌다: {missing}")
    constraints = _table(require(SECTION_CONSTRAINTS), SECTION_CONSTRAINTS)

    background = sections.get(SECTION_BACKGROUND)
    background = background.strip() if background and background.strip() else None
    vocabulary = (
        {k: _split_examples(v) for k, v in _table(sections[SECTION_VOCABULARY], SECTION_VOCABULARY).items()}
        if SECTION_VOCABULARY in sections else {}
    )
    prohibitions = _bullets(sections[SECTION_PROHIBITIONS], SECTION_PROHIBITIONS) if SECTION_PROHIBITIONS in sections else ()
    flows = _bullets(sections[SECTION_FLOWS], SECTION_FLOWS) if SECTION_FLOWS in sections else ()
    examples = tuple(parse_example_block(b) for b in _fenced_blocks(sections.get(SECTION_EXAMPLES, "")))

    persona = Persona(
        name=core["이름"],
        core=core,
        constraints=constraints,
        principles=_numbered(require(SECTION_PRINCIPLES), SECTION_PRINCIPLES),
        situations=_numbered(require(SECTION_SITUATIONS), SECTION_SITUATIONS),
        background=background,
        prohibitions=prohibitions,
        vocabulary=vocabulary,
        flows=flows,
        examples=examples,
        source=path,
    )
    if not persona.name:
        raise PersonaError(f"'## {SECTION_CORE}'의 이름이 비어 있다")
    return persona


@lru_cache(maxsize=8)
def _cached(path: str, required: tuple[str, ...]) -> Persona:
    return load(path, required_sections=required)


def load_cached(path: str | Path, required_sections: Iterable[str] = ()) -> Persona:
    """``load``와 같되 메모이즈. 단계마다 한 번씩 부른다."""
    return _cached(str(Path(path).resolve()), tuple(required_sections))
