"""페르소나 문서 파서: 엄격하고, 새 스키마의 모든 절을 읽는다."""
from pathlib import Path

import pytest

from persona_sft_data.core.persona import (
    SECTION_BACKGROUND, SECTION_CONSTRAINTS, SECTION_CORE, SECTION_PRINCIPLES,
    SECTION_SITUATIONS, PersonaError, load, load_cached, parse_example_block,
)

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "personas" / "mongle.md"


def _without(section: str, tmp_path: Path) -> Path:
    text = DOC.read_text(encoding="utf-8")
    start = text.index(f"## {section}")
    end = text.find("\n## ", start + 1)
    end = len(text) if end < 0 else end + 1
    out = tmp_path / "p.md"
    out.write_text(text[:start] + text[end:], encoding="utf-8")
    return out


def test_parses_the_shipped_document():
    p = load(DOC)
    assert p.name and p.core["말투"].startswith("항상 반말")
    assert p.constraints["말투"] == "반말" and p.constraints["발화 길이"] == "4~35글자"
    assert "규칙" not in p.constraints                    # 표 머리글은 행이 아니다
    assert len(p.constraints) == 11
    assert len(p.principles) == 6 and len(p.situations) == 15
    assert len(p.beats) > len(p.situations) and "배고픔" in p.beats
    assert len(p.vocabulary) == 10 and p.vocabulary["평온"][0] == "응"
    assert len(p.flows) == 7 and any("존댓말" in f for f in p.flows)
    assert len(p.examples) == 1 and len(p.examples[0]) == 10
    assert p.examples[0][0] == {"role": "user", "text": "안녕, 누구야?"}
    assert p.background is None
    assert len(p.prohibitions) == 8


@pytest.mark.parametrize("section", [SECTION_CORE, SECTION_CONSTRAINTS, SECTION_PRINCIPLES, SECTION_SITUATIONS])
def test_missing_required_section_raises(tmp_path, section):
    with pytest.raises(PersonaError, match=section):
        load(_without(section, tmp_path))


def test_profile_can_require_more_sections(tmp_path):
    with pytest.raises(PersonaError, match=SECTION_BACKGROUND):
        load(DOC, required_sections=(SECTION_BACKGROUND,))


def test_background_is_read_verbatim(tmp_path):
    text = DOC.read_text(encoding="utf-8") + "\n## 배경\n\n안개 낀 항구 도시 **세라**.\n두 번째 문단.\n"
    doc = tmp_path / "p.md"
    doc.write_text(text, encoding="utf-8")
    p = load(doc, required_sections=(SECTION_BACKGROUND,))
    assert p.background == "안개 낀 항구 도시 **세라**.\n두 번째 문단."
    assert "배경:" in p.system_prompt() and "세라" in p.system_prompt()


def test_constraint_table_with_a_malformed_row_raises(tmp_path):
    text = DOC.read_text(encoding="utf-8").replace("| 이모지 | 금지 |", "| 이모지 |")
    doc = tmp_path / "p.md"
    doc.write_text(text, encoding="utf-8")
    with pytest.raises(PersonaError, match="제약"):
        load(doc)


def test_example_block_must_alternate_and_end_with_assistant():
    assert parse_example_block("U: 안녕\nA: 응") == ({"role": "user", "text": "안녕"}, {"role": "assistant", "text": "응"})
    for bad in ("A: 응\nU: 안녕", "U: 안녕", "U: 안녕\nX: 응"):
        with pytest.raises(PersonaError):
            parse_example_block(bad)


def test_system_prompt_is_the_document_not_a_second_wording():
    p = load(DOC)
    prompt = p.system_prompt()
    assert prompt.startswith("이름: ")
    assert "발화 원칙:" in prompt and "1. " in prompt
    assert "하지 않는 말과 행동:" in prompt
    assert "4~35글자" not in prompt.split("발화 원칙:")[0].split("응답 길이")[0]   # 제약 표는 넣지 않는다
    assert "| 말투 |" not in prompt


def test_load_cached_returns_the_same_object():
    assert load_cached(DOC) is load_cached(DOC)
