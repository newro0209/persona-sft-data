"""프로필: 다섯 내장 프로필의 문서 골격은 파서를 통과하고, 프롬프트에 라벨이 들어간다."""
import random

import pytest

from persona_sft_data.core.gates import GateSettings, build_gate
from persona_sft_data.core.persona import load
from persona_sft_data.core.registry import PROFILES
from persona_sft_data.teacher import prompts

BUILTIN = ("companion", "npc", "novel", "trpg", "lore")


@pytest.mark.parametrize("name", BUILTIN)
def test_document_template_parses_and_builds_a_gate(tmp_path, name):
    prof = PROFILES.get(name)
    doc = tmp_path / f"{name}.md"
    doc.write_text(prof.document_template("테스트"), encoding="utf-8")
    persona = load(doc, required_sections=prof.required_sections)
    assert persona.name == "테스트" and persona.beats
    build_gate(persona, GateSettings())
    system = prompts.dialogue_system(persona, prof, random.Random(0))
    assert prof.assistant_label in system and prof.user_label in system
    if "배경" in prof.required_sections:
        assert persona.background


def test_profiles_differ_only_by_data():
    names = {PROFILES.get(n).assistant_label for n in BUILTIN}
    assert len(names) == 5
    assert "배경" in PROFILES.get("npc").required_sections
    assert PROFILES.get("companion").required_sections == ()
    assert any("존댓말" in f for f in PROFILES.get("companion").default_flows)
