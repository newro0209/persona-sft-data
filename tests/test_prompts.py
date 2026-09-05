"""프롬프트: 페르소나 문서와 프로필에서만 조립되고, 교사 출력 파싱은 만들어 내지 않는다."""
import random

from persona_sft_data.core.persona import load
from persona_sft_data.core.registry import PROFILES
from persona_sft_data.teacher import prompts
from tests.conftest import DOC


def test_system_prompts_carry_document_and_profile_and_constraints():
    p = load(DOC)
    prof = PROFILES.get("companion")
    rng = random.Random(0)
    system = prompts.dialogue_system(p, prof, rng)
    assert p.name in system and p.core["정체성"] in system
    assert prof.writer_framing in system
    assert "4~35글자" in system and "반말" in system          # 제약 표에서 렌더링
    assert "U:" in system and "A:" in system
    assert p.principles[0][:10] in system and p.prohibitions[0][:10] in system
    assert "한 줄만" in prompts.respond_system(p, prof, rng)
    assert "예시" in system and "U: " + p.examples[0][0]["text"] in system


def test_vocabulary_sample_rotates_but_the_full_table_is_available():
    p = load(DOC)
    prof = PROFILES.get("companion")
    a = prompts.persona_block(p, prof, vocabulary_sample=3, rng=random.Random(1))
    b = prompts.persona_block(p, prof, vocabulary_sample=3, rng=random.Random(2))
    assert a != b
    assert all(f"- {k}:" in prompts.persona_block(p, prof) for k in p.vocabulary)


def test_dialogue_user_prompt_names_situation_flow_and_line_count():
    p = load(DOC)
    prof = PROFILES.get("companion")
    text = prompts.dialogue_user(p, prof, "배고픔", "사용자가 걱정하며 묻는 흐름", 3)
    assert "상황: 배고픔" in text and "걱정하며" in text and "총 6줄" in text


def test_translate_prompt_names_both_languages():
    assert "영어" in prompts.translate_system("en", "ko") and "한국어" in prompts.translate_system("en", "ko")
    assert prompts.language_name("xx") == "xx"
    assert prompts.translate_user("hi") == "hi"


def test_parse_repair_render_roundtrip():
    text = "A: 먼저 말함\nU: 안녕\nA: 응, 안녕!\nA: 뭐 해?\nU: 그냥"
    turns = prompts.repair_dialogue(prompts.parse_dialogue(text))
    assert turns == [{"role": "user", "text": "안녕"}, {"role": "assistant", "text": "응, 안녕! 뭐 해?"}]
    assert prompts.render_dialogue(turns) == "U: 안녕\nA: 응, 안녕! 뭐 해?"
    assert prompts.parse_dialogue("U: \nA: x") == []
    assert prompts.repair_dialogue([{"role": "assistant", "text": "x"}]) == []


def test_reply_text_takes_one_bare_line():
    assert prompts.reply_text('  A: "응, 좋아."\n두 번째 줄') == "응, 좋아."
    assert prompts.reply_text("") == "" and prompts.reply_text(None) == ""
