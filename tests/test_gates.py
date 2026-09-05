"""게이트: 규칙은 제약 표의 행에서만 켜진다. 실제 교사 출력에서 나온 위반을 잡는다."""
from pathlib import Path

import pytest

from persona_sft_data.core.gates import Gate, GateSettings, build_gate
from persona_sft_data.core.persona import PersonaError, load

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "personas" / "mongle.md"


def _doc_with_constraints(tmp_path: Path, rows: dict[str, str]) -> Path:
    """제약 표를 통째로 바꾼 문서."""
    text = DOC.read_text(encoding="utf-8")
    start = text.index("## 제약")
    end = text.index("\n## ", start + 1)
    table = "## 제약\n\n| 규칙 | 값 |\n| --- | --- |\n" + "".join(f"| {k} | {v} |\n" for k, v in rows.items())
    out = tmp_path / "p.md"
    out.write_text(text[:start] + table + text[end:], encoding="utf-8")
    return out


def _session(assistant_text: str, user_text: str = "뭐 해?") -> dict:
    return {"turns": [{"role": "user", "text": user_text}, {"role": "assistant", "text": assistant_text}]}


@pytest.fixture(scope="module")
def gate() -> Gate:
    return build_gate(load(DOC), GateSettings())


@pytest.mark.parametrize("text, reason", [
    ("잘래 🐾", "emoji"),
    ("네, 잘 먹었어요.", "honorific"),
    ("저는 인공지능이야.", "claims_to_be_ai"),
    ("응, 네 옆에 꼭 붙어서宠받고 싶어.", "cjk_characters"),
    ("좋아 좋아 좋아", "repeated_phrase"),
    ("응… 그래… 알겠어", "multiple_ellipsis"),
    ("가" * 60, "assistant_too_long"),
    ("**좋아**", "markdown"),
    ("우리 재밌게 놀자. A:", "role_label_in_text"),
])
def test_rejects_observed_violations(gate, text, reason):
    verdict = gate.check(_session(text))
    assert not verdict.ok and reason in verdict.reasons, verdict.reasons


@pytest.mark.parametrize("text", ["응, 배 고파.", "같이 놀자!", "하암, 졸려.", "그건 잘 모르겠어.", "조금 삐졌어…", "히히, 좋아."])
def test_passes_ordinary_speech(gate, text):
    assert gate.check(_session(text)).ok, gate.check(_session(text)).reasons


def test_user_turns_are_not_bound_by_persona_rules(gate):
    assert gate.check(_session("응, 좋아.", user_text="밥 먹었어요? 🐾")).ok


def test_rejects_name_derived_babytalk_and_third_person(gate):
    name = load(DOC).name
    assert "name_suffix_babytalk" in gate.check(_session(f"알겠어, 기다릴{name[0]}!")).reasons
    assert "third_person_self" in gate.check(_session(f"{name}이도 잘 잤어.")).reasons


def test_structural_faults(gate):
    assert "does_not_start_with_user" in gate.check({"turns": [{"role": "assistant", "text": "응."}]}).reasons
    assert "roles_not_alternating" in gate.check({"turns": [
        {"role": "user", "text": "야"}, {"role": "user", "text": "야"}, {"role": "assistant", "text": "응."}]}).reasons
    assert "empty" in gate.check({"turns": []}).reasons
    small = build_gate(load(DOC), GateSettings(min_turns=4, max_turns=4))
    assert "too_few_turns" in small.check(_session("응.")).reasons


def test_a_missing_row_switches_the_rule_off(tmp_path):
    rows = {"말투": "반말", "발화 길이": "4~35글자"}
    g = build_gate(load(_doc_with_constraints(tmp_path, rows)), GateSettings())
    assert g.check(_session("잘래 🐾")).ok                   # 이모지 행이 없다
    assert not g.check(_session("잘 먹었어요.")).ok


@pytest.fixture
def honorific_gate(tmp_path) -> Gate:
    return build_gate(load(_doc_with_constraints(tmp_path, {"말투": "존댓말"})), GateSettings())


def test_honorific_persona_rejects_informal_endings(honorific_gate):
    assert "informal_ending" in honorific_gate.check(_session("응, 좋아.")).reasons
    assert "informal_ending" in honorific_gate.check(_session("그래 알았어.")).reasons
    assert honorific_gate.check(_session("네, 좋습니다.")).ok
    assert honorific_gate.check(_session("정말 그래요?")).ok


@pytest.mark.parametrize("text", [
    "네.", "예.", "그렇군요.", "알겠습니다.", "무엇을 찾으시는지요?", "어서 오십시오.",
    "아니오.", "여기 있습니다.", "무엇을 도와드릴까요?", "그리 하시오.", "음…",
])
def test_honorific_persona_passes_ordinary_polite_speech(honorific_gate, text):
    """화이트리스트가 좁으면 '네.' 같은 정상 존댓말이 거절되어 수율이 깎인다."""
    assert honorific_gate.check(_session(text)).ok, honorific_gate.check(_session(text)).reasons


@pytest.mark.parametrize("text", ["응, 좋아.", "그래 알았어.", "같이 가자!", "재밌네.", "배고파"])
def test_honorific_persona_still_rejects_informal_speech(honorific_gate, text):
    assert "informal_ending" in honorific_gate.check(_session(text)).reasons


def test_free_register_and_mixed_script_add_no_rule(tmp_path):
    g = build_gate(load(_doc_with_constraints(tmp_path, {"말투": "자유", "문자": "혼용"})), GateSettings())
    assert g.check(_session("OK, 좋아요.")).ok


def test_length_in_sentences(tmp_path):
    g = build_gate(load(_doc_with_constraints(tmp_path, {"발화 길이": "1~2문장"})), GateSettings())
    assert g.check(_session("좋아. 같이 가자.")).ok
    assert "assistant_too_long" in g.check(_session("좋아. 같이 가자. 지금 바로. 어서.")).reasons


def test_english_script_persona(tmp_path):
    g = build_gate(load(_doc_with_constraints(tmp_path, {"문자": "영문"})), GateSettings())
    assert g.check(_session("Sure, let's go.")).ok
    assert "hangul_characters" in g.check(_session("Sure, 가자.")).reasons


@pytest.mark.parametrize("rows, message", [
    ({"말투": "중얼중얼"}, "말투"),
    ({"발화 길이": "짧게"}, "발화 길이"),
    ({"말줄임표": "많이"}, "말줄임표"),
    ({"온도": "낮게"}, "온도"),
])
def test_unknown_keys_and_bad_values_raise(tmp_path, rows, message):
    with pytest.raises(PersonaError, match=message):
        build_gate(load(_doc_with_constraints(tmp_path, rows)), GateSettings())
