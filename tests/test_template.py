"""The template stage: does it cover the persona without quoting it?

Two properties matter here and neither is about the runner. Every record has to
pass the quality gate — that is the whole reason this slice exists, so a
regression that produces one bad record has broken the stage's purpose. And
every phrase the pet says has to be traceable to the persona document, because
the stage is what replaced 200 hand-written pet sentences.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from persona_sft_data import runner
from persona_sft_data.config import PipelineConfig, TeacherConfig
from persona_sft_data.gates import Gate
from persona_sft_data.persona import load_cached
from persona_sft_data.stages import template as T

ROOT = Path(__file__).resolve().parents[1]
PERSONA_DOC = ROOT / "personas" / "mongle.md"
TERMINAL = ".?!~…"


def _config(tmp_path: Path, **stage) -> PipelineConfig:
    """A config rooted in tmp_path. Every path the stage writes comes from it."""
    return PipelineConfig(
        path=tmp_path / "config.json",
        root=tmp_path,
        data_root=tmp_path / "data",
        seed=20260904,
        persona_doc=PERSONA_DOC,
        teachers={
            "bulk": TeacherConfig(
                name="bulk", model="test-model", base_url="http://127.0.0.1:1"
            )
        },
        stages={"template": stage, "filter": {}},
    )


def _run(tmp_path: Path, limit: int = 300):
    config = _config(tmp_path, limit=limit)
    logs: list[str] = []
    stats = runner.execute(T.TemplateStage(), config, log=logs.append)
    text = config.raw("template").read_text(encoding="utf-8")
    records = [json.loads(line) for line in text.splitlines() if line.strip()]
    return stats, records, logs


@pytest.fixture(scope="module")
def persona():
    return load_cached(PERSONA_DOC)


# --- the properties the slice exists for -----------------------------------

def test_every_record_passes_the_gate(tmp_path, persona):
    stats, records, _ = _run(tmp_path)
    gate = Gate.from_config(persona, {})

    assert records, "the stage produced nothing"
    for record in records:
        verdict = gate.check(record)
        assert verdict.ok, f"{verdict.reasons}: {record['turns']}"
    # The runner gates too, so a rejected record would never reach the file.
    # Assert on the count as well, or the check above passes vacuously.
    assert stats.rejected == 0
    assert stats.duplicates == 0
    assert stats.produced == len(records)


def test_pet_speech_is_assembled_only_from_the_persona_vocabulary(tmp_path, persona):
    _, records, _ = _run(tmp_path)
    allowed = {
        phrase.rstrip(TERMINAL)
        for phrases in persona.vocabulary.values()
        for phrase in phrases
    }

    for record in records:
        for turn in record["turns"]:
            if turn["role"] != "pet":
                continue
            for fragment in turn["text"].split(", "):
                assert fragment.rstrip(TERMINAL) in allowed, turn["text"]


def test_user_turns_raise_the_beat_they_are_filed_under(tmp_path):
    _, records, _ = _run(tmp_path)
    for record in records:
        users = [t["text"] for t in record["turns"] if t["role"] == "user"]
        assert len(users) == len(record["beats"])
        for beat, text in zip(record["beats"], users):
            assert beat in text


def test_records_carry_the_provenance_the_corpus_schema_expects(tmp_path, persona):
    stats, records, _ = _run(tmp_path, limit=120)
    beats = set(persona.beats)

    assert [r["id"] for r in records] == [
        f"template-{i:06d}" for i in range(len(records))
    ]
    for record in records:
        assert record["source"] == "template"
        assert record["license"] == "synthetic"
        assert record["generator"] == ["template"]
        assert record["scenario"] in beats
        assert record["scenario"] == record["beats"][0]
        assert len(record["turns"]) == 2 * len(record["beats"])
    assert stats.teacher_calls == 0, "this stage must never call a teacher"


def test_the_module_never_writes_the_pet_name(persona):
    """The design doc's rule: persona strings live in the document, not in code."""
    source = Path(T.__file__).read_text(encoding="utf-8")
    assert persona.name not in source


# --- the slot filling itself ------------------------------------------------

def test_the_limit_is_respected(tmp_path):
    stats, records, _ = _run(tmp_path, limit=7)
    assert stats.produced == 7
    assert len(records) == 7


def test_the_same_seed_produces_the_same_corpus(tmp_path):
    first = _run(tmp_path / "a")[1]
    second = _run(tmp_path / "b")[1]
    assert first == second


def test_beats_are_grouped_by_their_situation_line(persona):
    groups = T.beat_groups(persona)
    known = set(persona.beats)

    assert groups
    for group in groups:
        assert set(group) <= known
        assert all(len(beat) <= T.MAX_BEAT_CHARS for beat in group)
    # Grouping, not flattening: at least one line contributes several beats, or
    # a multi-exchange session could never walk one situation.
    assert any(len(group) > 1 for group in groups)


def test_a_two_pole_row_answers_the_pole_the_beat_names():
    """A label names two poles and the row lists their phrases in that order.

    Synthetic vocabulary rather than the persona's, so the rule is tested and
    not the document: answering 춥다 out of the 덥다 half is the failure this
    guards, and it is exactly what an earlier version did.
    """
    vocabulary = {
        "평온": ("가가", "나나"),
        "덥다·춥다": ("더워", "많이 더워", "추워", "많이 추워"),
    }
    assert set(T.heads_for("춥다", vocabulary)) == {"추워", "많이 추워"}
    assert set(T.heads_for("덥다", vocabulary)) == {"더워", "많이 더워"}


def test_a_beat_with_no_signal_falls_back_to_the_calm_row():
    vocabulary = {"평온": ("가가", "나나"), "덥다·춥다": ("더워", "추워")}
    assert T.heads_for("전혀 다른 말", vocabulary) == ("가가", "나나")


def test_a_matching_phrase_leads_its_own_reply():
    vocabulary = {"평온": ("가가",), "지루함·놀이": ("심심해", "같이 놀자")}
    assert T.heads_for("심심함", vocabulary) == ("심심해",)


def test_replies_never_join_two_phrases_that_repeat_each_other():
    """The gate rejects repeated fragments, so they are never composed.

    Here the only phrase available to pair with the head repeats a word of it,
    so the head has to stand alone.
    """
    vocabulary = {"평온": ("놀자 좋아",), "지루함·놀이": ("심심해 놀자",)}
    replies = T.replies_for("심심함", vocabulary, min_chars=2, max_chars=35)
    assert replies == ("심심해 놀자.",)


def test_replies_stay_inside_the_length_rule_of_the_document():
    vocabulary = {"평온": ("가가", "나나"), "지루함·놀이": ("심심해", "같이 놀자")}
    replies = T.replies_for("심심함", vocabulary, min_chars=6, max_chars=9)
    assert replies
    assert all(6 <= len(reply) <= 9 for reply in replies)


def test_the_particle_agrees_with_the_beats_final_syllable():
    """받침 decides 은/는, and the frames compute it rather than guess."""
    frame = T.USER_FRAMES[0]
    assert frame("배고픔").startswith("배고픔은")  # 픔 has a final consonant
    assert frame("인사").startswith("인사는")  # 사 does not
