"""The real stage: parsing, filtering, and the one column that must never load.

No network and no GPU. The SmileStyle side runs off an inline TSV dropped into
the stage's own cache directory, which is also what proves the cache is used;
``_fetch`` is replaced by a function that raises, so any attempt to reach the
network fails the test rather than passing it slowly.

The parquet side is tested without pyarrow — it is a training-side dependency
and this suite runs in the export environment. What is tested there is the part
that matters: ``jojo0217/korean_safe_conversation`` answers questions as an AI
assistant, and reading that column would inject the exact sentence the persona
document forbids into the corpus.
"""

from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from persona_sft_data import backend, runner
from persona_sft_data.config import PipelineConfig, TeacherConfig
from persona_sft_data.gates import Gate
from persona_sft_data.persona import load_cached
from persona_sft_data.stages import real

ROOT = Path(__file__).resolve().parents[1]
PERSONA_DOC = ROOT / "personas" / "mongle.md"

# Three columns of the real 17, so the test also covers "read the casual ones
# and leave the rest". The formal column is written so that its leaking into the
# corpus would be visible.
FIXTURE_TSV = "\n".join(
    [
        "formal\tinformal\tchat",
        "저는 지금 몹시 배가 고픕니다\t오늘따라 너무 배고프다\t나 배고파서 죽겠어",
        "졸음이 쏟아지고 있습니다\t너무 졸려서 눈이 감긴다\t",
        "함께 놀고 싶습니다\t우리 같이 놀자\t같이 놀래?",
        "환율 변동성이 큽니다\t환율 변동성이 커졌다는 기사\t",
        "\t\t",
    ]
)

# What the answer column of korean_safe_conversation actually contains.
AI_ANSWER = "저는 인공지능 챗봇이기 때문에 여행을 떠나지는 못했습니다."


class TrackingRow(dict):
    """A parquet row that records which columns anybody looked at."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.read: list[str] = []

    def get(self, key, default=None):
        self.read.append(key)
        return super().get(key, default)

    def __getitem__(self, key):
        self.read.append(key)
        return super().__getitem__(key)


def _config(tmp_path: Path, **stage) -> PipelineConfig:
    stage.setdefault("teacher", "bulk")
    return PipelineConfig(
        path=tmp_path / "config.json",
        root=tmp_path,
        data_root=tmp_path / "data",
        seed=20260904,
        persona_doc=PERSONA_DOC,
        teachers={
            "bulk": TeacherConfig(
                name="bulk",
                model="test-model",
                base_url="http://127.0.0.1:1",
                concurrency=4,
            )
        },
        stages={"real": stage, "filter": {}},
    )


@pytest.fixture
def offline(monkeypatch):
    """Make every download attempt fail, and record what was attempted."""
    attempted: list[str] = []

    def _raise(url: str, timeout: float) -> bytes:
        attempted.append(url)
        raise urllib.error.URLError("the tests are offline")

    monkeypatch.setattr(real, "_fetch", _raise)
    return attempted


def _run(tmp_path: Path, teacher, *, tsv: str | None = FIXTURE_TSV, **stage):
    config = _config(tmp_path, **stage)
    if tsv is not None:
        cache = config.data_root / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        (cache / real.SMILESTYLE_CACHE).write_text(tsv, encoding="utf-8")
    logs: list[str] = []
    stats = runner.execute(real.RealStage(teacher), config, log=logs.append)
    path = config.raw("real")
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return stats, records, logs


@pytest.fixture(scope="module")
def persona():
    return load_cached(PERSONA_DOC)


# --- the column that must never be read ------------------------------------

def test_the_answer_column_is_never_read(persona):
    """The instruction is taken; the AI-assistant answer is not even looked at."""
    rows = [
        TrackingRow({"instruction": "오늘 뭐 하고 놀까", "output": AI_ANSWER}),
        TrackingRow({"instruction": "배고픈데 뭐 먹지", "output": AI_ANSWER}),
    ]

    utterances = real.safe_conversation_utterances(rows)

    assert [u.text for u in utterances] == ["오늘 뭐 하고 놀까", "배고픈데 뭐 먹지"]
    for row in rows:
        assert set(row.read) == {"instruction"}, row.read
    assert all(AI_ANSWER not in u.text for u in utterances)


def test_the_parquet_reader_projects_only_the_instruction_column():
    """The projection is the second half of the guarantee.

    ``safe_conversation_utterances`` not reading the answers is not enough on
    its own — the column list handed to parquet is what stops them being decoded
    off disk at all, so it is asserted rather than assumed.
    """
    assert real.SAFE_CONVERSATION_COLUMNS == (real.INSTRUCTION_COLUMN,)
    assert real.INSTRUCTION_COLUMN == "instruction"


def test_safe_conversation_records_carry_its_licence():
    rows = [{"instruction": "같이 놀자"}, {"instruction": "   "}]
    utterances = real.safe_conversation_utterances(rows)

    assert len(utterances) == 1
    assert utterances[0].dataset == "korean_safe_conversation"
    assert utterances[0].license == "apache-2.0"
    assert utterances[0].url == real.SAFE_CONVERSATION_URL


# --- SmileStyle parsing -----------------------------------------------------

def test_only_the_casual_columns_of_smilestyle_are_read():
    utterances = real.smilestyle_utterances(FIXTURE_TSV.encode("utf-8"))
    texts = [u.text for u in utterances]

    assert "오늘따라 너무 배고프다" in texts
    assert "나 배고파서 죽겠어" in texts
    assert not any("습니다" in text or "됩니다" in text for text in texts)
    assert all(text.strip() for text in texts)
    assert {u.dataset for u in utterances} == {"smilestyle"}
    assert {u.license for u in utterances} == {"smilestyle"}


def test_smilestyle_returns_nothing_when_the_columns_are_missing():
    """A changed schema is reported by the caller, not guessed around."""
    assert real.smilestyle_utterances(b"a\tb\n1\t2\n") == []


# --- the topic filter -------------------------------------------------------

def test_the_topic_filter_keeps_the_persona_world_and_drops_the_rest(persona):
    signal = real.topic_signal(persona)

    assert real.in_scope("오늘따라 너무 배고프다", signal)
    assert real.in_scope("우리 같이 놀자", signal)
    assert not real.in_scope("환율 변동성이 커졌다는 기사", signal)
    # Cheaper to drop here than to spend a teacher call on a record the gate
    # would reject for its script anyway.
    assert not real.in_scope("오늘 배고픈데 delivery 시킬까", signal)
    assert not real.in_scope("漢字가 섞인 문장 배고파", signal)
    assert not real.in_scope("배고파 " * 20, signal)
    assert not real.in_scope("응", signal, min_chars=2, max_chars=60)


def test_slurs_are_dropped_without_taking_ordinary_words_with_them():
    """The stems match the start of a token, so a shrimp stays a shrimp.

    Both cases below are real utterances from the sources: the second is why a
    plain substring match cannot be used.
    """
    assert real.is_unsafe("개새끼들 진짜 돈 편하게번다")
    assert not real.is_unsafe("딱새우와 가시발새우는 같은 종류인가요?")
    assert not real.is_unsafe("도로변에 새끼곰이 있었음")
    assert not real.is_unsafe("소나무들이 수만그루씩 죽어서 그루터기만 남았어")


def test_the_blocked_stems_can_be_replaced_from_the_stage_config(tmp_path, offline):
    tsv = "\n".join(
        ["formal\tinformal\tchat", "\t오늘따라 너무 배고프다\t나 배고파서 죽겠어"]
    )
    stats, records, _ = _run(
        tmp_path,
        backend.FakeTeacher(default="응, 같이 있자."),
        tsv=tsv,
        blocked_stems=["배고"],
    )

    assert records == []
    assert stats.source_filter_reasons["unsafe_source"] == 2


# --- reply parsing ----------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("응, 배고파.", "응, 배고파."),
        ("P: 응, 배고파.", "응, 배고파."),
        ('"응, 배고파."', "응, 배고파."),
        ("\n\n  응, 배고파.\n두 번째 줄은 버린다", "응, 배고파."),
        ("", ""),
        (None, ""),
    ],
)
def test_reply_text_takes_one_bare_line(raw, expected):
    assert real.reply_text(raw) == expected


# --- the stage end to end ---------------------------------------------------

def test_the_stage_generates_a_reply_for_each_human_utterance(
    tmp_path, offline, persona
):
    teacher = backend.FakeTeacher(default="P: 응, 같이 있자.")
    stats, records, logs = _run(tmp_path, teacher)
    gate = Gate.from_config(persona, {})

    assert records
    for index, record in enumerate(records):
        assert record["id"] == f"real-{index:06d}"
        assert record["source"] == "real"
        assert record["scenario"] == real.SCENARIO
        assert record["generator"] == ["test-model"]
        assert record["real_source"] == "smilestyle"
        assert record["license"] == "smilestyle"
        assert record["source_url"] == real.SMILESTYLE_URL
        assert gate.check(record).ok
        # The user turn is human-written and the pet turn is generated: the
        # role tag the teacher added is stripped, not rejected.
        assert record["turns"][1]["text"] == "응, 같이 있자."

    user_turns = {r["turns"][0]["text"] for r in records}
    assert "오늘따라 너무 배고프다" in user_turns
    assert "환율 변동성이 커졌다는 기사" not in user_turns
    assert not any("습니다" in text for text in user_turns)

    assert stats.teacher_calls == len(records)
    assert stats.teacher_failures == 0
    assert stats.source_filter_reasons.get("off_topic", 0) >= 1
    # The teacher saw the human utterance itself, and a system prompt built
    # from the persona document.
    assert {req.user for req in teacher.seen} == user_turns
    assert all(persona.name in req.system for req in teacher.seen)


def test_a_cached_source_is_not_downloaded_again(tmp_path, offline):
    _run(tmp_path, backend.FakeTeacher(default="응, 같이 있자."))

    assert real.SMILESTYLE_URL not in offline
    assert real.SAFE_CONVERSATION_URL in offline


def test_a_failed_source_is_logged_and_the_other_one_still_runs(tmp_path, offline):
    stats, records, logs = _run(tmp_path, backend.FakeTeacher(default="응, 같이 있자."))
    report = "\n".join(logs)

    assert records, "the working source produced nothing"
    assert real.SAFE_CONVERSATION_URL in report
    assert "could not download" in report


def test_every_source_failing_produces_nothing_instead_of_raising(tmp_path, offline):
    stats, records, logs = _run(
        tmp_path, backend.FakeTeacher(default="응, 같이 있자."), tsv=None
    )

    assert records == []
    assert stats.produced == 0
    assert "no source could be read" in "\n".join(logs)


def test_an_empty_teacher_reply_is_counted_rather_than_written(tmp_path, offline):
    stats, records, _ = _run(tmp_path, backend.FakeTeacher(default="   "))

    assert records == []
    assert stats.reject_reasons["empty_reply"] > 0
    assert stats.teacher_calls > 0


def test_the_limit_bounds_the_number_of_teacher_calls(tmp_path, offline):
    teacher = backend.FakeTeacher(default="응, 같이 있자.")
    stats, records, _ = _run(tmp_path, teacher, limit=2)

    assert stats.teacher_calls == 2
    assert len(records) == 2


def test_the_module_never_writes_the_pet_name(persona):
    source = Path(real.__file__).read_text(encoding="utf-8")
    assert persona.name not in source
