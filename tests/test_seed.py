"""Seed stage tests. A FakeTeacher stands in for vLLM, so no GPU is involved."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from persona_sft_data import runner, schema
from persona_sft_data.backend import FakeTeacher, Request, Result, TeacherError
from persona_sft_data.config import PipelineConfig, TeacherConfig
from persona_sft_data.persona import load_cached
from persona_sft_data.stages.seed import SeedStage

REPO = Path(__file__).resolve().parent.parent
PERSONA_DOC = REPO / "personas" / "mongle.md"
MODEL = "test/teacher-reasoner"


def make_config(
    root: Path, *, seed: int = 11, per_situation: int = 2, concurrency: int = 16
) -> PipelineConfig:
    """A config that touches nothing outside ``root`` but the frozen persona."""
    return PipelineConfig(
        path=root / "configs" / "test.json",
        root=root,
        data_root=root / "data",
        seed=seed,
        persona_doc=PERSONA_DOC,
        teachers={
            "reasoner": TeacherConfig(
                name="reasoner",
                model=MODEL,
                base_url="http://127.0.0.1:0",
                concurrency=concurrency,
            )
        },
        stages={
            "seed": {
                "teacher": "reasoner",
                "per_situation": per_situation,
                "turns": [2, 3],
            }
        },
    )


def beats() -> tuple[str, ...]:
    return load_cached(PERSONA_DOC).beats


def replies(per_situation: int) -> dict[str, str]:
    """One distinct two-turn dialogue per request key, so nothing is a duplicate."""
    return {
        f"{b}:{n}": f"U: 오늘 뭐 했어 {b}-{n}\nP: 너랑 놀았지 {b}-{n}"
        for b in range(len(beats()))
        for n in range(per_situation)
    }


def silent(_message: str) -> None:
    return None


class BatchRecordingTeacher(FakeTeacher):
    """Remembers how many requests each round trip carried."""

    def __init__(self, replies: dict[str, str]) -> None:
        super().__init__(replies)
        self.batch_sizes: list[int] = []

    def generate(self, requests: Sequence[Request]) -> list[Result]:
        self.batch_sizes.append(len(requests))
        return super().generate(requests)


def test_seed_records_carry_the_corpus_shape(tmp_path):
    config = make_config(tmp_path, per_situation=2)
    teacher = FakeTeacher(replies(2))

    stats = runner.execute(SeedStage(teacher), config, log=silent)
    records = list(schema.read_jsonl(config.raw("seed")))

    assert stats.produced == len(beats()) * 2
    assert len(records) == stats.produced
    assert records[0]["id"] == "seed-000000"
    assert records[-1]["id"] == f"seed-{stats.produced - 1:06d}"
    assert records[0]["source"] == "teacher_seed"
    assert records[0]["license"] == "synthetic"
    # The model id comes from the config, never from the stage.
    assert records[0]["generator"] == [MODEL]
    assert stats.teacher_model == MODEL
    assert [t["role"] for t in records[0]["turns"]] == ["user", "pet"]
    # Every beat in the persona document is covered, not sampled.
    assert {r["scenario"] for r in records} == set(beats())
    assert stats.teacher_calls == len(beats()) * 2
    assert stats.completion_tokens > 0
    assert stats.teacher_failures == 0


def test_prompts_use_the_persona_document_and_the_configured_turn_counts(tmp_path):
    config = make_config(tmp_path, per_situation=1)
    teacher = FakeTeacher(replies(1))

    runner.execute(SeedStage(teacher), config, log=silent)

    assert [r.key for r in teacher.seen[:2]] == ["0:0", "1:0"]
    assert beats()[0] in teacher.seen[0].user
    assert load_cached(PERSONA_DOC).name in teacher.seen[0].system
    # Turn counts are sampled from settings["turns"]: both configured values
    # reach the teacher. Digits reach the user prompt only through that count.
    joined = "\n".join(r.user for r in teacher.seen)
    assert all(str(n * 2) in joined for n in config.stage("seed")["turns"])


def test_requests_are_issued_in_batches_of_the_teacher_concurrency(tmp_path):
    config = make_config(tmp_path, per_situation=2, concurrency=16)
    teacher = BatchRecordingTeacher(replies(2))

    runner.execute(SeedStage(teacher), config, log=silent)

    total = len(beats()) * 2
    assert sum(teacher.batch_sizes) == total
    assert max(teacher.batch_sizes) == 16
    assert len(teacher.batch_sizes) == -(-total // 16)


def test_unparseable_replies_are_counted_rather_than_crashing(tmp_path):
    config = make_config(tmp_path, per_situation=1)
    broken = replies(1)
    broken["0:0"] = ""  # empty completion
    broken["1:0"] = "죄송하지만 그 대화는 만들 수 없어."  # prose, no U:/P: lines
    teacher = FakeTeacher(broken)

    stats = runner.execute(SeedStage(teacher), config, log=silent)
    records = list(schema.read_jsonl(config.raw("seed")))

    assert stats.reject_reasons["unparseable"] == 2
    assert stats.produced == len(beats()) - 2
    assert len(records) == stats.produced
    # Ids stay contiguous: the index advances only when a record is yielded.
    assert [r["id"] for r in records[:2]] == ["seed-000000", "seed-000001"]


def test_failed_calls_are_reported_as_failures_and_rejects(tmp_path):
    class FlakyTeacher(FakeTeacher):
        def generate(self, requests: Sequence[Request]) -> list[Result]:
            out = super().generate(requests)
            return [
                Result(r.key, None, error="boom") if r.key.endswith(":0") else r
                for r in out
            ]

    config = make_config(tmp_path, per_situation=1)
    stats = runner.execute(SeedStage(FlakyTeacher(replies(1))), config, log=silent)

    assert stats.produced == 0
    assert stats.teacher_failures == len(beats())
    assert stats.reject_reasons["teacher_error"] == len(beats())


def test_the_stage_checks_the_server_before_generating_anything(tmp_path):
    class WrongModelTeacher(FakeTeacher):
        def check(self) -> None:
            raise TeacherError("server serves a different model")

    teacher = WrongModelTeacher({})
    with pytest.raises(TeacherError):
        runner.execute(SeedStage(teacher), make_config(tmp_path), log=silent)
    assert teacher.seen == []


def test_the_same_seed_produces_the_same_prompts(tmp_path):
    fixed = replies(2)

    def prompts_of(root: str, seed: int) -> list[tuple[str, str, str]]:
        teacher = FakeTeacher(fixed)
        config = make_config(tmp_path / root, seed=seed, per_situation=2)
        runner.execute(SeedStage(teacher), config, log=silent)
        return [(r.key, r.system, r.user) for r in teacher.seen]

    first = prompts_of("a", 11)
    again = prompts_of("b", 11)
    other = prompts_of("c", 12)

    assert first == again
    assert first != other  # the rng is actually consulted
