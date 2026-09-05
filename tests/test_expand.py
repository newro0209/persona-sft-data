"""Expand stage tests. A FakeTeacher stands in for vLLM, so no GPU is involved."""

from collections.abc import Sequence
from pathlib import Path

import pytest

from persona_sft_data import prompts, runner, schema
from persona_sft_data.backend import FakeTeacher, Request, Result, TeacherError
from persona_sft_data.config import PipelineConfig, TeacherConfig
from persona_sft_data.stages.expand import ExpandStage

REPO = Path(__file__).resolve().parent.parent
PERSONA_DOC = REPO / "personas" / "mongle.md"
MODEL = "test/teacher-bulk"
SEED_COUNT = 3


def make_config(
    root: Path, *, seed: int = 5, variants: int = 4, concurrency: int = 8
) -> PipelineConfig:
    """A config that touches nothing outside ``root`` but the frozen persona."""
    return PipelineConfig(
        path=root / "configs" / "test.json",
        root=root,
        data_root=root / "data",
        seed=seed,
        persona_doc=PERSONA_DOC,
        teachers={
            "bulk": TeacherConfig(
                name="bulk",
                model=MODEL,
                base_url="http://127.0.0.1:0",
                concurrency=concurrency,
            )
        },
        stages={"expand": {"teacher": "bulk", "variants_per_seed": variants}},
    )


def write_seeds(config: PipelineConfig, count: int = SEED_COUNT) -> list[dict]:
    seeds = [
        {
            "id": f"seed-{i:06d}",
            "source": "teacher_seed",
            "scenario": f"상황 {i}",
            "generator": ["some/upstream-model"],
            "license": "synthetic",
            "turns": [
                {"role": "user", "text": f"오늘 뭐 했어 {i}"},
                {"role": "pet", "text": f"너랑 놀았지 {i}"},
            ],
        }
        for i in range(count)
    ]
    schema.write_jsonl(config.raw("seed"), seeds)
    return seeds


def replies(count: int, variants: int) -> dict[str, str]:
    """One distinct rewrite per request key, so nothing is a duplicate."""
    return {
        f"{i}:{v}": f"U: 뭐 하고 지냈어 {i}-{v}\nP: 너 기다렸어 {i}-{v}"
        for i in range(count)
        for v in range(variants)
    }


def silent(_message: str) -> None:
    return None


def test_expand_records_carry_the_corpus_shape_and_their_seed(tmp_path):
    config = make_config(tmp_path, variants=4)
    seeds = write_seeds(config)
    teacher = FakeTeacher(replies(len(seeds), 4))

    stats = runner.execute(ExpandStage(teacher), config, log=silent)
    records = list(schema.read_jsonl(config.raw("expand")))

    assert stats.produced == len(seeds) * 4
    assert records[0]["id"] == "expand-000000"
    assert records[-1]["id"] == f"expand-{stats.produced - 1:06d}"
    assert records[0]["source"] == "teacher_expand"
    assert records[0]["license"] == "synthetic"
    # The model id comes from the config, never from the stage.
    assert records[0]["generator"] == [MODEL]
    assert stats.teacher_model == MODEL
    # Scenario is inherited so the seed stage's beat coverage survives.
    assert records[0]["scenario"] == seeds[0]["scenario"]
    assert records[0]["seed_id"] == seeds[0]["id"]
    assert {r["seed_id"] for r in records} == {s["id"] for s in seeds}
    assert [t["role"] for t in records[0]["turns"]] == ["user", "pet"]
    assert stats.teacher_calls == len(seeds) * 4
    assert stats.completion_tokens > 0


def test_the_teacher_is_shown_the_seed_dialogue(tmp_path):
    config = make_config(tmp_path, variants=2)
    seeds = write_seeds(config)
    teacher = FakeTeacher(replies(len(seeds), 2))

    runner.execute(ExpandStage(teacher), config, log=silent)

    assert [r.key for r in teacher.seen[:3]] == ["0:0", "0:1", "1:0"]
    assert prompts.render_dialogue(seeds[0]["turns"]) in teacher.seen[0].user


def test_variants_identical_to_their_seed_are_rejected(tmp_path):
    config = make_config(tmp_path, variants=2)
    seeds = write_seeds(config)
    echoed = replies(len(seeds), 2)
    # Verbatim echo, and an echo that differs only in whitespace: neither is a
    # variation, and the runner's dedupe cannot see it (the seed is elsewhere).
    echoed["0:0"] = prompts.render_dialogue(seeds[0]["turns"])
    echoed["1:0"] = prompts.render_dialogue(seeds[1]["turns"]).replace(" ", "  ")
    teacher = FakeTeacher(echoed)

    stats = runner.execute(ExpandStage(teacher), config, log=silent)
    records = list(schema.read_jsonl(config.raw("expand")))

    assert stats.reject_reasons["identical_to_seed"] == 2
    assert stats.produced == len(seeds) * 2 - 2
    assert len(records) == stats.produced
    seed_texts = {schema.session_text(s) for s in seeds}
    assert not any(schema.session_text(r) in seed_texts for r in records)


def test_unparseable_replies_are_counted_rather_than_crashing(tmp_path):
    config = make_config(tmp_path, variants=2)
    seeds = write_seeds(config)
    broken = replies(len(seeds), 2)
    broken["0:0"] = ""  # empty completion
    broken["1:1"] = "그 대화는 바꾸기 어려워."  # prose, no U:/P: lines
    teacher = FakeTeacher(broken)

    stats = runner.execute(ExpandStage(teacher), config, log=silent)
    records = list(schema.read_jsonl(config.raw("expand")))

    assert stats.reject_reasons["unparseable"] == 2
    assert stats.produced == len(seeds) * 2 - 2
    # Ids stay contiguous: the index advances only when a record is yielded.
    assert [r["id"] for r in records[:2]] == ["expand-000000", "expand-000001"]


def test_requests_are_issued_in_batches_of_the_teacher_concurrency(tmp_path):
    class BatchRecordingTeacher(FakeTeacher):
        def __init__(self, replies: dict[str, str]) -> None:
            super().__init__(replies)
            self.batch_sizes: list[int] = []

        def generate(self, requests: Sequence[Request]) -> list[Result]:
            self.batch_sizes.append(len(requests))
            return super().generate(requests)

    config = make_config(tmp_path, variants=6, concurrency=8)
    seeds = write_seeds(config)
    teacher = BatchRecordingTeacher(replies(len(seeds), 6))

    runner.execute(ExpandStage(teacher), config, log=silent)

    total = len(seeds) * 6
    assert sum(teacher.batch_sizes) == total
    assert max(teacher.batch_sizes) == 8
    assert len(teacher.batch_sizes) == -(-total // 8)


def test_a_missing_seed_file_fails_with_an_instruction(tmp_path):
    config = make_config(tmp_path)
    config.ensure_dirs()  # the directory exists; the seed output does not
    with pytest.raises(FileNotFoundError, match="seed"):
        runner.execute(ExpandStage(FakeTeacher({})), config, log=silent)


def test_the_stage_checks_the_server_before_generating_anything(tmp_path):
    class WrongModelTeacher(FakeTeacher):
        def check(self) -> None:
            raise TeacherError("server serves a different model")

    config = make_config(tmp_path)
    write_seeds(config)
    teacher = WrongModelTeacher({})
    with pytest.raises(TeacherError):
        runner.execute(ExpandStage(teacher), config, log=silent)
    assert teacher.seen == []


def test_the_same_seed_produces_the_same_prompts(tmp_path):
    fixed = replies(SEED_COUNT, 4)

    def prompts_of(root: str, seed: int) -> list[tuple[str, str, str]]:
        config = make_config(tmp_path / root, seed=seed, variants=4)
        write_seeds(config)
        teacher = FakeTeacher(fixed)
        runner.execute(ExpandStage(teacher), config, log=silent)
        return [(r.key, r.system, r.user) for r in teacher.seen]

    first = prompts_of("a", 5)
    again = prompts_of("b", 5)
    other = prompts_of("c", 6)

    assert first == again
    assert first != other  # the rng is actually consulted
