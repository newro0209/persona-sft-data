"""러너: 단계는 레코드를 낼 뿐이고 러너가 검증·중복 제거·게이트·통계·파일을 맡는다."""
import json
from dataclasses import dataclass

import pytest

from persona_sft_data.core.config import PipelineConfig
from persona_sft_data.core.registry import STAGES, TEACHERS
from persona_sft_data.core.runner import StageStats, execute, metric, reject_record
from persona_sft_data.core.schema import read_jsonl, write_jsonl
from tests.conftest import write_config


@dataclass(frozen=True)
class Empty:
    pass


@dataclass(frozen=True)
class WithTeacher:
    teacher: str


def _session(i, text):
    return {"id": f"g-{i}", "source": "gen", "turns": [{"role": "user", "text": "뭐 해?"}, {"role": "assistant", "text": text}]}


class Gen:
    name = config_name = "gen"
    mode, record_kind, produces = "records", "session", "raw"
    settings_type = WithTeacher
    def requires(self, config): return ()
    def instances(self, config): return [self]
    def preflight(self, ctx): pass
    def run(self, ctx):
        yield _session(1, "응, 같이 놀자.")
        yield _session(2, "응,  같이 놀자.")          # 지문이 같다
        yield {"id": "g-3", "source": "gen", "turns": [{"role": "user", "text": "야"}]}   # 스키마 위반
        yield _session(4, "잘래 🐾")                  # 게이트 위반
        yield metric(calls=3, failures=1, completion_tokens=10, rejected=1,
                     reject_reasons={"unparseable": 1}, source_filtered=2,
                     source_filter_reasons={"off_topic": 2})


class Utt:
    name = config_name = "utt"
    mode, record_kind, produces = "records", "utterance", "raw"
    settings_type = Empty
    def requires(self, config): return ()
    def instances(self, config): return [self]
    def preflight(self, ctx): pass
    def run(self, ctx):
        yield {"id": "u1", "text": "잘래 🐾", "source": "s", "language": "ko", "license": "mit"}  # 발화는 게이트 안 탄다
        yield {"id": "u2", "text": "야", "source": "s", "language": "korean", "license": "mit"}


class Art:
    name = config_name = "art"
    mode, record_kind, produces = "artifact", None, None
    settings_type = Empty
    def requires(self, config): return ("gen",)
    def instances(self, config): return [self]
    def preflight(self, ctx): pass
    def run(self, ctx):
        assert ctx.output is None and ctx.gate is None
        return StageStats(stage="art", output="x", started="now", produced=5)


class Reader:
    name = config_name = "reader"
    mode, record_kind, produces = "records", "session", "filtered"
    settings_type = Empty
    def requires(self, config): return ("gen",)
    def instances(self, config): return [self]
    def preflight(self, ctx): pass
    def run(self, ctx):
        yield from ctx.read("gen")


class Boom:
    """첫 반복에서 터지는 단계. 교사가 죽었거나 입력이 없을 때와 같은 자리다."""

    name = config_name = "boom"
    mode, record_kind, produces = "records", "session", "raw"
    settings_type = Empty
    def requires(self, config): return ()
    def instances(self, config): return [self]
    def preflight(self, ctx): pass
    def run(self, ctx):
        raise ValueError("교사가 죽었다")
        yield {}   # 제너레이터로 만들려고 둔다. 예외는 첫 반복에서 나온다.


class Judge:
    """자기 판단으로 거절한 레코드를 센티널로 넘기는 단계."""

    name = config_name = "judge"
    mode, record_kind, produces = "records", "session", "raw"
    settings_type = Empty
    def requires(self, config): return ()
    def instances(self, config): return [self]
    def preflight(self, ctx): pass
    def run(self, ctx):
        yield _session(1, "응, 같이 놀자.")
        yield reject_record(_session(2, "응, 같이 놀자아."), ["overused"])


class Fin:
    """finalize 훅을 선언한 단계. 러너가 통과시킨 것만 보고 파생 파일을 쓴다."""

    name = config_name = "fin"
    mode, record_kind, produces = "records", "session", "raw"
    settings_type = Empty
    def requires(self, config): return ()
    def instances(self, config): return [self]
    def preflight(self, ctx): pass
    def run(self, ctx):
        yield _session(1, "응, 같이 놀자.")
        yield _session(2, "잘래 🐾")                  # 게이트 위반
    def finalize(self, ctx, stats):
        ids = [r["id"] for r in read_jsonl(ctx.output)]
        (ctx.output.parent / "fin.txt").write_text(f"{stats.produced}:{','.join(ids)}", encoding="utf-8")


class FakeTeacherFactory:
    """설정의 ``kind: "fake"``가 통과하도록 두는 최소 팩토리. 내장 fake가 있으면 그것을 쓴다."""

    name = "fake"
    def build(self, cfg):
        return None


@pytest.fixture(autouse=True)
def _plugins():
    for cls in (Gen, Utt, Art, Reader, Boom, Judge, Fin):
        STAGES.add(cls.name, cls, origin="plugins")
    if "fake" not in TEACHERS.names():
        TEACHERS.add("fake", FakeTeacherFactory(), origin="plugins")


def test_records_stage_writes_output_rejects_sample_and_stats(tmp_path):
    cfg = PipelineConfig.load(write_config(tmp_path, stages={"gen": {"teacher": "fake"}}))
    stats = execute(Gen(), cfg, log=lambda m: None)
    assert (stats.produced, stats.duplicates, stats.rejected) == (1, 1, 3)
    assert stats.teacher_calls == 3 and stats.teacher_failures == 1 and stats.completion_tokens == 10
    assert stats.source_filtered == 2 and stats.source_filter_reasons == {"off_topic": 2}
    assert stats.teacher_model == "fake"
    assert stats.reject_reasons["emoji"] == 1 and stats.reject_reasons["unparseable"] == 1
    assert stats.reject_reasons["duplicate"] == 1 and any(k.startswith("schema:") for k in stats.reject_reasons)
    out = cfg.raw("gen")
    assert [r["id"] for r in read_jsonl(out)] == ["g-1"]
    rejected = list(read_jsonl(cfg.rejected_path(out)))
    assert {r["id"] for r in rejected} == {"g-2", "g-3", "g-4"} and all("_reject_reasons" in r for r in rejected)
    assert len(list(read_jsonl(cfg.sample_path(out)))) == 1
    written = json.loads(cfg.stats_path(out).read_text(encoding="utf-8"))
    assert written["yield_rate"] == 0.25 and written["environment"]["python"]


def test_utterance_stage_is_validated_but_not_gated(tmp_path):
    cfg = PipelineConfig.load(write_config(tmp_path, stages={"utt": {}}))
    stats = execute(Utt(), cfg, log=lambda m: None)
    assert stats.produced == 1 and stats.rejected == 1
    assert list(read_jsonl(cfg.raw("utt")))[0]["id"] == "u1"


def test_artifact_stage_returns_its_own_stats(tmp_path):
    cfg = PipelineConfig.load(write_config(tmp_path, stages={"art": {}}))
    assert execute(Art(), cfg, log=lambda m: None).produced == 5
    assert not (cfg.data_root / "raw").exists() or not list((cfg.data_root / "raw").iterdir())


def test_reading_a_missing_upstream_file_names_the_stage_to_run(tmp_path):
    cfg = PipelineConfig.load(write_config(tmp_path, stages={"reader": {}}))
    with pytest.raises(FileNotFoundError, match="'gen'"):
        execute(Reader(), cfg, log=lambda m: None)


def test_reader_sees_only_what_the_upstream_kept(tmp_path):
    cfg = PipelineConfig.load(write_config(tmp_path, stages={"gen": {"teacher": "fake"}, "reader": {}}))
    execute(Gen(), cfg, log=lambda m: None)
    stats = execute(Reader(), cfg, log=lambda m: None)
    assert stats.produced == 1 and cfg.filtered("reader").exists()


def test_a_failing_stage_leaves_the_previous_output_stats_and_sample_alone(tmp_path):
    """교사가 죽은 야간 재실행이 전날 산출물을 0바이트로 지우면 안 된다."""
    cfg = PipelineConfig.load(write_config(tmp_path, stages={"boom": {}}))
    out = cfg.raw("boom")
    out.parent.mkdir(parents=True, exist_ok=True)
    yesterday = [_session(9, "어제 만든 것.")]
    write_jsonl(out, yesterday)
    write_jsonl(cfg.sample_path(out), yesterday)
    cfg.rejected_path(out).write_text("", encoding="utf-8")
    cfg.stats_path(out).write_text('{"produced": 58}', encoding="utf-8")
    before = out.read_bytes()

    with pytest.raises(ValueError, match="교사가 죽었다"):
        execute(Boom(), cfg, log=lambda m: None)

    assert out.read_bytes() == before
    assert [r["id"] for r in read_jsonl(cfg.sample_path(out))] == ["g-9"]
    assert json.loads(cfg.stats_path(out).read_text(encoding="utf-8"))["produced"] == 58
    assert not list(out.parent.glob("*.tmp"))


def test_a_successful_stage_leaves_no_tmp_files(tmp_path):
    cfg = PipelineConfig.load(write_config(tmp_path, stages={"gen": {"teacher": "fake"}}))
    execute(Gen(), cfg, log=lambda m: None)
    assert list(read_jsonl(cfg.raw("gen")))
    assert not list((cfg.data_root / "raw").glob("*.tmp"))


def test_a_stage_rejection_lands_in_the_rejected_file_and_is_counted_once(tmp_path):
    cfg = PipelineConfig.load(write_config(tmp_path, stages={"judge": {}}))
    stats = execute(Judge(), cfg, log=lambda m: None)
    assert (stats.produced, stats.rejected, stats.duplicates) == (1, 1, 0)
    assert stats.reject_reasons == {"overused": 1}
    out = cfg.raw("judge")
    assert [r["id"] for r in read_jsonl(out)] == ["g-1"]
    rejected = list(read_jsonl(cfg.rejected_path(out)))
    assert [(r["id"], r["_reject_reasons"]) for r in rejected] == [("g-2", ["overused"])]


def test_finalize_is_optional_and_sees_only_what_passed(tmp_path):
    cfg = PipelineConfig.load(write_config(tmp_path, stages={"fin": {}, "gen": {"teacher": "fake"}}))
    execute(Fin(), cfg, log=lambda m: None)
    assert (cfg.raw("fin").parent / "fin.txt").read_text(encoding="utf-8") == "1:g-1"
    # 훅이 없는 단계는 영향받지 않는다.
    execute(Gen(), cfg, log=lambda m: None)
    assert not hasattr(Gen(), "finalize")
