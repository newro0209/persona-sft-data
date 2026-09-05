"""설정: 단일 출처, 단계 설정은 플러그인의 dataclass로 검증, 참조는 전부 존재해야 한다."""
from dataclasses import dataclass

import pytest

from persona_sft_data.core.config import ConfigError, PipelineConfig, TeacherConfig, build_settings
from persona_sft_data.core.registry import STAGES, TEACHERS, TRANSLATORS
from tests.conftest import write_config


@dataclass(frozen=True)
class GenSettings:
    teacher: str
    per_situation: int = 1


@dataclass(frozen=True)
class MixSettings:
    ratios: dict
    split: dict


class GenStage:
    name = config_name = "gen"
    mode, record_kind, produces = "records", "session", "raw"
    settings_type = GenSettings


class IngestLike:
    name = config_name = "ingest"
    mode, record_kind, produces = "records", "utterance", "raw"
    @dataclass(frozen=True)
    class S:
        teacher: str
        translator: str
        sources: list
    settings_type = S


class MixStage:
    name = config_name = "assemble"
    mode, record_kind, produces = "records", "session", "final"
    settings_type = MixSettings


class FakeTeacherFactory:
    name = "fake"
    def build(self, cfg):
        return None


class FakeTranslatorFactory:
    name = "teacher"
    def build(self, ctx, teacher):
        return None


@pytest.fixture(autouse=True)
def _plugins():
    STAGES.add("gen", GenStage, origin="plugins")
    STAGES.add("ingest", IngestLike, origin="plugins")
    STAGES.add("assemble", MixStage, origin="plugins")
    if "fake" not in TEACHERS.names():  # 내장 fake가 있으면 그것을 쓴다 (test_runner와 같은 규칙)
        TEACHERS.add("fake", FakeTeacherFactory(), origin="plugins")
    TRANSLATORS.add("teacher", FakeTranslatorFactory(), origin="plugins")


def test_loads_and_derives_every_path_from_data_root(tmp_path):
    cfg = PipelineConfig.load(write_config(tmp_path, stages={"gen": {"teacher": "fake"}}))
    assert cfg.root == tmp_path and cfg.data_root == tmp_path / "data"
    assert cfg.raw("gen") == tmp_path / "data" / "raw" / "gen.jsonl"
    assert cfg.filtered("gen").parent.name == "filtered" and cfg.final("x").parent.name == "final"
    assert cfg.stats_path(cfg.raw("gen")).name == "gen.jsonl.stats.json"
    assert cfg.rejected_path(cfg.raw("gen")).name == "gen.jsonl.rejected.jsonl"
    assert cfg.sample_path(cfg.raw("gen")).name == "gen.jsonl.sample.jsonl"
    assert cfg.datasets_root == tmp_path / "datasets"
    assert cfg.student.model == "org/student-base" and cfg.profile == "dummy"


def test_stage_settings_are_typed_and_unknown_keys_fail(tmp_path):
    cfg = PipelineConfig.load(write_config(tmp_path, stages={"gen": {"teacher": "fake", "per_situation": 3}}))
    assert cfg.stage_settings("gen") == GenSettings(teacher="fake", per_situation=3)
    assert cfg.teacher_for("gen") == cfg.teachers["fake"]
    with pytest.raises(ConfigError, match="stages.gen.*nope"):
        PipelineConfig.load(write_config(tmp_path, stages={"gen": {"teacher": "fake", "nope": 1}}))
    with pytest.raises(ConfigError, match="stages.gen.*teacher"):
        PipelineConfig.load(write_config(tmp_path, stages={"gen": {}}))


def test_references_must_exist(tmp_path):
    with pytest.raises(ConfigError, match="teacher 'ghost'"):
        PipelineConfig.load(write_config(tmp_path, stages={"gen": {"teacher": "ghost"}}))
    with pytest.raises(ConfigError, match="stage 'nostage'"):
        PipelineConfig.load(write_config(tmp_path, stages={"nostage": {}}))
    with pytest.raises(ConfigError, match="profile 'nobody'"):
        PipelineConfig.load(write_config(tmp_path, profile="nobody"))
    with pytest.raises(ConfigError, match="source 'missing'"):
        PipelineConfig.load(write_config(tmp_path, stages={"ingest": {"teacher": "fake", "translator": "teacher", "sources": ["missing"]}}))
    with pytest.raises(ConfigError, match="translator 'none'"):
        PipelineConfig.load(write_config(tmp_path, stages={"ingest": {"teacher": "fake", "translator": "none", "sources": []}}))


def test_teacher_config_defaults_and_unknown_keys():
    t = TeacherConfig.from_dict("t", {"model": "m", "base_url": "http://x"})
    assert t.kind == "openai" and t.concurrency == 64 and t.api_key is None
    with pytest.raises(ConfigError, match="teacher 't'.*bogus"):
        TeacherConfig.from_dict("t", {"model": "m", "base_url": "http://x", "bogus": 1})
    with pytest.raises(ConfigError, match="base_url"):
        TeacherConfig.from_dict("t", {"model": "m"})


@pytest.mark.xfail(reason="Task 9에서 포맷 플러그인이 등록되면 통과")
def test_source_config_needs_exactly_one_of_url_or_path(tmp_path):
    good = {"format": "tsv", "url": "http://x/a.tsv", "fields": ["a"], "language": "ko", "license": "mit"}
    cfg = PipelineConfig.load(write_config(tmp_path, sources={"s": good}))
    assert cfg.source("s").url == "http://x/a.tsv" and cfg.source("s").extract_kind == "field"
    for broken, msg in (
        ({**good, "path": "x.tsv"}, "url.*path"),
        ({k: v for k, v in good.items() if k != "url"}, "url.*path"),
        ({**good, "fields": []}, "fields"),
        ({k: v for k, v in good.items() if k != "license"}, "license"),
        ({**good, "language": "korean"}, "language"),
    ):
        with pytest.raises(ConfigError, match=msg):
            PipelineConfig.load(write_config(tmp_path, sources={"s": broken}))
    with_extract = {**good, "extract": {"kind": "regex", "pattern": "x"}}
    cfg = PipelineConfig.load(write_config(tmp_path, sources={"s": with_extract}))
    assert cfg.source("s").extract_kind == "regex" and cfg.source("s").extract == {"pattern": "x"}
    local = {**{k: v for k, v in good.items() if k != "url"}, "path": "fixtures/a.tsv"}
    cfg = PipelineConfig.load(write_config(tmp_path, sources={"s": local}))
    assert cfg.source("s").path == tmp_path / "fixtures" / "a.tsv"


def test_student_and_top_level_validation(tmp_path):
    with pytest.raises(ConfigError, match="student.model"):
        PipelineConfig.load(write_config(tmp_path, student={"model": ""}))
    with pytest.raises(ConfigError, match="chat_template"):
        PipelineConfig.load(write_config(tmp_path, student={"model": "m", "chat_template": "llama3"}))
    with pytest.raises(ConfigError, match="language"):
        PipelineConfig.load(write_config(tmp_path, language="kor"))
    with pytest.raises(ConfigError, match="'seed'"):
        PipelineConfig.load(write_config(tmp_path, seed=None))


def test_validate_pipeline_checks_the_dag(tmp_path):
    ok = PipelineConfig.load(write_config(tmp_path, stages={
        "gen": {"teacher": "fake"},
        "assemble": {"ratios": {"gen": 1.0}, "split": {"train": 0.8, "val": 0.1, "test": 0.1}},
    }))
    ok.validate_pipeline()
    assert ok.session_stages() == ("gen",)
    bad_ratio = PipelineConfig.load(write_config(tmp_path, stages={
        "gen": {"teacher": "fake"},
        "assemble": {"ratios": {"other": 1.0}, "split": {"train": 0.8, "val": 0.1, "test": 0.1}},
    }))
    with pytest.raises(ConfigError, match="ratios"):
        bad_ratio.validate_pipeline()
    bad_split = PipelineConfig.load(write_config(tmp_path, stages={
        "gen": {"teacher": "fake"},
        "assemble": {"ratios": {"gen": 1.0}, "split": {"train": 0.8, "val": 0.1, "test": 0.5}},
    }))
    with pytest.raises(ConfigError, match="split"):
        bad_split.validate_pipeline()
    no_assemble = PipelineConfig.load(write_config(tmp_path, stages={"gen": {"teacher": "fake"}}))
    with pytest.raises(ConfigError, match="assemble"):
        no_assemble.validate_pipeline()


def test_stage_seeds_are_deterministic_and_distinct(tmp_path):
    cfg = PipelineConfig.load(write_config(tmp_path))
    assert cfg.stage_seed("gen") != cfg.stage_seed("ingest")
    assert cfg.stage_seed("gen") == PipelineConfig.load(cfg.path).stage_seed("gen")


def test_build_settings_reports_where():
    @dataclass(frozen=True)
    class S:
        a: int
        b: int = 2
    assert build_settings(S, {"a": 1}, "x") == S(1, 2)
    with pytest.raises(ConfigError, match="x.*'a'"):
        build_settings(S, {}, "x")
