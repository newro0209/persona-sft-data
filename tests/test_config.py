"""설정: 단일 출처, 단계 설정은 플러그인의 dataclass로 검증, 참조는 전부 존재해야 한다."""
import sys
import textwrap
from dataclasses import dataclass

import pytest

from persona_sft_data.core.config import ConfigError, PipelineConfig, TeacherConfig, build_settings
from persona_sft_data.core.registry import FORMATS, STAGES, TEACHERS, TRANSLATORS
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
    """이 모듈의 자리 표시 등록. conftest의 격리 픽스처가 함수 끝에 되돌린다."""
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


def test_blocks_that_are_not_objects_are_config_errors_not_tracebacks(tmp_path):
    """``dict("hello")``는 맨 ``ValueError``, ``"hello".items()``는 ``AttributeError``다.

    ``cli.load_config``는 ``ConfigError``만 잡으므로 둘 다 한 줄 안내(종료 2) 대신
    트레이스백(종료 1)이 된다.
    """
    for overrides, msg in (
        ({"stages": {"gen": "hello"}}, r"stages\.gen: 객체여야 한다 \(받은 것: str\)"),
        ({"stages": {"gen": ["teacher", "fake"]}}, r"stages\.gen.*받은 것: list"),
        ({"stages": "hello"}, r"^stages: 객체여야 한다"),
        ({"teachers": "hello"}, r"^teachers: 객체여야 한다"),
        ({"teachers": {"fake": "hello"}}, r"teachers\.fake: 객체여야 한다"),
        ({"sources": "hello"}, r"^sources: 객체여야 한다"),
        ({"sources": {"s": "hello"}}, r"sources\.s: 객체여야 한다"),
        ({"student": "hello"}, r"^student: 객체여야 한다"),
    ):
        # stages 자체를 덮어쓰는 경우가 아니면 gen 단계를 하나 둔다 (stages는 필수 키다).
        base = {} if "stages" in overrides else {"stages": {"gen": {"teacher": "fake"}}}
        with pytest.raises(ConfigError, match=msg):
            PipelineConfig.load(write_config(tmp_path, **{**base, **overrides}))


def test_a_config_file_that_is_not_an_object_is_a_config_error(tmp_path):
    (tmp_path / "configs").mkdir(exist_ok=True)
    path = tmp_path / "configs" / "list.json"
    path.write_text('["not", "an", "object"]', encoding="utf-8")
    with pytest.raises(ConfigError, match="객체여야 한다.*받은 것: list"):
        PipelineConfig.load(path)


def test_source_extract_must_be_an_object(tmp_path):
    good = {"format": "tsv", "url": "http://x/a.tsv", "fields": ["a"], "language": "ko", "license": "mit"}
    with pytest.raises(ConfigError, match=r"source 's'\.extract: 객체여야 한다"):
        PipelineConfig.load(write_config(tmp_path, sources={"s": {**good, "extract": "field"}}))


def test_teacher_config_defaults_and_unknown_keys():
    t = TeacherConfig.from_dict("t", {"model": "m", "base_url": "http://x"})
    assert t.kind == "openai" and t.concurrency == 64 and t.api_key is None
    with pytest.raises(ConfigError, match="teacher 't'.*bogus"):
        TeacherConfig.from_dict("t", {"model": "m", "base_url": "http://x", "bogus": 1})
    with pytest.raises(ConfigError, match="base_url"):
        TeacherConfig.from_dict("t", {"model": "m"})


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


def test_local_plugins_load_from_the_config_root(tmp_path):
    """저장소에 둔 모듈은 sys.path에 없어도 붙는다 — 로드가 설정의 root를 넘긴다."""
    (tmp_path / "root_plugin.py").write_text(textwrap.dedent("""
        from persona_sft_data.core.registry import FORMATS
        @FORMATS.register("root_format")
        class RootFormat:
            name = "root_format"
            extensions = (".txt",)
            def rows(self, data, fields):
                return iter(())
    """), encoding="utf-8")
    assert str(tmp_path) not in sys.path
    try:
        cfg = PipelineConfig.load(write_config(tmp_path, plugins=["root_plugin"]))
        assert cfg.plugins == ("root_plugin",)
        assert FORMATS.get("root_format").name == "root_format"
    finally:
        sys.modules.pop("root_plugin", None)


def test_a_plugin_module_that_does_not_import_is_a_config_error(tmp_path):
    """ImportError만 잡으면 구문 오류가 트레이스백으로 새어 나간다."""
    (tmp_path / "broken.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="broken.*SyntaxError"):
        PipelineConfig.load(write_config(tmp_path, plugins=["broken"]))
    with pytest.raises(ConfigError, match="ghost_module"):
        PipelineConfig.load(write_config(tmp_path, plugins=["ghost_module"]))


def test_recipe_is_validated_at_load_time(tmp_path):
    """실행 시점까지 미루면 ``check``가 설정 오류(2)가 아니라 단계 실패(1)로 끝나고,
    ``run``은 교사 단계를 전부 돌린 뒤 내보내기에서 터진다."""
    def export(recipe):
        return write_config(tmp_path, stages={"export": {"name": "d", "recipe": recipe}})

    ok = PipelineConfig.load(export({"kind": "llamafactory", "lora_rank": 8}))
    assert ok.stage_settings("export").recipe["kind"] == "llamafactory"
    with pytest.raises(ConfigError, match=r"stages\.export\.recipe.*nope"):
        PipelineConfig.load(export({"kind": "llamafactory", "nope": 1}))
    with pytest.raises(ConfigError, match=r"stages\.export\.recipe.*'ghost'"):
        PipelineConfig.load(export({"kind": "ghost"}))
    with pytest.raises(ConfigError, match=r"stages\.export\.recipe\.kind"):
        PipelineConfig.load(export({}))
    with pytest.raises(ConfigError, match=r"stages\.export\.recipe"):
        PipelineConfig.load(export("llamafactory"))          # dict가 아니면 설정 오류다


def test_source_extract_settings_are_validated_at_load_time(tmp_path):
    base = {"format": "tsv", "path": "x.tsv", "fields": ["a"], "language": "ko", "license": "mit"}
    ok = PipelineConfig.load(write_config(tmp_path, sources={"s": {**base, "extract": {"kind": "regex", "pattern": "x"}}}))
    assert ok.source("s").extract == {"pattern": "x"}
    with pytest.raises(ConfigError, match="source 's' extract.*nope"):
        PipelineConfig.load(write_config(tmp_path, sources={"s": {**base, "extract": {"kind": "regex", "pattern": "x", "nope": 1}}}))
    with pytest.raises(ConfigError, match="source 's' extract.*pattern"):
        PipelineConfig.load(write_config(tmp_path, sources={"s": {**base, "extract": {"kind": "regex"}}}))


def test_build_settings_reports_where():
    @dataclass(frozen=True)
    class S:
        a: int
        b: int = 2
    assert build_settings(S, {"a": 1}, "x") == S(1, 2)
    with pytest.raises(ConfigError, match="x.*'a'"):
        build_settings(S, {}, "x")
