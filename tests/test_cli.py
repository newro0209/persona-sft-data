"""CLI: 스모크 설정으로 check → run → export가 GPU·네트워크 없이 끝까지 돈다."""
import json
import sys
from pathlib import Path

import pytest

from persona_sft_data import cli
from persona_sft_data.core.config import PipelineConfig
from persona_sft_data.sources.translate import TeacherTranslator
from persona_sft_data.teacher.base import TeacherError
from persona_sft_data.teacher.fake import EchoTeacher
from tests.conftest import DOC, ROOT

SMOKE = ROOT / "configs" / "smoke.json"


def _smoke_raw(tmp_path: Path) -> dict:
    raw = json.loads(SMOKE.read_text(encoding="utf-8"))
    raw["data_root"] = str(tmp_path / "data")
    raw["datasets_root"] = str(tmp_path / "datasets")
    raw["persona_doc"] = str(DOC)
    for s in raw["sources"].values():
        s["path"] = str(ROOT / s["path"])
    return raw


def _write(tmp_path: Path, raw: dict, name: str = "smoke") -> Path:
    (tmp_path / "configs").mkdir(exist_ok=True)
    path = tmp_path / "configs" / f"{name}.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def smoke(tmp_path: Path) -> Path:
    """저장소의 smoke.json을 임시 프로젝트로 옮긴다: 모든 경로를 절대 경로로."""
    return _write(tmp_path, _smoke_raw(tmp_path))


def test_check_run_export_end_to_end(smoke, capsys, monkeypatch):
    from persona_sft_data.stages import export as export_mod
    monkeypatch.setattr(export_mod, "_load_tokenizer", lambda student: None)
    assert cli.main(["check", "--config", str(smoke)]) == 0
    out = capsys.readouterr().out
    assert "persona" in out and "companion" in out and "fixture_en" in out
    assert cli.main(["run", "--config", str(smoke)]) == 0
    cfg = PipelineConfig.load(smoke)
    for name in ("ingest", "dialogue", "respond"):
        assert cfg.raw(name).exists() and cfg.stats_path(cfg.raw(name)).exists()
    assert cfg.filtered("dialogue").exists() and cfg.filtered("respond").exists()
    assert (cfg.data_root / "final" / "manifest.json").exists()
    dataset = cfg.datasets_root / "smoke"
    assert (dataset / "train.jsonl").exists() and (dataset / "recipe" / "llamafactory" / "lora_sft.yaml").exists()
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["records"] > 0 and manifest["sources"].get("dialogue", 0) > 0
    assert manifest["source_datasets"]["fixture_en"]["original_language"] == "en"
    # records > 0만 보면 filter가 raw의 거의 전부를 버려도 통과한다. 스모크 교사가 모든
    # 세션에 같은 후속 줄을 내면 assistant_line_overused가 코퍼스를 삼키므로, filter가
    # raw의 절반 이상을 남기고 dialogue가 assemble이 원하는 만큼 공급하는지도 본다.
    corpus = json.loads((cfg.data_root / "final" / "manifest.json").read_text(encoding="utf-8"))
    raw_produced = corpus["stages"]["raw/dialogue"]["produced"]
    assert corpus["stages"]["filtered/dialogue"]["produced"] >= raw_produced / 2
    # shortfall이 비어야 한다. 'dialogue만 없으면 된다'로 두면 smoke.json의 max_sessions가
    # 픽스처 규모보다 커서 respond가 늘 모자라는 상태를 통과시킨다 — 스모크는 정상 실행이
    # 어떤 모습인지 보여 주는 것이므로, 상한도 실제 가용량에 맞아야 한다.
    assert corpus["shortfall"] == {}, corpus["shortfall"]
    assert cli.main(["export", "--config", str(smoke), "--name", "smoke2"]) == 0
    assert (cfg.datasets_root / "smoke2" / "train.jsonl").exists()


def test_run_single_stage_and_ordering(smoke, capsys):
    cfg = PipelineConfig.load(smoke)
    assert [s.name for s in cli.ordered_stages(cfg)] == ["ingest", "dialogue", "respond", "filter", "assemble", "export"]
    assert cli.main(["run", "--config", str(smoke), "--stage", "dialogue"]) == 0
    assert cfg.raw("dialogue").exists() and not cfg.raw("ingest").exists()
    assert cli.main(["run", "--config", str(smoke), "--stage", "respond"]) == 1
    assert "ingest" in capsys.readouterr().err


def test_status_sources_and_plugins(smoke, capsys):
    assert cli.main(["run", "--config", str(smoke), "--stage", "ingest"]) == 0
    assert cli.main(["status", "--config", str(smoke)]) == 0
    assert "ingest" in capsys.readouterr().out
    assert cli.main(["sources", "--config", str(smoke), "--sample", "2", "--translate"]) == 0
    out = capsys.readouterr().out
    assert "fixture_ko" in out and "fixture_en" in out and "→" in out
    assert cli.main(["plugins"]) == 0
    out = capsys.readouterr().out
    assert "stages" in out and "llamafactory" in out and "builtin" in out


def test_init_scaffolds_a_parseable_persona_and_config(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["init", "세라", "--profile", "npc"]) == 0
    doc = tmp_path / "personas" / "세라.md"
    cfg_path = tmp_path / "configs" / "세라.json"
    assert doc.exists() and cfg_path.exists()
    cfg = PipelineConfig.load(cfg_path)
    assert cfg.profile == "npc" and cfg.persona_doc == doc.resolve()
    assert cli.main(["init", "세라", "--profile", "npc"]) == 2      # 이미 있으면 거부


def test_plugins_with_a_config_shows_local_plugins_and_their_origin(tmp_path, capsys):
    """스펙 §14의 출처 열(내장·entry point·plugins)은 설정을 붙여야 'plugins'까지 보인다."""
    (tmp_path / "table_plugin.py").write_text(
        "from persona_sft_data.core.registry import FORMATS\n"
        "@FORMATS.register('table_format')\n"
        "class TableFormat:\n"
        "    name = 'table_format'\n"
        "    extensions = ('.txt',)\n"
        "    def rows(self, data, fields):\n"
        "        return iter(())\n",
        encoding="utf-8",
    )
    raw = _smoke_raw(tmp_path)
    raw["plugins"] = ["table_plugin"]
    config = _write(tmp_path, raw, "with_plugin")
    try:
        assert cli.main(["plugins", "--config", str(config)]) == 0
        out = capsys.readouterr().out
        row = next(line for line in out.splitlines() if "table_format" in line)
        assert "plugins" in row and "table_plugin:TableFormat" in row
        assert "builtin" in out                                # 내장은 여전히 builtin이다
        assert cli.main(["plugins"]) == 0                      # --config는 선택이다
    finally:
        sys.modules.pop("table_plugin", None)


def test_sources_translate_stops_when_the_teacher_is_unreachable(smoke, capsys, monkeypatch):
    """번역 전후를 보러 온 명령이 서버가 죽은 줄도 모르고 종료 0으로 끝나면 안 된다."""
    def dead(self):
        raise TeacherError("교사 'bulk': 닿지 못했다")
    monkeypatch.setattr(EchoTeacher, "check", dead)
    assert cli.main(["sources", "--config", str(smoke), "--sample", "1", "--translate"]) == 1
    assert "교사 오류" in capsys.readouterr().err


def test_sources_marks_individual_translation_failures(smoke, capsys, monkeypatch):
    monkeypatch.setattr(TeacherTranslator, "translate", lambda self, texts, source_language: [None] * len(texts))
    assert cli.main(["sources", "--config", str(smoke), "--sample", "1", "--translate"]) == 0
    assert "(번역 실패)" in capsys.readouterr().out


def test_bad_config_exits_2(tmp_path, capsys):
    (tmp_path / "configs").mkdir()
    bad = tmp_path / "configs" / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert cli.main(["check", "--config", str(bad)]) == 2
    assert "profile" in capsys.readouterr().err


def test_recipe_extract_and_stage_values_exit_2_before_any_teacher_call(tmp_path, capsys):
    """실행 시점 검증이면 ``check``는 단계 실패(1)로, ``run``은 교사를 다 쓴 뒤 터진다."""
    broken = {
        "bad_recipe": lambda raw: raw["stages"]["export"]["recipe"].update({"nope": 1}),
        "bad_extract": lambda raw: raw["sources"]["fixture_ko"].setdefault("extract", {}).update({"nope": 1}),
        "bad_dialogue": lambda raw: raw["stages"]["dialogue"].update({"per_situation": 0}),
    }
    for name, break_it in broken.items():
        raw = _smoke_raw(tmp_path)
        break_it(raw)
        config = _write(tmp_path, raw, name)
        assert cli.main(["check", "--config", str(config)]) == 2, name
        assert "설정 오류" in capsys.readouterr().err, name
        assert cli.main(["run", "--config", str(config), "--stage", "dialogue"]) == 2, name
        assert not (tmp_path / "data" / "raw").exists(), name


def test_a_plugin_module_that_cannot_be_imported_exits_2(tmp_path, capsys):
    """구문 오류가 트레이스백으로 새면 사용자가 '설정 오류' 한 줄을 못 본다."""
    (tmp_path / "broken_plugin.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    raw = _smoke_raw(tmp_path)
    raw["plugins"] = ["broken_plugin"]
    config = _write(tmp_path, raw, "broken")
    assert cli.main(["check", "--config", str(config)]) == 2
    err = capsys.readouterr().err
    assert "설정 오류" in err and "SyntaxError" in err


BOOM_PLUGIN = '''"""preflight가 예상 밖 예외를 던지는 단계 플러그인."""
from dataclasses import dataclass

from persona_sft_data.core.registry import STAGES


@dataclass(frozen=True)
class BoomSettings:
    pass


@STAGES.register("boom", origin="plugins")
class BoomStage:
    name = config_name = "boom"
    mode, record_kind, produces = "records", "utterance", "raw"
    settings_type = BoomSettings

    def requires(self, config):
        return ()

    def instances(self, config):
        return [self]

    def preflight(self, ctx):
        raise RuntimeError("예상 밖으로 터졌다")

    def run(self, ctx):
        return iter(())
'''


def test_check_reports_an_unexpected_preflight_failure_and_keeps_going(tmp_path, capsys, monkeypatch):
    """``check``의 계약은 '단계마다 OK/FAILED, 하나라도 실패면 종료 1'이다.

    예상한 예외만 잡으면 플러그인 단계의 preflight가 던지는 아무 예외 하나가
    트레이스백(종료 1)으로 나머지 단계 점검을 통째로 앗아간다.
    """
    from persona_sft_data.stages import export as export_mod
    monkeypatch.setattr(export_mod, "_load_tokenizer", lambda student: None)
    (tmp_path / "boom_plugin.py").write_text(BOOM_PLUGIN, encoding="utf-8")
    raw = _smoke_raw(tmp_path)
    raw["plugins"] = ["boom_plugin"]
    raw["stages"] = {"boom": {}, **raw["stages"]}
    config = _write(tmp_path, raw, "boom")
    try:
        assert cli.main(["check", "--config", str(config)]) == 1
        out = capsys.readouterr().out
        assert "stage     : boom FAILED" in out
        assert "RuntimeError: 예상 밖으로 터졌다" in out       # 예외형까지 적어야 원인을 찾는다
        for name in ("ingest", "dialogue", "respond", "filter", "assemble", "export"):
            assert f"stage     : {name} OK" in out, name      # 나머지 점검은 이어진다
    finally:
        sys.modules.pop("boom_plugin", None)
