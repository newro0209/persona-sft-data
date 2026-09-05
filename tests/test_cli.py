"""CLI: 스모크 설정으로 check → run → export가 GPU·네트워크 없이 끝까지 돈다."""
import json
from pathlib import Path

import pytest

from persona_sft_data import cli
from persona_sft_data.core.config import PipelineConfig
from persona_sft_data.core.registry import STAGES, TRANSLATORS
from persona_sft_data.sources.translate import TeacherTranslatorFactory
from persona_sft_data.stages.assemble import AssembleStage
from persona_sft_data.stages.dialogue import DialogueStage
from persona_sft_data.stages.export import ExportStage
from persona_sft_data.stages.filter import FilterStage
from persona_sft_data.stages.ingest import IngestStage
from persona_sft_data.stages.respond import RespondStage
from tests.conftest import DOC, FIXTURES, ROOT

SMOKE = ROOT / "configs" / "smoke.json"

REAL_PLUGINS = (
    (STAGES, "ingest", IngestStage), (STAGES, "dialogue", DialogueStage), (STAGES, "respond", RespondStage),
    (STAGES, "filter", FilterStage), (STAGES, "assemble", AssembleStage), (STAGES, "export", ExportStage),
    (TRANSLATORS, "teacher", TeacherTranslatorFactory),
)


@pytest.fixture(autouse=True)
def _real_plugins():
    """test_config.py의 autouse 픽스처가 'ingest'·'assemble' 단계와 'teacher' 번역기를 자리 표시로
    덮어쓴 채 세션 레지스트리에 남긴다. CLI 스모크는 진짜 단계 여섯과 번역기로 돌아야 하므로
    이 모듈에서는 내장 구현을 다시 올리고, 끝나면 있던 그대로 되돌린다(test_stage_ingest와 같은 규칙)."""
    saved = [(registry, name, registry._items.get(name)) for registry, name, _ in REAL_PLUGINS]
    for registry, name, real in REAL_PLUGINS:
        registry.add(name, real, origin="plugins")
    yield
    for registry, name, previous in saved:
        if previous is not None:
            registry._items[name] = previous


@pytest.fixture
def smoke(tmp_path: Path) -> Path:
    """저장소의 smoke.json을 임시 프로젝트로 옮긴다: 모든 경로를 절대 경로로."""
    raw = json.loads(SMOKE.read_text(encoding="utf-8"))
    raw["data_root"] = str(tmp_path / "data")
    raw["datasets_root"] = str(tmp_path / "datasets")
    raw["persona_doc"] = str(DOC)
    for s in raw["sources"].values():
        s["path"] = str(ROOT / s["path"])
    (tmp_path / "configs").mkdir()
    path = tmp_path / "configs" / "smoke.json"
    path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")
    return path


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


def test_bad_config_exits_2(tmp_path, capsys):
    (tmp_path / "configs").mkdir()
    bad = tmp_path / "configs" / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert cli.main(["check", "--config", str(bad)]) == 2
    assert "profile" in capsys.readouterr().err
