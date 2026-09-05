"""레지스트리: 세 등록 경로와 우선순위, 오류 메시지."""
import sys
import textwrap

import pytest

from persona_sft_data.core import registry as reg
from persona_sft_data.core.registry import PluginError, Registry, load_plugins


def test_register_and_get_roundtrip():
    r: Registry[object] = Registry("test.group")
    obj = object()
    r.add("thing", obj, origin="builtin")
    assert r.get("thing") is obj
    assert r.names() == ["thing"]
    assert r.describe()[0].origin == "builtin"


def test_unknown_name_lists_what_is_registered():
    r: Registry[object] = Registry("test.group")
    r.add("a", object(), origin="builtin")
    with pytest.raises(PluginError, match="'zzz'.*\\['a'\\]"):
        r.get("zzz")


def test_plugins_override_entry_points_override_builtin():
    r: Registry[str] = Registry("test.group")
    r.add("x", "builtin", origin="builtin")
    r.add("x", "ep", origin="entry_point")
    assert r.get("x") == "ep"
    r.add("x", "plugins", origin="plugins")
    assert r.get("x") == "plugins"
    r.add("x", "ep2", origin="entry_point")     # 낮은 우선순위는 덮어쓰지 못한다
    assert r.get("x") == "plugins"


def test_same_object_from_a_lower_priority_origin_keeps_the_first_origin():
    r: Registry[object] = Registry("test.group")
    obj = object()
    r.add("x", obj, origin="builtin")
    r.add("x", obj, origin="entry_point")
    assert r.describe()[0].origin == "builtin"


def test_unknown_origin_is_rejected():
    r: Registry[object] = Registry("test.group")
    with pytest.raises(PluginError):
        r.add("x", object(), origin="magic")


def test_decorating_a_class_registers_an_instance_and_the_same_class_keeps_its_origin():
    r: Registry[object] = Registry("test.group")

    @r.register("k", origin="builtin")
    class K:
        name = "k"

    assert isinstance(r.get("k"), K) and r.describe()[0].path.endswith(":K")
    r.add("k", K, origin="entry_point")          # 같은 클래스가 entry point로 와도 내장을 유지
    assert r.describe()[0].origin == "builtin"


def test_entry_points_are_discovered_lazily(monkeypatch):
    class FakeEP:
        name = "from_ep"
        value = "somewhere:Thing"
        def load(self):
            return "loaded"
    monkeypatch.setattr(reg.metadata, "entry_points", lambda group: [FakeEP()] if group == "test.group" else [])
    r: Registry[str] = Registry("test.group")
    assert r.get("from_ep") == "loaded"
    assert [d.origin for d in r.describe()] == ["entry_point"]


def test_load_plugins_imports_modules_that_register_themselves(tmp_path, monkeypatch):
    (tmp_path / "my_plugin.py").write_text(textwrap.dedent("""
        from persona_sft_data.core.registry import STAGES
        @STAGES.register("custom_stage")
        class CustomStage:
            name = "custom_stage"
    """), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    assert load_plugins(["my_plugin"]) == ["my_plugin"]
    assert reg.STAGES.get("custom_stage").name == "custom_stage"
    assert next(d for d in reg.STAGES.describe() if d.name == "custom_stage").origin == "plugins"
    sys.modules.pop("my_plugin", None)


def test_load_plugins_reports_the_module_it_could_not_import():
    with pytest.raises(PluginError, match="no_such_module_xyz"):
        load_plugins(["no_such_module_xyz"])


def test_load_plugins_finds_a_local_module_under_search_path_only(tmp_path):
    """콘솔 스크립트의 sys.path에는 저장소가 아니라 .venv/Scripts가 들어간다.

    ``search_path``가 import 하는 동안만 저장소를 넣어야 로컬 플러그인이 붙는다.
    """
    (tmp_path / "path_plugin.py").write_text(textwrap.dedent("""
        from persona_sft_data.core.registry import FORMATS
        @FORMATS.register("path_format")
        class PathFormat:
            name = "path_format"
            extensions = (".txt",)
            def rows(self, data, fields):
                return iter(())
    """), encoding="utf-8")
    assert str(tmp_path) not in sys.path
    with pytest.raises(PluginError, match="path_plugin"):
        load_plugins(["path_plugin"])                      # sys.path에 없으면 못 찾는다
    try:
        assert load_plugins(["path_plugin"], search_path=tmp_path) == ["path_plugin"]
        assert reg.FORMATS.get("path_format").name == "path_format"
        assert str(tmp_path.resolve()) not in sys.path      # 끝나면 원상 복구
    finally:
        sys.modules.pop("path_plugin", None)


def test_load_plugins_wraps_failures_that_are_not_importerror(tmp_path):
    """구문 오류·이름 오류가 그대로 전파되면 사용자가 한 줄 안내 대신 트레이스백을 본다."""
    (tmp_path / "broken_plugin.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    (tmp_path / "naming_plugin.py").write_text("undefined_name\n", encoding="utf-8")
    with pytest.raises(PluginError, match="broken_plugin.*SyntaxError"):
        load_plugins(["broken_plugin"], search_path=tmp_path)
    with pytest.raises(PluginError, match="naming_plugin.*NameError"):
        load_plugins(["naming_plugin"], search_path=tmp_path)


def test_snapshot_and_restore_undo_later_registrations():
    r: Registry[str] = Registry("test.group")
    r.add("kept", "before", origin="builtin")
    saved = r.snapshot()
    r.add("kept", "after", origin="plugins")
    r.add("added", "later", origin="plugins")
    assert r.get("kept") == "after" and r.names() == ["added", "kept"]
    r.restore(saved)
    assert r.get("kept") == "before" and r.names() == ["kept"]
    r.restore(saved)                                        # 여러 번 되돌려도 같다
    assert r.names() == ["kept"]


def test_snapshot_discovers_first_so_restore_cannot_lose_what_loads_once(monkeypatch):
    """``builtins.load()``는 한 번만 돈다.

    발견 전 상태를 담아 되돌리면 재발견이 이미 import 된 모듈을 다시 import 해도
    데코레이터가 돌지 않아 내장이 영구히 사라진다. ``snapshot()``이 먼저 발견한다.
    """
    loads = []

    class FakeEP:
        name = "discovered"
        value = "somewhere:Thing"
        def load(self):
            loads.append(1)
            return f"loaded-{len(loads)}"

    monkeypatch.setattr(reg.metadata, "entry_points", lambda group: [FakeEP()] if group == "test.group" else [])
    r: Registry[str] = Registry("test.group")
    saved = r.snapshot()
    r.add("temporary", "x", origin="plugins")
    r.restore(saved)
    assert r.get("discovered") == "loaded-1" and loads == [1]
    assert "temporary" not in r.names()


def test_registries_are_isolated_between_test_functions():
    """conftest의 autouse 픽스처가 함수마다 여덟 레지스트리를 되돌린다.

    다른 모듈(test_config.py)이 'ingest'·'assemble'·'gen'을 자리 표시로 등록해도
    그 함수 밖으로는 새지 않으므로 여기서는 내장이 보인다 — 테스트 순서와 무관하다.
    """
    from persona_sft_data.stages.assemble import AssembleStage
    from persona_sft_data.stages.ingest import IngestStage

    assert type(reg.STAGES.get("ingest")) is IngestStage
    assert type(reg.STAGES.get("assemble")) is AssembleStage
    assert "gen" not in reg.STAGES.names()


def test_pyproject_declares_every_builtin_as_an_entry_point():
    """내장도 entry point로 선언돼야 `plugins` 표가 내장과 외부를 같은 방식으로 보여 준다."""
    import tomllib
    from pathlib import Path
    from persona_sft_data.core.registry import GROUPS

    pyproject = tomllib.loads((Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8"))
    declared = pyproject["project"]["entry-points"]
    for group, registry in GROUPS.items():
        builtin = {d.name for d in registry.describe() if d.origin == "builtin"}
        assert builtin <= set(declared[f"persona_sft_data.{group}"]), (group, builtin)
        for name in builtin:
            module, _, attr = declared[f"persona_sft_data.{group}"][name].partition(":")
            obj = getattr(__import__(module, fromlist=[attr]), attr)
            registered = registry.get(name)
            assert obj is registered or type(registered) is obj, (group, name)
