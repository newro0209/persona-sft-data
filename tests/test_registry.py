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
