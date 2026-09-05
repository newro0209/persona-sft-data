"""불변식. 이 재작성이 지키려는 것을 테스트로 고정한다."""
import ast
import re
from pathlib import Path

from persona_sft_data.core.persona import load
from tests.conftest import DOC, ROOT

PKG = ROOT / "persona_sft_data"

MODEL_ID = re.compile(r"(?:hf\.co/|kakaocorp/|LGAI-|NotoriousH2/|Qwen)")
DATA_PREFIX = re.compile(r"^data[/\\]")
URL = re.compile(r"^https?://")


def _sources():
    return sorted(p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts)


def _live_strings(path: Path):
    """독스트링을 뺀 문자열 리터럴. 실수를 설명하는 산문은 실수가 아니다."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            yield node.lineno, node.value


def test_persona_name_appears_in_no_source_file():
    name = load(DOC).name
    offenders = {p.relative_to(ROOT).as_posix() for p in _sources() if name in p.read_text(encoding="utf-8")}
    assert offenders == set()


def test_no_model_ids_data_paths_or_urls_in_source():
    offenders = [f"{p.relative_to(ROOT).as_posix()}:{line} {value!r}"
                 for p in _sources() for line, value in _live_strings(p)
                 if MODEL_ID.search(value) or DATA_PREFIX.match(value) or URL.match(value)]
    assert offenders == [], offenders


def test_no_profile_branching_in_source():
    offenders = [p.relative_to(ROOT).as_posix() for p in _sources()
                 if re.search(r"profile(\.name)?\s*==\s*['\"]", p.read_text(encoding="utf-8"))]
    assert offenders == []


def test_smoke_and_main_configs_point_at_different_data_roots():
    from persona_sft_data.core.config import PipelineConfig
    a = PipelineConfig.load(ROOT / "configs" / "mongle.json")
    b = PipelineConfig.load(ROOT / "configs" / "smoke.json")
    assert a.data_root != b.data_root
    a.validate_pipeline()
    b.validate_pipeline()
