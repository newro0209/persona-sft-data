"""모든 테스트가 같이 쓰는 것: 더미 프로필, 임시 프로젝트 설정 작성."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from persona_sft_data.core.registry import PROFILES

ROOT = Path(__file__).resolve().parents[1]
DOC = ROOT / "personas" / "mongle.md"
FIXTURES = ROOT / "tests" / "fixtures"


class DummyProfile:
    """프로필 구현이 나오기 전에도 설정·러너를 테스트하려고 두는 최소 프로필."""

    name = "dummy"
    assistant_label = "캐릭터"
    user_label = "사용자"
    writer_framing = "너는 캐릭터와 사용자의 짧은 대화를 쓰는 작가다."
    required_sections: tuple[str, ...] = ()
    default_flows = ("사용자가 말을 거는 흐름",)
    default_turns = (2,)
    extra_rules: tuple[str, ...] = ()

    def document_template(self, persona_name: str) -> str:
        return DOC.read_text(encoding="utf-8")


PROFILES.add("dummy", DummyProfile(), origin="plugins")

BASE_CONFIG = {
    "profile": "dummy",
    "language": "ko",
    "data_root": "data",
    "datasets_root": "datasets",
    "seed": 7,
    "persona_doc": "personas/mongle.md",
    "plugins": [],
    "student": {"model": "org/student-base", "trust_remote_code": True, "chat_template": "chatml"},
    "teachers": {
        "fake": {"kind": "fake", "model": "fake", "base_url": "http://localhost:1"},
    },
    "sources": {},
    "stages": {},
}


def write_config(tmp_path: Path, **overrides) -> Path:
    """임시 프로젝트를 만든다: personas/mongle.md 복사 + configs/test.json."""
    (tmp_path / "personas").mkdir(exist_ok=True)
    shutil.copy(DOC, tmp_path / "personas" / "mongle.md")
    (tmp_path / "configs").mkdir(exist_ok=True)
    raw = json.loads(json.dumps(BASE_CONFIG))
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key] = {**raw[key], **value}
        else:
            raw[key] = value
    path = tmp_path / "configs" / "test.json"
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


@pytest.fixture
def project(tmp_path: Path) -> Path:
    return tmp_path
