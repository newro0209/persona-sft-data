"""The invariants the pipeline rewrite exists to hold.

These are the design's success criteria expressed as tests, so they cannot
quietly stop being true. The old pipeline satisfied none of them: the pet's
name appeared in ten modules, model ids in eight places, and fifteen
``Path("data/...")`` defaults meant a stage's output location was whatever its
argparse block said.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from persona_sft_data import gates
from persona_sft_data.config import ConfigError, PipelineConfig
from persona_sft_data.persona import PersonaError, load as load_persona

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "persona_sft_data"
CONFIG = ROOT / "configs" / "mongle.json"
PERSONA_DOC = ROOT / "personas" / "mongle.md"


def gen_sources() -> list[Path]:
    return sorted(p for p in GEN.rglob("*.py") if "__pycache__" not in p.parts)


# --- the persona lives in one place ---------------------------------------


def test_persona_name_appears_in_no_source_file():
    """The whole point of parsing the document. If the name can be written in
    code, so can the rest of the persona, and the two drift."""
    name = load_persona(PERSONA_DOC).name
    offenders = {
        path.relative_to(ROOT).as_posix(): text.count(name)
        for path in gen_sources()
        if name in (text := path.read_text(encoding="utf-8"))
    }
    assert offenders == {}, (
        f"persona name {name!r} found in persona_sft_data: {offenders}. "
        "Read it from the document instead."
    )


def test_persona_parses_the_real_document():
    persona = load_persona(PERSONA_DOC)
    assert persona.name
    assert len(persona.principles) >= 5
    assert len(persona.prohibitions) >= 5
    assert len(persona.situations) >= 10
    assert len(persona.beats) > len(persona.situations)
    assert persona.preamble.startswith("<|u|>")
    lo, hi = persona.utterance_char_range()
    assert 0 < lo < hi


@pytest.mark.parametrize(
    "section",
    ["## 핵심 정의", "## 발화 원칙", "## 하지 않는 말과 행동", "## 다룰 상황"],
)
def test_persona_parser_raises_on_a_missing_section(tmp_path, section):
    """A frozen document that changes shape must raise, not return empty rules.
    An empty prohibition list would disarm the gate silently."""
    text = PERSONA_DOC.read_text(encoding="utf-8")
    start = text.index(section)
    end = text.index("\n## ", start + 1)
    doc = tmp_path / "persona.md"
    doc.write_text(text[:start] + text[end + 1 :], encoding="utf-8")
    with pytest.raises(PersonaError):
        load_persona(doc)


# --- no literals that make a run unrepeatable ------------------------------

MODEL_ID = re.compile(r"(?:hf\.co/|kakaocorp/|LGAI-|NotoriousH2/|Qwen)")
DATA_PREFIX = re.compile(r"^data[/\\]")


def _live_strings(path: Path):
    """Yield (lineno, value) for string literals that are actually code.

    Docstrings and comments are excluded on purpose: config.py's docstring
    quotes the very path literal this test bans, as the example of what went
    wrong. Prose about a mistake is not the mistake.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and isinstance(
                body[0].value, ast.Constant
            ) and isinstance(body[0].value.value, str):
                docstrings.add(id(body[0].value))
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            yield node.lineno, node.value


def test_no_model_ids_in_source():
    """Model choice belongs to the config. Eight hardcoded ids is why swapping
    a teacher used to mean editing eight files."""
    offenders = [
        f"{path.relative_to(ROOT).as_posix()}:{line} {value!r}"
        for path in gen_sources()
        for line, value in _live_strings(path)
        if MODEL_ID.search(value)
    ]
    assert offenders == [], f"model ids in source: {offenders}"


def test_no_data_path_literals_in_source():
    """Every path derives from data_root. Fifteen of these made the old
    pipeline's output locations argparse defaults rather than configuration."""
    offenders = [
        f"{path.relative_to(ROOT).as_posix()}:{line} {value!r}"
        for path in gen_sources()
        for line, value in _live_strings(path)
        if DATA_PREFIX.match(value)
    ]
    assert offenders == [], f"data/ path literals in source: {offenders}"


# --- config drives every path ---------------------------------------------


def test_config_derives_paths_from_data_root():
    cfg = PipelineConfig.load(CONFIG)
    assert cfg.raw("seed") == cfg.data_root / "raw" / "seed.jsonl"
    assert cfg.filtered("seed") == cfg.data_root / "filtered" / "seed.jsonl"
    assert cfg.final("assemble") == cfg.data_root / "final" / "assemble.jsonl"
    out = cfg.raw("seed")
    assert cfg.stats_path(out).name.endswith(".stats.json")
    assert cfg.rejected_path(out).name.endswith(".rejected.jsonl")


def test_config_rejects_an_unknown_teacher():
    cfg = PipelineConfig.load(CONFIG)
    broken = dict(cfg.stages)
    broken["seed"] = {**broken["seed"], "teacher": "nope"}
    cfg = PipelineConfig(**{**cfg.__dict__, "stages": broken})
    with pytest.raises(ConfigError):
        cfg.teacher_for("seed")


def test_stage_seeds_differ_but_are_deterministic():
    cfg = PipelineConfig.load(CONFIG)
    assert cfg.stage_seed("seed") != cfg.stage_seed("expand")
    assert cfg.stage_seed("seed") == PipelineConfig.load(CONFIG).stage_seed("seed")


def test_smoke_config_points_somewhere_else():
    """Experiments are distinguished by config, not by one-off directories."""
    smoke = ROOT / "configs" / "smoke.json"
    if not smoke.exists():
        pytest.skip("no smoke config")
    assert PipelineConfig.load(smoke).data_root != PipelineConfig.load(CONFIG).data_root


# --- the gate catches what a hand-written check missed ---------------------


def _session(pet_text: str) -> dict:
    return {"turns": [{"role": "user", "text": "뭐 해?"},
                      {"role": "pet", "text": pet_text}]}


@pytest.fixture(scope="module")
def gate() -> gates.Gate:
    return gates.Gate.from_config(load_persona(PERSONA_DOC), {})


@pytest.mark.parametrize(
    "text, reason",
    [
        ("잘래 🐾", "emoji"),
        ("네, 잘 먹었어요.", "honorific"),
        ("저는 인공지능이야.", "claims_to_be_ai"),
        ("응, 네 옆에 꼭 붙어서宠받고 싶어.", "cjk_characters"),
        ("좋아 좋아 좋아", "repeated_phrase"),
        ("응… 그래… 알겠어", "multiple_ellipsis"),
        ("가" * 60, "pet_utterance_too_long"),
    ],
)
def test_gate_rejects_observed_violations(gate, text, reason):
    """Every one of these came out of a real teacher run during the rewrite."""
    verdict = gate.check(_session(text))
    assert not verdict.ok
    assert reason in verdict.reasons, verdict.reasons


@pytest.mark.parametrize(
    "text",
    ["응, 배 고파.", "같이 놀자!", "하암, 졸려.", "그건 잘 모르겠어.",
     "조금 삐졌어…", "네 글씨 예쁘다", "히히, 좋아."],
)
def test_gate_passes_ordinary_speech(gate, text):
    verdict = gate.check(_session(text))
    assert verdict.ok, verdict.reasons


def test_gate_rejects_babytalk_derived_from_the_name(gate):
    """기달몽/놀랐몽 — a syllable of the name worn as a verb ending, which the
    persona document forbids and an ad-hoc check missed."""
    name = load_persona(PERSONA_DOC).name
    verdict = gate.check(_session(f"알겠어, 기다릴{name[0]}!"))
    assert "name_suffix_babytalk" in verdict.reasons


def test_gate_rejects_structural_faults(gate):
    assert "does_not_start_with_user" in gate.check(
        {"turns": [{"role": "pet", "text": "응."}]}
    ).reasons
    assert "roles_not_alternating" in gate.check(
        {"turns": [{"role": "user", "text": "야"}, {"role": "user", "text": "야"},
                   {"role": "pet", "text": "응."}]}
    ).reasons


def test_gate_length_limit_comes_from_the_document(gate):
    _, hi = load_persona(PERSONA_DOC).utterance_char_range()
    assert gate.max_chars == hi
