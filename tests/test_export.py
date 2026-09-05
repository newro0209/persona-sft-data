"""The export stage: sessions in, a trainer-ready dataset out.

What can go wrong here is quiet. A role mapped the wrong way, a system prompt
that drifts from the document, a manifest count that does not match the file
-- none of it raises, all of it ruins a fine-tune. So the checks are exact:
every field, every count, every hash.
"""

import hashlib
import json
from pathlib import Path

import pytest

from persona_sft_data.config import PipelineConfig
from persona_sft_data.persona import load as load_persona
from persona_sft_data.schema import write_jsonl
from persona_sft_data.stages.export import dataset_card, run_export, to_messages

ROOT = Path(__file__).resolve().parents[1]
PERSONA_DOC = ROOT / "personas" / "mongle.md"


def _session(sid: str, source: str, turns: list[str], **extra) -> dict:
    roles = ("user", "pet")
    return {
        "id": sid, "source": source, "scenario": "테스트", "license": "synthetic",
        "generator": ["fake"],
        "turns": [{"role": roles[i % 2], "text": t} for i, t in enumerate(turns)],
        **extra,
    }


@pytest.fixture
def project(tmp_path: Path) -> PipelineConfig:
    """A throwaway project root with three tiny splits already assembled."""
    (tmp_path / "configs").mkdir()
    config_path = tmp_path / "configs" / "t.json"
    config_path.write_text(json.dumps({
        "data_root": "data",
        "datasets_root": "out",
        "seed": 1,
        "persona_doc": str(PERSONA_DOC),
        "teachers": {"t": {"model": "fake", "base_url": "http://localhost:1"}},
        "stages": {"export": {"name": "unit"}},
    }), encoding="utf-8")
    config = PipelineConfig.load(config_path)

    final = config.data_root / "final"
    final.mkdir(parents=True)
    write_jsonl(final / "train.jsonl", [
        _session("a", "teacher_seed", ["안녕", "응, 안녕", "뭐 해?", "누워 있어"], split="train"),
        _session("b", "template", ["배고파?", "응, 배고파"], split="train", beats=["배고픔"]),
        _session("c", "real", ["비 와", "그래, 같이 있자"], split="train",
                 license="mit", real_source="open_korean_instructions",
                 source_url="https://example.invalid/x.parquet"),
    ])
    write_jsonl(final / "val.jsonl", [
        # a second `real` record under a different licence: the card must say
        # both, not whichever came first
        _session("d", "real", ["졸려", "나도"], split="val", license="apache-2.0",
                 real_source="korean_safe_conversation", source_url="https://example.invalid/y"),
    ])
    write_jsonl(final / "test.jsonl", [_session("e", "teacher_expand", ["놀자", "좋아"], split="test")])
    return config


# --- one record ------------------------------------------------------------


def test_to_messages_maps_roles_and_keeps_provenance():
    persona = load_persona(PERSONA_DOC)
    prompt = persona.system_prompt()
    record = to_messages(_session("x", "real", ["안녕", "응", "잘 자", "너도"],
                                  real_source="r", source_url="u"), prompt)

    assert record["messages"][0] == {"role": "system", "content": prompt}
    assert [m["role"] for m in record["messages"][1:]] == ["user", "assistant", "user", "assistant"]
    assert [m["content"] for m in record["messages"][1:]] == ["안녕", "응", "잘 자", "너도"]
    # provenance survives; the session-shaped fields do not
    assert record["id"] == "x" and record["source"] == "real"
    assert record["real_source"] == "r" and record["source_url"] == "u"
    assert "turns" not in record and "split" not in record


def test_to_messages_rejects_what_the_schema_rejects():
    """A pet turn first, or an odd count, must not become a silently misaligned
    chat -- the schema already knows these are wrong."""
    bad = {"id": "z", "source": "s", "turns": [{"role": "pet", "text": "응"}]}
    with pytest.raises(Exception):
        to_messages(bad, "sys")


# --- the system prompt is the document -------------------------------------


def test_system_prompt_is_rendered_from_the_document():
    persona = load_persona(PERSONA_DOC)
    prompt = persona.system_prompt()
    assert prompt.startswith(f"이름: {persona.name}")
    for key, value in persona.core.items():
        assert f"{key}: {value}" in prompt
    for i, principle in enumerate(persona.principles, 1):
        assert f"{i}. {principle}" in prompt
    for prohibition in persona.prohibitions:
        assert f"- {prohibition}" in prompt
    # plain text: no table pipes, no bold, no code spans
    assert "|" not in prompt and "**" not in prompt and "`" not in prompt


# --- the whole export ------------------------------------------------------


def test_export_writes_every_split_and_an_honest_manifest(project: PipelineConfig):
    out = run_export(project, log=lambda *_: None)
    assert out == project.datasets_root / "unit"

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    prompt = (out / "system_prompt.txt").read_text(encoding="utf-8").rstrip("\n")
    assert prompt == load_persona(PERSONA_DOC).system_prompt()

    for split, expected in (("train", 3), ("val", 1), ("test", 1)):
        path = out / f"{split}.jsonl"
        records = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines()]
        assert len(records) == expected == manifest["files"][split]["records"]
        assert manifest["files"][split]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        for r in records:
            assert r["messages"][0]["content"] == prompt

    assert manifest["records"] == 5
    assert manifest["turns"] == 4 + 2 + 2 + 2 + 2
    assert manifest["sources"] == {"real": 2, "teacher_seed": 1, "template": 1, "teacher_expand": 1}
    assert manifest["licenses"]["real"] == ["apache-2.0", "mit"]
    assert manifest["licenses"]["open_korean_instructions"] == ["mit"]
    assert manifest["real_sources"] == {
        "open_korean_instructions": "https://example.invalid/x.parquet",
        "korean_safe_conversation": "https://example.invalid/y",
    }
    assert manifest["generators"] == {"fake": 5}
    assert manifest["persona_sha256"] == hashlib.sha256(PERSONA_DOC.read_bytes()).hexdigest()


def test_export_name_override_lands_beside_not_over(project: PipelineConfig):
    run_export(project, log=lambda *_: None)
    other = run_export(project, name="unit-2", log=lambda *_: None)
    assert other.name == "unit-2"
    assert (project.datasets_root / "unit" / "train.jsonl").exists()


def test_export_refuses_without_an_assembled_corpus(project: PipelineConfig):
    (project.data_root / "final" / "val.jsonl").unlink()
    with pytest.raises(FileNotFoundError):
        run_export(project, log=lambda *_: None)


def test_dataset_card_states_counts_sources_and_prompt(project: PipelineConfig):
    out = run_export(project, log=lambda *_: None)
    card = (out / "README.md").read_text(encoding="utf-8")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert card.startswith("---\nlanguage:\n- ko\n")
    assert "| train | 3 |" in card and "| val | 1 |" in card
    assert "| real | 2 | apache-2.0, mit |" in card
    assert "`open_korean_instructions` (mit)" in card
    assert "`korean_safe_conversation` (apache-2.0)" in card
    assert (out / "system_prompt.txt").read_text(encoding="utf-8").rstrip("\n") in card
    assert manifest["config_path"] in card
    # size bucket comes from the count, not a guess
    assert "- n<1K" in dataset_card(manifest, "p")
