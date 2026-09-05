"""export: messages JSONL, 템플릿, 길이 보고, 레시피, 카드가 한 디렉터리에 나온다."""
import json

import pytest

from persona_sft_data.core.config import PipelineConfig
from persona_sft_data.core.runner import execute
from persona_sft_data.core.schema import read_jsonl, write_jsonl
from persona_sft_data.recipes.chat_template import CHATML_JINJA, render_chatml
from persona_sft_data.stages import export as export_mod
from persona_sft_data.stages.export import ExportStage, to_messages
from tests.conftest import write_config


def _s(i, source, **extra):
    return {"id": f"{source}-{i}", "source": source, "scenario": "x", "license": "synthetic" if source == "dialogue" else "cc-by-4.0",
            "generator": ["m"], "split": "train",
            "turns": [{"role": "user", "text": f"질문 {i}"}, {"role": "assistant", "text": f"응, 좋아 {i}."}], **extra}


def _project(tmp_path, recipe=None, **overrides):
    cfg = PipelineConfig.load(write_config(tmp_path, stages={
        "dialogue": {"teacher": "fake"}, "filter": {},
        "assemble": {"ratios": {"dialogue": 1.0}, "split": {"train": 0.8, "val": 0.1, "test": 0.1}},
        "export": {"name": "demo", "recipe": recipe or {"kind": "llamafactory", "lora_rank": 8}},
    }, **overrides))
    final = cfg.data_root / "final"
    write_jsonl(final / "train.jsonl", [_s(i, "dialogue") for i in range(4)] + [
        _s(9, "respond", source_dataset="soda", source_url="http://x", original_language="en",
           translator="teacher/model-3b")])
    write_jsonl(final / "val.jsonl", [_s(5, "dialogue")])
    write_jsonl(final / "test.jsonl", [_s(6, "dialogue")])
    (final / "manifest.json").write_text(json.dumps({"seed": 7}), encoding="utf-8")
    return cfg


def test_render_chatml_matches_the_trainer_template_byte_for_byte():
    messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}, {"role": "assistant", "content": "A"}]
    assert render_chatml(messages) == "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n<|im_start|>assistant\nA<|im_end|>\n"
    assert render_chatml(messages[:2], add_generation_prompt=True).endswith("<|im_start|>assistant\n")
    assert "{% generation %}" in CHATML_JINJA and "<|im_end|>" in CHATML_JINJA
    # 개행은 블록 태그 앞의 리터럴이어야 한다. 태그 뒤에 있으면 trim_blocks가 먹는다.
    assert "<|im_end|>\n{% endgeneration %}" in CHATML_JINJA
    assert "<|im_end|>{% endgeneration %}" not in CHATML_JINJA


def _generation_extension():
    """``{% generation %}``을 통과시키는 최소 jinja2 확장. 표준 jinja2는 이 태그를 모른다.

    태그를 문자열로 지워 렌더하면 블록 구조가 사라져 ``trim_blocks``가 걸릴 자리가
    없어지고, 정작 확인하려던 결함이 테스트를 통과해 버린다. 그래서 진짜 블록으로 파싱한다.
    """
    from jinja2 import nodes
    from jinja2.ext import Extension

    class Generation(Extension):
        tags = {"generation"}

        def parse(self, parser):
            lineno = next(parser.stream).lineno
            body = parser.parse_statements(("name:endgeneration",), drop_needle=True)
            return nodes.CallBlock(self.call_method("_keep", []), [], [], body).set_lineno(lineno)

        def _keep(self, caller):
            return caller()

    return Generation


@pytest.mark.parametrize("env_kwargs", [{}, {"trim_blocks": True, "lstrip_blocks": True},
                                        {"trim_blocks": True}, {"lstrip_blocks": True}])
def test_jinja_template_renders_the_same_bytes_in_every_environment(env_kwargs):
    """트레이너가 어떤 jinja 환경으로 컴파일해도 파이썬 렌더러와 같은 바이트를 내야 한다.

    ``trim_blocks=True``면 블록 태그 바로 뒤의 개행이 사라진다. 개행이 태그 뒤에 있으면
    assistant 턴 뒤 개행이 없어져 ``<|im_end|>``와 다음 ``<|im_start|>``가 붙어 버린다.
    """
    jinja2 = pytest.importorskip("jinja2")
    env = jinja2.Environment(extensions=[_generation_extension()], **env_kwargs)
    template = env.from_string(CHATML_JINJA)
    system, user, assistant = ({"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                               {"role": "assistant", "content": "A"})
    cases = [
        ([system, user, assistant], False),
        ([system, user, assistant, {"role": "user", "content": "U2"}, {"role": "assistant", "content": "A2"}], False),
        ([system, user], True),
        ([system, user, assistant], True),
    ]
    for messages, add_generation_prompt in cases:
        rendered = template.render(messages=messages, add_generation_prompt=add_generation_prompt)
        assert rendered == render_chatml(messages, add_generation_prompt=add_generation_prompt)


def test_to_messages_maps_roles_and_keeps_provenance():
    out = to_messages(_s(1, "dialogue"), "SYS")
    assert out["messages"][0] == {"role": "system", "content": "SYS"}
    assert [m["role"] for m in out["messages"]] == ["system", "user", "assistant"]
    assert "turns" not in out and "split" not in out and out["source"] == "dialogue"


def test_export_writes_every_file_with_character_lengths(tmp_path, monkeypatch):
    monkeypatch.setattr(export_mod, "_load_tokenizer", lambda student: None)
    cfg = _project(tmp_path)
    stats = execute(ExportStage(), cfg, log=lambda m: None)
    out = cfg.datasets_root / "demo"
    for name in ("train.jsonl", "val.jsonl", "test.jsonl", "system_prompt.txt", "chat_template.jinja",
                 "rendered_sample.txt", "manifest.json", "README.md",
                 "recipe/llamafactory/dataset_info.json", "recipe/llamafactory/lora_sft.yaml", "recipe/llamafactory/README.md"):
        assert (out / name).exists(), name
    assert stats.produced == 7
    train = list(read_jsonl(out / "train.jsonl"))
    assert len(train) == 5 and train[0]["messages"][0]["content"] == (out / "system_prompt.txt").read_text(encoding="utf-8").rstrip("\n")
    assert (out / "chat_template.jinja").read_text(encoding="utf-8") == CHATML_JINJA
    assert "<|im_start|>system" in (out / "rendered_sample.txt").read_text(encoding="utf-8")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["length_report"]["method"] == "characters" and manifest["length_report"]["count"] == 7
    assert manifest["student"]["model"] == "org/student-base" and manifest["chat_template"] == "chatml"
    assert manifest["sources"] == {"dialogue": 6, "respond": 1}
    assert manifest["language"] == cfg.language
    assert manifest["source_datasets"]["soda"] == {"url": "http://x", "original_language": "en",
                                                   "translator": "teacher/model-3b", "license": "cc-by-4.0", "records": 1}
    assert manifest["files"]["train"]["records"] == 5 and manifest["files"]["train"]["sha256"]
    info = json.loads((out / "recipe/llamafactory/dataset_info.json").read_text(encoding="utf-8"))
    assert info["demo"]["file_name"] == "../../train.jsonl" and info["demo"]["formatting"] == "sharegpt"
    assert info["demo"]["tags"]["assistant_tag"] == "assistant" and info["demo_val"]["file_name"] == "../../val.jsonl"
    yaml = (out / "recipe/llamafactory/lora_sft.yaml").read_text(encoding="utf-8")
    for line in ("model_name_or_path: org/student-base", "trust_remote_code: true", "template: chatml",
                 "finetuning_type: lora", "lora_rank: 8", "lora_target: all", "dataset: demo", "eval_dataset: demo_val",
                 "train_on_prompt: false", f"cutoff_len: {manifest['length_report']['cutoff_len']}", "bf16: true"):
        assert line in yaml, line
    assert str(out / "recipe" / "llamafactory").replace("\\", "/") in yaml.replace("\\", "/")
    card = (out / "README.md").read_text(encoding="utf-8")
    # 머리말의 언어와 언어 태그는 설정값에서 나온다 -- 코드에 박힌 'ko'가 아니다.
    assert card.startswith(f"---\nlanguage:\n- {cfg.language}\n") and "soda" in card and "org/student-base" in card
    assert ("- korean" in card) == (cfg.language == "ko")
    # 카드의 외부 데이터셋 줄에 원어와 번역 모델이 같이 적힌다.
    assert "원어 en → `teacher/model-3b` 번역" in card


def test_data_card_language_comes_from_the_config_not_the_code(tmp_path, monkeypatch):
    """설정의 language가 카드 머리말과 manifest를 정한다. ko가 아니면 korean 태그도 없다."""
    monkeypatch.setattr(export_mod, "_load_tokenizer", lambda student: None)
    cfg = _project(tmp_path, language="ja")
    execute(ExportStage(), cfg, log=lambda m: None)
    out = cfg.datasets_root / "demo"
    assert json.loads((out / "manifest.json").read_text(encoding="utf-8"))["language"] == "ja"
    card = (out / "README.md").read_text(encoding="utf-8")
    assert card.startswith("---\nlanguage:\n- ja\n") and "- korean" not in card


def test_export_uses_the_student_tokenizer_when_available(tmp_path, monkeypatch):
    class Tok:
        def encode(self, text):
            class Enc:
                ids = list(range(len(text) // 2))
            return Enc()
    monkeypatch.setattr(export_mod, "_load_tokenizer", lambda student: Tok())
    cfg = _project(tmp_path)
    execute(ExportStage(), cfg, log=lambda m: None)
    report = json.loads((cfg.datasets_root / "demo" / "manifest.json").read_text(encoding="utf-8"))["length_report"]
    assert report["method"] == "tokens:org/student-base" and report["cutoff_len"] % 64 == 0 and report["cutoff_len"] >= 256


def test_export_name_override_and_missing_input(tmp_path, monkeypatch):
    monkeypatch.setattr(export_mod, "_load_tokenizer", lambda student: None)
    cfg = _project(tmp_path)
    execute(ExportStage(name_override="other"), cfg, log=lambda m: None)
    assert (cfg.datasets_root / "other" / "train.jsonl").exists()
    (cfg.data_root / "final" / "val.jsonl").unlink()
    with pytest.raises(FileNotFoundError, match="assemble"):
        execute(ExportStage(), cfg, log=lambda m: None)


def test_unknown_recipe_kind_is_a_config_error(tmp_path):
    from persona_sft_data.core.config import ConfigError
    with pytest.raises(ConfigError, match="recipe"):
        execute(ExportStage(), _project(tmp_path, recipe={"kind": "nope"}), log=lambda m: None)
