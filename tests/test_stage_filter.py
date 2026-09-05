"""filter: raw 세션 파일마다 게이트를 다시 적용하고, 파일 전체를 봐야 하는 과다 반복만 여기서 거른다."""
import pytest

from persona_sft_data.core.config import PipelineConfig
from persona_sft_data.core.runner import execute
from persona_sft_data.core.schema import read_jsonl, write_jsonl
from persona_sft_data.stages.filter import FilterStage
from tests.conftest import write_config


def _s(i, text, source="dialogue"):
    return {"id": f"{source}-{i}", "source": source, "scenario": "x", "license": "synthetic", "generator": ["m"],
            "turns": [{"role": "user", "text": f"질문 {i}"}, {"role": "assistant", "text": text}]}


def _config(tmp_path, **filter_settings):
    return PipelineConfig.load(write_config(tmp_path, stages={
        "dialogue": {"teacher": "fake"}, "respond": {"teacher": "fake"},
        "filter": filter_settings,
    }))


def test_one_instance_per_existing_raw_session_file(tmp_path):
    cfg = _config(tmp_path)
    write_jsonl(cfg.raw("dialogue"), [_s(1, "응, 좋아.")])
    names = [inst.name for inst in FilterStage().instances(cfg)]
    assert names == ["dialogue"]
    write_jsonl(cfg.raw("respond"), [_s(1, "응.", "respond")])
    assert [inst.name for inst in FilterStage().instances(cfg)] == ["dialogue", "respond"]
    assert FilterStage().requires(cfg) == ("dialogue", "respond")


def test_overused_assistant_lines_are_dropped_and_the_gate_still_applies(tmp_path):
    cfg = _config(tmp_path, max_identical_assistant_turns=2)
    write_jsonl(cfg.raw("dialogue"), [
        _s(1, "응, 좋아."), _s(2, "응, 좋아."), _s(3, "응, 좋아."),   # 세 번째부터 과다
        _s(4, "잘래 🐾"),                                          # 게이트
        _s(5, "히히, 같이 놀자."),
    ])
    stats = execute(FilterStage("dialogue"), cfg, log=lambda m: None)
    kept = [r["id"] for r in read_jsonl(cfg.filtered("dialogue"))]
    assert kept == ["dialogue-1", "dialogue-2", "dialogue-5"]
    assert stats.reject_reasons["assistant_line_overused"] == 1 and stats.reject_reasons["emoji"] == 1
    assert stats.produced == 3 and stats.rejected == 2


def test_turn_bounds_come_from_filter_settings(tmp_path):
    cfg = _config(tmp_path, max_turns=2)
    long = {"id": "d-1", "source": "dialogue", "turns": [
        {"role": "user", "text": "하나"}, {"role": "assistant", "text": "응."},
        {"role": "user", "text": "둘"}, {"role": "assistant", "text": "그래."}]}
    write_jsonl(cfg.raw("dialogue"), [long])
    stats = execute(FilterStage("dialogue"), cfg, log=lambda m: None)
    assert stats.reject_reasons.get("too_many_turns") == 1


def test_no_raw_file_at_all_is_an_error_that_names_the_stages(tmp_path):
    cfg = _config(tmp_path)
    with pytest.raises(FileNotFoundError, match="dialogue"):
        FilterStage().instances(cfg)
