"""assemble: 개수 비율로 섞고, 세션 단위로 나누고, manifest가 재현에 필요한 것을 담는다."""
import json

import pytest

from persona_sft_data.core.config import PipelineConfig
from persona_sft_data.core.registry import STAGES
from persona_sft_data.core.runner import execute
from persona_sft_data.core.schema import read_jsonl, write_jsonl
from persona_sft_data.stages.assemble import AssembleStage
from tests.conftest import write_config


@pytest.fixture(autouse=True)
def _real_assemble():
    """test_config.py의 autouse 픽스처가 'assemble' 단계를 ``max_sessions`` 없는 자리 표시로
    덮어쓴 채 세션 레지스트리에 남긴다. 설정 검증이 진짜 ``AssembleSettings``를 쓰도록 이
    모듈에서는 내장 구현을 다시 올리고, 끝나면 있던 그대로 되돌린다(test_stage_ingest와 같은 규칙)."""
    previous = STAGES._items.get("assemble")
    STAGES.add("assemble", AssembleStage, origin="plugins")
    yield
    if previous is not None:
        STAGES._items["assemble"] = previous


def _s(i, source):
    # 러너가 내용 지문으로 중복을 걸러 내므로, 소스가 달라도 턴 텍스트가 같으면 하나만 남는다.
    # 버킷별 선택 개수를 세는 테스트라 사용자 발화에 소스 이름을 넣어 지문을 갈라 둔다.
    return {"id": f"{source}-{i}", "source": source, "scenario": "x", "license": "synthetic", "generator": ["m"],
            "turns": [{"role": "user", "text": f"{source} 질문 {i}"}, {"role": "assistant", "text": f"응, 좋아 {i}."}]}


def _config(tmp_path, ratios, max_sessions=8):
    cfg = PipelineConfig.load(write_config(tmp_path, stages={
        "dialogue": {"teacher": "fake"}, "respond": {"teacher": "fake"}, "filter": {},
        "assemble": {"ratios": ratios, "max_sessions": max_sessions, "split": {"train": 0.5, "val": 0.25, "test": 0.25}},
    }))
    (cfg.data_root / "raw").mkdir(parents=True, exist_ok=True)
    (cfg.raw("dialogue")).write_text("", encoding="utf-8")
    cfg.stats_path(cfg.raw("dialogue")).write_text(json.dumps({"produced": 10, "rejected": 1, "yield_rate": 0.9}), encoding="utf-8")
    return cfg


def test_draws_each_bucket_to_its_share_and_reports_shortfall(tmp_path):
    cfg = _config(tmp_path, {"dialogue": 0.5, "respond": 0.5})
    write_jsonl(cfg.filtered("dialogue"), [_s(i, "dialogue") for i in range(10)])
    write_jsonl(cfg.filtered("respond"), [_s(i, "respond") for i in range(2)])
    stats = execute(AssembleStage(), cfg, log=lambda m: None)
    records = list(read_jsonl(cfg.final("assemble")))
    by_source = {s: sum(r["source"] == s for r in records) for s in ("dialogue", "respond")}
    assert by_source == {"dialogue": 4, "respond": 2} and stats.produced == 6
    splits = {s: len(list(read_jsonl(cfg.final(s)))) for s in ("train", "val", "test")}
    assert sum(splits.values()) == 6 and splits["val"] == 1 and splits["test"] == 1
    assert {r["split"] for r in records} == {"train", "val", "test"}
    manifest = json.loads((cfg.data_root / "final" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["shortfall"] == {"respond": 2} and manifest["requested_ratios"] == {"dialogue": 0.5, "respond": 0.5}
    assert manifest["selected"] == {"dialogue": 4, "respond": 2} and manifest["split_sessions"]["train"] == 4
    assert manifest["config"]["seed"] == 7 and manifest["persona_sha256"] and manifest["student"]["model"] == "org/student-base"
    assert manifest["files"]["train"]["sha256"] and manifest["stages"]["raw/dialogue"]["produced"] == 10
    assert manifest["profile"] == "dummy"


def test_missing_filtered_input_names_filter(tmp_path):
    cfg = _config(tmp_path, {"dialogue": 1.0})
    with pytest.raises(FileNotFoundError, match="filter"):
        execute(AssembleStage(), cfg, log=lambda m: None)


def test_same_seed_same_selection_and_split(tmp_path):
    cfg = _config(tmp_path, {"dialogue": 1.0}, max_sessions=5)
    write_jsonl(cfg.filtered("dialogue"), [_s(i, "dialogue") for i in range(20)])
    execute(AssembleStage(), cfg, log=lambda m: None)
    first = [(r["id"], r["split"]) for r in read_jsonl(cfg.final("assemble"))]
    execute(AssembleStage(), cfg, log=lambda m: None)
    assert first == [(r["id"], r["split"]) for r in read_jsonl(cfg.final("assemble"))]
