"""dialogue: 모든 beat를 빠짐없이 돌고, 교사 출력은 파싱·수선하며, 실패는 센다."""
import re

from persona_sft_data.core.config import PipelineConfig
from persona_sft_data.core.persona import load
from persona_sft_data.core.runner import execute
from persona_sft_data.core.schema import read_jsonl
from persona_sft_data.stages.dialogue import DialogueStage
from persona_sft_data.teacher.fake import FakeTeacher
from tests.conftest import DOC, write_config


def _reply(req):
    situation = req.user.splitlines()[0].split(":", 1)[1].strip()
    n = int(re.search(r"(\d+)번", req.user).group(1))
    lines = []
    for i in range(n):
        lines.append(f"U: {situation} 어때? {i}")
        lines.append(f"A: 응, 좋아 {i}.")          # 상황을 되풀이하면 긴 beat가 길이 규칙에 걸린다
    return "\n".join(lines)


def _config(tmp_path, **dialogue):
    return PipelineConfig.load(write_config(tmp_path, stages={"dialogue": {"teacher": "fake", "per_situation": 1, **dialogue}}))


def test_every_beat_gets_a_dialogue_with_the_corpus_shape(tmp_path):
    fake = FakeTeacher(reply_fn=_reply)
    cfg = _config(tmp_path, turns=[2])
    stats = execute(DialogueStage(teacher=fake), cfg, log=lambda m: None)
    beats = load(DOC).beats
    records = list(read_jsonl(cfg.raw("dialogue")))
    assert stats.produced == len(beats) and {r["scenario"] for r in records} == set(beats)
    r = records[0]
    assert r["id"].startswith("dialogue-") and r["source"] == "dialogue" and r["generator"] == ["fake"]
    assert len(r["turns"]) == 4 and r["turns"][0]["role"] == "user"
    assert fake.checked and stats.teacher_calls == len(beats) and stats.teacher_model == "fake"


def test_prompts_use_document_flows_and_configured_turns(tmp_path):
    fake = FakeTeacher(reply_fn=_reply)
    cfg = _config(tmp_path, turns=[3])
    execute(DialogueStage(teacher=fake), cfg, log=lambda m: None)
    flows = load(DOC).flows
    assert all("총 6줄" in r.user for r in fake.seen)
    assert all(any(f in r.user for f in flows) for r in fake.seen)
    assert all("[반드시 지킬 것]" in r.system for r in fake.seen)


def test_unparseable_and_failed_replies_are_counted(tmp_path):
    fake = FakeTeacher(reply_fn=lambda r: "그냥 산문")
    stats = execute(DialogueStage(teacher=fake), _config(tmp_path), log=lambda m: None)
    assert stats.produced == 0 and stats.reject_reasons["unparseable"] == stats.rejected > 0


def test_same_seed_same_prompts(tmp_path):
    a, b = FakeTeacher(reply_fn=_reply), FakeTeacher(reply_fn=_reply)
    execute(DialogueStage(teacher=a), _config(tmp_path), log=lambda m: None)
    execute(DialogueStage(teacher=b), _config(tmp_path), log=lambda m: None)
    assert [r.user for r in a.seen] == [r.user for r in b.seen] and [r.system for r in a.seen] == [r.system for r in b.seen]
