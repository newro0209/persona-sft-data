"""respond: ingest의 발화마다 교사가 한 줄 답하고, 출처 필드가 레코드로 옮겨진다."""
from persona_sft_data.core.config import PipelineConfig
from persona_sft_data.core.runner import execute
from persona_sft_data.core.schema import read_jsonl, write_jsonl
from persona_sft_data.stages.respond import RespondStage
from persona_sft_data.teacher.fake import FakeTeacher
from tests.conftest import write_config

UTTERANCES = [
    {"id": "ko-000001", "text": "밥 먹었어?", "source": "ko", "language": "ko", "license": "smilestyle", "url": None},
    {"id": "en-000000", "text": "같이 놀자", "source": "en", "language": "ko", "license": "cc-by-4.0", "url": "http://x",
     "original_text": "Let's play", "original_language": "en", "translator": "fake"},
    {"id": "ko-000002", "text": "졸려?", "source": "ko", "language": "ko", "license": "smilestyle", "url": None},
]


def _config(tmp_path, **respond):
    cfg = PipelineConfig.load(write_config(tmp_path, stages={"respond": {"teacher": "fake", **respond}}))
    write_jsonl(cfg.raw("ingest"), UTTERANCES)
    return cfg


def test_each_utterance_becomes_a_two_turn_session_with_provenance(tmp_path):
    fake = FakeTeacher(reply_fn=lambda r: f'A: "응, 좋아 {r.user[:2]}."')
    cfg = _config(tmp_path)
    stats = execute(RespondStage(teacher=fake), cfg, log=lambda m: None)
    records = {r["utterance_id"]: r for r in read_jsonl(cfg.raw("respond"))}
    assert stats.produced == 3 and fake.checked
    en = records["en-000000"]
    assert en["id"] == "respond-en-000000" and en["source"] == "respond" and en["scenario"] == "source:en"
    assert en["source_dataset"] == "en" and en["source_url"] == "http://x" and en["original_language"] == "en"
    assert en["license"] == "cc-by-4.0" and en["generator"] == ["fake"]
    assert en["turns"] == [{"role": "user", "text": "같이 놀자"}, {"role": "assistant", "text": "응, 좋아 같이."}]
    assert all("[반드시 지킬 것]" in r.system for r in fake.seen)


def test_empty_replies_and_limit(tmp_path):
    fake = FakeTeacher(reply_fn=lambda r: "" if "졸려" in r.user else "응.")
    stats = execute(RespondStage(teacher=fake), _config(tmp_path, limit=2), log=lambda m: None)
    assert len(fake.seen) == 2 and stats.produced + stats.rejected == 2


def test_missing_ingest_output_says_which_stage_to_run(tmp_path):
    cfg = PipelineConfig.load(write_config(tmp_path, stages={"respond": {"teacher": "fake"}}))
    try:
        execute(RespondStage(teacher=FakeTeacher()), cfg, log=lambda m: None)
    except FileNotFoundError as exc:
        assert "'ingest'" in str(exc)
    else:
        raise AssertionError("FileNotFoundError expected")
