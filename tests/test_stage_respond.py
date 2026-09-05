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
    # 번역 모델도 세션으로 옮겨진다 -- 데이터 카드가 소스별 번역 모델을 적는 근거다.
    assert en["translator"] == "fake" and records["ko-000001"]["translator"] is None
    assert en["turns"] == [{"role": "user", "text": "같이 놀자"}, {"role": "assistant", "text": "응, 좋아 같이."}]
    assert all("[반드시 지킬 것]" in r.system for r in fake.seen)


def test_empty_replies_are_rejected_and_limit_caps_the_requests(tmp_path):
    """빈 답은 ``empty_reply``로 거절되고 그 발화는 출력에 없다. limit은 요청 수를 자른다.

    답은 게이트(4~35글자 반말)를 통과하는 것으로 쓴다 -- 게이트에 걸리는 답을 쓰면
    ``empty_reply`` 경로와 길이 거절이 섞여 무엇이 걸렀는지 구분되지 않는다.
    """
    fake = FakeTeacher(reply_fn=lambda r: "" if "졸려" in r.user else "응, 좋아.")
    cfg = _config(tmp_path, limit=2)
    stats = execute(RespondStage(teacher=fake), cfg, log=lambda m: None)
    assert len(fake.seen) == 2                      # 발화 3개 중 2개만 교사에게 간다
    assert stats.produced == 1
    assert stats.reject_reasons.get("empty_reply") == 1
    asked = {r.key for r in fake.seen}
    produced_ids = {r["utterance_id"] for r in read_jsonl(cfg.raw("respond"))}
    assert "ko-000002" in asked and produced_ids == asked - {"ko-000002"}


def test_missing_ingest_output_says_which_stage_to_run(tmp_path):
    cfg = PipelineConfig.load(write_config(tmp_path, stages={"respond": {"teacher": "fake"}}))
    try:
        execute(RespondStage(teacher=FakeTeacher()), cfg, log=lambda m: None)
    except FileNotFoundError as exc:
        assert "'ingest'" in str(exc)
    else:
        raise AssertionError("FileNotFoundError expected")
