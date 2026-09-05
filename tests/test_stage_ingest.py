"""ingest: 소스마다 읽고, 싸게 거르고, 표집하고, 필요하면 번역하고, 주제·안전 필터를 건다."""
import json

from persona_sft_data.core.config import PipelineConfig
from persona_sft_data.core.runner import build_context, execute
from persona_sft_data.core.schema import read_jsonl, write_jsonl
from persona_sft_data.stages.ingest import IngestStage
from persona_sft_data.stages.respond import RespondStage
from persona_sft_data.teacher.fake import FakeTeacher
from tests.conftest import FIXTURES, write_config


def _sources():
    return {
        "ko": {"format": "tsv", "path": str(FIXTURES / "utterances.tsv"), "fields": ["informal", "chat"], "language": "ko", "license": "smilestyle"},
        "en": {"format": "jsonl", "path": str(FIXTURES / "english.jsonl"), "fields": ["dialog"],
               "extract": {"kind": "list", "keep": "even"}, "language": "en", "license": "cc-by-4.0"},
    }


def _config(tmp_path, **ingest):
    return PipelineConfig.load(write_config(
        tmp_path, sources=_sources(),
        stages={"ingest": {"teacher": "fake", "sources": ["ko", "en"], **ingest}},
    ))


def test_reads_translates_and_filters_each_source(tmp_path):
    counter = iter(range(1, 100))
    fake = FakeTeacher(reply_fn=lambda r: f"같이 놀자 {next(counter)}")
    cfg = _config(tmp_path)
    stats = execute(IngestStage(teacher=fake), cfg, log=lambda m: None)
    records = list(read_jsonl(cfg.raw("ingest")))
    assert stats.produced == len(records) >= 4
    ko = [r for r in records if r["source"] == "ko"]
    en = [r for r in records if r["source"] == "en"]
    assert all(r["language"] == "ko" and r["license"] == "smilestyle" and "original_text" not in r for r in ko)
    assert len(en) == 2 and all(r["original_language"] == "en" and r["original_text"] and r["translator"] == "fake" for r in en)
    assert len(fake.seen) == 2 and "영어" in fake.seen[0].system                 # 한국어 소스는 번역하지 않는다
    assert fake.checked
    per_source = json.loads(cfg.stats_path(cfg.raw("ingest")).read_text(encoding="utf-8"))["sources"]
    assert per_source["en"]["translated"] == 2 and per_source["ko"]["raw"] == 5
    assert stats.source_filtered >= 1 and "off_topic" in stats.source_filter_reasons


def test_translation_failures_are_rejects_and_limit_bounds_the_teacher(tmp_path):
    fake = FakeTeacher(reply_fn=lambda r: "")
    cfg = _config(tmp_path, limit_per_source=1)
    stats = execute(IngestStage(teacher=fake), cfg, log=lambda m: None)
    assert len(fake.seen) == 1 and stats.reject_reasons.get("translation_failed") == 1
    assert stats.teacher_calls == 1 and stats.teacher_failures == 1


def test_a_source_that_cannot_be_read_is_skipped_not_fatal(tmp_path):
    cfg = PipelineConfig.load(write_config(
        tmp_path,
        sources={"gone": {"format": "text", "path": "nope.txt", "fields": ["text"], "language": "ko", "license": "mit"},
                 **{"ko": _sources()["ko"]}},
        stages={"ingest": {"teacher": "fake", "sources": ["gone", "ko"]}},
    ))
    logs = []
    stats = execute(IngestStage(teacher=FakeTeacher()), cfg, log=logs.append)
    assert stats.produced >= 3 and any("gone" in m for m in logs)


def test_preflight_skips_a_source_it_cannot_parse(tmp_path):
    """check는 소스 하나의 파싱 실패로 죽지 않는다.

    ``run``만 감싸고 ``preflight``를 감싸지 않으면 둘째 줄이 JSON이 아닌 jsonl 하나가
    ``check`` 전체를 트레이스백으로 끝내고, 나머지 소스·단계는 점검조차 못 한다.
    """
    cfg = PipelineConfig.load(write_config(
        tmp_path,
        sources={"broken": {"format": "jsonl", "path": str(FIXTURES / "broken.jsonl"), "fields": ["text"],
                            "language": "ko", "license": "mit"},
                 "ko": _sources()["ko"]},
        stages={"ingest": {"teacher": "fake", "sources": ["broken", "ko"]}},
    ))
    stage = IngestStage(teacher=FakeTeacher())
    logs = []
    stage.preflight(build_context(stage, cfg, log=logs.append))
    assert any("broken" in m and "읽을 수 없다" in m for m in logs)
    assert any("밥 먹었어?" in m for m in logs)                  # 뒤 소스는 그대로 점검된다


def test_all_sources_failing_still_writes_an_empty_output_and_stats(tmp_path):
    """소스 전부가 실패해도 산출물과 통계는 남는다: 0바이트 jsonl과 produced 0."""
    cfg = PipelineConfig.load(write_config(
        tmp_path,
        sources={"gone": {"format": "text", "path": "nope.txt", "fields": ["text"], "language": "ko", "license": "mit"}},
        stages={"ingest": {"teacher": "fake", "sources": ["gone"]}},
    ))
    stats = execute(IngestStage(teacher=FakeTeacher()), cfg, log=lambda m: None)
    out = cfg.raw("ingest")
    assert stats.produced == 0 and out.exists() and out.stat().st_size == 0
    assert list(read_jsonl(out)) == []
    written = json.loads(cfg.stats_path(out).read_text(encoding="utf-8"))
    assert written["produced"] == 0 and written["stage"] == "ingest"


def test_respond_says_the_ingest_output_is_empty(tmp_path):
    """빈(파일은 있는) ingest 출력에 respond는 알리고 아무것도 만들지 않는다.

    respond 쪽 테스트는 입력 파일이 아예 없는 경우만 덮는다. 이 경계는 ingest가
    빈 파일을 쓰는 위 동작과 짝이라서 여기에 둔다.
    """
    cfg = PipelineConfig.load(write_config(tmp_path, stages={"respond": {"teacher": "fake"}}))
    write_jsonl(cfg.raw("ingest"), [])
    fake = FakeTeacher()
    logs = []
    stats = execute(RespondStage(teacher=fake), cfg, log=logs.append)
    assert stats.produced == 0 and fake.seen == []
    assert any("ingest 출력이 비어 있어" in m for m in logs)


def test_blocked_stems_and_topic_hits_come_from_settings(tmp_path):
    cfg = _config(tmp_path, blocked_stems=["밥"], topic_min_hits=1)
    stats = execute(IngestStage(teacher=FakeTeacher(reply_fn=lambda r: "같이 놀자")), cfg, log=lambda m: None)
    assert stats.source_filter_reasons.get("unsafe_source", 0) >= 1
    assert all("밥" not in r["text"] for r in read_jsonl(cfg.raw("ingest")))


def test_same_seed_same_sample(tmp_path):
    a = execute(IngestStage(teacher=FakeTeacher(reply_fn=lambda r: "같이 놀자")), _config(tmp_path, limit_per_source=2), log=lambda m: None)
    first = [r["text"] for r in read_jsonl(_config(tmp_path).raw("ingest"))]
    b = execute(IngestStage(teacher=FakeTeacher(reply_fn=lambda r: "같이 놀자")), _config(tmp_path, limit_per_source=2), log=lambda m: None)
    assert first == [r["text"] for r in read_jsonl(_config(tmp_path).raw("ingest"))] and a.produced == b.produced
