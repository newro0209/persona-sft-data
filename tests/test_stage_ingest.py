"""ingest: 소스마다 읽고, 싸게 거르고, 표집하고, 필요하면 번역하고, 주제·안전 필터를 건다."""
import json

import pytest

from persona_sft_data.core.config import PipelineConfig
from persona_sft_data.core.registry import STAGES, TRANSLATORS
from persona_sft_data.core.runner import execute
from persona_sft_data.core.schema import read_jsonl
from persona_sft_data.sources.translate import TeacherTranslatorFactory
from persona_sft_data.stages.ingest import IngestStage
from persona_sft_data.teacher.fake import FakeTeacher
from tests.conftest import FIXTURES, write_config


@pytest.fixture(autouse=True)
def _real_plugins():
    """test_config.py의 autouse 픽스처가 'ingest' 단계와 'teacher' 번역기를 자리 표시로 덮어쓴 채
    세션 레지스트리에 남긴다. 설정 검증이 진짜 ``IngestSettings``를 쓰고 번역이 실제로 돌도록
    이 모듈에서는 내장 구현을 다시 올리고, 끝나면 있던 그대로 되돌린다."""
    targets = ((STAGES, "ingest", IngestStage), (TRANSLATORS, "teacher", TeacherTranslatorFactory))
    saved = [(registry, name, registry._items.get(name)) for registry, name, _ in targets]
    for registry, name, real in targets:
        registry.add(name, real, origin="plugins")
    yield
    for registry, name, previous in saved:
        if previous is not None:
            registry._items[name] = previous


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
