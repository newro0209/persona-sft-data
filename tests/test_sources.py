"""소스: 포맷은 어댑터, 추출은 전략, 번역은 교사, 위험·주제 필터는 페르소나에서."""
import json
from dataclasses import dataclass

import pytest

from persona_sft_data.core.config import ConfigError, SourceConfig
from persona_sft_data.core.persona import load
from persona_sft_data.core.registry import EXTRACTORS, FORMATS, TRANSLATORS
from persona_sft_data.sources import base, safety, topic
from persona_sft_data.sources.translate import TeacherTranslator
from persona_sft_data.teacher.fake import FakeTeacher
from tests.conftest import DOC, FIXTURES


def _cfg(name, fmt, filename, fields, extract=None, language="ko"):
    raw = {"format": fmt, "path": f"tests/fixtures/{filename}", "fields": fields, "language": language, "license": "mit"}
    if extract:
        raw["extract"] = extract
    return SourceConfig.from_dict(name, raw, FIXTURES.parents[1])


def test_formats_are_registered():
    assert {"tsv", "csv", "jsonl", "json", "parquet", "text"} <= set(FORMATS.names())
    assert {"field", "regex", "conversation", "list"} <= set(EXTRACTORS.names())
    assert "teacher" in TRANSLATORS.names()


def test_tsv_reads_only_the_selected_columns_and_skips_empty_cells():
    cfg = _cfg("s", "tsv", "utterances.tsv", ["informal", "chat"])
    data = cfg.path.read_bytes()
    assert list(base.read_utterances(cfg, data)) == ["밥 먹었어?", "밥 먹음?", "같이 놀자", "놀자ㅋㅋ", "졸려"]


def test_jsonl_with_field_regex_and_conversation_extractors():
    data = (FIXTURES / "utterances.jsonl").read_bytes()
    assert list(base.read_utterances(_cfg("a", "jsonl", "utterances.jsonl", ["instruction"]), data)) == ["오늘 기분 어때?", "심심한데 뭐 하지"]
    rx = {"kind": "regex", "pattern": r"<usr>\s*(.*?)\s*(?=<bot>|<usr>|$)"}
    assert list(base.read_utterances(_cfg("b", "jsonl", "utterances.jsonl", ["text"], rx), data)) == ["배고파", "졸려"]
    conv = {"kind": "conversation", "exclude_roles": ["assistant", "bot"]}
    assert list(base.read_utterances(_cfg("c", "jsonl", "utterances.jsonl", ["conv"], conv), data)) == ["같이 있자", "궁금해"]
    only = {"kind": "conversation", "include_roles": ["human"]}
    assert list(base.read_utterances(_cfg("d", "jsonl", "utterances.jsonl", ["conv"], only), data)) == ["궁금해"]


def test_json_list_extractor_keeps_even_or_odd_or_all():
    data = (FIXTURES / "utterances.json").read_bytes()
    even = _cfg("e", "json", "utterances.json", ["dialog"], {"kind": "list", "keep": "even"})
    assert list(base.read_utterances(even, data)) == ["안녕", "뭐 해?", "졸려"]
    odd = _cfg("o", "json", "utterances.json", ["dialog"], {"kind": "list", "keep": "odd"})
    assert list(base.read_utterances(odd, data)) == ["응 안녕", "놀아"]
    bad = _cfg("x", "json", "utterances.json", ["dialog"], {"kind": "list", "keep": "some"})
    with pytest.raises(ConfigError, match="keep"):
        list(base.read_utterances(bad, data))


def test_text_format_is_one_line_per_utterance():
    cfg = _cfg("t", "text", "lines.txt", ["text"])
    assert list(base.read_utterances(cfg, cfg.path.read_bytes())) == ["배고파", "같이 놀자", "이제 배불러"]


def test_parquet_projects_only_the_named_columns(tmp_path):
    pq = pytest.importorskip("pyarrow.parquet")
    pa = pytest.importorskip("pyarrow")
    table = pa.table({"instruction": ["배고파", "졸려"], "output": ["저는 챗봇입니다", "x"]})
    pq.write_table(table, tmp_path / "p.parquet")
    rows = list(FORMATS.get("parquet").rows((tmp_path / "p.parquet").read_bytes(), ["instruction"]))
    assert rows == [{"instruction": "배고파"}, {"instruction": "졸려"}]


def test_unknown_extract_setting_is_a_config_error():
    cfg = _cfg("r", "jsonl", "utterances.jsonl", ["text"], {"kind": "regex", "pattern": "x", "flags": 1})
    with pytest.raises(ConfigError, match="flags"):
        list(base.read_utterances(cfg, b'{"text": "x"}\n'))


def test_fetch_source_uses_path_or_downloads_once_into_cache(tmp_path, monkeypatch):
    logs = []
    local = _cfg("l", "text", "lines.txt", ["text"])
    assert base.fetch_source(local, tmp_path, timeout=1, log=logs.append) == local.path.read_bytes()
    calls = []
    monkeypatch.setattr(base, "_fetch", lambda url, timeout: calls.append(url) or b"a\n")
    remote = SourceConfig.from_dict("r", {"format": "text", "url": "http://x/a.txt", "fields": ["text"], "language": "ko", "license": "mit"}, tmp_path)
    assert base.fetch_source(remote, tmp_path, timeout=1, log=logs.append) == b"a\n"
    assert base.fetch_source(remote, tmp_path, timeout=1, log=logs.append) == b"a\n"
    assert calls == ["http://x/a.txt"] and (tmp_path / "r.txt").exists()
    monkeypatch.setattr(base, "_fetch", lambda url, timeout: (_ for _ in ()).throw(OSError("offline")))
    broken = SourceConfig.from_dict("b", {"format": "text", "url": "http://x/b.txt", "fields": ["text"], "language": "ko", "license": "mit"}, tmp_path)
    assert base.fetch_source(broken, tmp_path, timeout=1, log=logs.append) is None
    assert any("offline" in m for m in logs)
    missing = SourceConfig.from_dict("m", {"format": "text", "path": "nope.txt", "fields": ["text"], "language": "ko", "license": "mit"}, tmp_path)
    assert base.fetch_source(missing, tmp_path, timeout=1, log=logs.append) is None


def test_teacher_translator_batches_and_reports_failures():
    fake = FakeTeacher(reply_fn=lambda r: "" if r.user == "bad" else f'A: "{r.user} 번역"')
    tr = TeacherTranslator(fake, "ko", log=lambda m: None, batch_size=2)
    out = tr.translate(["hello", "bad", "bye"], "en")
    assert out == ["hello 번역", None, "bye 번역"]
    assert len(fake.seen) == 3 and "영어" in fake.seen[0].system and "한국어" in fake.seen[0].system


def test_safety_matches_token_initial_stems_only():
    assert safety.is_unsafe("이 씨발 뭐야")
    assert not safety.is_unsafe("가시발새우 먹고 싶다")
    assert safety.is_unsafe("아무말", stems=("아무",))


def test_topic_signal_comes_from_the_document_and_in_scope_is_coarse():
    p = load(DOC)
    sig = topic.signal(p)
    assert topic.bigrams("배고파") & sig
    assert topic.in_scope("배고픈데 밥 줄래?", sig)
    assert not topic.in_scope("양자역학의 파동함수를 설명해줘", sig)
    assert not topic.in_scope("hello there", sig)
    assert not topic.in_scope("배" * 70, sig)
    assert not topic.in_scope("배고파", sig, min_hits=50)
