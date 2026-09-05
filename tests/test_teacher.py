"""교사: 팩토리로 만들고, 실패는 Result로 돌아오며, 스모크용 EchoTeacher는 형식을 지킨다."""
import json

import pytest

from persona_sft_data.core.config import TeacherConfig
from persona_sft_data.core.registry import TEACHERS
from persona_sft_data.teacher import openai_compat
from persona_sft_data.teacher.base import Request, TeacherError, batched
from persona_sft_data.teacher.fake import EchoTeacher, FakeTeacher
from persona_sft_data.teacher.prompts import parse_dialogue


def test_batched_bounds_memory_not_concurrency():
    assert list(batched(range(5), 2)) == [[0, 1], [2, 3], [4]]
    assert list(batched([], 3)) == []


def test_factories_are_registered_and_build_by_kind():
    cfg = TeacherConfig.from_dict("t", {"kind": "fake", "model": "m", "base_url": "http://x"})
    teacher = TEACHERS.get(cfg.kind).build(cfg)
    assert isinstance(teacher, EchoTeacher) and teacher.name == "t"
    cfg2 = TeacherConfig.from_dict("r", {"model": "m", "base_url": "http://x"})
    assert isinstance(TEACHERS.get(cfg2.kind).build(cfg2), openai_compat.OpenAICompatTeacher)


def test_fake_teacher_records_requests_and_answers_by_key():
    fake = FakeTeacher({"a": "답"}, default="기본")
    results = fake.generate([Request("a", "s", "u"), Request("b", "s", "u")])
    assert [r.text for r in results] == ["답", "기본"] and [r.key for r in fake.seen] == ["a", "b"]
    assert all(r.ok for r in results)


def test_echo_teacher_writes_a_parseable_dialogue_and_one_line_replies():
    echo = EchoTeacher("e")
    dialogue = echo.generate([Request("k", "sys", "상황: 배고픔\n흐름: 다정하게\n길이: 사용자 2번, 캐릭터 2번 (총 4줄)\n\n써라.")])[0].text
    turns = parse_dialogue(dialogue)
    assert len(turns) == 4 and turns[0]["role"] == "user" and turns[-1]["role"] == "assistant"
    reply = echo.generate([Request("k", "sys", "밥 먹었어?")])[0].text
    assert "\n" not in reply and reply
    translated = echo.generate([Request("k", "sys", "What do you want for dinner?")])[0].text
    assert any("가" <= ch <= "힣" for ch in translated)


class _Response:
    def __init__(self, payload, status=200):
        self._payload = json.dumps(payload).encode()
        self.status = status
    def read(self):
        return self._payload
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


def test_openai_compat_check_rejects_a_server_serving_another_model(monkeypatch):
    cfg = TeacherConfig.from_dict("t", {"model": "wanted", "base_url": "http://x"})
    monkeypatch.setattr(openai_compat.urllib.request, "urlopen",
                        lambda url, timeout=0: _Response({"data": [{"id": "other"}]}))
    with pytest.raises(TeacherError, match="wanted.*other"):
        openai_compat.OpenAICompatTeacher(cfg).check()


def test_openai_compat_generate_returns_results_in_order_and_failures_as_results(monkeypatch):
    cfg = TeacherConfig.from_dict("t", {"model": "m", "base_url": "http://x", "api_key": "k", "concurrency": 2})
    calls = []
    def fake_urlopen(req, timeout=0):
        body = json.loads(req.data)
        calls.append((req.get_header("Authorization"), body["messages"][1]["content"]))
        if body["messages"][1]["content"] == "boom":
            raise OSError("down")
        return _Response({"choices": [{"message": {"content": " 답: " + body["messages"][1]["content"] + " "}}],
                          "usage": {"completion_tokens": 3}})
    monkeypatch.setattr(openai_compat.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(openai_compat.time, "sleep", lambda s: None)
    teacher = openai_compat.OpenAICompatTeacher(cfg, retries=1)
    results = teacher.generate([Request("1", "s", "a"), Request("2", "s", "boom"), Request("3", "s", "c")])
    assert [r.key for r in results] == ["1", "2", "3"]
    assert results[0].text == "답: a" and results[0].completion_tokens == 3
    assert not results[1].ok and "down" in results[1].error
    assert calls[0][0] == "Bearer k"
