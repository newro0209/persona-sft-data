"""OpenAI 호환 chat-completions 백엔드. vLLM이 대표다.

표준 라이브러리 ``urllib``과 스레드 풀이면 충분하다. 실제 일은 vLLM의 연속 배칭이
하므로 중요한 것은 요청을 한꺼번에 던지는 것이지 어떤 클라이언트를 쓰느냐가
아니다.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

from persona_sft_data.core.config import TeacherConfig
from persona_sft_data.core.registry import TEACHERS
from persona_sft_data.teacher.base import Request, Result, TeacherError


class OpenAICompatTeacher:
    def __init__(self, cfg: TeacherConfig, *, retries: int = 2) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self.retries = retries
        self._url = cfg.base_url.rstrip("/") + "/v1/chat/completions"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.cfg.api_key:
            headers["Authorization"] = f"Bearer {self.cfg.api_key}"
        return headers

    def check(self) -> None:
        """서버가 떠 있고 설정의 모델을 서빙하는지. 다른 모델이 떠 있으면 생성 전에 멈춘다."""
        models_url = self.cfg.base_url.rstrip("/") + "/v1/models"
        try:
            req = urllib.request.Request(models_url, headers=self._headers())
            with urllib.request.urlopen(req, timeout=10) as r:
                served = [m["id"] for m in json.loads(r.read())["data"]]
        except Exception as exc:  # noqa: BLE001 - 어떤 실패든 설정 문제로 보고
            raise TeacherError(
                f"교사 {self.name!r}: {models_url}에 닿지 못했다 ({exc}).\n  서버가 떠 있는가? docs/wsl-vllm.md"
            ) from exc
        if self.cfg.model not in served:
            raise TeacherError(
                f"교사 {self.name!r}은(는) {self.cfg.model!r}을(를) 원하는데 {self.cfg.base_url}은(는) {served}을(를) 서빙한다.\n"
                "  이 단계가 필요로 하는 모델로 서버를 다시 띄워라."
            )

    def _once(self, req: Request) -> Result:
        body = json.dumps({
            "model": self.cfg.model,
            "messages": [{"role": "system", "content": req.system}, {"role": "user", "content": req.user}],
            "temperature": self.cfg.temperature if req.temperature is None else req.temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": req.max_tokens or self.cfg.max_tokens,
        }).encode()
        http = urllib.request.Request(self._url, body, self._headers())
        last = "unknown"
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(http, timeout=self.cfg.timeout) as r:
                    payload = json.loads(r.read())
                return Result(
                    key=req.key,
                    text=payload["choices"][0]["message"]["content"].strip(),
                    completion_tokens=payload.get("usage", {}).get("completion_tokens", 0),
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:200]
                if exc.code == 404:
                    return Result(req.key, None, error=f"404 {detail}")
                last = f"HTTP {exc.code} {detail}"
            except Exception as exc:  # noqa: BLE001 - 네트워크 오류는 일시적일 수 있다
                last = f"{type(exc).__name__}: {exc}"
            if attempt < self.retries:
                time.sleep(1.0 + attempt)
        return Result(req.key, None, error=last)

    def generate(self, requests: Sequence[Request]) -> list[Result]:
        """전부 한꺼번에 보내고 입력 순서로 돌려준다. 실패는 예외가 아니라 ``text=None``."""
        if not requests:
            return []
        workers = min(self.cfg.concurrency, len(requests))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._once, requests))


@TEACHERS.register("openai", origin="builtin")
class OpenAIFactory:
    name = "openai"

    def build(self, cfg: TeacherConfig) -> OpenAICompatTeacher:
        return OpenAICompatTeacher(cfg)
