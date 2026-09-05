"""The teacher interface, and the one implementation that talks to vLLM.

There is a single boundary between this pipeline and the models: an
OpenAI-compatible HTTP endpoint. The pipeline runs on Windows, the server runs
in WSL2, and neither knows anything about the other beyond that URL. See
``docs/wsl-vllm.md`` for standing the server up — it must be served under
NAT networking with ``HF_HUB_OFFLINE=1``, for reasons measured there.

Only stdlib is used. ``urllib`` plus a thread pool is enough for a
request/response API, and vLLM's continuous batching does the real work: 400
concurrent requests measured 2,517 tok/s against 275 tok/s at 20, so what
matters is issuing them together, not which client library does it.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Protocol

from persona_sft_data.config import TeacherConfig


class TeacherError(RuntimeError):
    """Raised when the server cannot serve what the config asked for."""


@dataclass(frozen=True)
class Request:
    """One generation. ``key`` travels through so callers can match results
    back to whatever produced them without relying on list order."""

    key: str
    system: str
    user: str
    max_tokens: int | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class Result:
    key: str
    text: str | None
    completion_tokens: int = 0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.text is not None


class Teacher(Protocol):
    """What the stages depend on. ``VLLMTeacher`` is the real one;
    tests use a fake so that no unit test needs a GPU."""

    name: str

    def generate(self, requests: Sequence[Request]) -> list[Result]: ...


class VLLMTeacher:
    """An OpenAI-compatible chat-completions client."""

    def __init__(self, cfg: TeacherConfig, *, retries: int = 2) -> None:
        self.cfg = cfg
        self.name = cfg.name
        self.retries = retries
        self._url = cfg.base_url.rstrip("/") + "/v1/chat/completions"

    # ---- preflight -------------------------------------------------------

    def check(self) -> None:
        """Confirm the server is up and serving the model this config names.

        vLLM 404s on an id it is not serving, so a config pointing at a server
        running the other teacher fails here rather than quietly generating
        eight million tokens with the wrong model.
        """
        models_url = self.cfg.base_url.rstrip("/") + "/v1/models"
        try:
            with urllib.request.urlopen(models_url, timeout=10) as r:
                served = [m["id"] for m in json.loads(r.read())["data"]]
        except Exception as exc:  # noqa: BLE001 - report anything as a setup fault
            raise TeacherError(
                f"teacher {self.name!r}: cannot reach {models_url} ({exc}).\n"
                "  Is the vLLM server up? See docs/wsl-vllm.md"
            ) from exc
        if self.cfg.model not in served:
            raise TeacherError(
                f"teacher {self.name!r} wants {self.cfg.model!r} but the server "
                f"at {self.cfg.base_url} serves {served}.\n"
                "  The two teachers share a port and are loaded one at a time; "
                "restart the server with the model this stage needs."
            )

    # ---- generation ------------------------------------------------------

    def _once(self, req: Request) -> Result:
        body = json.dumps(
            {
                "model": self.cfg.model,
                "messages": [
                    {"role": "system", "content": req.system},
                    {"role": "user", "content": req.user},
                ],
                "temperature": (
                    self.cfg.temperature if req.temperature is None else req.temperature
                ),
                "top_p": self.cfg.top_p,
                "max_tokens": req.max_tokens or self.cfg.max_tokens,
            }
        ).encode()
        http = urllib.request.Request(
            self._url, body, {"Content-Type": "application/json"}
        )
        last: str = "unknown"
        for attempt in range(self.retries + 1):
            try:
                with urllib.request.urlopen(http, timeout=self.cfg.timeout) as r:
                    payload = json.loads(r.read())
                return Result(
                    key=req.key,
                    text=payload["choices"][0]["message"]["content"].strip(),
                    completion_tokens=payload.get("usage", {}).get(
                        "completion_tokens", 0
                    ),
                )
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")[:200]
                if exc.code == 404:
                    # Not transient: the server is serving a different model.
                    return Result(req.key, None, error=f"404 {detail}")
                last = f"HTTP {exc.code} {detail}"
            except Exception as exc:  # noqa: BLE001 - network faults are transient
                last = f"{type(exc).__name__}: {exc}"
            if attempt < self.retries:
                time.sleep(1.0 + attempt)
        return Result(req.key, None, error=last)

    def generate(self, requests: Sequence[Request]) -> list[Result]:
        """Issue every request at once and return results in input order.

        Failures come back as ``Result`` with ``text=None`` rather than raising,
        so a stage can record how many calls failed instead of losing a batch.
        """
        if not requests:
            return []
        workers = min(self.cfg.concurrency, len(requests))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._once, requests))


class FakeTeacher:
    """Deterministic stand-in so stage tests need neither GPU nor WSL."""

    def __init__(self, replies: dict[str, str] | None = None, *, default: str = "") -> None:
        self.name = "fake"
        self.replies = replies or {}
        self.default = default
        self.seen: list[Request] = []

    def check(self) -> None:
        return None

    def generate(self, requests: Sequence[Request]) -> list[Result]:
        out = []
        for req in requests:
            self.seen.append(req)
            text = self.replies.get(req.key, self.default)
            out.append(Result(req.key, text, completion_tokens=len(text)))
        return out


def build(cfg: TeacherConfig, *, overrides: dict | None = None) -> VLLMTeacher:
    """Construct a teacher, optionally with per-stage sampling overrides."""
    if overrides:
        cfg = replace(cfg, **overrides)
    return VLLMTeacher(cfg)


def batched(items: Iterable, size: int):
    """Yield lists of at most ``size``. Stages use this to bound how much sits
    in memory, not to limit concurrency — vLLM wants the whole batch at once."""
    batch: list = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch
