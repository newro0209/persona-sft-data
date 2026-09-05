"""assemble: 개수 비율로 섞고 세션 단위로 나누고 manifest를 쓴다.

토큰 예산은 없다. PEFT 데이터는 규모가 아니라 구성이 문제라 세션 개수로 센다.
비율이 안 맞으면 조용히 바꾸지 않고 SHORTFALL로 적는다. manifest는 "이 코퍼스가
어떤 설정·시드·문서·학생에서 나왔는가"에 혼자 답한다.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from persona_sft_data.core import schema
from persona_sft_data.core.registry import STAGES
from persona_sft_data.core.runner import StageContext, metric

SPLITS = ("train", "val", "test")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class AssembleSettings:
    ratios: dict[str, float]
    split: dict[str, float]
    max_sessions: int = 8000


@STAGES.register("assemble", origin="builtin")
class AssembleStage:
    name = "assemble"
    config_name = "assemble"
    mode = "records"
    record_kind = "session"
    produces = "final"
    settings_type = AssembleSettings

    def __init__(self) -> None:
        # run이 뽑은 것과 finalize가 세는 것을 잇는 장부. 한 execute 안에서 run 다음에
        # finalize가 오는 한 쌍이므로 인스턴스에 둔다.
        self._bucket_of: dict[str, str] = {}
        self._shortfall: dict[str, int] = {}

    def requires(self, config: Any) -> tuple[str, ...]:
        return ("filter",)

    def instances(self, config: Any) -> list[Any]:
        return [self]

    def preflight(self, ctx: StageContext) -> None:
        return None

    def run(self, ctx: StageContext) -> Iterator[dict[str, Any]]:
        s = ctx.settings
        ratios = dict(s.ratios)
        split = dict(s.split)
        pools: dict[str, list[dict[str, Any]]] = {}
        for bucket in ratios:
            path = ctx.config.filtered(bucket)
            if not path.exists():
                raise FileNotFoundError(f"{path}가 없다. 먼저 filter 단계를 돌려라.")
            pools[bucket] = list(schema.read_jsonl(path))

        selected: list[dict[str, Any]] = []
        shortfall: dict[str, int] = {}
        self._bucket_of = {}
        for bucket, pool in pools.items():
            ctx.rng.shuffle(pool)
            want = int(round(int(s.max_sessions) * ratios[bucket]))
            take = pool[:want]
            if len(take) < want:
                shortfall[bucket] = want - len(take)
            for record in take:
                self._bucket_of[str(record.get("id"))] = bucket
            selected.extend(take)
            ctx.log(f"[assemble] {bucket}: {len(pool):,} available, {want:,} wanted, {len(take):,} taken")
        if shortfall:
            ctx.log(f"[assemble] SHORTFALL (sessions): {shortfall}")
        self._shortfall = shortfall

        # split은 여기서 붙인다. 파일로 나누는 것은 러너가 거절을 걸러 낸 뒤 finalize가 한다.
        ctx.rng.shuffle(selected)
        n = len(selected)
        n_val = int(n * split["val"])
        n_test = int(n * split["test"])
        for i, record in enumerate(selected):
            record["split"] = "val" if i < n_val else "test" if i < n_val + n_test else "train"
            yield record

        ctx.log(f"[assemble] {n:,} sessions을 러너에 넘겼다")
        yield metric(extra={"shortfall": shortfall})

    def finalize(self, ctx: StageContext, stats: Any) -> None:
        """러너가 통과시킨 것만으로 split 파일과 manifest를 쓴다.

        run에서 쓰면 러너의 정규화·지문 중복 제거·게이트를 우회해, 러너가 거절한
        세션이 split 파일에 남고 export가 그것을 데이터셋에 싣는다. 그래서 러너가
        제자리에 옮겨 놓은 자기 출력(``final/assemble.jsonl``)을 다시 읽어 나눈다.
        """
        assert ctx.output is not None
        s = ctx.settings
        ratios = dict(s.ratios)
        split = dict(s.split)

        by_split: dict[str, list[dict[str, Any]]] = {k: [] for k in SPLITS}
        selected: dict[str, int] = {b: 0 for b in ratios}
        for record in schema.read_jsonl(ctx.output):
            name = str(record.get("split", ""))
            if name not in by_split:
                raise schema.SchemaError(f"{ctx.output}에 split이 없는 레코드가 있다: id={record.get('id')!r}")
            by_split[name].append(record)
            bucket = self._bucket_of.get(str(record.get("id"))) or str(record.get("source") or "unknown")
            selected[bucket] = selected.get(bucket, 0) + 1

        final_dir = ctx.config.data_root / "final"
        files: dict[str, dict[str, Any]] = {}
        for name in SPLITS:
            path = final_dir / f"{name}.jsonl"
            schema.write_jsonl(path, by_split[name])
            files[name] = {"path": _describe(path, ctx.config.root), "sha256": sha256_of(path), "sessions": len(by_split[name])}

        manifest = {
            "generated_by": "persona_sft_data",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "config_path": _describe(ctx.config.path, ctx.config.root),
            "config": json.loads(ctx.config.path.read_text(encoding="utf-8")),
            "seed": ctx.config.seed,
            "profile": ctx.config.profile,
            "persona_doc": _describe(ctx.config.persona_doc, ctx.config.root),
            "persona_sha256": sha256_of(ctx.config.persona_doc),
            "student": {"model": ctx.config.student.model, "chat_template": ctx.config.student.chat_template},
            "requested_ratios": ratios,
            "max_sessions": int(s.max_sessions),
            "selected": selected,
            "shortfall": self._shortfall,
            "requested_split": split,
            "split_sessions": {k: len(v) for k, v in by_split.items()},
            "files": files,
            "stages": _stage_stats(ctx.config),
        }
        (final_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n"
        )
        total = sum(len(v) for v in by_split.values())
        ctx.log(f"[assemble] {total:,} sessions; manifest -> final/manifest.json")


def _describe(path: Path, root: Path) -> str:
    """manifest에 적는 경로. 저장소 루트 기준 상대 경로(posix), 밖이면 절대 경로."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _stage_stats(config: Any) -> dict[str, Any]:
    """raw/·filtered/의 stats 파일을 요약해 manifest에 접는다."""
    out: dict[str, Any] = {}
    for sub in ("raw", "filtered"):
        for stats_file in sorted((config.data_root / sub).glob("*.jsonl.stats.json")):
            key = f"{sub}/{stats_file.name[: -len('.jsonl.stats.json')]}"
            try:
                data = json.loads(stats_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            out[key] = {k: data.get(k) for k in ("produced", "rejected", "yield_rate", "seconds", "teacher_model",
                                                  "teacher_calls", "teacher_failures", "reject_reasons")}
    return out
