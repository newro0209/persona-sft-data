"""export: 조립된 코퍼스를 학습용 데이터셋 디렉터리로.

세션(``turns``)을 OpenAI ``messages``로 바꾸고, 시스템 프롬프트는 페르소나 문서에서
렌더링한다 — 모델이 학습하는 정의와 코퍼스를 만들고 검열한 정의가 같다. 채팅
템플릿, 길이 보고, 레시피, 카드를 같이 낸다. 러너 단계가 아니라 파일을 직접 쓴다.
"""

from __future__ import annotations

import json
import math
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from persona_sft_data.core import schema
from persona_sft_data.core.config import ConfigError, StudentConfig, build_settings
from persona_sft_data.core.registry import RECIPES, STAGES, PluginError
from persona_sft_data.core.runner import StageContext, StageStats
from persona_sft_data.recipes.base import ExportInfo, LengthReport
from persona_sft_data.recipes.chat_template import jinja_for, renderer_for
from persona_sft_data.stages.assemble import SPLITS, sha256_of

# 세션 레코드에서 messages 레코드로 옮기지 않는 필드. turns는 messages가 되고, split은 파일 이름이 말한다.
DROPPED = frozenset({"turns", "split"})


@dataclass(frozen=True)
class ExportSettings:
    name: str
    recipe: dict[str, Any]


def to_messages(record: Mapping[str, Any], system_prompt: str) -> dict[str, Any]:
    """세션 하나를 OpenAI ``messages`` 레코드로. 출처 필드는 그대로 실린다."""
    session = schema.RECORD_KINDS["session"].normalize(record)
    out = {k: v for k, v in session.items() if k not in DROPPED}
    out["messages"] = [{"role": "system", "content": system_prompt}] + [
        {"role": t["role"], "content": t["text"]} for t in session["turns"]
    ]
    return out


def _load_tokenizer(student: StudentConfig) -> Any | None:
    """``student`` extra가 있으면 학생 토크나이저. 없거나 실패하면 None → 글자 수로 잰다."""
    try:
        from huggingface_hub import hf_hub_download
        from tokenizers import Tokenizer
    except ImportError:
        return None
    try:
        return Tokenizer.from_file(hf_hub_download(student.model, "tokenizer.json"))
    except Exception:  # noqa: BLE001 - 오프라인, 없는 모델, 권한 등 전부 "측정 불가"
        return None


def _percentile(sorted_values: Sequence[int], q: float) -> int:
    if not sorted_values:
        return 0
    index = min(len(sorted_values) - 1, max(0, math.ceil(q * len(sorted_values)) - 1))
    return int(sorted_values[index])


def measure_lengths(texts: Sequence[str], student: StudentConfig) -> LengthReport:
    """렌더링된 학습 텍스트의 길이 분포. 토크나이저가 있으면 토큰, 없으면 글자 수.

    ``cutoff_len``은 토큰 기준이면 p99를 64의 배수로 올림(최소 256), 글자 기준이면 글자 수
    p99 그대로 — 한국어는 글자당 1토큰 미만이라 넉넉한 쪽이다.
    """
    tokenizer = _load_tokenizer(student)
    if tokenizer is not None:
        lengths = sorted(len(tokenizer.encode(t).ids) for t in texts)
        method = f"tokens:{student.model}"
    else:
        lengths = sorted(len(t) for t in texts)
        method = "characters"
    p99 = _percentile(lengths, 0.99)
    cutoff = max(256, math.ceil(p99 / 64) * 64) if tokenizer is not None else max(1, p99)
    return LengthReport(method=method, count=len(lengths), p50=_percentile(lengths, 0.5),
                        p95=_percentile(lengths, 0.95), p99=p99, max=lengths[-1] if lengths else 0, cutoff_len=cutoff)


@STAGES.register("export", origin="builtin")
class ExportStage:
    name = "export"
    config_name = "export"
    mode = "artifact"
    record_kind = None
    produces = None
    settings_type = ExportSettings

    def __init__(self, name_override: str | None = None) -> None:
        # CLI의 ``--name``. 설정의 이름 대신 이 이름의 디렉터리에 쓴다.
        self.name_override = name_override

    def requires(self, config: Any) -> tuple[str, ...]:
        return ("assemble",)

    def instances(self, config: Any) -> list[Any]:
        return [self]

    def preflight(self, ctx: StageContext) -> None:
        self._recipe(ctx)
        if _load_tokenizer(ctx.config.student) is None:
            ctx.log("[export] 학생 토크나이저를 쓸 수 없어 길이는 글자 수로 잰다 ([student] extra 설치 시 토큰으로)")

    def _recipe(self, ctx: StageContext) -> tuple[Any, Any]:
        """설정 ``recipe``에서 레시피 플러그인과 그 설정 dataclass를. ``kind``가 플러그인을 고른다."""
        raw = dict(ctx.settings.recipe or {})
        kind = raw.pop("kind", None)
        if not kind:
            raise ConfigError("stages.export.recipe.kind가 없다")
        try:
            recipe = RECIPES.get(kind)
        except PluginError as exc:
            raise ConfigError(f"stages.export.recipe: {exc}") from None
        return recipe, build_settings(recipe.settings_type, raw, "stages.export.recipe")

    def run(self, ctx: StageContext) -> StageStats:
        recipe, recipe_settings = self._recipe(ctx)
        name = self.name_override or ctx.settings.name
        out_dir = ctx.config.datasets_root / name
        out_dir.mkdir(parents=True, exist_ok=True)
        system_prompt = ctx.persona.system_prompt()
        template_name = ctx.config.student.chat_template
        render = renderer_for(template_name)
        started = time.time()

        files: dict[str, dict[str, Any]] = {}
        sources: Counter[str] = Counter()
        generators: Counter[str] = Counter()
        licenses: dict[str, set[str]] = {}
        source_datasets: dict[str, dict[str, Any]] = {}
        rendered: list[str] = []
        sample: list[str] = []
        turns_total = 0
        for split in SPLITS:
            src = ctx.config.final(split)
            if not src.exists():
                raise FileNotFoundError(f"{src}가 없다. 먼저 assemble 단계를 돌려라.")
            dst = out_dir / f"{split}.jsonl"
            n = 0
            with dst.open("w", encoding="utf-8", newline="\n") as handle:
                for record in schema.read_jsonl(src):
                    chat = to_messages(record, system_prompt)
                    schema.append_jsonl(handle, chat)
                    n += 1
                    sources[chat["source"]] += 1
                    licenses.setdefault(chat["source"], set()).add(str(chat.get("license", "")))
                    for g in chat.get("generator", ()):
                        generators[g] += 1
                    if chat.get("source_dataset"):
                        entry = source_datasets.setdefault(chat["source_dataset"], {
                            "url": chat.get("source_url"), "original_language": chat.get("original_language"),
                            "license": str(chat.get("license", "")), "records": 0})
                        entry["records"] += 1
                    turns_total += len(chat["messages"]) - 1
                    text = render(chat["messages"])
                    rendered.append(text)
                    if split == "train" and len(sample) < 3:
                        sample.append(text)
            files[split] = {"path": dst.name, "records": n, "sha256": sha256_of(dst)}
            ctx.log(f"[export] {split}: {n:,} records -> {_describe(dst, ctx.config.root)}")

        (out_dir / "system_prompt.txt").write_text(system_prompt + "\n", encoding="utf-8", newline="\n")
        (out_dir / "chat_template.jinja").write_text(jinja_for(template_name), encoding="utf-8", newline="\n")
        (out_dir / "rendered_sample.txt").write_text("\n---\n".join(sample), encoding="utf-8", newline="\n")
        report = measure_lengths(rendered, ctx.config.student)
        ctx.log(f"[export] length ({report.method}): p50 {report.p50} p95 {report.p95} p99 {report.p99} "
                f"max {report.max} -> cutoff_len {report.cutoff_len}")

        manifest: dict[str, Any] = {
            "name": name, "format": "openai-messages", "generated_by": "persona_sft_data",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "persona": ctx.persona.name, "profile": ctx.config.profile,
            "persona_doc": _describe(ctx.config.persona_doc, ctx.config.root),
            "persona_sha256": sha256_of(ctx.config.persona_doc),
            "config_path": _describe(ctx.config.path, ctx.config.root), "seed": ctx.config.seed,
            "student": {"model": ctx.config.student.model, "trust_remote_code": ctx.config.student.trust_remote_code},
            "chat_template": template_name,
            "records": sum(f["records"] for f in files.values()), "turns": turns_total,
            "files": files,
            "sources": dict(sources.most_common()),
            "licenses": {k: sorted(v) for k, v in licenses.items()},
            "generators": dict(generators.most_common()),
            "source_datasets": source_datasets,
            "length_report": report.to_dict(),
            "recipe": {"kind": recipe.name, **recipe_settings.__dict__},
        }
        corpus_manifest = ctx.config.data_root / "final" / "manifest.json"
        if corpus_manifest.exists():
            manifest["corpus_manifest_sha256"] = sha256_of(corpus_manifest)

        info = ExportInfo(name=name, out_dir=out_dir, root=ctx.config.root, files=files, student=ctx.config.student,
                          system_prompt=system_prompt, chat_template_name=template_name, length_report=report,
                          persona_name=ctx.persona.name, profile=ctx.config.profile, seed=ctx.config.seed)
        written = recipe.write(out_dir, info, recipe_settings)
        manifest["recipe"]["files"] = [p.relative_to(out_dir).as_posix() for p in written]

        (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                               encoding="utf-8", newline="\n")
        (out_dir / "README.md").write_text(dataset_card(manifest, system_prompt), encoding="utf-8", newline="\n")
        output = _describe(out_dir, ctx.config.root)
        ctx.log(f"[export] {manifest['records']:,} records, {turns_total:,} turns in {time.time() - started:.1f}s -> {output}")
        return StageStats(stage="export", output=output,
                          started=time.strftime("%Y-%m-%dT%H:%M:%S"), seconds=round(time.time() - started, 2),
                          produced=manifest["records"], extra={"length_report": report.to_dict(), "files": files})


def dataset_card(manifest: Mapping[str, Any], system_prompt: str) -> str:
    """Hugging Face 데이터 카드(YAML 머리말 + 마크다운). manifest만 보고 만든다."""
    records = manifest["records"]
    size_bucket = next(label for bound, label in ((1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"),
                                                  (1_000_000, "100K<n<1M"), (float("inf"), "1M<n<10M")) if records < bound)
    all_licenses = sorted({lic for lics in manifest["licenses"].values() for lic in lics})
    lines = ["---", "language:", "- ko", f"license: {'other' if len(all_licenses) != 1 else all_licenses[0]}",
             "task_categories:", "- text-generation", "tags:", "- persona", "- roleplay", "- sft", "- korean",
             "size_categories:", f"- {size_bucket}", "---", "",
             f"# {manifest['name']}", "",
             f"`{manifest['persona']}` 페르소나(프로필 `{manifest['profile']}`) PEFT 미세조정 데이터셋. "
             f"OpenAI `messages` 포맷, {records:,}개 대화 / {manifest['turns']:,}턴.", "",
             "## 분할", "", "| split | records | sha256 |", "| --- | --- | --- |"]
    lines += [f"| {s} | {f['records']:,} | `{f['sha256'][:16]}` |" for s, f in manifest["files"].items()]
    lines += ["", "## 출처와 라이선스", "", "| source | records | license |", "| --- | --- | --- |"]
    lines += [f"| {s} | {n:,} | {', '.join(manifest['licenses'].get(s, []))} |" for s, n in manifest["sources"].items()]
    if manifest["source_datasets"]:
        lines += ["", "외부 데이터셋 (사용자 발화의 원본):", ""]
        for name, d in manifest["source_datasets"].items():
            lang = f", 원어 {d['original_language']} → 교사 번역" if d.get("original_language") else ""
            lines.append(f"- `{name}` ({d['license']}{lang}) — {d.get('url') or '로컬 파일'} — {d['records']:,} records")
    lines += ["", "## 생성 모델", ""] + [f"- `{m}`: {n:,} records" for m, n in manifest["generators"].items()]
    lr = manifest["length_report"]
    lines += ["", "## 학생 모델과 템플릿", "",
              f"- 학생: `{manifest['student']['model']}` (trust_remote_code: {manifest['student']['trust_remote_code']})",
              f"- 채팅 템플릿: `{manifest['chat_template']}` (`chat_template.jinja`)",
              f"- 길이 ({lr['method']}): p50 {lr['p50']} · p95 {lr['p95']} · p99 {lr['p99']} · max {lr['max']} → cutoff_len {lr['cutoff_len']}",
              f"- 레시피: `recipe/{manifest['recipe']['kind']}/`"]
    lines += ["", "## 포맷", "",
              "한 줄에 한 대화. `messages[0]`은 항상 아래 시스템 프롬프트이고, 그 뒤로 `user`/`assistant`가 번갈아 온다.",
              "나머지 필드는 출처 추적용이며 학습에는 쓰지 않아도 된다.", "", "```", system_prompt, "```", "",
              "## 재현", "",
              f"- 페르소나 문서: `{manifest['persona_doc']}` (sha256 `{manifest['persona_sha256'][:16]}`)",
              f"- 설정: `{manifest['config_path']}`, seed {manifest['seed']}",
              f"- 생성: `persona-sft-data run --config {manifest['config_path']}`", ""]
    return "\n".join(lines)


def _describe(path: Path, root: Path) -> str:
    """manifest·로그에 적는 경로. 저장소 루트 기준 상대 경로(posix), 밖이면 절대 경로."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()
