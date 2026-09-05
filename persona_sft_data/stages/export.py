"""Turn the assembled corpus into a fine-tuning dataset.

The pipeline's own record is a *session*: ``turns`` of ``user``/``pet`` plus
provenance -- which source, which teacher, which licence, which scenario. That
is the right shape for gating and mixing and the wrong shape for a trainer.
Axolotl, LLaMA-Factory and TRL all read the OpenAI ``messages`` layout, with a
``system`` turn first and ``user``/``assistant`` alternating after it.

This stage writes that layout, one file per split, next to a manifest and a
dataset card. It changes nothing about the text. The system prompt is rendered
from the persona document by :meth:`Persona.system_prompt`, so the prompt a
model is trained against is the same definition the corpus was generated and
gated against.

Unlike the generators this is not a :class:`runner.Stage`: it neither gates nor
dedupes, it reads three files and writes five, and the runner's single-output
contract would fit badly. It is a function the CLI calls after assemble.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from persona_sft_data import schema
from persona_sft_data.config import PipelineConfig
from persona_sft_data.persona import Persona, load_cached

SPLITS = ("train", "val", "test")

# Session roles -> chat roles. ``pet`` is what the persona is to the pipeline;
# ``assistant`` is what it is to a chat template.
ROLES = {"user": "user", "pet": "assistant"}

# Session fields that do not survive the conversion: ``turns`` becomes
# ``messages`` and ``split`` becomes the file the record is in. Everything else
# -- id, source, scenario, licence, generator, real_source, source_url -- is
# carried through so a record can still say where it came from.
DROPPED = frozenset({"turns", "split"})


def to_messages(record: Mapping[str, Any], system_prompt: str) -> dict[str, Any]:
    """One session -> one chat record. Raises on a session the schema rejects,
    which cannot happen for assemble's output and should be loud if it does."""
    session = schema.normalize_session(record)
    out = {k: v for k, v in session.items() if k not in DROPPED}
    out["messages"] = [{"role": "system", "content": system_prompt}] + [
        {"role": ROLES[turn["role"]], "content": turn["text"]}
        for turn in session["turns"]
    ]
    return out


def run_export(config: PipelineConfig, name: str | None = None, *, log=print) -> Path:
    """Write ``datasets/<name>/`` from ``final/{train,val,test}.jsonl``.

    Returns the dataset directory. ``name`` defaults to the config's
    ``stages.export.name``; passing one lets a re-export land beside the last
    instead of over it.
    """
    settings = config.stage("export")
    name = name or settings.get("name")
    if not name:
        raise ValueError("stages.export needs a 'name' (or pass --name)")

    persona = load_cached(config.persona_doc)
    system_prompt = persona.system_prompt()
    out_dir = config.datasets_root / name
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    files: dict[str, dict[str, Any]] = {}
    sources: Counter[str] = Counter()
    # A source can carry more than one licence: `real` mixes apache-2.0, mit
    # and smilestyle. Keep the set, not the first one seen.
    licenses: dict[str, set[str]] = {}
    generators: Counter[str] = Counter()
    real_sources: dict[str, str] = {}
    turns_total = 0

    for split in SPLITS:
        src = config.final(split)
        if not src.exists():
            raise FileNotFoundError(f"{src} is missing: run assemble first")
        dst = out_dir / f"{split}.jsonl"
        n = 0
        with dst.open("w", encoding="utf-8", newline="\n") as handle:
            for record in schema.read_jsonl(src):
                chat = to_messages(record, system_prompt)
                handle.write(json.dumps(chat, ensure_ascii=False) + "\n")
                n += 1
                sources[chat["source"]] += 1
                licenses.setdefault(chat["source"], set()).add(str(chat.get("license", "")))
                for g in chat.get("generator", ()):
                    generators[g] += 1
                if chat.get("real_source"):
                    real_sources.setdefault(chat["real_source"], str(chat.get("source_url", "")))
                    licenses.setdefault(chat["real_source"], set()).add(str(chat.get("license", "")))
                turns_total += len(chat["messages"]) - 1
        files[split] = {"path": dst.name, "records": n, "sha256": _sha256(dst)}
        log(f"[export] {split}: {n:,} records -> {dst.relative_to(config.root)}")

    (out_dir / "system_prompt.txt").write_text(system_prompt + "\n", encoding="utf-8", newline="\n")

    manifest = {
        "name": name,
        "format": "openai-messages",
        "generated_by": "persona_sft_data",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "persona": persona.name,
        "persona_doc": _describe(config.persona_doc, config.root),
        "persona_sha256": _sha256(config.persona_doc),
        "config_path": _describe(config.path, config.root),
        "seed": config.seed,
        "records": sum(f["records"] for f in files.values()),
        "turns": turns_total,
        "files": files,
        "sources": dict(sources.most_common()),
        "licenses": {k: sorted(v) for k, v in licenses.items()},
        "generators": dict(generators.most_common()),
        "real_sources": real_sources,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode("utf-8")).hexdigest(),
    }
    corpus_manifest = config.data_root / "final" / "manifest.json"
    if corpus_manifest.exists():
        manifest["corpus_manifest_sha256"] = _sha256(corpus_manifest)

    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    (out_dir / "README.md").write_text(dataset_card(manifest, system_prompt), encoding="utf-8", newline="\n")
    log(f"[export] {manifest['records']:,} records, {turns_total:,} turns in "
        f"{time.time() - t0:.1f}s -> {out_dir.relative_to(config.root)}")
    return out_dir


def dataset_card(manifest: Mapping[str, Any], system_prompt: str) -> str:
    """A Hugging Face style card: YAML front matter, then what a reader needs
    to decide whether to train on this."""
    records = manifest["records"]
    size_bucket = next(
        label for bound, label in (
            (1_000, "n<1K"), (10_000, "1K<n<10K"), (100_000, "10K<n<100K"),
            (1_000_000, "100K<n<1M"), (float("inf"), "1M<n<10M"),
        ) if records < bound
    )
    licenses = sorted({l for ls in manifest["licenses"].values() for l in ls})

    def licence_cell(key: str) -> str:
        return ", ".join(manifest["licenses"].get(key, []))

    lines = [
        "---",
        "language:",
        "- ko",
        f"license: {'other' if len(licenses) != 1 else licenses[0]}",
        "task_categories:",
        "- text-generation",
        "tags:",
        "- persona",
        "- roleplay",
        "- sft",
        "- korean",
        f"size_categories:",
        f"- {size_bucket}",
        "---",
        "",
        f"# {manifest['name']}",
        "",
        f"`{manifest['persona']}` 페르소나 미세조정 데이터셋. OpenAI `messages` 포맷, "
        f"{records:,}개 대화 / {manifest['turns']:,}턴.",
        "",
        "## 분할",
        "",
        "| split | records | sha256 |",
        "| --- | --- | --- |",
    ]
    for split, info in manifest["files"].items():
        lines.append(f"| {split} | {info['records']:,} | `{info['sha256'][:16]}` |")

    lines += ["", "## 출처와 라이선스", "", "| source | records | license |", "| --- | --- | --- |"]
    for source, n in manifest["sources"].items():
        lines.append(f"| {source} | {n:,} | {licence_cell(source)} |")
    if manifest["real_sources"]:
        lines += ["", "`real` 레코드의 원본:", ""]
        for name, url in manifest["real_sources"].items():
            lines.append(f"- `{name}` ({licence_cell(name)}) — {url}")

    lines += ["", "## 생성 모델", ""]
    for model, n in manifest["generators"].items():
        lines.append(f"- `{model}`: {n:,} records")

    lines += [
        "",
        "## 포맷",
        "",
        "한 줄에 한 대화. `messages[0]`은 항상 아래 시스템 프롬프트이고, 그 뒤로",
        "`user`/`assistant`가 번갈아 온다. 나머지 필드(`id`, `source`, `scenario`,",
        "`license`, `generator`, …)는 출처 추적용이며 학습에는 쓰지 않아도 된다.",
        "",
        "시스템 프롬프트는 모든 레코드에서 같고 `system_prompt.txt`에도 있다. 파일",
        "크기의 대부분이 그 반복이다 — 트레이너가 system 필드를 따로 받는다면",
        "`messages[1:]`만 넘기고 프롬프트는 한 번만 주는 편이 가볍다.",
        "",
        "```",
        system_prompt,
        "```",
        "",
        "## 재현",
        "",
        f"- 페르소나 문서: `{manifest['persona_doc']}` (sha256 `{manifest['persona_sha256'][:16]}`)",
        f"- 설정: `{manifest['config_path']}`, seed {manifest['seed']}",
        f"- 생성: `python -m persona_sft_data run --config {manifest['config_path']}`",
        "",
    ]
    return "\n".join(lines)


def _describe(path: Path, root: Path) -> str:
    """A path for the manifest: relative to the project when it is inside it,
    absolute when it is not. A persona document shared between projects is a
    legitimate thing to point a config at, and must not crash the export."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
