"""Run the corpus pipeline.

    python -m persona_sft_data run --config configs/mongle.json
    python -m persona_sft_data run --config configs/mongle.json --stage seed
    python -m persona_sft_data check --config configs/mongle.json
    python -m persona_sft_data export --config configs/mongle.json

There is one entry point and one config. Which teacher, which ratios, how many
records and where everything lands are all in the config file; nothing is a
command-line default that quietly becomes the real setting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

from persona_sft_data import backend, runner
from persona_sft_data.config import ConfigError, PipelineConfig
from persona_sft_data.persona import PersonaError, load as load_persona

# Order matters: expand reads seed's output, filter reads raw, assemble reads
# filtered. Named here rather than inferred so a run is predictable.
GENERATORS = ("seed", "expand", "real", "template")


def _import_stage(name: str):
    """Stages are imported lazily so a missing one does not break the others."""
    from importlib import import_module

    module = import_module(f"persona_sft_data.stages.{name}")
    for attr in dir(module):
        obj = getattr(module, attr)
        if (
            isinstance(obj, type)
            and getattr(obj, "name", None) == name
            and hasattr(obj, "run")
        ):
            return obj()
    raise SystemExit(f"persona_sft_data/stages/{name}.py defines no stage named {name!r}")


def cmd_check(config: PipelineConfig) -> int:
    """Preflight: config parses, persona parses, teachers answer."""
    persona = load_persona(config.persona_doc)
    print(f"persona   : {persona.source.name} — "
          f"{len(persona.beats)} beats, {len(persona.prohibitions)} prohibitions, "
          f"{len(persona.principles)} principles")
    lo, hi = persona.utterance_char_range()
    print(f"length    : {lo}~{hi} characters (from the document)")
    print(f"data_root : {config.data_root}")

    ok = True
    for name, teacher_cfg in config.teachers.items():
        client = backend.build(teacher_cfg)
        try:
            client.check()
            print(f"teacher   : {name} -> {teacher_cfg.model} at {teacher_cfg.base_url} OK")
        except backend.TeacherError as exc:
            print(f"teacher   : {name} FAILED\n  {exc}")
            ok = False
    return 0 if ok else 1


def cmd_run(config: PipelineConfig, only: str | None) -> int:
    config.ensure_dirs()
    t0 = time.time()

    # filter and assemble are handled below: filter runs once per raw source and
    # has no single stage class to import, and assemble needs filter's output.
    wanted = [only] if only in GENERATORS else ([] if only else list(GENERATORS))
    for name in wanted:
        if name not in config.stages:
            raise SystemExit(f"config has no stage {name!r}")
        runner.execute(_import_stage(name), config)

    if only in GENERATORS:
        return 0

    # filter runs once per raw source
    from persona_sft_data.stages import filter as filter_stage

    if only in (None, "filter"):
        for stage in filter_stage.stages_for(config):
            runner.execute(stage, config)
    if only == "filter":
        return 0

    if only in (None, "assemble"):
        stats = runner.execute(_import_stage("assemble"), config)
        _finalize_manifest(config)
        print(f"\ncorpus: {stats.produced:,} sessions in {time.time() - t0:.0f}s total")

    # export is the one stage a config may leave out: a corpus is useful on
    # its own, and the dataset is a view of it.
    if only in (None, "export") and "export" in config.stages:
        cmd_export(config, None)
    return 0


def cmd_export(config: PipelineConfig, name: str | None) -> int:
    """Write the fine-tuning dataset from an already assembled corpus."""
    from persona_sft_data.stages.export import run_export

    run_export(config, name)
    return 0


def _finalize_manifest(config: PipelineConfig) -> None:
    """Record corpus.jsonl's hash, which only exists once the runner has
    finished writing it."""
    manifest_path = config.data_root / "final" / "manifest.json"
    corpus = config.final("assemble")
    if not (manifest_path.exists() and corpus.exists()):
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    h = hashlib.sha256()
    with corpus.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    manifest.setdefault("files", {}).setdefault("corpus", {})["sha256"] = h.hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="persona_sft_data", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    for name, help_text in (
        ("run", "생성 실행"),
        ("check", "설정·페르소나·교사 점검"),
        ("export", "조립된 코퍼스를 미세조정 데이터셋(messages JSONL)으로 내보내기"),
    ):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--config", required=True, type=Path)
        if name == "run":
            p.add_argument(
                "--stage",
                choices=[*GENERATORS, "filter", "assemble", "export"],
                help="이 단계만 실행 (기본: 전부)",
            )
        if name == "export":
            p.add_argument("--name", help="데이터셋 이름 (기본: 설정의 stages.export.name)")

    args = parser.parse_args(argv)
    try:
        config = PipelineConfig.load(args.config)
    except (ConfigError, PersonaError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2

    if args.command == "check":
        return cmd_check(config)
    if args.command == "export":
        return cmd_export(config, args.name)
    return cmd_run(config, getattr(args, "stage", None))


if __name__ == "__main__":
    raise SystemExit(main())
