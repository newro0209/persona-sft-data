"""명령줄 도구.

    persona-sft-data check   --config configs/<이름>.json
    persona-sft-data run     --config configs/<이름>.json [--stage <단계>]
    persona-sft-data export  --config configs/<이름>.json [--name <데이터셋 이름>]
    persona-sft-data sources --config configs/<이름>.json [--sample N] [--translate]
    persona-sft-data status  --config configs/<이름>.json [--watch]
    persona-sft-data plugins [--config configs/<이름>.json]
    persona-sft-data init <이름> [--profile <프로필>]

명령 하나가 클래스 하나다(Command). ``main``은 파서 구성과 디스패치만 한다. 어떤
교사·비율·한도·경로를 쓰는지는 전부 설정 파일에 있고 명령줄 기본값이 조용히 실제
설정이 되는 일은 없다.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import islice
from pathlib import Path
from typing import Any

from persona_sft_data.core import runner
from persona_sft_data.core.config import ConfigError, PipelineConfig
from persona_sft_data.core.gates import build_gate
from persona_sft_data.core.persona import PersonaError, load_cached
from persona_sft_data.core.registry import GROUPS, PROFILES, STAGES, TEACHERS, TRANSLATORS, PluginError
from persona_sft_data.sources.base import fetch_source, read_utterances
from persona_sft_data.teacher.base import TeacherError


def load_config(path: Path) -> PipelineConfig:
    """설정을 읽는다. 설정·페르소나·플러그인 오류는 한 줄로 알리고 종료 코드 2."""
    try:
        return PipelineConfig.load(path)
    except (ConfigError, PersonaError, PluginError) as exc:
        print(f"설정 오류: {exc}", file=sys.stderr)
        raise SystemExit(2)


def ordered_stages(config: PipelineConfig) -> list[Any]:
    """설정에 있는 단계를 requires()로 위상 정렬. 같은 층에서는 설정에 적힌 순서."""
    names = list(config.stages)
    stages = {n: STAGES.get(n) for n in names}
    deps = {n: [d for d in stages[n].requires(config) if d in names] for n in names}
    done: list[str] = []
    while len(done) < len(names):
        ready = [n for n in names if n not in done and all(d in done for d in deps[n])]
        if not ready:
            raise ConfigError(f"단계 의존성에 순환이 있다: {deps}")
        done.extend(ready)
    return [stages[n] for n in done]


class Command:
    """서브커맨드 하나. ``configure``가 인자를 붙이고 ``run``이 종료 코드를 돌려준다."""

    name = ""
    help = ""

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--config", required=True, type=Path, help="설정 파일 경로")

    def run(self, args: argparse.Namespace) -> int:
        raise NotImplementedError


class Check(Command):
    name = "check"
    help = "설정·페르소나·프로필·교사·소스를 점검한다"

    def run(self, args: argparse.Namespace) -> int:
        config = load_config(args.config)
        try:
            config.validate_pipeline()
        except ConfigError as exc:
            print(f"설정 오류: {exc}", file=sys.stderr)
            return 2
        profile = PROFILES.get(config.profile)
        try:
            persona = load_cached(config.persona_doc, profile.required_sections)
            gate = build_gate(persona, runner.gate_settings_for(config))
        except PersonaError as exc:
            print(f"페르소나 오류: {exc}", file=sys.stderr)
            return 2
        print(f"persona   : {persona.name} ({persona.source.name}) — beats {len(persona.beats)}, "
              f"principles {len(persona.principles)}, constraints {len(persona.constraints)}")
        print(f"profile   : {profile.name} ({profile.assistant_label}/{profile.user_label})")
        print(f"gate      : {', '.join(r.name for r in gate.rules)}")
        print(f"student   : {config.student.model} (template {config.student.chat_template})")
        print(f"data_root : {config.data_root}")
        ok = True
        for stage in ordered_stages(config):
            ctx = runner.build_context(stage, config)
            try:
                stage.preflight(ctx)
                print(f"stage     : {stage.name} OK")
            except (TeacherError, ConfigError, PersonaError, FileNotFoundError) as exc:
                print(f"stage     : {stage.name} FAILED\n  {exc}")
                ok = False
        return 0 if ok else 1


class Run(Command):
    name = "run"
    help = "설정된 단계를 순서대로 실행한다"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        super().configure(parser)
        parser.add_argument("--stage", help="이 단계만 실행 (기본: 전부)")

    def run(self, args: argparse.Namespace) -> int:
        config = load_config(args.config)
        try:
            config.validate_pipeline()
        except ConfigError as exc:
            print(f"설정 오류: {exc}", file=sys.stderr)
            return 2
        config.ensure_dirs()
        stages = ordered_stages(config)
        if args.stage:
            stages = [s for s in stages if s.name == args.stage]
            if not stages:
                print(f"설정에 stage {args.stage!r}이(가) 없다", file=sys.stderr)
                return 2
        t0 = time.time()
        for stage in stages:
            try:
                for instance in stage.instances(config):
                    runner.execute(instance, config)
            except (FileNotFoundError, TeacherError, ConfigError, PersonaError) as exc:
                print(f"[{stage.name}] 실패: {exc}", file=sys.stderr)
                return 1
        print(f"\n완료: {time.time() - t0:.0f}s")
        return 0


class Export(Command):
    name = "export"
    help = "조립된 코퍼스를 데이터셋과 레시피로 내보낸다"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        super().configure(parser)
        parser.add_argument("--name", help="데이터셋 이름 (기본: 설정의 stages.export.name)")

    def run(self, args: argparse.Namespace) -> int:
        config = load_config(args.config)
        stage = type(STAGES.get("export"))(name_override=args.name)
        try:
            runner.execute(stage, config)
        except (FileNotFoundError, ConfigError, PersonaError) as exc:
            print(f"[export] 실패: {exc}", file=sys.stderr)
            return 1
        return 0


class Sources(Command):
    name = "sources"
    help = "소스별 발화 표본을 보여 준다 (번역 전후)"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        super().configure(parser)
        parser.add_argument("--sample", type=int, default=5, help="소스당 표본 수")
        parser.add_argument("--translate", action="store_true", help="다른 언어 소스는 교사로 번역해 같이 보여 준다")

    def run(self, args: argparse.Namespace) -> int:
        config = load_config(args.config)
        translator = None
        if args.translate and config.has_stage("ingest"):
            stage = STAGES.get("ingest")
            ctx = runner.build_context(stage, config)
            tcfg = config.teacher_for("ingest")
            teacher = TEACHERS.get(tcfg.kind).build(tcfg)
            try:
                # 교사에 닿는지 먼저 본다. 이걸 건너뛰면 서버가 죽었을 때 재시도 대기
                # 뒤 원문만 찍고 조용히 성공으로 끝나 번역 전후를 볼 수 없다.
                teacher.check()
            except TeacherError as exc:
                print(f"교사 오류: {exc}", file=sys.stderr)
                return 1
            translator = TRANSLATORS.get(ctx.settings.translator).build(ctx, teacher)
        cache = config.data_root / "cache"
        for name, source in config.sources.items():
            data = fetch_source(source, cache, timeout=60.0, log=print)
            if data is None:
                continue
            sample = list(islice(read_utterances(source, data), args.sample))
            print(f"\n[{name}] format={source.format} language={source.language} license={source.license}")
            translating = translator is not None and source.language != config.language
            translated = translator.translate(sample, source.language) if translating else [None] * len(sample)
            for text, tr in zip(sample, translated):
                if not translating:
                    print(f"  {text}")
                else:
                    print(f"  {text}  →  {tr or '(번역 실패)'}")
        return 0


class Plugins(Command):
    name = "plugins"
    help = "등록된 플러그인을 그룹별로 보여 준다"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        # --config는 선택이다. 주면 설정의 로컬 플러그인까지 붙여 'plugins' 출처를 보여 준다.
        parser.add_argument("--config", type=Path, help="이 설정의 plugins 목록을 먼저 붙인다")

    def run(self, args: argparse.Namespace) -> int:
        if args.config is not None:
            load_config(args.config)
        for group, registry in GROUPS.items():
            print(f"\n{group}")
            for reg in registry.describe():
                print(f"  {reg.name:<20} {reg.origin:<12} {reg.path}")
        return 0


class Init(Command):
    name = "init"
    help = "새 페르소나 문서와 설정 파일 골격을 만든다"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument("name", help="페르소나 이름 (파일 이름으로도 쓴다)")
        parser.add_argument("--profile", default="companion", help="프로필 (기본: companion)")

    def run(self, args: argparse.Namespace) -> int:
        try:
            profile = PROFILES.get(args.profile)
        except PluginError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        doc = Path("personas") / f"{args.name}.md"
        cfg = Path("configs") / f"{args.name}.json"
        if doc.exists() or cfg.exists():
            print(f"이미 있다: {doc if doc.exists() else cfg}", file=sys.stderr)
            return 2
        doc.parent.mkdir(parents=True, exist_ok=True)
        cfg.parent.mkdir(parents=True, exist_ok=True)
        doc.write_text(profile.document_template(args.name), encoding="utf-8", newline="\n")
        config = {
            "profile": profile.name, "language": "ko", "data_root": (Path("data") / args.name).as_posix(),
            "datasets_root": "datasets",
            "seed": int(time.strftime("%Y%m%d")), "persona_doc": doc.as_posix(), "plugins": [],
            "student": {"model": "<학생 모델 id>", "trust_remote_code": True, "chat_template": "chatml"},
            "teachers": {"reasoner": {"kind": "openai", "model": "<교사 모델 id>", "base_url": "<교사 서버 base_url>"}},
            "sources": {},
            "stages": {
                "dialogue": {"teacher": "reasoner", "per_situation": 40},
                "filter": {},
                "assemble": {"ratios": {"dialogue": 1.0}, "max_sessions": 4000, "split": {"train": 0.9, "val": 0.05, "test": 0.05}},
                "export": {"name": f"{args.name}-peft-v1", "recipe": {"kind": "llamafactory"}},
            },
        }
        cfg.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(f"만들었다: {doc}, {cfg}\n다음: 문서의 핵심 정의·배경·다룰 상황을 채우고, 설정의 학생·교사 모델 id를 적은 뒤 `check`.")
        return 0


class Status(Command):
    name = "status"
    help = "단계별 산출 개수와 수율을 한 화면으로"

    def configure(self, parser: argparse.ArgumentParser) -> None:
        super().configure(parser)
        parser.add_argument("--watch", action="store_true", help="2초마다 갱신")

    @staticmethod
    def _count(path: Path) -> int:
        if not path.exists():
            return 0
        with path.open("rb") as f:
            return sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(1 << 20), b""))

    def _render(self, config: PipelineConfig) -> str:
        lines = [f"  {config.data_root}  ({time.strftime('%H:%M:%S')})", ""]
        for stage in ordered_stages(config):
            if stage.mode != "records":
                continue
            for inst in (stage.instances(config) if stage.name != "filter" else [stage]):
                if stage.name == "filter":
                    for n in config.session_stages():
                        out = config.filtered(n)
                        kept, rej = self._count(out), self._count(config.rejected_path(out))
                        lines.append(f"  filter/{n:<10} {kept:>8,} kept {rej:>8,} rejected")
                    continue
                out = getattr(config, inst.produces)(inst.name)
                kept, rej = self._count(out), self._count(config.rejected_path(out))
                rate = f"{kept / (kept + rej):.1%}" if kept + rej else "-"
                lines.append(f"  {inst.name:<17} {kept:>8,} kept {rej:>8,} rejected  {rate}")
        return "\n".join(lines)

    def run(self, args: argparse.Namespace) -> int:
        config = load_config(args.config)
        if not args.watch:
            print(self._render(config))
            return 0
        try:
            while True:
                sys.stdout.write("\x1b[H\x1b[J" + self._render(config) + "\n")
                sys.stdout.flush()
                time.sleep(2)
        except KeyboardInterrupt:
            return 0


COMMANDS: tuple[Command, ...] = (Check(), Run(), Export(), Sources(), Status(), Plugins(), Init())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="persona-sft-data", description="페르소나 PEFT 데이터셋·레시피 도구")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        command.configure(sub.add_parser(command.name, help=command.help))
    args = parser.parse_args(argv)
    command = next(c for c in COMMANDS if c.name == args.command)
    try:
        return command.run(args)
    except SystemExit as exc:
        return int(exc.code or 0)


if __name__ == "__main__":
    raise SystemExit(main())
