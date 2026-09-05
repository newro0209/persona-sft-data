"""Watch a running pipeline.

    python -m persona_sft_data.progress --config configs/mongle.json

Reads the output files a stage is writing and refreshes in place. Nothing here
touches the pipeline, so it is safe to start, stop and restart while a
generation is running — and safe to run from a second terminal.

The targets come from the same config the run uses, so the percentages mean
what the run means by them rather than being typed in twice.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from persona_sft_data.config import PipelineConfig
from persona_sft_data.persona import load_cached

BEATS_STAGE = "seed"


@dataclass
class StageView:
    name: str
    kept: int
    rejected: int
    target: int
    started: float | None

    @property
    def done(self) -> int:
        return self.kept + self.rejected

    @property
    def yield_rate(self) -> float | None:
        return self.kept / self.done if self.done else None

    @property
    def fraction(self) -> float:
        return min(1.0, self.done / self.target) if self.target else 0.0


def _count(path: Path) -> int:
    if not path.exists():
        return 0
    # Counting bytes for newlines is much faster than decoding 170k JSON lines,
    # and this runs every second next to a saturated GPU.
    total = 0
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            total += chunk.count(b"\n")
    return total


def targets(config: PipelineConfig) -> dict[str, int]:
    """How many records each stage intends to produce, from the config."""
    persona = load_cached(config.persona_doc)
    beats = len(persona.beats)
    seed_n = beats * int(config.stages["seed"].get("per_situation", 0))
    return {
        "seed": seed_n,
        "expand": seed_n * int(config.stages["expand"].get("variants_per_seed", 0)),
        "real": int(config.stages["real"].get("limit", 0)),
        "template": int(config.stages["template"].get("limit", 0)),
    }


def read_stages(config: PipelineConfig) -> list[StageView]:
    out = []
    for name, target in targets(config).items():
        path = config.raw(name)
        out.append(
            StageView(
                name=name,
                kept=_count(path),
                rejected=_count(config.rejected_path(path)),
                target=target,
                started=os.path.getctime(path) if path.exists() else None,
            )
        )
    return out


def gpu_line() -> str:
    """Best-effort; the pipeline runs on Windows and the GPU is seen from WSL."""
    for cmd in (
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader"],
        ["wsl.exe", "bash", "-c",
         "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu "
         "--format=csv,noheader"],
    ):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=6)
        except (OSError, subprocess.SubprocessError):
            continue
        line = r.stdout.replace("\r", "").replace("\x00", "").strip().splitlines()
        if line and "," in line[0]:
            return line[0].strip()
    return "GPU 정보 없음"


def bar(fraction: float, width: int) -> str:
    filled = int(fraction * width)
    return "█" * filled + "·" * (width - filled)


def render(config: PipelineConfig, stages: list[StageView]) -> str:
    cols = min(shutil.get_terminal_size((100, 24)).columns, 100)
    width = max(20, cols - 52)
    now = time.strftime("%H:%M:%S")
    lines = [f"  코퍼스 생성 — {config.data_root.name}/  ({now})", ""]

    active = None
    for s in stages:
        if 0 < s.done < s.target:
            active = s

    for s in stages:
        if s.done == 0:
            state, detail = "대기", ""
        elif s.done >= s.target:
            state, detail = "완료", f"수율 {s.yield_rate:.1%}"
        else:
            state, detail = "진행", f"수율 {s.yield_rate:.1%}"
        mark = "▶" if s is active else " "
        lines.append(
            f" {mark} {s.name:<9} {bar(s.fraction, width)} "
            f"{s.done:>7,}/{s.target:<7,} {s.fraction:5.1%}  {state} {detail}"
        )

    lines.append("")
    if active and active.started:
        elapsed = time.time() - active.started
        rate = active.done / elapsed if elapsed > 0 else 0
        remain = (active.target - active.done) / rate if rate > 0 else 0
        lines.append(
            f"  {active.name}: {rate:5.1f} 건/초 · 경과 {elapsed/60:5.1f}분 · "
            f"남은 약 {remain/60:.0f}분"
        )
        total_target = sum(s.target for s in stages)
        total_done = sum(s.done for s in stages)
        lines.append(
            f"  전체: {total_done:,}/{total_target:,} ({total_done/total_target:.1%})"
        )
    elif all(s.done >= s.target for s in stages if s.target):
        lines.append("  생성 단계 완료 — filter·assemble이 남았다")
    else:
        lines.append("  시작 대기 중")

    lines.append(f"  {gpu_line()}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="persona_sft_data.progress", description=__doc__)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--interval", type=float, default=2.0, help="갱신 주기(초)")
    ap.add_argument("--once", action="store_true", help="한 번만 출력하고 종료")
    args = ap.parse_args(argv)

    config = PipelineConfig.load(args.config)
    if args.once:
        print(render(config, read_stages(config)))
        return 0

    print("\x1b[?25l", end="")  # hide cursor
    try:
        while True:
            frame = render(config, read_stages(config))
            # Redraw in place rather than scrolling, so the terminal stays
            # readable next to a long-running job.
            sys.stdout.write("\x1b[H\x1b[J" + frame + "\n")
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        return 0
    finally:
        print("\x1b[?25h", end="")  # show cursor


if __name__ == "__main__":
    raise SystemExit(main())
