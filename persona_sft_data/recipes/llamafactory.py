"""LLaMA-Factory 레시피: dataset_info.json + LoRA SFT YAML + 실행 안내.

YAML은 표준 라이브러리로 직접 쓴다 — 값이 전부 스칼라라 라이브러리가 필요 없다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from persona_sft_data.core.registry import RECIPES
from persona_sft_data.recipes.base import ExportInfo

# 이 프로젝트의 템플릿 이름 → LLaMA-Factory의 template 값
LLAMAFACTORY_TEMPLATES = {"chatml": "chatml"}


@dataclass(frozen=True)
class LlamaFactorySettings:
    """설정 ``stages.export.recipe``의 ``kind`` 이외 키. ``cutoff_len="auto"``면 길이 보고에서 정한다."""

    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    epochs: float = 3.0
    cutoff_len: int | str = "auto"
    batch_size: int = 8
    gradient_accumulation: int = 2
    warmup_ratio: float = 0.05


@RECIPES.register("llamafactory", origin="builtin")
class LlamaFactoryRecipe:
    name = "llamafactory"
    settings_type = LlamaFactorySettings

    def write(self, out_dir: Path, info: ExportInfo, settings: LlamaFactorySettings) -> list[Path]:
        recipe_dir = out_dir / "recipe" / "llamafactory"
        recipe_dir.mkdir(parents=True, exist_ok=True)
        tags = {"role_tag": "role", "content_tag": "content", "user_tag": "user",
                "assistant_tag": "assistant", "system_tag": "system"}
        dataset_info = {
            info.name: {"file_name": "../../train.jsonl", "formatting": "sharegpt",
                        "columns": {"messages": "messages"}, "tags": tags},
            f"{info.name}_val": {"file_name": "../../val.jsonl", "formatting": "sharegpt",
                                 "columns": {"messages": "messages"}, "tags": tags},
        }
        info_path = recipe_dir / "dataset_info.json"
        info_path.write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

        cutoff = info.length_report.cutoff_len if settings.cutoff_len == "auto" else int(settings.cutoff_len)
        template = LLAMAFACTORY_TEMPLATES[info.chat_template_name]
        yaml = f"""### model
model_name_or_path: {info.student.model}
trust_remote_code: {str(info.student.trust_remote_code).lower()}

### method
stage: sft
do_train: true
finetuning_type: lora
lora_rank: {settings.lora_rank}
lora_alpha: {settings.lora_alpha}
lora_dropout: {settings.lora_dropout}
lora_target: all

### dataset
dataset: {info.name}
eval_dataset: {info.name}_val
dataset_dir: {recipe_dir.resolve().as_posix()}
template: {template}
cutoff_len: {cutoff}
train_on_prompt: false
preprocessing_num_workers: 4

### output
output_dir: saves/{info.name}/lora
logging_steps: 10
save_steps: 200
plot_loss: true
overwrite_output_dir: true
report_to: none

### train
per_device_train_batch_size: {settings.batch_size}
gradient_accumulation_steps: {settings.gradient_accumulation}
learning_rate: {settings.learning_rate}
num_train_epochs: {settings.epochs}
lr_scheduler_type: cosine
warmup_ratio: {settings.warmup_ratio}
bf16: true

### eval
per_device_eval_batch_size: {settings.batch_size}
eval_strategy: steps
eval_steps: 100
"""
        yaml_path = recipe_dir / "lora_sft.yaml"
        yaml_path.write_text(yaml, encoding="utf-8", newline="\n")

        readme = f"""# {info.name} — LLaMA-Factory LoRA 레시피

```bash
llamafactory-cli train {yaml_path.resolve().as_posix()}
```

- 학생 모델 `{info.student.model}`은 커스텀 아키텍처라 `trust_remote_code: true`가 필요하다.
- 채팅 템플릿은 `{template}`이다. 데이터셋의 `chat_template.jinja`와 같은 형식이며, 추론 때도 같은 템플릿을 써야 한다.
- `cutoff_len {cutoff}`은 길이 보고({info.length_report.method}, p99 {info.length_report.p99})에서 정했다.
- 손실은 assistant 발화에만 건다(`train_on_prompt: false`).
- 검증은 `{info.name}_val`(val.jsonl)로 100 스텝마다.
"""
        readme_path = recipe_dir / "README.md"
        readme_path.write_text(readme, encoding="utf-8", newline="\n")
        return [info_path, yaml_path, readme_path]
