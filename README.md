# persona-sft-data

페르소나 문서 하나, 교사 모델, 그리고 임의 포맷·임의 언어의 외부 텍스트 데이터셋에서
**사전학습 LLM을 PEFT(LoRA)로 미세조정할 데이터셋과 학습 레시피**를 만든다.

- 페르소나가 무엇을 말하고 무엇을 말하지 않는지는 `personas/<이름>.md` 한 곳에만 있다.
  코드에는 페르소나 문자열이 없고, 테스트가 그것을 강제한다.
- 반려 펫만이 아니다. `companion` `npc` `novel` `trpg` `lore` 프로필이 각각 다른
  용도(AI 반려로봇, 게임 NPC, 소설 인물, TRPG 진행자, 세계관 안내자)의 기본값을 준다.
- 단계·포맷·추출기·교사·번역기·레시피·프로필·규칙 여덟 확장점이 전부 플러그인이다.
  내장도 같은 경로로 등록되고, `plugins` 명령이 한 표로 보여 준다.

원래는 ESP32용 소형 모델을 처음부터 학습시킬 코퍼스를 만드는 파이프라인이었고, 그
결과(38만 세션)는 [my-llm](../my-llm)이 소비했다. 2026-09-05에 목적을 PEFT로 바꿨다 —
배경은 `docs/superpowers/specs/2026-09-05-peft-persona-toolkit-design.md`.

## 빠르게

```powershell
uv venv .venv --python 3.12
uv pip install --python .venv\Scripts\python.exe -e ".[dev,parquet]"
.venv\Scripts\python.exe -m pytest                                   # GPU·네트워크 없이 전부

persona-sft-data check   --config configs/mongle.json   # 설정·페르소나·교사·소스 점검
persona-sft-data run     --config configs/mongle.json   # 전부 (교사 서버 필요)
persona-sft-data export  --config configs/mongle.json   # 조립된 코퍼스 → 데이터셋 + 레시피
```

교사 없이 전체 흐름을 보려면 `configs/smoke.json`(`kind: fake` 교사, 로컬 픽스처 소스).
`run --stage <단계>`로 한 단계만 돌린다.

## 파이프라인

```
sources ─ ingest(포맷·추출 → 표집 → 번역 → 주제·안전 필터) ─ respond(발화에 교사가 답함) ─┐
personas/<이름>.md ─ dialogue(추론 교사가 상황·흐름별 대화 작성) ──────────────────────┼─ filter ─ assemble ─ export
```

- **레코드**는 둘이다. 세션 `{id, source, scenario, license, generator, turns:[{role: user|assistant, text}]}`,
  발화 `{id, text, source, language, license, url, original_text?, original_language?, translator?}`.
- **모든 경로는 `data_root` 하나에서 파생된다.** `raw/` 생성물, `filtered/` 게이트 통과분,
  `final/` 혼합·분할 코퍼스. 단계마다 `.stats.json`, `.rejected.jsonl`, `.sample.jsonl`이 같이 나온다.
- **게이트는 문서의 `## 제약` 표에서만 켜진다.** 행이 없는 규칙은 꺼진 것이다. 그래서
  존댓말 NPC와 반말 펫이 같은 코드로 다른 검열을 받는다.
- **교사·학생·소스는 설정에만 있다.** 모델 id, 데이터셋 URL, `data/` 경로가 코드에
  나타나면 테스트가 실패한다.

## 페르소나 문서

| 절 | 필수 | 쓰임 |
| --- | --- | --- |
| `## 핵심 정의` 표 (이름·정체성·사용자와의 관계·말투·성격·응답 길이·지식 범위) | 필수 | 시스템 프롬프트, 교사 프롬프트 |
| `## 제약` 표 (`\| 규칙 \| 값 \|`) | 필수 | 게이트 규칙 생성 |
| `## 발화 원칙` 번호 목록 | 필수 | 프롬프트 |
| `## 다룰 상황` 번호 목록 (쉼표로 여러 순간) | 필수 | dialogue의 커버리지 단위 |
| `## 배경` 자유 서술 | 프로필이 요구하면 | 세계관·설정을 프롬프트에 그대로 |
| `## 하지 않는 말과 행동` · `## 어휘와 표현` · `## 대화 흐름` · `## 예시 대화` | 선택 | 프롬프트 |

제약 표의 규칙 키: 말투(반말·존댓말·서술체·자유), 발화 길이(`N~M글자`/`N~M문장`),
문자(한글·영문·혼용), 이모지, 마크다운, 역할 표기, AI 자칭, 반복, 3인칭 자칭, 이름 어미,
말줄임표(`최대 N개`). 새 페르소나는 `persona-sft-data init <이름> --profile <프로필>`로
골격을 만들고 채운다.

## 외부 소스

설정의 `sources`에 한 항목이면 된다. 포맷 `tsv` `csv` `jsonl` `json` `parquet` `text`,
추출기 `field`(열 그대로) `regex`(패턴 그룹) `conversation`(role/content 목록의 사용자 역할)
`list`(교대 발화 목록의 홀·짝). 목표 언어(`language`)와 다른 소스는 `ingest`가 교사로
번역하고 원문을 함께 남긴다. `sources --sample 5 --translate`로 번역 전후를 눈으로 본다.

```json
"soda": {"format": "parquet", "url": "https://huggingface.co/datasets/allenai/soda/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
         "fields": ["dialogue"], "extract": {"kind": "list", "keep": "even"}, "language": "en", "license": "cc-by-4.0"}
```

## 내보내기

`datasets/<name>/`에 `train|val|test.jsonl`(OpenAI `messages`, 시스템 프롬프트 포함),
`system_prompt.txt`, `chat_template.jinja`(ChatML), `rendered_sample.txt`, `manifest.json`,
데이터 카드 `README.md`, 그리고 `recipe/llamafactory/`에 `dataset_info.json`·`lora_sft.yaml`·
실행 안내. `[student]` extra(`tokenizers`, `huggingface_hub`)가 있으면 학생 토크나이저로
길이 분포를 재 `cutoff_len`을 제안하고, 없으면 글자 수로 재고 그렇다고 적는다.

```bash
llamafactory-cli train datasets/<name>/recipe/llamafactory/lora_sft.yaml
```

학생 모델은 기본값이 없는 필수 설정이다(`student.model`). 이 저장소의 설정
(`configs/mongle.json`·`configs/smoke.json`)이 겨냥하는 학생은
`kakaocorp/kanana-2-1.3b-base`이고, 그 문자열은 설정에만 있다. base 모델에는 채팅
템플릿이 없어 이 프로젝트가 ChatML을 정하며, 추론 때도 `chat_template.jinja`를 써야 한다.

## 플러그인

```python
from persona_sft_data.core.registry import FORMATS

@FORMATS.register("xml")           # 설정의 "plugins": ["my_formats"]로 import 되면 등록된다
class XmlFormat:
    name = "xml"
    extensions = (".xml",)
    def rows(self, data, fields): ...
```

설치되는 패키지라면 `pyproject.toml`의 `[project.entry-points."persona_sft_data.formats"]`에
선언한다. 인터페이스는 `persona_sft_data/core/plugin.py`. 우선순위는 설정 `plugins` >
entry point > 내장이다.

## 명령

| 명령 | 하는 일 |
| --- | --- |
| `check` | 설정·페르소나·프로필·게이트, 단계별 preflight(교사 접속, 소스 표본, 학생 토크나이저) |
| `run [--stage X]` | 설정된 단계를 의존 순서로 |
| `export [--name N]` | assemble 결과에서 데이터셋·레시피만 |
| `sources [--sample N] [--translate]` | 소스별 발화 표본 |
| `status [--watch]` | 단계별 산출·거절 개수 |
| `plugins` | 그룹별 등록 목록 |
| `init <이름> [--profile P]` | 페르소나 문서와 설정 골격 |

## 교사 서버

WSL2의 vLLM으로 `kakaocorp/kanana-2-3b-instruct`(대량)와 30B MoE의 AWQ w4a16(추론)을
번갈아 띄운다. 설치는 `setup/wsl_vllm_setup.sh`, 함정과 해법은
[docs/wsl-vllm.md](docs/wsl-vllm.md), 무인 실행은 `setup/overnight.sh`.

## 저장소 구조

```
persona_sft_data/
  cli.py          명령
  core/           registry · plugin · config · persona · schema · runner · gates · builtins
  rules/          제약 표 규칙 플러그인 11개 + 구조 규칙
  teacher/        base · openai_compat · fake · prompts
  profiles/       companion · npc · novel · trpg · lore
  sources/        base · formats · extractors · translate · safety · topic
  stages/         ingest · dialogue · respond · filter · assemble · export
  recipes/        base · chat_template · llamafactory
personas/         페르소나 문서 (단일 진실)
configs/          mongle.json (본 실행), smoke.json (교사 없이)
tests/            GPU·네트워크 없이 전부. fixtures/에 로컬 소스
docs/             wsl-vllm.md, superpowers/specs·plans
setup/            vLLM 설치·네트워크 모드·야간 실행
```

## 측정하지 않은 것

- `companion` 프롬프트의 실제 교사 수율은 구 파이프라인에서 측정됐고(seed 89%, real 28%),
  새 코드로는 아직 재측정하지 않았다.
- `npc` `novel` `trpg` `lore` 프롬프트는 FakeTeacher로 형식만 검증했다. 실제 교사 수율은
  **미측정**이다.
- 3B 교사의 영→한 번역 품질은 **미측정**이다. 짧은 구어 문장에 한정해 쓴다.
- kanana-2-1.3b-base + ChatML LoRA 학습은 **미실행**이다. 레시피는 LLaMA-Factory 문서의
  필드로 구성했다.
