# WSL2 vLLM 교사 서버 — 설치와 트러블슈팅

- 대상: `persona_sft_data` 파이프라인이 붙을 교사 서버
- 검증 환경: Windows 11 (10.0.26200), WSL2 2.6.3.0, Ubuntu 24.04, RTX 5090 32GB
- 작성일: 2026-09-04

**이 문서는 서브에이전트가 그대로 따라 하도록 쓴다.** §1의 스크립트 한 줄로
끝나는 것이 정상이고, 실패하면 §3의 증상표에서 찾는다. §2는 vLLM과 무관하게
**이 기계에서 명령을 실행할 때 밟는 함정**이며, 처음 구축할 때 여기서 세 번
헛돌았다.

파이프라인은 이 서버를 **띄우지 않는다.** 사람이 띄우고, 파이프라인은
`http://localhost:8000/v1`에 HTTP로 붙을 뿐이다(설계 §3.2).

## 1. 한 줄 설치

```bash
MSYS_NO_PATHCONV=1 wsl.exe bash -lc   'bash /mnt/c/Users/newro/projects/persona-sft-data/setup/wsl_vllm_setup.sh'
```

멱등하다. 이미 된 단계는 건너뛰므로 몇 번을 돌려도 안전하다. 끝에 `SETUP OK`가
찍히면 성공이고, 그때 `~/vllm-teacher-env.sh`가 만들어진다. 서버를 띄우기 전에
이것을 `source` 한다 — CUDA_HOME과 FlashInfer 우회가 들어 있다.

```bash
source ~/vllm-teacher-env.sh
vllm serve <model> --port 8000 --max-model-len 4096     --gpu-memory-utilization 0.90 --served-model-name teacher
```

## 2. 이 기계에서 명령을 실행할 때의 함정

**vLLM 문제가 아니다. 셋 다 실제로 밟았고, 셋 다 조용히 실패한다.**

### 2.1 파이프가 종료 코드를 삼킨다

```bash
wsl.exe bash -lc '... | tr -d "\r"'    # 종료 코드는 tr의 것 -- 항상 0
```

첫 설치가 이렇게 **실패했는데 "exit code 0"으로 보고**됐다. 스크립트를 돌릴
때는 파이프를 붙이지 않는다. 출력을 다듬어야 하면 스크립트 안에서 한다.

### 2.2 Windows에서 파이썬으로 셸 스크립트를 쓰면 CRLF가 된다

`pathlib.Path.write_text()`는 Windows에서 `\n`을 `\r\n`으로 바꾼다. bash가
`set -euo pipefail\r`을 읽고 `set: pipefail: invalid option name`으로 죽는다.

- 셸 스크립트는 **bash 히어독으로 쓴다** (`cat > f <<'EOF'`).
- 파이썬을 써야 하면 `write_text(s, newline="\n")`.
- 의심되면 `sed -i 's/\r$//' f`, 확인은 `cat -A f | head`(`^M$`이 보이면 CRLF).

### 2.3 Git Bash(MSYS)가 경로를 변환한다

`S=/mnt/c/...` 같은 `VAR=경로` 토큰을 MSYS가 Windows 경로로 바꿔 버려서
`$S`가 빈 값이 됐다(`bash: /wsl_vllm_install.sh: No such file or directory`).

- `wsl.exe`를 부를 때는 **`MSYS_NO_PATHCONV=1`을 앞에 붙인다.**
- `-lc` 문자열 안에서 경로를 변수에 담지 말고 **전체 경로를 그대로 쓴다.**

### 2.4 vLLM 스크립트는 히어독으로 넘기지 않는다

`python - <<'PY' ... PY`는 이 프로젝트의 다른 곳에서는 잘 동작하지만
**vLLM에서는 안 된다.** 엔진이 워커를 `multiprocessing` spawn으로 띄우면서
`__main__`을 파일 경로로 다시 읽는데, 히어독으로 넘긴 코드에는 경로가 없다.

```
FileNotFoundError: [Errno 2] No such file or directory: '/.../<stdin>'
RuntimeError: Engine core initialization failed. Failed core proc(s): {'EngineCore': 1}
```

파이썬을 파일로 쓰고 `python script.py`로 실행한다. `/mnt/c`가 아니라 리눅스
파일시스템에 두는 편이 빠르다.

### 2.5 진짜 원인은 래퍼 예외 아래에 있다

vLLM이 죽으면 마지막에 보이는 것은 늘 이 문장이다.

```
RuntimeError: Engine core initialization failed. See root cause above.
```

**여기엔 정보가 없다.** 서브프로세스 로그를 직접 뒤진다. 위 §2.4처럼 진짜
원인이 출력 맨 위에 있을 수도 있으니 뒤만 보지 말고 앞도 본다.

```bash
MSYS_NO_PATHCONV=1 wsl.exe bash -lc \
  'grep -nE "fatal error|Could not find|Unsupported|NotImplementedError|out of memory|No available memory" ~/vllm-serve.log | head -20'
```

## 3. 증상별 진단표

실제로 만난 것만 적는다. 추측은 넣지 않는다.

| 증상 | 원인 | 조치 |
| --- | --- | --- |
| `curl`이 WSL에서만 타임아웃(Windows는 200) | WSL2 NAT이 DNS는 되고 TCP가 안 나감 | §4 미러 네트워킹 |
| `ImportError: libcudart.so.13` | `--torch-backend=cu128`을 강제해 CUDA 12.8 torch가 깔림. vLLM 0.28은 CUDA 13 링크 | `--torch-backend=auto`로 재설치 |
| `cuda_utils.c:9:10: fatal error: Python.h` | 파이썬 개발 헤더 없음. Triton이 런타임 컴파일에 실패 | `sudo apt-get install -y python3.12-dev build-essential` |
| `Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist` | `CUDA_HOME` 미설정 | pip이 깐 것을 가리킨다(§5). 시스템 CUDA 툴킷 설치 불필요 |
| `A virtual environment already exists` | 앞선 실패가 남긴 venv | `uv venv ... --clear` |
| `RuntimeError: Ninja build failed` + `CUDA compiler and CUDA toolkit headers are incompatible` | FlashInfer가 샘플링 커널을 JIT 빌드하는데 번들 cccl 헤더가 CUDA 13 헤더와 충돌 | `export VLLM_USE_FLASHINFER_SAMPLER=0` |
| 서버가 `Application startup complete`까지 갔는데 `/health`가 `000` | 미러 모드 + Hyper-V 방화벽이 WSL 리슨 포트를 차단 | 서빙은 NAT으로 (§4). 미러가 꼭 필요하면 `firewall=false` (§4.2) |
| 위 조치 뒤에도 `/health`가 `000`이고 `ss`에는 `127.0.0.1:8000`이 보임 | **미러 모드에서 vLLM은 루프백으로 닿지 않는다.** `--host`와 무관하게 `127.0.0.1`은 `SYN-SENT`에서 멈춘다 | **서빙은 NAT 모드로 한다** (§4) |
| `FileNotFoundError: ... '<stdin>'` 뒤에 `Engine core initialization failed` | vLLM이 워커를 `multiprocessing` spawn으로 띄우며 `__main__`을 **파일 경로로** 다시 읽는다. `python - <<EOF`로 넘긴 스크립트에는 경로가 없다 | 파이썬을 **실제 파일**로 쓰고 실행한다 |
| NAT 모드에서 `non-default args:` 로그 뒤 침묵, GPU 2GB에서 멈춤 | `huggingface_hub`가 캐시된 모델의 리비전을 Hub에 확인하려 하고, NAT이 그 SYN을 조용히 버린다. 실측: `SYN-SENT ... -> 3.168.178.31:443`(CloudFront/HF) | `export HF_HUB_OFFLINE=1` — `~/vllm-teacher-env.sh`에 들어 있다 |
| `RuntimeError: UVA is not available` (모델 적재 전에 죽음) | vLLM이 WSL에서 pinned memory를 기본으로 끈다. 일부 모델의 워커 경로가 `UvaBuffer`를 요구한다 — **qwen3는 요구하고 deepseek_v3는 아니라서 모델 문제처럼 보인다** | `export VLLM_WSL2_ENABLE_PIN_MEMORY=1` (커널 4.19.121 이상). `~/vllm-teacher-env.sh`에 들어 있다 |
| `set: pipefail: invalid option name` | 스크립트가 CRLF | §2.2 |
| `bash: /<script>.sh: No such file or directory` | MSYS 경로 변환 | §2.3 |

**모델이 정상 적재된 뒤에 죽는 경우가 많다.** 로그에
`Model loading took NN GiB` 가 있으면 가중치·아키텍처·양자화는 문제가 아니고,
그 뒤의 JIT 컴파일 단계다.

## 3.1 정상 기동 시 로그에 나와야 하는 것

기동에 성공하면 아래가 찍힌다. **다른 것이 찍히면 설계가 가정한 경로가 아니므로
멈추고 확인한다.**

```
Using 'MARLIN' WNA16 MoE backend.                          # W4A16 MoE
Using TRITON_MLA attention backend                          # MLA
Model loading took 15.9x GiB memory                         # 30B-A3B AWQ
Compiling a graph for compile range (1, 2048) takes ~6 s    # torch.compile
```

`non-default args:` 다음 줄이 안 나오고 GPU가 2GB 근처면 **네트워크로 나가려다
매달린 것**이다(§3 표의 `HF_HUB_OFFLINE` 행). 모델 문제가 아니다.

`Model loading took ...`까지 갔다면 **가중치·아키텍처·양자화는 문제가 아니다.**
그 뒤에 죽으면 원인은 JIT 컴파일 쪽이고, §3 표의 아래 세 줄 중 하나다.

## 4. WSL 네트워크 — 두 모드가 서로 반대로 고장 나 있다

이 기계에서 WSL2의 두 네트워킹 모드는 **정반대 방향으로** 실패한다. 둘 다 새로
설치한 Ubuntu-24.04에서 측정했으므로 배포판 문제가 아니라 기계 문제다.

| | NAT (기본, `.wslconfig` 없음) | mirrored |
| --- | --- | --- |
| 아웃바운드 TCP | **죽어 있음.** DNS는 되는데 pypi·HF가 타임아웃 | 정상 |
| Windows → WSL `localhost:PORT` | **정상** (표준 포워딩) | 미니 서버는 됨, **vLLM은 안 됨** |
| WSL → `127.0.0.1:PORT` | 정상 | 미니 서버는 됨, **vLLM은 안 됨** |
| WSL → 인터페이스 IP:PORT | — | vLLM 됨 (WSL 안에서만) |

mirrored에서 vLLM만 안 되는 증상은 진단하기 나쁜 모양이다. `ss`에는
`LISTEN 0.0.0.0:8000`이 멀쩡히 보이고, 커널 카운터(`ListenDrops`,
`ListenOverflows`)는 0인데, 클라이언트 소켓이 `SYN-SENT`에서 멈춘다. 같은 순간
같은 네임스페이스에서 파이썬 `http.server`는 200을 준다.

**따라서 단계별로 모드를 바꾼다.**

```powershell
tools\setup\wsl_net_mode.ps1 mirrored   # pip install, hf download 전에
tools\setup\wsl_net_mode.ps1 nat        # vllm serve 전에
```

각 전환은 `wsl --shutdown`을 동반하므로 WSL에서 돌던 것이 전부 죽는다. 모델을
다 받은 뒤에는 `nat`으로 두고 쓴다.

### 4.1 mirrored를 켜는 방법 (참고)

기본 NAT에서 **DNS는 되는데 TCP가 전혀 안 나가는** 상태가 될 수 있다.

```bash
MSYS_NO_PATHCONV=1 wsl.exe bash -c 'curl -sS -o /dev/null -w "%{http_code}\n" --max-time 15 https://pypi.org/simple/'
```

`000`이면 `C:\Users\<user>\.wslconfig`에 미러 네트워킹을 켠다. NAT을 건너뛰고
Windows 네트워크 스택을 그대로 쓴다.

```ini
[wsl2]
networkingMode=mirrored
dnsTunneling=true
autoProxy=true
firewall=true
```

`wsl --shutdown` 뒤 재확인. **되돌리려면 파일을 지우고 다시 `wsl --shutdown`.**

### 4.2 `firewall=false`가 필요한 이유와 그 대가

미러 모드만으로는 부족했다. Hyper-V 방화벽이 켜져 있으면 **WSL 안에서 리슨 중인
서비스에 WSL 자신도, Windows도 붙지 못한다.** vLLM이 `0.0.0.0:8000`에 떠 있는데
모든 접속이 `000`으로 끝났다. `firewall=false`가 이를 푼다.
(`hostAddressLoopback`은 WSL 2.6.3.0이 모르는 키라 해결책이 아니다 — 넣으면
경고만 나온다.)

**대가:** WSL 트래픽이 Windows 방화벽 필터를 거치지 않으므로, WSL에서
`0.0.0.0`에 바인딩한 서비스는 같은 네트워크에서 보인다.

**`--host 127.0.0.1`로 묶으려 하지 마라.** 직관에 반하지만, 미러 모드에서
WSL의 `127.0.0.1`은 일반 루프백이 아니어서 **서버가 아예 접속 불가가 된다.**
`ss`에는 `LISTEN 127.0.0.1:8000`이 멀쩡히 보이는데 `curl`은 연결 거부도 아닌
타임아웃으로 끝난다 — 진단하기 나쁜 형태로 실패한다.

노출이 신경 쓰이면 바인딩이 아니라 인증으로 막는다.

```bash
vllm serve <model> --api-key "$(openssl rand -hex 16)" ...
```

교사 생성이 끝나면 `.wslconfig`를 지우고 `wsl --shutdown` 하면 원래대로 돌아온다.

## 5. 설치 상세

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv ~/vllm-env --python 3.12 --clear
source ~/vllm-env/bin/activate
uv pip install vllm --torch-backend=auto      # auto -- 손으로 고정하지 않는다

sudo apt-get install -y python3.12-dev build-essential

export CUDA_HOME="$HOME/vllm-env/lib/python3.12/site-packages/nvidia/cu13"
export PATH="$CUDA_HOME/bin:$PATH"
```

`nvidia/cu13`에 `bin/ include/ lib/ nvvm/`이 모두 있어 시스템 CUDA 툴킷이
필요 없다. **경로의 `cu13`은 CUDA 메이저 버전이므로 vLLM이 올라가면 바뀔 수
있다.** 하드코딩 대신 찾는다.

```bash
find ~/vllm-env -name nvcc -type f | head -1
```

검증:

```bash
python -c "import torch; print('sm_120' in torch.cuda.get_arch_list())"
```

측정값: vLLM 0.28.0 / torch 2.13.0+cu130에서 `arch_list`는
`['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120']`.

## 6. 교사 서버 띄우기

두 교사를 **순차로** 올린다. 동시 적재는 KV 캐시까지 고려하면 빠듯하고, 단계가
어차피 순차라 얻을 것이 없다.

```bash
# 시드·추론 (seed 단계) -- 적재 실측 15.92 GiB
vllm serve NotoriousH2/kanana-2-30b-a3b-instruct-2601-awq-w4a16 \
    --port 8000 --max-model-len 4096 \
    --gpu-memory-utilization 0.90 --served-model-name teacher

# 대량 대화 (expand·real 단계)
vllm serve kakaocorp/kanana-2-3b-instruct \
    --port 8000 --max-model-len 4096 \
    --gpu-memory-utilization 0.90 --served-model-name teacher
```

`--max-model-len`을 모델 최대치(32768)가 아니라 4096으로 낮추는 것은 의도된
선택이다. 이 파이프라인의 프롬프트는 짧고, 줄인 만큼 KV 캐시가 커져 배치가
커진다.

두 명령의 포트가 같은 것은 오타가 아니다. 시점만 다르게 뜬다. 요청의 `model`
필드가 어느 쪽인지 판정하며, vLLM은 서빙하지 않는 모델 id를 404로 거절하므로
**잘못된 서버에 붙었을 때 조용히 다른 모델로 생성되는 일이 구조적으로 없다.**

`--host`를 주지 않는 것이 의도된 선택이다(§4.2). 기동은 느리다(모델 적재 + JIT 컴파일). `/health`를 폴링하되, **프로세스가
죽었는지도 같이 본다** — 안 그러면 죽은 서버를 타임아웃까지 기다린다.

```bash
for i in $(seq 1 120); do
    curl -s -o /dev/null -m 2 http://127.0.0.1:8000/health && { echo up; break; }
    pgrep -f "vllm serve" >/dev/null || { echo "DIED"; tail -40 ~/vllm-serve.log; break; }
    sleep 5
done
```
