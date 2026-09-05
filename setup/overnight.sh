#!/usr/bin/env bash
#
# 생성 체인 전체를 무인으로: dialogue(추론 교사) → ingest·respond(대량 교사) →
# filter → assemble → export. 모델 교체는 파이프라인 밖의 일이라 여기 있다 —
# 파이프라인은 서버에 붙을 뿐 서버를 관리하지 않는다.
#
#   MSYS_NO_PATHCONV=1 wsl.exe bash -lc \
#     'bash /mnt/c/Users/newro/projects/persona-sft-data/setup/overnight.sh'
#
# 단계마다 $LOG에 덧붙이고, 안전한 곳에서는 계속 가므로 아침에 어디까지 갔는지
# 보인다. 학습은 여기 없다 — 이 프로젝트는 데이터셋과 레시피에서 끝나고,
# recipe/llamafactory/lora_sft.yaml을 LLaMA-Factory에 넘기는 것은 다음 일이다.
set -uo pipefail

REPO=/mnt/c/Users/newro/projects/persona-sft-data
WINPY="$REPO/.venv/Scripts/python.exe"
CONFIG=configs/mongle.json
LOG="$HOME/overnight.log"
REASONER=NotoriousH2/kanana-2-30b-a3b-instruct-2601-awq-w4a16
BULK=kakaocorp/kanana-2-3b-instruct

say()  { printf '\n=== %s  %s ===\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; }
fail() { printf '!!! %s  FAILED: %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$LOG"; exit 1; }

serve() {
    local model="$1"
    say "serving $model"
    # shellcheck disable=SC1091
    source "$HOME/vllm-teacher-env.sh"
    pkill -f "vllm serve" 2>/dev/null || true
    sleep 8
    nohup vllm serve "$model" --port 8000 --max-model-len 4096 \
        --gpu-memory-utilization 0.90 > "$HOME/vllm-serve.log" 2>&1 &
    for i in $(seq 1 150); do
        if grep -aq "Application startup complete" "$HOME/vllm-serve.log" 2>/dev/null; then
            echo "  ready after ~$((i * 5))s" | tee -a "$LOG"
            return 0
        fi
        if ! pgrep -f "vllm serve" >/dev/null; then
            echo "  server died:" | tee -a "$LOG"
            tr '\r' '\n' < "$HOME/vllm-serve.log" \
                | grep -aE "Error|error:|Traceback" | head -5 | tee -a "$LOG"
            return 1
        fi
        sleep 5
    done
    echo "  timeout waiting for startup" | tee -a "$LOG"
    return 1
}

stage() {
    local name="$1"
    say "$name"
    "$WINPY" -m persona_sft_data run --config "$CONFIG" --stage "$name" 2>&1 | tee -a "$LOG"
    [ "${PIPESTATUS[0]}" -eq 0 ] || fail "$name"
}

cd "$REPO" || fail "cannot cd $REPO"
: > "$LOG"
say "start"

serve "$REASONER" || fail "reasoner would not start"
stage dialogue

serve "$BULK" || fail "bulk teacher would not start"
stage ingest
stage respond

# 이 뒤는 교사가 필요 없다. GPU를 돌려준다.
say "stopping vllm"
pkill -f "vllm serve" 2>/dev/null || true
sleep 10

stage filter
stage assemble
stage export

say "done -- dataset and recipe in datasets/, corpus in data/final/"
