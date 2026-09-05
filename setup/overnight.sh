#!/usr/bin/env bash
#
# The whole generation chain, unattended: seed, expand, real, template, filter,
# assemble, export. About six hours at the measured rates -- seed 23.3 calls/s
# on the 30B, expand 27.2 on the 3B.
#
#   MSYS_NO_PATHCONV=1 wsl.exe bash -lc \
#     'bash /mnt/c/Users/newro/projects/persona-sft-data/setup/overnight.sh'
#
# Model swapping lives here rather than in the pipeline on purpose: the
# pipeline connects to a server, it does not manage one.
#
# Every step appends to $LOG and the script keeps going where it safely can, so
# a morning reader sees how far it got rather than one truncated failure.
#
# Training is not here. This project ends at a dataset; what consumes it
# (my-llm's tokenizer, pack and sweep, or an SFT trainer) is the consumer's
# script. The GPU is released at the end for whatever that is.
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
    echo "  timeout waiting for /health" | tee -a "$LOG"
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
stage seed

serve "$BULK" || fail "bulk teacher would not start"
stage expand
stage real

# Nothing after this needs a teacher; give the GPU back.
say "stopping vllm"
pkill -f "vllm serve" 2>/dev/null || true
sleep 10

stage template
stage filter
stage assemble
stage export

say "done -- dataset in datasets/, corpus in data/final/"
