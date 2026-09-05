#!/usr/bin/env bash
#
# Idempotent WSL2 setup for the vLLM teacher server.
#
#   MSYS_NO_PATHCONV=1 wsl.exe bash -lc 'bash /mnt/c/<repo>/setup/wsl_vllm_setup.sh'
#
# Re-running is safe: every step checks before acting. Prints SETUP OK on success.
# See docs/wsl-vllm.md for the symptom table behind each step.
#
# NOTE: this file must keep LF line endings. A CRLF copy dies on line 1 of the
# body with "set: pipefail: invalid option name". .gitattributes pins it.
set -euo pipefail

VENV="$HOME/vllm-env"

# Ubuntu 24.04 LTS. Its system Python is 3.12, which is the combination this
# repo actually measured working end to end (vLLM 0.28.0, MARLIN W4A16 MoE,
# 201.5 tok/s on the 30B-A3B). Newer Ubuntu releases ship a Python ahead of
# what vLLM supports, so the distro is chosen to match vLLM, not the reverse.
PYVER=3.12

step() { printf '\n=== %s ===\n' "$1"; }

step "1/6 distro check"
# Fail loudly rather than half-installing on a release whose Python vLLM does
# not support. The alternative is a silent failure much later, at engine start.
. /etc/os-release
echo "$PRETTY_NAME"
if ! command -v "python$PYVER" >/dev/null 2>&1; then
    echo "FAIL: python$PYVER not found on this distro."
    echo "      This script targets Ubuntu 24.04 LTS, whose system Python is $PYVER."
    echo "      Install Ubuntu-24.04 (wsl --install -d Ubuntu-24.04) or adjust PYVER"
    echo "      to a version vLLM supports and that this distro provides."
    exit 1
fi

step "2/6 build tools"
# vLLM JIT-compiles CUDA helpers at engine start. Without these the server dies
# AFTER the model loads, which looks like a model fault but is not.
if [ -f "/usr/include/python$PYVER/Python.h" ] && command -v gcc >/dev/null 2>&1; then
    echo "already present"
else
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq         "python$PYVER-dev" build-essential ninja-build curl
fi
test -f "/usr/include/python$PYVER/Python.h" || { echo "FAIL: Python.h missing"; exit 1; }
gcc --version | head -1

step "3/6 uv"
if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

step "4/6 venv + vllm"
# --torch-backend=auto is load-bearing. Pinning cu128 out of Blackwell caution
# installs a torch whose runtime vLLM's compiled extension cannot load
# (ImportError: libcudart.so.13). Let uv match vLLM's own CUDA build.
if [ ! -x "$VENV/bin/python" ]; then
    uv venv "$VENV" --python "$PYVER" --clear
fi
source "$VENV/bin/activate"
python -c "import vllm" 2>/dev/null || uv pip install vllm --torch-backend=auto

step "5/6 verify GPU"
python - <<'PY'
import torch, vllm
cap = torch.cuda.get_device_capability(0)
tag = f"sm_{cap[0]}{cap[1]}"
archs = torch.cuda.get_arch_list()
print("vllm ", vllm.__version__)
print("torch", torch.__version__)
print("gpu  ", torch.cuda.get_device_name(0), tag)
assert tag in archs, f"{tag} not in {archs}"
print(f"OK: {tag} supported")
PY

step "6/6 write the serve env"
# No system CUDA toolkit here; the pip-installed one has the full
# bin/include/lib/nvvm layout. Discover it -- the directory name carries the
# CUDA major version and moves when vLLM upgrades.
NVCC="$(find "$VENV" -name nvcc -type f 2>/dev/null | head -1)"
test -n "$NVCC" || { echo "FAIL: no nvcc under $VENV"; exit 1; }
CUDA_HOME="$(dirname "$(dirname "$NVCC")")"
echo "CUDA_HOME=$CUDA_HOME"

# FlashInfer JIT-builds its sampling kernel and its bundled cccl headers collide
# with the CUDA 13 toolkit headers. Only the sampler is affected -- MARLIN W4A16
# MoE, TRITON_MLA attention and the torch.compile graph all succeed -- and vLLM
# has a native top-k/top-p sampler, so disabling it costs almost nothing.
cat > "$HOME/vllm-teacher-env.sh" <<ENVEOF
export PATH="\$HOME/.local/bin:\$PATH"
source "$VENV/bin/activate"
export CUDA_HOME="$CUDA_HOME"
export PATH="\$CUDA_HOME/bin:\$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0
# Serving runs under NAT, where outbound TCP is silently dropped on this
# machine. huggingface_hub still tries to reach the Hub to validate a cached
# model's revision, and that request hangs for minutes instead of failing.
# The models are local; never let serve-time code phone home.
export HF_HUB_OFFLINE=1
# vLLM turns pinned memory off on WSL by default, conservatively. Kernels from
# 4.19.121 support it and vLLM gates it behind this flag. Without it, models
# whose worker path allocates a UvaBuffer die at startup with "UVA is not
# available" -- qwen3 does, deepseek_v3 does not, so it looks model-specific.
export VLLM_WSL2_ENABLE_PIN_MEMORY=1
ENVEOF
echo "wrote ~/vllm-teacher-env.sh -- source it before 'vllm serve'"

echo
echo "SETUP OK"
