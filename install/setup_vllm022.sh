#!/bin/bash
# Install vLLM 0.22.0 into a FRESH conda env (vllm022) on the client PAI box.
#
# Why a fresh env: vLLM 0.22 drags a specific torch/torchvision/flashinfer set;
# upgrading `base` (which has ray etc.) in place tends to break it. conda itself
# is fine on this box — we only need a new vLLM, isolated.
#
# Network here is flaky, so this script:
#   - runs fully non-interactive (sources conda.sh; `conda activate` works in nohup)
#   - uses long pip timeout + many retries (auto-resumes from ~/.cache/pip on rerun)
#   - logs everything and prints SETUP_DONE rc=<code> at the end
#
# RUN IT DETACHED (survives SSH disconnect):
#   nohup bash install/setup_vllm022.sh > ~/setup_vllm022.log 2>&1 &
#   tail -f ~/setup_vllm022.log
#   grep -E "VLLM_VERSION|SETUP_DONE" ~/setup_vllm022.log   # done?
#
# If it dies before SETUP_DONE, just rerun the same nohup line — finished wheels
# are cached, so it picks up where it left off.
#
# Overridable knobs:
#   CONDA_SH   path to conda.sh           (default /home/ray/anaconda3/etc/profile.d/conda.sh)
#   ENV_NAME   conda env name             (default vllm022)
#   PY_VER     python version             (default 3.10)
#   VLLM_VER   vLLM version to install    (default 0.22.0)
#   PIP_INDEX  pip index URL              (default aliyun mirror)

set -x
set -o pipefail

CONDA_SH="${CONDA_SH:-/home/ray/anaconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-vllm022}"
PY_VER="${PY_VER:-3.10}"
VLLM_VER="${VLLM_VER:-0.22.0}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"

if [ ! -f "$CONDA_SH" ]; then
    echo "[fatal] conda.sh not found at $CONDA_SH — set CONDA_SH=/path/to/anaconda3/etc/profile.d/conda.sh"
    echo "SETUP_DONE rc=1"
    exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"

# Create env only if missing (idempotent — safe to rerun after a drop).
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python="$PY_VER" -y
fi
conda activate "$ENV_NAME"

export PIP_DEFAULT_TIMEOUT=300
python -m pip install --upgrade pip -i "$PIP_INDEX" || true

# --retries 30 + long timeout ride out flaky network; cached wheels skip on rerun.
pip install --retries 30 --timeout 300 \
    "vllm==${VLLM_VER}" -i "$PIP_INDEX"
RC=$?

echo "===== VERIFY ====="
python -c "import vllm; print('VLLM_VERSION', vllm.__version__)" || RC=1
python -c "import torch; print('TORCH', torch.__version__, 'CUDA', torch.version.cuda)" || true

echo "===== ARCH CHECK (does this vLLM support the client model?) ====="
python -c "from vllm import ModelRegistry; a=ModelRegistry.get_supported_archs(); print('Qwen3_5MoeForConditionalGeneration supported:', 'Qwen3_5MoeForConditionalGeneration' in a); print('qwen3 archs:', [x for x in a if 'wen3' in x.lower()])" || true

echo "SETUP_DONE rc=$RC"
exit "$RC"
