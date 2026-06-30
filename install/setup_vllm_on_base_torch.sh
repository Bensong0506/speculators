#!/bin/bash
# Find the newest vLLM that (a) rides the base image's WORKING torch (2.9.0+cu12,
# which runs on this box's 12.8 driver) and (b) supports the client model arch
# Qwen3_5MoeForConditionalGeneration.
#
# Why not vLLM 0.22.0: it hard-pins torch 2.11 (cu13), which needs a CUDA-13 driver.
# The base torch 2.9.0 is cu12 and already works here, so we keep it and let pip pick
# the highest vllm compatible with torch==2.9.0, then check arch support.
#
# Safe: clones `base` into a NEW env (default vllm_qwen); never modifies base.
#
# RUN detached (survives SSH drops):
#   nohup bash install/setup_vllm_on_base_torch.sh > ~/setup_vllm_qwen.log 2>&1 &
#   tail -f ~/setup_vllm_qwen.log
#   grep -E "PICKED_VLLM|CUDA_OK|CUDA_FAIL|ARCH_SUPPORTED|SETUP_DONE" ~/setup_vllm_qwen.log
#
# Knobs: CONDA_SH, BASE_ENV (default base), ENV_NAME (default vllm_qwen),
#        PIP_INDEX (aliyun pypi), VLLM_SPEC (default "vllm" = newest compatible;
#        set e.g. VLLM_SPEC='vllm<0.22' to cap).

set -x
set -o pipefail
CONDA_SH="${CONDA_SH:-/home/ray/anaconda3/etc/profile.d/conda.sh}"
BASE_ENV="${BASE_ENV:-base}"
ENV_NAME="${ENV_NAME:-vllm_qwen}"
PIP_INDEX="${PIP_INDEX:-https://mirrors.aliyun.com/pypi/simple/}"
VLLM_SPEC="${VLLM_SPEC:-vllm}"
ARCH="${ARCH:-Qwen3_5MoeForConditionalGeneration}"

# shellcheck disable=SC1090
source "$CONDA_SH"

# Read the base env's working torch stack (so we can pin it through the upgrade).
conda activate "$BASE_ENV"
TV="$(python  -c "import torch;print(torch.__version__.split('+')[0])"       2>/dev/null || echo "")"
TVV="$(python -c "import torchvision;print(torchvision.__version__.split('+')[0])" 2>/dev/null || echo "")"
TVA="$(python -c "import torchaudio;print(torchaudio.__version__.split('+')[0])"   2>/dev/null || echo "")"
echo "base torch stack: torch=${TV:-?} torchvision=${TVV:-?} torchaudio=${TVA:-?} (cu12 -> works on 12.8 driver)"
[ -n "$TV" ] || { echo "[fatal] could not read torch from $BASE_ENV"; echo "SETUP_DONE rc=1"; exit 1; }

# Clone base into a throwaway env (keeps the proven cu12 torch + flashinfer); never
# touches base itself.
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" --clone "$BASE_ENV" -y
fi
conda activate "$ENV_NAME"

export PIP_DEFAULT_TIMEOUT=300
# Pin torch to the base version so pip keeps the cu12 build and backtracks to the
# HIGHEST vllm that accepts torch==$TV (instead of dragging in torch 2.11/cu13).
PINS=("torch==${TV}")
[ -n "$TVV" ] && PINS+=("torchvision==${TVV}")
[ -n "$TVA" ] && PINS+=("torchaudio==${TVA}")

echo "===== upgrade vLLM while pinning torch==${TV} ====="
pip install --no-cache-dir -U "$VLLM_SPEC" "${PINS[@]}" \
    --retries 20 --timeout 300 -i "$PIP_INDEX"
RC=$?

echo "===== which vllm did pip pick, and does torch still init? ====="
python - <<PY
import torch
print("torch", torch.__version__, "built-cuda", torch.version.cuda)
try:
    import vllm
    print("PICKED_VLLM", vllm.__version__)
except Exception as e:
    print("VLLM_IMPORT_FAIL", repr(e))
try:
    ok = torch.cuda.is_available()
    print("cuda.is_available:", ok)
    print("CUDA_OK" if ok else "CUDA_FAIL (is_available False)")
except Exception as e:
    print("CUDA_FAIL", repr(e))
try:
    from vllm import ModelRegistry
    a = ModelRegistry.get_supported_archs()
    print("ARCH_SUPPORTED", "$ARCH" in a)
    print("qwen3 archs:", [x for x in a if "wen3" in x.lower()])
except Exception as e:
    print("ARCH_CHECK_FAIL", repr(e))
PY
echo "SETUP_DONE rc=$RC"
exit "$RC"
