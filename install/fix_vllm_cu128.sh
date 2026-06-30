#!/bin/bash
# FIX: after torch is cu128, vLLM 0.22.0 from PyPI still fails with
#   ImportError: libcudart.so.13: cannot open shared object file
# because the PyPI vllm wheel's compiled _C was built for CUDA 13. vLLM also
# publishes a cu128 build as a GitHub release asset (vllm-<ver>+cu128-...whl).
# Install THAT over the PyPI one (--no-deps so it won't touch our cu128 torch).
#
# RUN (in vllm022, after fix_torch_cu128.sh shows CUDA_OK):
#   nohup bash install/fix_vllm_cu128.sh > ~/fix_vllm.log 2>&1 &
#   tail -f ~/fix_vllm.log
#   grep -E "WHEEL_URL|VLLM_C_OK|ARCH_SUPPORTED|VLLM_FAIL|FIX_DONE" ~/fix_vllm.log
#
# Knobs: CONDA_SH, ENV_NAME, VLLM_VER (default 0.22.0), CPU_ARCH (default x86_64).

set -x
set -o pipefail
CONDA_SH="${CONDA_SH:-/home/ray/anaconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-vllm022}"
VLLM_VER="${VLLM_VER:-0.22.0}"
CPU_ARCH="${CPU_ARCH:-x86_64}"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

echo "===== find the cu128 vLLM ${VLLM_VER} wheel (GitHub release asset) ====="
# Prefer the GitHub releases API (gives the real filename); fall back to the
# documented URL pattern if the API isn't reachable.
API="https://api.github.com/repos/vllm-project/vllm/releases/tags/v${VLLM_VER}"
WHEEL_URL="$(curl -s --max-time 30 "$API" \
    | grep -oE 'https://[^"]*vllm-[^"]*\+cu128[^"]*'"${CPU_ARCH}"'\.whl' | head -1)"
if [ -z "$WHEEL_URL" ]; then
    WHEEL_URL="https://github.com/vllm-project/vllm/releases/download/v${VLLM_VER}/vllm-${VLLM_VER}+cu128-cp38-abi3-manylinux_2_35_${CPU_ARCH}.whl"
    echo "  API gave nothing; trying documented URL pattern."
fi
echo "WHEEL_URL $WHEEL_URL"
curl -sfI --max-time 30 "$WHEEL_URL" >/dev/null 2>&1 && echo "  (asset reachable)" || \
    echo "  WARN: HEAD failed; pip will still try (or set WHEEL_URL=... by hand)."

echo "===== install the cu128 vLLM over the cu13 one (--no-deps: keep our cu128 torch) ====="
PIP_CONFIG_FILE=/dev/null pip install --no-cache-dir --force-reinstall --no-deps "$WHEEL_URL"
RC=$?

echo "===== verify: the _C extension that was failing now loads + arch supported ====="
python - <<'PY'
try:
    import torch
    print("torch", torch.__version__, "built-cuda", torch.version.cuda, "cuda.is_available", torch.cuda.is_available())
    import vllm
    print("VLLM", vllm.__version__)
    import vllm._C  # this is exactly what raised libcudart.so.13
    print("VLLM_C_OK")
    from vllm import ModelRegistry
    print("ARCH_SUPPORTED", "Qwen3_5MoeForConditionalGeneration" in ModelRegistry.get_supported_archs())
except Exception as e:
    print("VLLM_FAIL", repr(e))
PY
echo "FIX_DONE rc=$RC"
exit "$RC"
