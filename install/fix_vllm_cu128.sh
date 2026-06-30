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
# vLLM 0.22.0 has NO cu128 wheel — the lowest CUDA-12 build is cu129 (CUDA 12.9).
# cu129 runs on this 12.8 driver via CUDA minor-version compatibility, and it links
# libcudart.so.12 (provided by our cu128 torch) — NOT the .so.13 the PyPI wheel needed.
CUDA_TAG="${CUDA_TAG:-cu129}"
MANYLINUX="${MANYLINUX:-manylinux_2_28}"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

echo "===== fetch the ${CUDA_TAG} vLLM ${VLLM_VER} wheel (GitHub release asset) ====="
WHEEL="vllm-${VLLM_VER}+${CUDA_TAG}-cp38-abi3-${MANYLINUX}_${CPU_ARCH}.whl"
# %2B = '+' (GitHub asset paths url-encode the plus).
WHEEL_URL="${WHEEL_URL:-https://github.com/vllm-project/vllm/releases/download/v${VLLM_VER}/vllm-${VLLM_VER}%2B${CUDA_TAG}-cp38-abi3-${MANYLINUX}_${CPU_ARCH}.whl}"
echo "WHEEL_URL $WHEEL_URL"

# Download to a local file first (curl -L follows the CDN redirect; more reliable
# than pip fetching the github URL directly), then pip-install the local wheel.
DEST="/tmp/${WHEEL}"
if ! curl -L --fail --retry 10 --retry-delay 5 --max-time 1800 -o "$DEST" "$WHEEL_URL"; then
    echo "DOWNLOAD_FAIL: could not fetch $WHEEL_URL"
    echo "  Try a proxy/mirror, or download the wheel by hand and rerun with WHEEL_URL=file:///path/to.whl"
    echo "FIX_DONE rc=1"; exit 1
fi
ls -la "$DEST"

echo "===== install the ${CUDA_TAG} vLLM over the cu13 one (--no-deps: keep our cu128 torch) ====="
PIP_CONFIG_FILE=/dev/null pip install --no-cache-dir --force-reinstall --no-deps "$DEST"
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
