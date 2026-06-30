#!/bin/bash
# FIX: "NVIDIA driver too old (found version 12080)" when starting vLLM.
#
# The box's driver supports CUDA 12.8, but `pip install vllm==0.22.0` pulled a
# torch built for a newer CUDA (12.9/13.x), whose self-check refuses to init on a
# 12.8 driver. Fix = reinstall the SAME torch version but the cu128 build, from the
# aliyun pytorch-wheels mirror. vLLM links against torch's CUDA runtime, so this
# usually makes the whole stack run on 12.8.
#
# RUN (in the vllm022 env), detached so it survives SSH drops:
#   nohup bash install/fix_torch_cu128.sh > ~/fix_torch.log 2>&1 &
#   tail -f ~/fix_torch.log
#   grep -E "FIX_DONE|CUDA_OK|CUDA_FAIL" ~/fix_torch.log
#
# Knobs: CONDA_SH, ENV_NAME, TORCH_INDEX (default aliyun cu128 mirror).

set -x
set -o pipefail
CONDA_SH="${CONDA_SH:-/home/ray/anaconda3/etc/profile.d/conda.sh}"
ENV_NAME="${ENV_NAME:-vllm022}"
TORCH_INDEX="${TORCH_INDEX:-https://mirrors.aliyun.com/pytorch-wheels/cu128}"

# shellcheck disable=SC1090
source "$CONDA_SH"
conda activate "$ENV_NAME"

echo "===== driver / current torch ====="
nvidia-smi | head -4 || true
TV="$(python -c "import torch;print(torch.__version__.split('+')[0])" 2>/dev/null || echo "")"
echo "installed torch (version only): ${TV:-<none>}"

echo "===== reinstall torch stack as cu128 ====="
# Pin the same torch version if we detected one; otherwise let the index pick.
if [ -n "$TV" ]; then
    pip install --no-cache-dir "torch==${TV}" torchvision torchaudio --index-url "$TORCH_INDEX" || \
        pip install --no-cache-dir torch torchvision torchaudio --index-url "$TORCH_INDEX"
else
    pip install --no-cache-dir torch torchvision torchaudio --index-url "$TORCH_INDEX"
fi

echo "===== verify (this actually inits CUDA on the GPUs) ====="
python - <<'PY'
import torch
print("torch", torch.__version__, "built-cuda", torch.version.cuda)
try:
    ok = torch.cuda.is_available()
    print("cuda.is_available:", ok)
    if ok:
        print("device0:", torch.cuda.get_device_name(0))
        print("CUDA_OK")
    else:
        print("CUDA_FAIL (is_available False)")
except Exception as e:
    print("CUDA_FAIL", repr(e))
PY

echo "===== sanity: vllm still imports ====="
python -c "import vllm; print('VLLM', vllm.__version__)" || echo "VLLM_IMPORT_FAIL"
echo "FIX_DONE rc=$?"
