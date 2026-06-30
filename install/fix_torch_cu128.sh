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
TV="$(python  -c "import torch;print(torch.__version__.split('+')[0])"       2>/dev/null || echo "")"
TVV="$(python -c "import torchvision;print(torchvision.__version__.split('+')[0])" 2>/dev/null || echo "")"
TVA="$(python -c "import torchaudio;print(torchaudio.__version__.split('+')[0])"   2>/dev/null || echo "")"
echo "installed (version only): torch=${TV:-?} torchvision=${TVV:-?} torchaudio=${TVA:-?} (current build is +cu130 -> too new for the 12.8 driver)"

echo "===== FORCE-reinstall the SAME versions as cu128 builds ====="
# The earlier run skipped because pip saw '2.11.0' already satisfied (it ignores the
# +cu130 local tag). --force-reinstall + an index that ONLY has cu128 wheels forces
# the cu128 build; deps (nvidia-*-cu12) come from the same mirror, non-torch deps
# from pypi. Pin the detected versions so vllm's torch requirement stays satisfied.
PKGS=()
[ -n "$TV" ]  && PKGS+=("torch==${TV}")
[ -n "$TVV" ] && PKGS+=("torchvision==${TVV}")
[ -n "$TVA" ] && PKGS+=("torchaudio==${TVA}")
[ ${#PKGS[@]} -eq 0 ] && PKGS=(torch torchvision torchaudio)

pip install --no-cache-dir --force-reinstall "${PKGS[@]}" \
    --index-url "$TORCH_INDEX" \
    --extra-index-url https://mirrors.aliyun.com/pypi/simple/

echo "===== verify (this actually inits CUDA on the GPUs) ====="
python - <<'PY'
import torch
v = torch.version.cuda or ""
print("torch", torch.__version__, "built-cuda", v)
if v.startswith("13"):
    print("CUDA_FAIL still a cu13 build -> the cu128 wheel for this torch version was NOT on the mirror.")
    print("  -> report this; we'll pin a torch version that has a cu128 build, or use download.pytorch.org/whl/cu128.")
else:
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
