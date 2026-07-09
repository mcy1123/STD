#!/usr/bin/env bash
set -euo pipefail

# ----------------------------------------------------------------
# STD Reproduction — one-click environment setup
# ----------------------------------------------------------------
# Creates the 'specvlm' conda environment with all dependencies.
#
# Usage:
#   bash scripts/setup_env.sh
#
# After setup, verify with:
#   conda run -n specvlm python -c "import torch; print(torch.cuda.get_device_name(0))"
# ----------------------------------------------------------------

ENV_NAME="specvlm"
PYTHON_VER="3.10"

echo "=== Creating conda environment: $ENV_NAME (Python $PYTHON_VER) ==="
conda create -n "$ENV_NAME" "python=$PYTHON_VER" -y

echo ""
echo "=== Installing PyTorch 2.6.0 (CUDA 12.4) ==="
conda run -n "$ENV_NAME" pip install torch==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124

echo ""
echo "=== Installing core dependencies ==="
conda run -n "$ENV_NAME" pip install \
  transformers==4.48.0 \
  "datasets>=2.14,<3" \
  accelerate \
  "numpy<2.0" \
  qwen-vl-utils==0.0.10 \
  av==14.0.0 \
  triton==3.2.0 \
  huggingface_hub

echo ""
echo "=== Verifying installation ==="
conda run -n "$ENV_NAME" python -c "
import sys
errors = []

try:
    import torch
    assert torch.cuda.is_available(), 'CUDA not available'
    print(f'  PyTorch {torch.__version__}  |  GPU: {torch.cuda.get_device_name(0)}  |  Count: {torch.cuda.device_count()}')
except Exception as e:
    errors.append(f'PyTorch: {e}')

try:
    import transformers
    print(f'  transformers {transformers.__version__}')
except Exception as e:
    errors.append(f'transformers: {e}')

try:
    import av
    print(f'  av {av.__version__}')
except Exception as e:
    errors.append(f'av: {e}')

try:
    import triton
    print(f'  triton OK')
except Exception as e:
    errors.append(f'triton: {e}')

try:
    import qwen_vl_utils
    print(f'  qwen-vl-utils OK')
except Exception as e:
    errors.append(f'qwen-vl-utils: {e}')

try:
    from datasets import load_dataset
    print(f'  datasets OK')
except Exception as e:
    errors.append(f'datasets: {e}')

if errors:
    print()
    print('WARNINGS:')
    for err in errors:
        print(f'  - {err}')
else:
    print()
    print('All dependencies OK.')
"

echo ""
echo "=== Setup complete ==="
echo ""
echo "Next steps:"
echo "  1. Download the model:"
echo "     huggingface-cli download Qwen/Qwen2.5-VL-7B-Instruct --local-dir models/Qwen2.5-VL-7B-Instruct"
echo ""
echo "  2. Download Video-MME dataset:"
echo "     conda run -n $ENV_NAME python scripts/prepare_std_datasets.py --dataset Video-MME"
echo "     conda run -n $ENV_NAME python scripts/extract_videomme_chunks.py --chunks 01"
echo ""
echo "  3. Run smoke test:"
echo "     SPECVLM_MAX_CACHE_LEN=40960 conda run -n $ENV_NAME python scripts/benchmark_std.py \\"
echo "       --model-path models/Qwen2.5-VL-7B-Instruct \\"
echo "       --dataset Video-MME \\"
echo "       --data-path datasets/Video-MME \\"
echo "       --video-root datasets/Video-MME/videos \\"
echo "       --eval-num 3 --frame-num 32 --max-new-tokens 64 \\"
echo "       --gamma 9 --target-k-plus-text 1024 --prompt-style cot --gpu-ids 0"
