#!/usr/bin/env bash
# One-time setup on a GCP g2-standard-16 (8 cores, 64 GB RAM, 1x L4 24 GB).
set -euo pipefail
nvidia-smi
pip install -r "$(dirname "$0")/../requirements.txt"
python -c "import onnxruntime as o; p=o.get_available_providers(); print(p); assert 'CUDAExecutionProvider' in p, 'onnxruntime-gpu not using CUDA'"
python -c "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name())"
