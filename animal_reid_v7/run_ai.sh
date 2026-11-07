#!/usr/bin/env bash
set -euo pipefail
cd /home/liu/CV/animal_reid_v7
export PYTHONPATH="/home/liu/CV/.deps:/home/liu/CV/references/romav2/src:/home/liu/CV/references/lightglue${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_HOME="/home/liu/CV/animal_reid_v7/models/torch_home"
export MPLCONFIGDIR="/home/liu/CV/animal_reid_v7/.matplotlib"
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
exec /home/liu/miniconda3/envs/ai/bin/python -u "$@"
