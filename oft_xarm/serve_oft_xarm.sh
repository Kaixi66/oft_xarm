#!/usr/bin/env bash
set -euo pipefail

REALWORLD_ROOT="${REALWORLD_ROOT:-/workspace/kaixi/RealWorld}"
REPO_DIR="${REPO_DIR:-/workspace/kaixi/openvla-oft}"
CONDA_ENV="${CONDA_ENV:-/workspace/kaixi/.conda/envs/openvla-oft}"

PYTHON="${PYTHON:-${CONDA_ENV}/bin/python}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8777}"
CHECKPOINT="${CHECKPOINT:-${REALWORLD_ROOT}/openvla_oft_runs/checkpoints/openvla-oft_setting2}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-openvla/openvla-7b}"
UNNORM_KEY="${UNNORM_KEY:-utokyo_xarm_pick_and_place_converted_externally_to_rlds}"

USE_L1_REGRESSION="${USE_L1_REGRESSION:-True}"
USE_DIFFUSION="${USE_DIFFUSION:-False}"
USE_FILM="${USE_FILM:-False}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"
USE_PROPRIO="${USE_PROPRIO:-True}"
CENTER_CROP="${CENTER_CROP:-True}"
LORA_RANK="${LORA_RANK:-32}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-False}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-False}"
DRY_RUN="${DRY_RUN:-false}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "[serve_oft_xarm] missing python: ${PYTHON}" >&2
    exit 1
fi

if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "[serve_oft_xarm] missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${REALWORLD_ROOT}/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

cmd=(
    "${PYTHON}"
    vla-scripts/deploy.py
    --host "${HOST}"
    --port "${PORT}"
    --pretrained_checkpoint "${CHECKPOINT}"
    --base_checkpoint "${BASE_CHECKPOINT}"
    --use_l1_regression "${USE_L1_REGRESSION}"
    --use_diffusion "${USE_DIFFUSION}"
    --use_film "${USE_FILM}"
    --num_images_in_input "${NUM_IMAGES_IN_INPUT}"
    --use_proprio "${USE_PROPRIO}"
    --center_crop "${CENTER_CROP}"
    --lora_rank "${LORA_RANK}"
    --unnorm_key "${UNNORM_KEY}"
    --load_in_8bit "${LOAD_IN_8BIT}"
    --load_in_4bit "${LOAD_IN_4BIT}"
)

echo "============================================"
echo "  OpenVLA-OFT xArm Server"
echo "============================================"
echo "  checkpoint: ${CHECKPOINT}"
echo "  base:       ${BASE_CHECKPOINT}"
echo "  endpoint:   http://${HOST}:${PORT}/act"
echo "  unnorm_key: ${UNNORM_KEY}"
echo "  images:     ${NUM_IMAGES_IN_INPUT}"
echo "  proprio:    ${USE_PROPRIO}"
echo "============================================"
printf '[serve_oft_xarm] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN,,}" == "true" ]]; then
    exit 0
fi

exec "${cmd[@]}"
