#!/usr/bin/env bash
set -euo pipefail

# OpenVLA-OFT xArm serving launcher. Serves a MERGED checkpoint via deploy.py.
# Common usage:
#   TASK=setting1 ./serve_oft_xarm.sh
#   TASK=setting2 PORT=8778 ./serve_oft_xarm.sh
#   CHECKPOINT=/path/to/merged_ckpt ./serve_oft_xarm.sh
#   DRY_RUN=true TASK=setting1 ./serve_oft_xarm.sh

#########################
# User-facing settings
#########################

# setting1: cube -> plastic cup | setting2: cup stacking
# The served model must match the physical scene AND the client --prompt.
TASK="${TASK:-setting2}"

# Merged checkpoint dir (must contain config.json + model safetensors).
# Leave empty to use the TASK default under merged_public_checkpoints/.
CHECKPOINT="${CHECKPOINT:-}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8777}"

# Model options — keep in sync with how the checkpoint was trained.
USE_L1_REGRESSION="${USE_L1_REGRESSION:-True}"
USE_DIFFUSION="${USE_DIFFUSION:-False}"
USE_FILM="${USE_FILM:-False}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"
USE_PROPRIO="${USE_PROPRIO:-True}"
CENTER_CROP="${CENTER_CROP:-True}"
LOAD_IN_8BIT="${LOAD_IN_8BIT:-False}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-False}"
UNNORM_KEY="${UNNORM_KEY:-utokyo_xarm_pick_and_place_converted_externally_to_rlds}"

DRY_RUN="${DRY_RUN:-false}"

#################
# Internal wiring
#################

REALWORLD_ROOT="${REALWORLD_ROOT:-/workspace/kaixi/RealWorld}"
REPO_DIR="${REPO_DIR:-/workspace/kaixi/openvla-oft}"
CONDA_ENV="${CONDA_ENV:-/workspace/kaixi/.conda/envs/openvla-oft}"
PYTHON="${PYTHON:-${CONDA_ENV}/bin/python}"

case "${TASK}" in
    setting1)
        DEFAULT_CHECKPOINT="${REALWORLD_ROOT}/openvla_oft_runs/merged_public_checkpoints/openvla-oft_setting1"
        EXAMPLE_PROMPT="put the red cube into the plastic cup"
        ;;
    setting2)
        DEFAULT_CHECKPOINT="${REALWORLD_ROOT}/openvla_oft_runs/merged_public_checkpoints/openvla-oft_setting2"
        EXAMPLE_PROMPT="stack the red cup on top of the green cup"
        ;;
    *)
        echo "[serve_oft_xarm] TASK must be setting1 or setting2" >&2
        exit 2
        ;;
esac
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "[serve_oft_xarm] missing python: ${PYTHON}" >&2
    exit 1
fi

if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "[serve_oft_xarm] missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

if [[ ! -f "${CHECKPOINT}/config.json" ]]; then
    echo "[serve_oft_xarm] ${CHECKPOINT} is not a merged checkpoint (no config.json)." >&2
    echo "[serve_oft_xarm] merge the LoRA run first:" >&2
    echo "  ${PYTHON} ${REALWORLD_ROOT}/merge_oft_lora_to_base.py --checkpoint-dir <lora_run_dir> --output-dir <merged_dir>" >&2
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
    --use_l1_regression "${USE_L1_REGRESSION}"
    --use_diffusion "${USE_DIFFUSION}"
    --use_film "${USE_FILM}"
    --num_images_in_input "${NUM_IMAGES_IN_INPUT}"
    --use_proprio "${USE_PROPRIO}"
    --center_crop "${CENTER_CROP}"
    --unnorm_key "${UNNORM_KEY}"
    --load_in_8bit "${LOAD_IN_8BIT}"
    --load_in_4bit "${LOAD_IN_4BIT}"
)

echo "============================================"
echo "  OpenVLA-OFT xArm Server"
echo "============================================"
echo "  task:       ${TASK}"
echo "  checkpoint: ${CHECKPOINT}"
echo "  endpoint:   http://${HOST}:${PORT}/act"
echo "  unnorm_key: ${UNNORM_KEY}"
echo "  images:     ${NUM_IMAGES_IN_INPUT}"
echo "  proprio:    ${USE_PROPRIO}"
echo "  prompt e.g. \"${EXAMPLE_PROMPT}\""
echo "  (client --prompt must be one of this task's training instructions)"
echo "============================================"
printf '[serve_oft_xarm] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN,,}" == "true" ]]; then
    exit 0
fi

exec "${cmd[@]}"
