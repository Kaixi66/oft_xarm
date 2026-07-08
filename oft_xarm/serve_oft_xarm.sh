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
TASK="${TASK:-${SETTING:-setting2}}"

# Merged checkpoint dir (must contain config.json + model safetensors).
# Leave empty to use the TASK default under CHECKPOINT_ROOT.
CHECKPOINT="${CHECKPOINT:-}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8777}"

# Model options — keep in sync with how the checkpoint was trained.
USE_L1_REGRESSION="${USE_L1_REGRESSION:-True}"
USE_DIFFUSION="${USE_DIFFUSION:-False}"
USE_FILM="${USE_FILM:-}"
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

REALWORLD_ROOT="${REALWORLD_ROOT:-/home/zheyu/kaixi/RealWorld}"
REPO_DIR="${REPO_DIR:-/home/zheyu/0517_lab_xarm/openvla-oft}"
CONDA_ENV="${CONDA_ENV:-/home/zheyu/miniforge3/envs/openvla-oft-thor}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/home/zheyu/kaixi/RealWorld-OFT-merged-checkpoints}"
PYTHON="${PYTHON:-${CONDA_ENV}/bin/python}"

case "${TASK}" in
    setting1)
        DEFAULT_CHECKPOINT="${CHECKPOINT_ROOT}/openvla-oft_setting1"
        DEFAULT_USE_FILM="False"
        EXAMPLE_PROMPT="put the red cube into the plastic cup"
        ;;
    setting2|paper_setting2)
        DEFAULT_CHECKPOINT="${CHECKPOINT_ROOT}/AAyano_oft_setting2_stacking_paper_10k"
        DEFAULT_USE_FILM="True"
        EXAMPLE_PROMPT="stack the red cup on top of the green cup"
        ;;
    setting2_legacy)
        DEFAULT_CHECKPOINT="${CHECKPOINT_ROOT}/openvla-oft_setting2"
        DEFAULT_USE_FILM="False"
        EXAMPLE_PROMPT="stack the red cup on top of the green cup"
        ;;
    *)
        echo "[serve_oft_xarm] TASK must be setting1, setting2, paper_setting2, or setting2_legacy" >&2
        exit 2
        ;;
esac
CHECKPOINT="${CHECKPOINT:-${DEFAULT_CHECKPOINT}}"
USE_FILM="${USE_FILM:-${DEFAULT_USE_FILM}}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "[serve_oft_xarm] missing python: ${PYTHON}" >&2
    exit 1
fi

if [[ ! -d "${REPO_DIR}" ]]; then
    echo "[serve_oft_xarm] missing openvla-oft repo: ${REPO_DIR}" >&2
    exit 1
fi

if [[ ! -d "${CHECKPOINT}" ]]; then
    echo "[serve_oft_xarm] missing checkpoint: ${CHECKPOINT}" >&2
    exit 1
fi

required_checkpoint_files=(
    "config.json"
    "dataset_statistics.json"
    "model.safetensors.index.json"
    "action_head--latest_checkpoint.pt"
    "proprio_projector--latest_checkpoint.pt"
)

for required_file in "${required_checkpoint_files[@]}"; do
    if [[ ! -f "${CHECKPOINT}/${required_file}" ]]; then
        echo "[serve_oft_xarm] checkpoint missing ${required_file}: ${CHECKPOINT}" >&2
        echo "[serve_oft_xarm] merge the LoRA run first:" >&2
        echo "  ${PYTHON} ${REALWORLD_ROOT}/merge_oft_lora_to_base.py --checkpoint-dir <lora_run_dir> --output-dir <merged_dir>" >&2
        exit 1
    fi
done

"${PYTHON}" - "${CHECKPOINT}" "${USE_FILM}" <<'PY'
import json
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1])
requested_film = sys.argv[2].lower() in {"1", "true", "yes", "y"}
metadata_path = checkpoint / "oft_training_config.json"
vision_backbones = sorted(checkpoint.glob("vision_backbone--*.pt"))

if metadata_path.exists():
    metadata = json.loads(metadata_path.read_text())
    trained_film = bool(metadata.get("use_film", False))
    if trained_film != requested_film:
        raise SystemExit(
            f"[serve_oft_xarm] FiLM mismatch: checkpoint use_film={trained_film}, "
            f"but USE_FILM={requested_film}. Checkpoint: {checkpoint}"
        )
else:
    print(f"[serve_oft_xarm] warning: no oft_training_config.json in {checkpoint}; checking sidecars only")

if requested_film and not vision_backbones:
    raise SystemExit(
        f"[serve_oft_xarm] USE_FILM=True but no vision_backbone--*.pt found in {checkpoint}. "
        "Use a FiLM-trained merged checkpoint."
    )
if not requested_film and vision_backbones:
    raise SystemExit(
        f"[serve_oft_xarm] USE_FILM=False but FiLM vision backbone sidecar exists in {checkpoint}. "
        "Set USE_FILM=True or use a non-FiLM checkpoint."
    )
PY

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/home/zheyu/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false
export PATH="/usr/local/cuda/bin:${PATH}"
export TRITON_PTXAS_PATH="${TRITON_PTXAS_PATH:-/usr/local/cuda/bin/ptxas}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

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
echo "  repo:       ${REPO_DIR}"
echo "  python:     ${PYTHON}"
echo "  checkpoint: ${CHECKPOINT}"
echo "  endpoint:   http://${HOST}:${PORT}/act"
echo "  unnorm_key: ${UNNORM_KEY}"
echo "  use_film:   ${USE_FILM}"
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
