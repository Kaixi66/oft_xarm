#!/usr/bin/env bash
set -euo pipefail

# OpenVLA-OFT launcher for the local real-world LeRobot -> RLDS XArm data.
# Common usage:
#   TASK=setting1 ./train_oft_realworld.sh
#   TASK=setting2 GPUS=0,1 BATCH_SIZE=16 ./train_oft_realworld.sh
#   DRY_RUN=true TASK=setting1 ./train_oft_realworld.sh

#########################
# User-facing settings
#########################

# setting1: cube -> plastic cup
# setting2: cup stacking
# merged: the earlier combined dataset
TASK="${TASK:-setting2}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUMULATION_STEPS="${GRAD_ACCUMULATION_STEPS:-1}"
LEARNING_RATE="${LEARNING_RATE:-5e-4}"
LORA_RANK="${LORA_RANK:-32}"
MAX_STEPS="${MAX_STEPS:-100005}"
NUM_STEPS_BEFORE_DECAY="${NUM_STEPS_BEFORE_DECAY:-50000}"
SAVE_FREQ="${SAVE_FREQ:-10000}"
SAVE_LATEST_CHECKPOINT_ONLY="${SAVE_LATEST_CHECKPOINT_ONLY:-True}"
SHUFFLE_BUFFER_SIZE="${SHUFFLE_BUFFER_SIZE:-100000}"

USE_L1_REGRESSION="${USE_L1_REGRESSION:-True}"
USE_DIFFUSION="${USE_DIFFUSION:-False}"
USE_FILM="${USE_FILM:-True}"
NUM_IMAGES_IN_INPUT="${NUM_IMAGES_IN_INPUT:-2}"
USE_PROPRIO="${USE_PROPRIO:-True}"
IMAGE_AUG="${IMAGE_AUG:-True}"
USE_LORA="${USE_LORA:-True}"
MERGE_LORA_DURING_TRAINING="${MERGE_LORA_DURING_TRAINING:-False}"

USE_VAL_SET="${USE_VAL_SET:-False}"
VAL_FREQ="${VAL_FREQ:-2500}"
VAL_TIME_LIMIT="${VAL_TIME_LIMIT:-180}"

# Leave empty to use the per-TASK default run name below. A hardcoded default
# here would ignore TASK and make e.g. TASK=setting2 write into setting1's run.
RUN_NAME="${RUN_NAME:-}"

WANDB_ENTITY="${WANDB_ENTITY:-optimal-training-strategy}"
WANDB_PROJECT="${WANDB_PROJECT:-RealWorld}"
WANDB_LOG_FREQ="${WANDB_LOG_FREQ:-10}"
export WANDB_MODE="${WANDB_MODE:-online}"

DRY_RUN="${DRY_RUN:-false}"
BACKGROUND="${BACKGROUND:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

#################
# Internal wiring
#################

REALWORLD_ROOT="${REALWORLD_ROOT:-/workspace/kaixi/RealWorld}"
REPO_DIR="${REPO_DIR:-/workspace/kaixi/openvla-oft}"
CONDA_ENV="${CONDA_ENV:-/workspace/kaixi/.conda/envs/openvla-oft}"
CONDA_SH="${CONDA_SH:-/workspace/ghsun/miniconda3/etc/profile.d/conda.sh}"
ACTIVATE_CONDA="${ACTIVATE_CONDA:-true}"
NPROC_PER_NODE="${NPROC_PER_NODE:-auto}"

DATASET_NAME="${DATASET_NAME:-utokyo_xarm_pick_and_place_converted_externally_to_rlds}"
VLA_PATH="${VLA_PATH:-openvla/openvla-7b}"
RUN_ROOT_DIR="${RUN_ROOT_DIR:-${REALWORLD_ROOT}/openvla_oft_runs/checkpoints}"
LOG_DIR="${LOG_DIR:-${REALWORLD_ROOT}/openvla_oft_runs/logs}"

case "${TASK}" in
    setting1)
        DEFAULT_DATA_ROOT_DIR="${REALWORLD_ROOT}/rlds_data_setting1"
        DEFAULT_EXP_PREFIX="xarm_setting1_oft"
        DEFAULT_RUN_NAME="oft_setting1_paper"
        ;;
    setting2)
        DEFAULT_DATA_ROOT_DIR="${REALWORLD_ROOT}/rlds_data_setting2"
        DEFAULT_EXP_PREFIX="xarm_setting2_oft"
        DEFAULT_RUN_NAME="oft_setting2_paper"
        ;;
    merged)
        DEFAULT_DATA_ROOT_DIR="${REALWORLD_ROOT}/rlds_data"
        DEFAULT_EXP_PREFIX="xarm_merged_oft"
        DEFAULT_RUN_NAME="oft_merged_paper"
        ;;
    *)
        echo "[train_oft_realworld] TASK must be one of: setting1, setting2, merged" >&2
        exit 2
        ;;
esac

DATA_ROOT_DIR="${DATA_ROOT_DIR:-${DEFAULT_DATA_ROOT_DIR}}"

############################
# Derived settings
############################

current_time="${CURRENT_TIME:-$(date +%Y%m%d_%H%M%S)}"
gpu_list="${GPUS// /}"
IFS=',' read -r -a gpu_array <<< "${gpu_list}"
if [[ "${NPROC_PER_NODE}" == "auto" ]]; then
    NPROC_PER_NODE="${#gpu_array[@]}"
fi

if [[ -n "${RUN_NAME}" ]]; then
    EXP_NAME="${RUN_NAME}"
else
    EXP_NAME="${EXP_NAME:-${DEFAULT_RUN_NAME}}"
fi
log_file="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"
dataset_dir="${DATA_ROOT_DIR}/${DATASET_NAME}"

if [[ ! -d "${dataset_dir}" ]]; then
    echo "[train_oft_realworld] missing TFDS dataset: ${dataset_dir}" >&2
    echo "[train_oft_realworld] run the converter first, for example:" >&2
    echo "  /workspace/kaixi/.conda/envs/vla-adapter/bin/python ${REALWORLD_ROOT}/lerobot_to_rlds.py --overwrite --dataset-root /root/angli/hf_cache/lerobot/lab/xarm_setting1_51 --tfds-data-dir ${REALWORLD_ROOT}/rlds_data_setting1" >&2
    exit 1
fi

if [[ "${ACTIVATE_CONDA}" == "true" ]]; then
    source "${CONDA_SH}"
    conda activate "${CONDA_ENV}"
fi

cd "${REPO_DIR}"
mkdir -p "${LOG_DIR}" "${RUN_ROOT_DIR}" "${REALWORLD_ROOT}/.cache/huggingface"

export CUDA_VISIBLE_DEVICES="${gpu_list}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-${REALWORLD_ROOT}/.cache/huggingface}"
export PYTHONPATH="${REPO_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

torchrun_bin="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
hash -r

cmd=(
    "${torchrun_bin}"
    --standalone
    --nnodes 1
    --nproc-per-node "${NPROC_PER_NODE}"
    vla-scripts/finetune.py
    --vla_path "${VLA_PATH}"
    --data_root_dir "${DATA_ROOT_DIR}"
    --dataset_name "${DATASET_NAME}"
    --run_root_dir "${RUN_ROOT_DIR}"
    --shuffle_buffer_size "${SHUFFLE_BUFFER_SIZE}"
    --use_l1_regression "${USE_L1_REGRESSION}"
    --use_diffusion "${USE_DIFFUSION}"
    --use_film "${USE_FILM}"
    --num_images_in_input "${NUM_IMAGES_IN_INPUT}"
    --use_proprio "${USE_PROPRIO}"
    --batch_size "${BATCH_SIZE}"
    --grad_accumulation_steps "${GRAD_ACCUMULATION_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --num_steps_before_decay "${NUM_STEPS_BEFORE_DECAY}"
    --max_steps "${MAX_STEPS}"
    --use_val_set "${USE_VAL_SET}"
    --val_freq "${VAL_FREQ}"
    --val_time_limit "${VAL_TIME_LIMIT}"
    --save_freq "${SAVE_FREQ}"
    --save_latest_checkpoint_only "${SAVE_LATEST_CHECKPOINT_ONLY}"
    --image_aug "${IMAGE_AUG}"
    --use_lora "${USE_LORA}"
    --lora_rank "${LORA_RANK}"
    --merge_lora_during_training "${MERGE_LORA_DURING_TRAINING}"
    --wandb_entity "${WANDB_ENTITY}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_log_freq "${WANDB_LOG_FREQ}"
    --run_id_override "${EXP_NAME}"
)

if [[ -n "${EXTRA_ARGS}" ]]; then
    # shellcheck disable=SC2206
    extra_args_array=(${EXTRA_ARGS})
    cmd+=("${extra_args_array[@]}")
fi

cat <<EOF
[train_oft_realworld] task: ${TASK}
[train_oft_realworld] dataset: ${DATASET_NAME}
[train_oft_realworld] data_root: ${DATA_ROOT_DIR}
[train_oft_realworld] gpus: ${CUDA_VISIBLE_DEVICES}
[train_oft_realworld] nproc_per_node: ${NPROC_PER_NODE}
[train_oft_realworld] effective_batch: $((NPROC_PER_NODE * BATCH_SIZE * GRAD_ACCUMULATION_STEPS))
[train_oft_realworld] vla_path: ${VLA_PATH}
[train_oft_realworld] use_film: ${USE_FILM}
[train_oft_realworld] num_images_in_input: ${NUM_IMAGES_IN_INPUT}
[train_oft_realworld] use_proprio: ${USE_PROPRIO}
[train_oft_realworld] expected_constants: XARM chunk=25 action_dim=7 proprio_dim=6
[train_oft_realworld] save_latest_checkpoint_only: ${SAVE_LATEST_CHECKPOINT_ONLY}
[train_oft_realworld] merge_lora_during_training: ${MERGE_LORA_DURING_TRAINING}
[train_oft_realworld] wandb: ${WANDB_ENTITY}/${WANDB_PROJECT} (${WANDB_MODE})
[train_oft_realworld] run_name: ${EXP_NAME}
[train_oft_realworld] wandb_run_name: ft+${EXP_NAME}
[train_oft_realworld] run_root: ${RUN_ROOT_DIR}
[train_oft_realworld] log_file: ${log_file}
EOF

printf '[train_oft_realworld] command:'
printf ' %q' "${cmd[@]}"
printf '\n'

if [[ "${DRY_RUN}" == "true" ]]; then
    exit 0
fi

if [[ "${BACKGROUND}" == "true" ]]; then
    nohup "${cmd[@]}" > "${log_file}" 2>&1 &
    pid="$!"
    echo "[train_oft_realworld] started in background: pid=${pid}"
    echo "[train_oft_realworld] tail log: tail -f ${log_file}"
else
    "${cmd[@]}" 2>&1 | tee "${log_file}"
fi
