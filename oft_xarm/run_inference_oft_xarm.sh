#!/usr/bin/env bash
set -euo pipefail

CLIENT_PYTHON="${CLIENT_PYTHON:-/home/zheyu/code/openpi_xarm/.venv/bin/python}"
# setting1: put the red cube into the plastic cup
# paper setting2: stack the red cup on top of the green cup
INSTRUCTION="${INSTRUCTION:-stack the red cup on top of the green cup}"
ACTION_HZ="${ACTION_HZ:-25.0}"
SERVO_HZ="${SERVO_HZ:-100.0}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-25}"
PROPRIO_DIM="${PROPRIO_DIM:-6}"
MAX_STEPS="${MAX_STEPS:-30000}"
SPEED_SCALE="${SPEED_SCALE:-1.0}"
MAX_DELTA_MM="${MAX_DELTA_MM:-200.0}"
MAX_DELTA_RAD="${MAX_DELTA_RAD:-1.0}"
ASYNC_REQUERY="${ASYNC_REQUERY:-false}"
OVERLAP_K="${OVERLAP_K:-5}"
GRIPPER_OPEN_HOLD="${GRIPPER_OPEN_HOLD:-2.8}"
GRIPPER_CLOSE_HOLD="${GRIPPER_CLOSE_HOLD:-1.6}"
DISABLE_GRIPPER="${DISABLE_GRIPPER:-false}"
DEBUG_IMAGE_DIR="${DEBUG_IMAGE_DIR:-debug_images/inference_run}"
DEBUG_IMAGE_EVERY="${DEBUG_IMAGE_EVERY:-10}"
LOG_FILE="${LOG_FILE:-inference_oft.log}"
TEE_LOG="${TEE_LOG:-true}"

if [[ ! -x "${CLIENT_PYTHON}" ]]; then
    echo "[run_inference_oft_xarm] missing client python: ${CLIENT_PYTHON}" >&2
    exit 1
fi

cd "$(dirname "$0")"

cmd=(
    "${CLIENT_PYTHON}"
    inference_oft_xarm.py
    --instruction "${INSTRUCTION}"
    --action-hz "${ACTION_HZ}"
    --servo-hz "${SERVO_HZ}"
    --num-open-loop-steps "${NUM_OPEN_LOOP_STEPS}"
    --proprio-dim "${PROPRIO_DIM}"
    --max-steps "${MAX_STEPS}"
    --speed-scale "${SPEED_SCALE}"
    --max-delta-mm "${MAX_DELTA_MM}"
    --max-delta-rad "${MAX_DELTA_RAD}"
    --overlap-k "${OVERLAP_K}"
    --gripper-open-hold "${GRIPPER_OPEN_HOLD}"
    --gripper-close-hold "${GRIPPER_CLOSE_HOLD}"
)

if [[ "${ASYNC_REQUERY,,}" == "true" ]]; then
    cmd+=(--async-requery)
fi

if [[ "${DISABLE_GRIPPER,,}" == "true" ]]; then
    cmd+=(--disable-gripper)
fi

if [[ -n "${DEBUG_IMAGE_DIR}" ]]; then
    cmd+=(
        --debug-image-dir "${DEBUG_IMAGE_DIR}"
        --debug-image-every "${DEBUG_IMAGE_EVERY}"
    )
fi

cmd+=("$@")

for arg in "$@"; do
    if [[ "${arg}" == "-h" || "${arg}" == "--help" ]]; then
        exec "${cmd[@]}"
    fi
done

if [[ "${TEE_LOG,,}" == "true" ]]; then
    export PYTHONUNBUFFERED=1
    exec "${cmd[@]}" 2>&1 | tee "${LOG_FILE}"
fi

exec "${cmd[@]}"
