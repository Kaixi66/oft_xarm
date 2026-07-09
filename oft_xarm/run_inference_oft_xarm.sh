#!/usr/bin/env bash
set -euo pipefail

CLIENT_PYTHON="${CLIENT_PYTHON:-/home/zheyu/code/openpi_xarm/.venv/bin/python}"

# setting1 eg: put the red cube into the plastic cup
# setting2 eg: stack the blue cup on top of the red cup  (RGB order)
INSTRUCTION="${INSTRUCTION:-stack the blue cup on top of the red cup}"

# Old data-collection reset: 6 joint angles in degrees.
RESET_POSITION_DEG="${RESET_POSITION_DEG:-55.399232 7.733498 -48.980042 -1.039517 -57.38115 -0.614669}"
ACTION_HZ="${ACTION_HZ:-10.0}"
SERVO_HZ="${SERVO_HZ:-100.0}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-10}"
PROPRIO_DIM="${PROPRIO_DIM:-6}"
MAX_STEPS="${MAX_STEPS:-30000}"
SPEED_SCALE="${SPEED_SCALE:-1.0}"
MAX_DELTA_MM="${MAX_DELTA_MM:-200.0}"
MAX_DELTA_RAD="${MAX_DELTA_RAD:-1.0}"
ASYNC_REQUERY="${ASYNC_REQUERY:-false}"
OVERLAP_K="${OVERLAP_K:-5}"
RESET_SPEED="${RESET_SPEED:-30.0}"
RESET_PAUSE="${RESET_PAUSE:-2.0}"
RESET_TIMEOUT="${RESET_TIMEOUT:-15.0}"
RESET_GRIPPER_POS="${RESET_GRIPPER_POS:-}"
RESET_TRIGGER_FILE="${RESET_TRIGGER_FILE:-/tmp/oft_xarm_reset}"
GRIPPER_OPEN_HOLD="${GRIPPER_OPEN_HOLD:-2.8}"
GRIPPER_CLOSE_HOLD="${GRIPPER_CLOSE_HOLD:-1.6}"
DISABLE_GRIPPER="${DISABLE_GRIPPER:-false}"
DEBUG_IMAGE_DIR="${DEBUG_IMAGE_DIR:-debug_images/inference_run}"
DEBUG_IMAGE_EVERY="${DEBUG_IMAGE_EVERY:-}"
LOG_FILE="${LOG_FILE:-inference_oft.log}"
TEE_LOG="${TEE_LOG:-true}"
LOG_ACTION_CHUNKS="${LOG_ACTION_CHUNKS:-true}"

if [[ ! -x "${CLIENT_PYTHON}" ]]; then
    echo "[run_inference_oft_xarm] missing client python: ${CLIENT_PYTHON}" >&2
    exit 1
fi

cd "$(dirname "$0")"

if [[ -z "${INSTRUCTION}" ]]; then
    echo "[run_inference_oft_xarm] INSTRUCTION cannot be empty" >&2
    exit 1
fi

cmd=(
    "${CLIENT_PYTHON}"
    inference_oft_xarm.py
    --task custom
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
    --reset-speed "${RESET_SPEED}"
    --reset-pause "${RESET_PAUSE}"
    --reset-timeout "${RESET_TIMEOUT}"
    --reset-trigger-file "${RESET_TRIGGER_FILE}"
    --gripper-open-hold "${GRIPPER_OPEN_HOLD}"
    --gripper-close-hold "${GRIPPER_CLOSE_HOLD}"
)

if [[ -n "${RESET_POSITION_DEG}" ]]; then
    read -r -a reset_position_args <<< "${RESET_POSITION_DEG}"
    if [[ "${#reset_position_args[@]}" -ne 6 ]]; then
        echo "[run_inference_oft_xarm] RESET_POSITION_DEG must contain exactly 6 degree values" >&2
        exit 1
    fi
    cmd+=(--reset-position-deg "${reset_position_args[@]}")
fi

if [[ -n "${RESET_GRIPPER_POS}" ]]; then
    cmd+=(--reset-gripper-pos "${RESET_GRIPPER_POS}")
fi

if [[ "${ASYNC_REQUERY,,}" == "true" ]]; then
    cmd+=(--async-requery)
fi

if [[ "${DISABLE_GRIPPER,,}" == "true" ]]; then
    cmd+=(--disable-gripper)
fi

if [[ -n "${DEBUG_IMAGE_EVERY}" ]]; then
    if [[ -z "${DEBUG_IMAGE_DIR}" ]]; then
        echo "[run_inference_oft_xarm] DEBUG_IMAGE_DIR cannot be empty when DEBUG_IMAGE_EVERY is set" >&2
        exit 1
    fi
    cmd+=(
        --debug-image-dir "${DEBUG_IMAGE_DIR}"
        --debug-image-every "${DEBUG_IMAGE_EVERY}"
    )
fi

if [[ "${LOG_ACTION_CHUNKS,,}" == "true" ]]; then
    cmd+=(--log-action-chunks)
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
