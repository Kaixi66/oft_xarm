#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLIENT_PYTHON="${CLIENT_PYTHON:-/home/zheyu/code/openpi_xarm/.venv/bin/python}"
ACTIVE_TASK_FILE="${ACTIVE_TASK_FILE:-${SCRIPT_DIR}/.runtime/active_oft_task.env}"

# serve_oft_xarm.sh writes the selected task here after validating its
# checkpoint. An explicit OFT_TASK_OVERRIDE remains available for automation.
if [[ -f "${ACTIVE_TASK_FILE}" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "${ACTIVE_TASK_FILE}"
    set +a
fi
OFT_TASK="${OFT_TASK_OVERRIDE:-${OFT_TASK:-}}"
if [[ -z "${OFT_TASK}" ]]; then
    echo "[run_inference_oft_xarm] no active task; start ./serve_oft_xarm.sh first" >&2
    exit 2
fi

case "${OFT_TASK}" in
    put-blue-bowl-in-second-drawer)
        TASK_INSTRUCTION="put the blue bowl in the second drawer"
        TASK_MAX_STEPS=650
        ;;
    erase-circle-from-whiteboard)
        TASK_INSTRUCTION="erase the circle from the whiteboard"
        TASK_MAX_STEPS=1950
        ;;
    *)
        echo "[run_inference_oft_xarm] unsupported OFT_TASK=${OFT_TASK}" >&2
        exit 2
        ;;
esac
INSTRUCTION="${INSTRUCTION:-${TASK_INSTRUCTION}}"
if [[ "${INSTRUCTION}" != "${TASK_INSTRUCTION}" ]]; then
    echo "[run_inference_oft_xarm] instruction/task mismatch for ${OFT_TASK}" >&2
    echo "  expected: ${TASK_INSTRUCTION}" >&2
    echo "  got:      ${INSTRUCTION}" >&2
    exit 2
fi

# Proven UF850 data-collection reset: 6 joint angles in degrees.
RESET_POSITION_DEG="${RESET_POSITION_DEG:-55.399232 7.733498 -48.980042 -1.039517 -57.38115 -0.614669}"
ACTION_HZ="${ACTION_HZ:-10.0}"
SERVO_HZ="${SERVO_HZ:-100.0}"
NUM_OPEN_LOOP_STEPS="${NUM_OPEN_LOOP_STEPS:-8}"
PROPRIO_DIM="${PROPRIO_DIM:-6}"
MAX_STEPS="${MAX_STEPS:-${TASK_MAX_STEPS}}"
SPEED_SCALE="${SPEED_SCALE:-1.0}"
MAX_DELTA_MM="${MAX_DELTA_MM:-200.0}"
MAX_DELTA_RAD="${MAX_DELTA_RAD:-1.0}"
ASYNC_REQUERY="${ASYNC_REQUERY:-false}"
OVERLAP_K="${OVERLAP_K:-5}"
RESET_SPEED="${RESET_SPEED:-20.0}"
RESET_PAUSE="${RESET_PAUSE:-2.0}"
RESET_TIMEOUT="${RESET_TIMEOUT:-30.0}"
RESET_GRIPPER_POS="${RESET_GRIPPER_POS:-}"
RESET_TRIGGER_FILE="${RESET_TRIGGER_FILE:-/tmp/oft_xarm_reset}"
GRIPPER_OPEN_HOLD="${GRIPPER_OPEN_HOLD:-2.8}"
GRIPPER_CLOSE_HOLD="${GRIPPER_CLOSE_HOLD:-1.6}"
DISABLE_GRIPPER="${DISABLE_GRIPPER:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/outputs}"
RECORD_VIDEO="${RECORD_VIDEO:-true}"
VIDEO_DIR="${VIDEO_DIR:-${OUTPUT_DIR}/videos}"
VIDEO_FPS="${VIDEO_FPS:-30}"
SAVE_VIDEO_FRAMES="${SAVE_VIDEO_FRAMES:-true}"
FRAME_JPEG_QUALITY="${FRAME_JPEG_QUALITY:-95}"
DEBUG_IMAGE_DIR="${DEBUG_IMAGE_DIR:-debug_images/inference_run}"
DEBUG_IMAGE_EVERY="${DEBUG_IMAGE_EVERY:-}"
LOG_FILE="${LOG_FILE:-inference_oft.log}"
TEE_LOG="${TEE_LOG:-true}"
LOG_ACTION_CHUNKS="${LOG_ACTION_CHUNKS:-true}"

# Exact geometry recorded in both supported CASE-Lab training manifests.
CAMERA_WIDTH="${CAMERA_WIDTH:-960}"
CAMERA_HEIGHT="${CAMERA_HEIGHT:-540}"
CAMERA_FPS="${CAMERA_FPS:-60}"
CAMERA_WARMUP_FRAMES="${CAMERA_WARMUP_FRAMES:-30}"
EXTERNAL_CROP_LEFT="${EXTERNAL_CROP_LEFT:-270}"
WRIST_CROP_LEFT="${WRIST_CROP_LEFT:-380}"
CROP_SIZE="${CROP_SIZE:-540}"

if [[ ! -x "${CLIENT_PYTHON}" ]]; then
    echo "[run_inference_oft_xarm] missing client python: ${CLIENT_PYTHON}" >&2
    exit 1
fi

cd "${SCRIPT_DIR}"

if [[ -z "${INSTRUCTION}" ]]; then
    echo "[run_inference_oft_xarm] INSTRUCTION cannot be empty" >&2
    exit 1
fi

cmd=(
    "${CLIENT_PYTHON}"
    "${SCRIPT_DIR}/inference_oft_xarm.py"
    --task "${OFT_TASK}"
    --instruction "${INSTRUCTION}"
    --action-hz "${ACTION_HZ}"
    --servo-hz "${SERVO_HZ}"
    --num-open-loop-steps "${NUM_OPEN_LOOP_STEPS}"
    --proprio-dim "${PROPRIO_DIM}"
    --max-steps "${MAX_STEPS}"
    --speed-scale "${SPEED_SCALE}"
    --max-delta-mm "${MAX_DELTA_MM}"
    --max-delta-rad "${MAX_DELTA_RAD}"
    --camera-width "${CAMERA_WIDTH}"
    --camera-height "${CAMERA_HEIGHT}"
    --camera-fps "${CAMERA_FPS}"
    --camera-warmup-frames "${CAMERA_WARMUP_FRAMES}"
    --external-crop-left "${EXTERNAL_CROP_LEFT}"
    --wrist-crop-left "${WRIST_CROP_LEFT}"
    --crop-size "${CROP_SIZE}"
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

if [[ "${RECORD_VIDEO,,}" == "true" ]]; then
    cmd+=(--record-video --video-dir "${VIDEO_DIR}" --video-fps "${VIDEO_FPS}")
    if [[ "${SAVE_VIDEO_FRAMES,,}" == "true" ]]; then
        cmd+=(--save-video-frames --frame-jpeg-quality "${FRAME_JPEG_QUALITY}")
    fi
elif [[ "${SAVE_VIDEO_FRAMES,,}" == "true" ]]; then
    echo "[run_inference_oft_xarm] SAVE_VIDEO_FRAMES=true requires RECORD_VIDEO=true" >&2
    exit 2
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
