#!/usr/bin/env python3
"""xArm6 + OpenVLA-OFT deployment client.

This script reuses the lab xArm hardware loop:
- two RealSense RGB streams
- 6-dim xArm joint state in radians
- 25-step open-loop action chunks by default
- interpolated Cartesian servo execution for each action
- basic gripper control from action[6]

The model call is adapted for OpenVLA-OFT's HTTP /act server. Gripper output
action[6] is binarized: positive closes the xArm gripper, non-positive opens it.
"""

import argparse
import collections
import os
import select
import signal
import sys
import termios
import time
import threading
import tty
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    from xarm.wrapper import XArmAPI
except ImportError:
    XArmAPI = None

try:
    import json_numpy

    json_numpy.patch()
except ImportError:
    json_numpy = None

DEFAULT_EXTERNAL_CAM_SERIAL = "215222078407"
DEFAULT_WRIST_CAM_SERIAL = "845112070404"
DEFAULT_RESET_POSITION_DEG = [-87.862963, -11.723519, -70.560039, 2.650331, -56.14253, 180.583577]
DEFAULT_HARDWARE_PYTHON = "/home/zheyu/code/openpi_xarm/.venv/bin/python"

MAX_POS_DELTA_MM = 200.0
MAX_ROT_DELTA_RAD = 1.0

DEG2RAD = np.pi / 180.0


def require_hardware_dependencies() -> None:
    missing = []
    if cv2 is None:
        missing.append("opencv-python")
    if rs is None:
        missing.append("pyrealsense2")
    if XArmAPI is None:
        missing.append("xarm-python-sdk")
    if missing:
        hardware_python = os.environ.get("OFT_XARM_CLIENT_PYTHON", DEFAULT_HARDWARE_PYTHON)
        already_reexeced = os.environ.get("OFT_XARM_REEXECED") == "1"
        current_python = os.path.realpath(sys.executable)
        target_python = os.path.realpath(hardware_python)
        invoked_as_script = os.path.exists(sys.argv[0]) and os.path.realpath(sys.argv[0]) == os.path.realpath(__file__)

        if invoked_as_script and not already_reexeced and os.path.exists(hardware_python) and current_python != target_python:
            print(
                "Wrong Python environment for RealSense/xArm client; "
                f"re-executing with {hardware_python}",
                flush=True,
            )
            os.environ["OFT_XARM_REEXECED"] = "1"
            os.execv(hardware_python, [hardware_python, os.path.abspath(__file__), *sys.argv[1:]])

        raise RuntimeError(
            "Missing hardware runtime dependencies: "
            + ", ".join(missing)
            + f". Current Python: {sys.executable}. "
            + f"Run with {hardware_python}, or use ./run_inference_oft_xarm.sh."
        )


class KeyListener:
    """Non-blocking single-key listener for interactive reset."""

    def __init__(self, reset_key: str = "r", enabled: bool = True):
        self.reset_key = reset_key.lower()
        self.enabled = enabled
        self._pressed = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._old_settings = None

    def start(self) -> None:
        if not self.enabled:
            return
        if not sys.stdin.isatty():
            self.enabled = False
            print("  Keyboard reset disabled: stdin is not a TTY")
            return
        self._old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            ch = sys.stdin.read(1)
            if ch.lower() == self.reset_key:
                self._pressed.set()

    def check_and_clear(self) -> bool:
        if self._pressed.is_set():
            self._pressed.clear()
            return True
        return False

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._old_settings is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)
            self._old_settings = None


class RealsenseCapture:
    """Captures RGB frames from a RealSense camera identified by serial number."""

    def __init__(self, serial: str, width: int = 1920, height: int = 1080, fps: int = 30):
        self.serial = serial
        self.width = width
        self.height = height
        self.pipeline = rs.pipeline()

        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.yuyv, fps)
        self.pipeline.start(config)

        for _ in range(30):
            self.pipeline.wait_for_frames()
        print(f"Camera {serial} ready ({width}x{height} YUYV)")

    def get_frame(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError(f"No color frame from camera {self.serial}")

        raw = np.asanyarray(color_frame.get_data())
        return cv2.cvtColor(
            raw.view(np.uint8).reshape(self.height, self.width, 2),
            cv2.COLOR_YUV2RGB_YUYV,
        )

    def close(self) -> None:
        self.pipeline.stop()
        print(f"Camera {self.serial} stopped")


class OFTActionClient:
    """Small HTTP client for OpenVLA-OFT deploy.py."""

    def __init__(self, endpoint: str, timeout: float):
        self.endpoint = endpoint
        self.timeout = timeout
        self.session = requests.Session()

    def _jsonable(self, value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, dict):
            return {key: self._jsonable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(val) for val in value]
        return value

    def infer(self, observation: dict) -> np.ndarray:
        if json_numpy is not None:
            payload = {"encoded": json_numpy.dumps(observation)}
        else:
            payload = self._jsonable(observation)
        response = self.session.post(self.endpoint, json=payload, timeout=self.timeout)
        response.raise_for_status()

        result = response.json()
        if result == "error":
            raise RuntimeError("OFT server returned error; check server traceback")
        if isinstance(result, str):
            if json_numpy is None:
                raise RuntimeError("OFT server returned encoded numpy JSON, but json_numpy is not installed")
            result = json_numpy.loads(result)

        actions = np.asarray(result, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] < 6:
            raise RuntimeError(f"Expected action chunk shape (T, >=6), got {actions.shape}")

        # Training actions store xyz deltas in centimeters; xArm Cartesian servo uses millimeters.
        actions[:, 0:3] *= 10.0
        return actions


def get_xarm_state_cached(arm: XArmAPI, proprio_dim: int) -> np.ndarray:
    """Read the 6 xArm joint angles in radians."""
    if proprio_dim != 6:
        raise ValueError(
            f"xArm OFT proprio is 6-dim without padding; got --proprio-dim {proprio_dim}. "
            "Use a checkpoint trained with PROPRIO_DIM=6."
        )

    angles_deg = arm.angles
    if angles_deg is None:
        raise RuntimeError("arm.angles returned None (report stream not ready?)")

    return np.asarray(angles_deg[:6], dtype=np.float32) * DEG2RAD


def crop_and_resize(
    image_rgb: np.ndarray,
    crop_size: int = 1080,
    target_size: int = 224,
    crop_mode: str = "center",
) -> np.ndarray:
    """Crop with numpy slicing, then resize with PIL LANCZOS."""
    h, w = image_rgb.shape[:2]

    if crop_mode == "center":
        left = (w - crop_size) // 2
        cropped = image_rgb[0:crop_size, left:left + crop_size]
    elif crop_mode == "right":
        right = w - 80
        left = right - crop_size
        cropped = image_rgb[0:crop_size, left:right]
    elif crop_mode == "right_0":
        right = w
        left = right - crop_size
        cropped = image_rgb[0:crop_size, left:right]
    elif crop_mode == "right_3/4":
        left = w // 4
        cropped = image_rgb[0:h, left:w]
    elif crop_mode == "left_180":
        left = 180
        cropped = image_rgb[0:crop_size, left:left + crop_size]
    elif crop_mode == "left_540":
        left = 540
        cropped = image_rgb[0:crop_size, left:left + crop_size]
    elif crop_mode == "resize_only":
        cropped = image_rgb
    else:
        raise ValueError(f"Invalid crop_mode: {crop_mode}")

    pil_img = Image.fromarray(cropped)
    pil_resized = pil_img.resize((target_size, target_size), Image.LANCZOS)
    return np.array(pil_resized)


def save_debug_image(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image).save(path, quality=95)


class AsyncInferenceWorker:
    """Keeps the action queue filled using background camera capture and OFT inference."""

    def __init__(
        self,
        client: OFTActionClient,
        cam_wrist: RealsenseCapture,
        cam_external: RealsenseCapture,
        prompt: str,
        arm: XArmAPI,
        overlap_k: int,
        num_open_loop_steps: int,
        proprio_dim: int,
        wrist_crop: str,
        external_crop: str,
        debug_image_dir: str,
        debug_image_every: int,
    ):
        self.client = client
        self.cam_wrist = cam_wrist
        self.cam_external = cam_external
        self.prompt = prompt
        self.arm = arm
        self.overlap_k = overlap_k
        self.num_open_loop_steps = num_open_loop_steps
        self.proprio_dim = proprio_dim
        self.wrist_crop = wrist_crop
        self.external_crop = external_crop
        self.debug_image_dir = Path(debug_image_dir).expanduser() if debug_image_dir else None
        self.debug_image_every = max(1, debug_image_every)
        self._debug_capture_count = 0
        if self.debug_image_dir is not None:
            self.debug_image_dir.mkdir(parents=True, exist_ok=True)

        self._cam_pool = ThreadPoolExecutor(max_workers=2)
        self._queue = collections.deque()
        self._lock = threading.Lock()
        # Monotonic count of actions actually popped by the executor; used to
        # measure how many actions ran while an inference was in flight.
        self._popped_total = 0
        # Incremented on every queue flush; a chunk whose observation predates
        # the latest flush (e.g. taken mid-gripper-motion) must be discarded.
        self._flush_epoch = 0
        self._thread = None
        self._running = False

        self.infer_count = 0
        self.last_cam_ms = 0.0
        self.last_infer_ms = 0.0
        self._log_queue = collections.deque()

    def _capture_wrist(self) -> tuple[np.ndarray, np.ndarray]:
        raw = self.cam_wrist.get_frame()
        return raw, crop_and_resize(raw, crop_mode=self.wrist_crop)

    def _capture_ext(self) -> tuple[np.ndarray, np.ndarray]:
        raw = self.cam_external.get_frame()
        return raw, crop_and_resize(raw, crop_mode=self.external_crop)

    def _save_debug_images(
        self,
        raw_wrist: np.ndarray,
        raw_ext: np.ndarray,
        img_wrist: np.ndarray,
        img_ext: np.ndarray,
    ) -> None:
        if self.debug_image_dir is None:
            return
        self._debug_capture_count += 1
        if (self._debug_capture_count - 1) % self.debug_image_every != 0:
            return

        seq = self._debug_capture_count
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        prefix = f"{seq:06d}_{timestamp}"
        save_debug_image(self.debug_image_dir / f"{prefix}_external_raw.jpg", raw_ext)
        save_debug_image(
            self.debug_image_dir / f"{prefix}_external_{self.external_crop}_224.jpg",
            img_ext,
        )
        save_debug_image(self.debug_image_dir / f"{prefix}_wrist_raw.jpg", raw_wrist)
        save_debug_image(
            self.debug_image_dir / f"{prefix}_wrist_{self.wrist_crop}_224.jpg",
            img_wrist,
        )

    def _infer_once(self, state: np.ndarray) -> np.ndarray:
        t0 = time.time()
        fut_w = self._cam_pool.submit(self._capture_wrist)
        fut_e = self._cam_pool.submit(self._capture_ext)
        raw_wrist, img_wrist = fut_w.result()
        raw_ext, img_ext = fut_e.result()
        self.last_cam_ms = (time.time() - t0) * 1000
        self._save_debug_images(raw_wrist, raw_ext, img_wrist, img_ext)

        observation = {
            "full_image": img_ext,
            "wrist_image": img_wrist,
            "state": state,
            "instruction": self.prompt,
        }

        t0 = time.time()
        actions = self.client.infer(observation)
        self.last_infer_ms = (time.time() - t0) * 1000
        return actions

    def run_first_sync(self):
        return self.refill_sync()

    def refill_sync(self):
        state = get_xarm_state_cached(self.arm, self.proprio_dim)
        actions = self._infer_once(state)
        actions = actions[: self.num_open_loop_steps]
        self.infer_count += 1
        with self._lock:
            self._queue.extend(actions)
        return len(actions), self.last_cam_ms, self.last_infer_ms

    def start(self) -> None:
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        while self._running:
            with self._lock:
                qlen = len(self._queue)

            if qlen <= self.overlap_k:
                if self.arm.error_code != 0 or self.arm.state == 4:
                    time.sleep(0.1)
                    continue

                try:
                    with self._lock:
                        popped_before = self._popped_total
                        epoch_before = self._flush_epoch
                    state = get_xarm_state_cached(self.arm, self.proprio_dim)
                    actions = self._infer_once(state)
                    actions = actions[: self.num_open_loop_steps]
                    self.infer_count += 1

                    with self._lock:
                        if self._flush_epoch != epoch_before:
                            self._log_queue.append(
                                f"  INFER #{self.infer_count} | "
                                f"discarded chunk from pre-flush observation | "
                                f"cam={self.last_cam_ms:.0f}ms infer={self.last_infer_ms:.0f}ms"
                            )
                            continue

                        consumed = min(self._popped_total - popped_before, len(actions))
                        remaining = list(self._queue)
                        self._queue.clear()

                        n_overlap = min(len(remaining), len(actions) - consumed)

                        for i in range(n_overlap):
                            w_new = (i + 1) / (n_overlap + 1)
                            blended = (1.0 - w_new) * remaining[i] + w_new * actions[consumed + i]
                            # The gripper command is binarized by sign downstream;
                            # blending -1/+1 across chunks crosses zero and can
                            # trigger it early, so take the newer chunk's value.
                            blended[6:] = actions[consumed + i][6:]
                            remaining[i] = blended

                        self._queue.extend(remaining[:n_overlap])
                        self._queue.extend(actions[consumed + n_overlap:])
                        new_qlen = len(self._queue)

                    self._log_queue.append(
                        f"  INFER #{self.infer_count} | "
                        f"cam={self.last_cam_ms:.0f}ms infer={self.last_infer_ms:.0f}ms | "
                        f"got {len(actions)}, consumed={consumed}, "
                        f"blend={n_overlap}, total Q:{new_qlen}"
                    )
                except Exception as exc:
                    self._log_queue.append(f"  INFER ERROR: {exc}")
                    time.sleep(0.05)
            else:
                time.sleep(0.005)

    def pop_action(self):
        with self._lock:
            if not self._queue:
                return None
            self._popped_total += 1
            return self._queue.popleft()

    def queue_len(self) -> int:
        with self._lock:
            return len(self._queue)

    def drain_logs(self):
        logs = []
        while self._log_queue:
            try:
                logs.append(self._log_queue.popleft())
            except IndexError:
                break
        return logs

    def shutdown(self) -> None:
        self.stop()
        self._cam_pool.shutdown(wait=False)


def servo_hold(arm: XArmAPI, tracked_pose: np.ndarray, servo_dt: float, duration: float) -> None:
    pose_list = tracked_pose.tolist()
    end_time = time.perf_counter() + duration
    while time.perf_counter() < end_time:
        t_h = time.perf_counter()
        arm.set_servo_cartesian(pose_list, is_radian=True)
        elapsed = time.perf_counter() - t_h
        remaining = servo_dt - elapsed
        if remaining > 0:
            time.sleep(remaining)


def init_gripper(arm: XArmAPI, open_pos: int, speed: int) -> float:
    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(speed)
    arm.set_gripper_position(open_pos, wait=True)
    return -1.0


def flush_action_queue(worker: AsyncInferenceWorker) -> int:
    with worker._lock:
        stale = len(worker._queue)
        worker._queue.clear()
        worker._flush_epoch += 1
    return stale


def enter_servo_mode(arm: XArmAPI) -> None:
    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(0.1)
    print("  Entered servo mode (mode=1)")


def reset_to_home(
    arm: XArmAPI,
    worker: AsyncInferenceWorker,
    reset_angles_deg: list[float],
    *,
    reset_speed: float,
    reset_pause: float,
    servo_dt: float,
    dry_run: bool,
    async_requery: bool,
) -> np.ndarray:
    """Move to the configured reset joint pose and return a re-synced TCP pose."""
    print("\n  [RESET] 'R' pressed: stopping policy actions and moving to reset pose...")

    if async_requery:
        worker.stop()

    stale = flush_action_queue(worker)
    print(f"  [RESET] Cleared {stale} queued actions")

    if dry_run:
        print(f"  [RESET] dry-run: would move joints to {reset_angles_deg}")
    else:
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.5)
        code = arm.set_servo_angle(angle=reset_angles_deg, speed=reset_speed, is_radian=False, wait=True)
        if code != 0:
            raise RuntimeError(f"set_servo_angle reset failed: code={code}")
        print(f"  [RESET] Reached reset joint pose: {reset_angles_deg}")

    if reset_pause > 0:
        print(f"  [RESET] Pausing {reset_pause:.1f}s...")
        if dry_run:
            time.sleep(reset_pause)
        else:
            code, pose = arm.get_position(is_radian=True)
            if code == 0:
                servo_hold(arm, np.array(pose[:6], dtype=np.float64), servo_dt, reset_pause)
            else:
                time.sleep(reset_pause)

    code, new_pose = arm.get_position(is_radian=True)
    if code != 0:
        raise RuntimeError(f"get_position failed after reset: code={code}")
    tracked_pose = np.array(new_pose[:6], dtype=np.float64)
    print(f"  [RESET] Tracked pose re-synced: [{', '.join(f'{v:.2f}' for v in tracked_pose)}]")

    if not dry_run:
        enter_servo_mode(arm)

    if async_requery:
        worker.start()

    print("  [RESET] Done. Next step will use a fresh observation/action chunk.\n")
    return tracked_pose


def build_server_endpoint(args) -> str:
    if args.server_url:
        return args.server_url.rstrip("/")
    return f"http://{args.host}:{args.port}/act"


def normalize_prompt(prompt: str) -> str:
    prompt = " ".join(prompt.strip().split())
    if not prompt:
        raise ValueError("--prompt/--instruction cannot be empty")
    return prompt


def warn_if_full_openvla_prompt(prompt: str) -> None:
    lowered = prompt.lower()
    if "what action should the robot take" in lowered or lowered.startswith("in:"):
        print(
            "WARNING: pass only the raw task instruction, not the full OpenVLA prompt. "
            f"Using instruction text as provided: {prompt!r}"
        )


def parse_args():
    parser = argparse.ArgumentParser(description="xArm6 + OpenVLA-OFT HTTP inference client")
    parser.add_argument("--xarm-ip", default="192.168.1.230", help="xArm IP address")
    parser.add_argument(
        "--prompt",
        "--instruction",
        dest="prompt",
        default="put the red cube into the plastic cup",
        help="Raw task instruction, e.g. 'put the red cube into the plastic cup'. Do not include the OpenVLA 'In: ... Out:' wrapper.",
    )
    parser.add_argument("--max-steps", type=int, default=30000)
    parser.add_argument("--action-hz", type=float, default=25.0)
    parser.add_argument("--servo-hz", type=float, default=100.0)
    parser.add_argument("--num-open-loop-steps", type=int, default=25)
    parser.add_argument("--async-requery", action="store_true", help="Use legacy overlap/blending requery")
    parser.add_argument("--overlap-k", type=int, default=5)

    parser.add_argument("--host", default="127.0.0.1", help="OFT server host")
    parser.add_argument("--port", type=int, default=8777, help="OFT server port")
    parser.add_argument("--server-url", default="", help="Full OFT /act URL; overrides host/port")
    parser.add_argument("--request-timeout", type=float, default=120.0)

    parser.add_argument("--proprio-dim", type=int, default=6, help="xArm OFT proprio dimension; must be 6")
    parser.add_argument("--external-cam-serial", default=DEFAULT_EXTERNAL_CAM_SERIAL)
    parser.add_argument("--wrist-cam-serial", default=DEFAULT_WRIST_CAM_SERIAL)
    parser.add_argument("--camera-width", type=int, default=1920)
    parser.add_argument("--camera-height", type=int, default=1080)
    parser.add_argument("--camera-fps", type=int, default=30)
    parser.add_argument("--external-crop", default="left_540")
    parser.add_argument("--wrist-crop", default="right")
    parser.add_argument(
        "--debug-image-dir",
        default="",
        help="If set, save raw camera frames and cropped 224x224 model inputs during inference.",
    )
    parser.add_argument(
        "--debug-image-every",
        type=int,
        default=1,
        help="Save one image set every N inference requests when --debug-image-dir is set.",
    )

    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--max-delta-mm", type=float, default=200.0)
    parser.add_argument("--max-delta-rad", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true", help="Run inference and timing without moving the arm")
    parser.add_argument("--verbose-actions", action="store_true")
    parser.add_argument("--disable-keyboard-reset", action="store_true", help="Disable press-R reset to home pose")
    parser.add_argument("--reset-position-deg", type=float, nargs=6, default=DEFAULT_RESET_POSITION_DEG)
    parser.add_argument("--reset-speed", type=float, default=30.0)
    parser.add_argument("--reset-pause", type=float, default=2.0)

    parser.add_argument("--disable-gripper", action="store_true", help="Ignore action[6] and do not command gripper")
    parser.add_argument("--gripper-open-pos", type=int, default=850)
    parser.add_argument("--gripper-close-pos", type=int, default=0)
    parser.add_argument("--gripper-init-pos", type=int, default=800)
    parser.add_argument("--gripper-init-speed", type=int, default=1000)
    parser.add_argument("--gripper-close-speed", type=int, default=5000)
    parser.add_argument("--gripper-open-hold", type=float, default=2.8)
    parser.add_argument("--gripper-close-hold", type=float, default=1.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.prompt = normalize_prompt(args.prompt)
    warn_if_full_openvla_prompt(args.prompt)

    substeps = max(1, round(args.servo_hz / args.action_hz))
    servo_dt = 1.0 / args.servo_hz
    action_dt = substeps * servo_dt
    endpoint = build_server_endpoint(args)

    if args.num_open_loop_steps <= 0:
        raise ValueError(f"--num-open-loop-steps must be positive, got {args.num_open_loop_steps}")

    if not 30.0 <= args.servo_hz <= 250.0:
        print(f"WARNING: servo_hz={args.servo_hz} is outside servo mode range [30, 250]")

    if args.async_requery:
        min_overlap_k = int(np.ceil(0.45 / (1.0 / args.action_hz)))
        if args.overlap_k < min_overlap_k:
            print(f"WARNING: overlap_k={args.overlap_k} may be too small. Recommended >= {min_overlap_k}")

    require_hardware_dependencies()

    arm = None
    cam_external = None
    cam_wrist = None
    worker = None
    key_listener = None

    def cleanup(signum=None, frame=None):
        print("\nCleaning up...")
        if key_listener is not None:
            try:
                key_listener.stop()
            except Exception:
                pass
        if worker is not None:
            try:
                worker.shutdown()
            except Exception:
                pass
        if arm is not None:
            try:
                arm.set_mode(0)
                arm.set_state(4)
                arm.disconnect()
                print("xArm stopped and disconnected")
            except Exception as exc:
                print(f"xArm cleanup error: {exc}")
        if cam_external is not None:
            try:
                cam_external.close()
            except Exception:
                pass
        if cam_wrist is not None:
            try:
                cam_wrist.close()
            except Exception:
                pass
        if signum is not None:
            sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)

    try:
        global MAX_POS_DELTA_MM, MAX_ROT_DELTA_RAD
        MAX_POS_DELTA_MM = args.max_delta_mm
        MAX_ROT_DELTA_RAD = args.max_delta_rad

        print(f"Task instruction: {args.prompt!r}")
        print(f"Connecting to OFT server at {endpoint}...")
        client = OFTActionClient(endpoint=endpoint, timeout=args.request_timeout)

        print(f"Connecting to xArm at {args.xarm_ip}...")
        arm = XArmAPI(args.xarm_ip)
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.5)

        gripper_enabled = not args.disable_gripper
        current_gripper_state = -1.0
        if gripper_enabled:
            if args.dry_run:
                print(
                    "Dry run enabled: gripper init skipped "
                    f"(would open to {args.gripper_init_pos})"
                )
            else:
                current_gripper_state = init_gripper(
                    arm,
                    open_pos=args.gripper_init_pos,
                    speed=args.gripper_init_speed,
                )
                print(f"xArm gripper initialized open at position {args.gripper_init_pos}")

        print("xArm initialized")

        code, init_pose = arm.get_position(is_radian=True)
        if code != 0:
            raise RuntimeError(f"Initial get_position failed: code={code}")
        tracked_pose = np.array(init_pose[:6], dtype=np.float64)
        print(f"  Initial TCP pose: [{', '.join(f'{v:.2f}' for v in tracked_pose)}]")

        print("Starting cameras...")
        cam_external = RealsenseCapture(
            args.external_cam_serial,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        )
        cam_wrist = RealsenseCapture(
            args.wrist_cam_serial,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
        )

        worker = AsyncInferenceWorker(
            client=client,
            cam_wrist=cam_wrist,
            cam_external=cam_external,
            prompt=args.prompt,
            arm=arm,
            overlap_k=args.overlap_k,
            num_open_loop_steps=args.num_open_loop_steps,
            proprio_dim=args.proprio_dim,
            wrist_crop=args.wrist_crop,
            external_crop=args.external_crop,
            debug_image_dir=args.debug_image_dir,
            debug_image_every=args.debug_image_every,
        )

        print("Running first inference (sync, in mode 0)...")
        n_first, cam_ms_first, infer_ms_first = worker.run_first_sync()
        print(f"  First inference: {n_first} actions, cam={cam_ms_first:.0f}ms, infer={infer_ms_first:.0f}ms")

        if not args.dry_run:
            enter_servo_mode(arm)
        else:
            print("  Dry run enabled: servo mode and motion commands are skipped")

        if args.async_requery:
            worker.start()

        key_listener = KeyListener(enabled=not args.disable_keyboard_reset)
        key_listener.start()
        if key_listener.enabled:
            print("  Press 'R' to reset arm to the configured joint pose.\n")

        move_ok = 0
        hold_steps = 0
        reset_count = 0

        print("\nStarting control loop (OpenVLA-OFT):")
        print(f"  action_hz={args.action_hz}, servo_hz={args.servo_hz}")
        print(f"  substeps={substeps}, servo_dt={servo_dt * 1000:.1f}ms, action_dt={action_dt * 1000:.0f}ms")
        print(
            f"  open_loop_steps={args.num_open_loop_steps}, "
            f"async_requery={args.async_requery}, overlap_k={args.overlap_k}, "
            f"proprio_dim={args.proprio_dim}\n"
        )
        if gripper_enabled:
            print(
                "  gripper=enabled "
                f"(close if action[6] > 0, open otherwise; "
                f"open_pos={args.gripper_open_pos}, close_pos={args.gripper_close_pos})\n"
            )
        else:
            print("  gripper=disabled\n")

        for step in range(args.max_steps):
            t_step_start = time.perf_counter()

            if key_listener is not None and key_listener.check_and_clear():
                reset_count += 1
                tracked_pose = reset_to_home(
                    arm,
                    worker,
                    list(args.reset_position_deg),
                    reset_speed=args.reset_speed,
                    reset_pause=args.reset_pause,
                    servo_dt=servo_dt,
                    dry_run=args.dry_run,
                    async_requery=args.async_requery,
                )
                print(f"  Reset #{reset_count} complete. Continuing from step {step}.")
                continue

            if not args.dry_run and (arm.error_code != 0 or arm.state == 4):
                print(f"  Step {step}: arm error={arm.error_code} state={arm.state}, recovering...")
                arm.set_mode(0)
                arm.set_state(0)
                time.sleep(0.1)
                arm.clean_error()
                arm.clean_warn()
                arm.motion_enable(enable=True)
                arm.set_mode(0)
                arm.set_state(0)
                time.sleep(0.5)
                code, recovered_pose = arm.get_position(is_radian=True)
                if code == 0:
                    tracked_pose = np.array(recovered_pose[:6], dtype=np.float64)
                    print("  Tracked pose re-synced after error recovery")
                enter_servo_mode(arm)

            for log in worker.drain_logs():
                print(log)

            action = worker.pop_action()

            if action is None:
                if args.async_requery:
                    if args.dry_run:
                        time.sleep(action_dt)
                    else:
                        servo_hold(arm, tracked_pose, servo_dt, action_dt)
                    hold_steps += 1
                    print(f"  Step {step:03d} | HOLD (queue empty) | total_holds={hold_steps}")
                    continue

                print("  Requerying model for next open-loop chunk...")
                n_next, cam_ms_next, infer_ms_next = worker.refill_sync()
                print(f"  Requery done: {n_next} actions, cam={cam_ms_next:.0f}ms, infer={infer_ms_next:.0f}ms")
                action = worker.pop_action()
                if action is None:
                    hold_steps += 1
                    print(f"  Step {step:03d} | HOLD (empty chunk) | total_holds={hold_steps}")
                    continue

            queue_len = worker.queue_len()
            if args.verbose_actions:
                print(action)

            t0 = time.perf_counter()
            gripper_trigger = False
            if gripper_enabled:
                if len(action) <= 6:
                    raise RuntimeError(
                        f"Gripper is enabled but model returned action dim {len(action)}; expected >= 7. "
                        "Use --disable-gripper for old 6-dim checkpoints."
                    )

                gripper_cmd = 1.0 if action[6] > 0 else -1.0
                gripper_trigger = (gripper_cmd > 0) != (current_gripper_state > 0)

            if args.dry_run:
                print(f"  [{step}] action: {np.asarray(action)}")
                if gripper_enabled and gripper_trigger:
                    gripper_text = "close" if gripper_cmd > 0 else "open"
                    print(f"  [{step}] dry-run gripper trigger: {gripper_text}")
                    current_gripper_state = gripper_cmd
                time.sleep(action_dt)
            else:
                if gripper_trigger:
                    if gripper_cmd > 0:
                        print("  [GRASP] Closing gripper...")
                        arm.set_gripper_speed(args.gripper_close_speed)
                        arm.set_gripper_position(args.gripper_close_pos, wait=False)
                        hold_dur = args.gripper_close_hold
                    else:
                        print("  [RELEASE] Opening gripper...")
                        arm.set_gripper_position(args.gripper_open_pos, wait=False)
                        hold_dur = args.gripper_open_hold

                    if hold_dur > 0:
                        servo_hold(arm, tracked_pose, servo_dt, hold_dur)
                    current_gripper_state = gripper_cmd

                    if args.async_requery:
                        stale = flush_action_queue(worker)
                        print(f"  Flushed {stale} stale actions from queue after gripper change")
                    else:
                        print("  Keeping queued open-loop actions after gripper change")

                pos_delta = action[:3].astype(np.float64) * args.speed_scale
                pos_norm = np.linalg.norm(pos_delta)
                if pos_norm > MAX_POS_DELTA_MM:
                    pos_delta *= MAX_POS_DELTA_MM / pos_norm

                rot_delta = action[3:6].astype(np.float64) * args.speed_scale
                rot_norm = np.linalg.norm(rot_delta)
                if rot_norm > MAX_ROT_DELTA_RAD:
                    rot_delta *= MAX_ROT_DELTA_RAD / rot_norm

                sub_pos = pos_delta / substeps
                sub_rot = rot_delta / substeps

                for _ in range(substeps):
                    t_sub = time.perf_counter()
                    tracked_pose[:3] += sub_pos
                    tracked_pose[3:6] += sub_rot
                    arm.set_servo_cartesian(tracked_pose.tolist(), is_radian=True)

                    remaining = servo_dt - (time.perf_counter() - t_sub)
                    if remaining > 0:
                        time.sleep(remaining)

                move_ok += 1

            t_act = time.perf_counter() - t0
            actual_dt = time.perf_counter() - t_step_start

            print(
                f"  Step {step:03d} | {actual_dt * 1000:>5.0f}ms | "
                f"Act: {t_act * 1000:>3.0f}ms | Q:{queue_len}"
            )

        print(
            f"\nControl loop finished: ok={move_ok}, inferences={worker.infer_count}, "
            f"holds={hold_steps}, resets={reset_count}"
        )

    except Exception as exc:
        print(f"Error: {exc}")
        import traceback

        traceback.print_exc()
    finally:
        cleanup()


if __name__ == "__main__":
    main()
