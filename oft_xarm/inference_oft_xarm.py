#!/usr/bin/env python3
"""xArm6 + OpenVLA-OFT deployment client.

This script reuses the lab xArm hardware loop:
- two RealSense RGB streams
- 6-dim xArm joint state in radians
- 8-step open-loop action chunks at 10 Hz
- interpolated Cartesian servo execution for each action
- basic gripper control from action[6]

The model call is adapted for OpenVLA-OFT's HTTP /act server. Gripper output
action[6] is binarized: positive closes the xArm gripper, non-positive opens it.
"""

import argparse
import collections
import json
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

# The two supported task datasets were collected at 960x540 and converted to RLDS with
# these exact square crops before resizing to 224x224. Keep these values in
# sync with the CASE-Lab conversion manifests; using the old 1920x1080 crop
# modes silently feeds the model a different view.
TRAIN_CAMERA_WIDTH = 960
TRAIN_CAMERA_HEIGHT = 540
TRAIN_CAMERA_FPS = 60
TRAIN_CROP_SIZE = 540
TRAIN_EXTERNAL_CROP_LEFT = 270  # cam_1: [270, 0, 810, 540]
TRAIN_WRIST_CROP_LEFT = 380     # cam_0: [380, 0, 920, 540]
POLICY_IMAGE_SIZE = 224

# Per-task instruction, duration cap, and press-R reset pose. Both available
# tasks use the proven UF850 collection reset pose and start with the gripper
# open.
COLLECTION_RESET_POSITION_DEG = [
    55.399232,
    7.733498,
    -48.980042,
    -1.039517,
    -57.38115,
    -0.614669,
]
TASK_PRESETS = {
    "put-blue-bowl-in-second-drawer": {
        "instruction": "put the blue bowl in the second drawer",
        "reset_position_deg": COLLECTION_RESET_POSITION_DEG,
        "max_steps": 650,
    },
    "erase-circle-from-whiteboard": {
        "instruction": "erase the circle from the whiteboard",
        "reset_position_deg": COLLECTION_RESET_POSITION_DEG,
        "max_steps": 1950,
    },
}

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

    def __init__(
        self,
        serial: str,
        width: int = TRAIN_CAMERA_WIDTH,
        height: int = TRAIN_CAMERA_HEIGHT,
        fps: int = TRAIN_CAMERA_FPS,
        warmup_frames: int = 30,
    ):
        self.serial = serial
        self.width = width
        self.height = height
        self._lock = threading.Lock()
        self._closed = False
        self.pipeline = rs.pipeline()

        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.yuyv, fps)
        self.pipeline.start(config)

        for _ in range(warmup_frames):
            self.pipeline.wait_for_frames()
        print(
            f"Camera {serial} ready "
            f"({width}x{height} YUYV @ {fps} FPS, warmup={warmup_frames})"
        )

    def get_frame(self) -> np.ndarray:
        with self._lock:
            if self._closed:
                raise RuntimeError(f"RealSense {self.serial} is closed")
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                raise RuntimeError(f"No color frame from camera {self.serial}")

            raw = np.asanyarray(color_frame.get_data())
            image = cv2.cvtColor(
                raw.view(np.uint8).reshape(self.height, self.width, 2),
                cv2.COLOR_YUV2RGB_YUYV,
            )
            return np.ascontiguousarray(image)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self.pipeline.stop()
        print(f"Camera {self.serial} stopped")


class InferenceVideoRecorder:
    """Record both full-resolution RealSense streams into one run directory."""

    def __init__(
        self,
        output_dir: str,
        external_camera: RealsenseCapture,
        wrist_camera: RealsenseCapture,
        fps: float,
        task: str,
        instruction: str,
        save_frames: bool = False,
        frame_jpeg_quality: int = 95,
    ):
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        task_name = task or "custom"
        self.run_dir = Path(output_dir).expanduser() / task_name / f"{timestamp}_{os.getpid()}"
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.external_camera = external_camera
        self.wrist_camera = wrist_camera
        self.fps = fps
        self.save_frames = save_frames
        self.frame_jpeg_quality = frame_jpeg_quality
        self._frame_interval = 1.0 / fps
        self._stop_event = threading.Event()
        self._thread = None
        self._stopped = False
        self._frame_count = 0
        self._started_at = None
        self._error = None
        self._capture_pool = ThreadPoolExecutor(max_workers=2)

        self._frame_dirs = {}
        if self.save_frames:
            self._frame_dirs = {
                "external": self.run_dir / "external_frames",
                "wrist": self.run_dir / "wrist_frames",
            }
            for frame_dir in self._frame_dirs.values():
                frame_dir.mkdir()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writers = {
            "external": cv2.VideoWriter(
                str(self.run_dir / "external.mp4"),
                fourcc,
                fps,
                (external_camera.width, external_camera.height),
            ),
            "wrist": cv2.VideoWriter(
                str(self.run_dir / "wrist.mp4"),
                fourcc,
                fps,
                (wrist_camera.width, wrist_camera.height),
            ),
        }
        failed = [name for name, writer in self._writers.items() if not writer.isOpened()]
        if failed:
            for writer in self._writers.values():
                writer.release()
            raise RuntimeError(f"Could not open MP4 writer(s) {failed} in {self.run_dir}")

        (self.run_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "task": task,
                    "instruction": instruction,
                    "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                    "fps": fps,
                    "save_frames": save_frames,
                    "frame_jpeg_quality": frame_jpeg_quality if save_frames else None,
                    "frame_filename_pattern": "frame_%06d.jpg" if save_frames else None,
                    "resolution": [external_camera.width, external_camera.height],
                    "external_camera_serial": external_camera.serial,
                    "wrist_camera_serial": wrist_camera.serial,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def start(self) -> None:
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._record_loop, daemon=True)
        self._thread.start()
        print(
            f"[VIDEO] Recording {self.external_camera.width}x{self.external_camera.height} "
            f"at {self.fps:g} FPS to {self.run_dir}"
        )
        if self.save_frames:
            print(
                f"[VIDEO] Saving every recorded frame as JPEG "
                f"(quality={self.frame_jpeg_quality})"
            )

    def _record_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                started = time.perf_counter()
                external_future = self._capture_pool.submit(self.external_camera.get_frame)
                wrist_future = self._capture_pool.submit(self.wrist_camera.get_frame)
                external_rgb = external_future.result()
                wrist_rgb = wrist_future.result()
                external_bgr = cv2.cvtColor(external_rgb, cv2.COLOR_RGB2BGR)
                wrist_bgr = cv2.cvtColor(wrist_rgb, cv2.COLOR_RGB2BGR)
                self._writers["external"].write(external_bgr)
                self._writers["wrist"].write(wrist_bgr)
                if self.save_frames:
                    filename = f"frame_{self._frame_count:06d}.jpg"
                    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, self.frame_jpeg_quality]
                    external_ok = cv2.imwrite(
                        str(self._frame_dirs["external"] / filename),
                        external_bgr,
                        jpeg_params,
                    )
                    wrist_ok = cv2.imwrite(
                        str(self._frame_dirs["wrist"] / filename),
                        wrist_bgr,
                        jpeg_params,
                    )
                    if not external_ok or not wrist_ok:
                        raise RuntimeError(f"Failed to save JPEG frame pair {filename}")
                self._frame_count += 1
                remaining = self._frame_interval - (time.perf_counter() - started)
                self._stop_event.wait(max(remaining, 0.001))
        except Exception as exc:
            self._error = str(exc)
            print(f"[VIDEO] Recording stopped after error: {exc}", flush=True)
        finally:
            for writer in self._writers.values():
                writer.release()

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)
            self._thread = None
        self._capture_pool.shutdown(wait=False)
        elapsed = time.perf_counter() - self._started_at if self._started_at else 0.0
        effective_fps = self._frame_count / elapsed if elapsed > 0 else 0.0
        print(
            f"[VIDEO] Saved {self._frame_count} frames per camera "
            f"({effective_fps:.1f} effective FPS) to {self.run_dir}"
        )
        if self._error:
            print(f"[VIDEO] WARNING: {self._error}")


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
    """Read xArm joint angles in radians; legacy OFT pads the 6 joints to 8D."""
    if proprio_dim not in (6, 8):
        raise ValueError(
            f"xArm OFT proprio must be 6D or legacy padded 8D; got --proprio-dim {proprio_dim}."
        )

    angles_deg = arm.angles
    if angles_deg is None:
        raise RuntimeError("arm.angles returned None (report stream not ready?)")

    state = np.asarray(angles_deg[:6], dtype=np.float32) * DEG2RAD
    if proprio_dim == 8:
        state = np.concatenate([state, np.zeros(2, dtype=np.float32)])
    return state


def crop_and_resize(
    image_rgb: np.ndarray,
    crop_left: int,
    crop_size: int = TRAIN_CROP_SIZE,
    target_size: int = POLICY_IMAGE_SIZE,
) -> np.ndarray:
    """Apply the exact RLDS square crop, then resize with PIL LANCZOS."""
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image with shape (H,W,3), got {image_rgb.shape}")
    h, w = image_rgb.shape[:2]
    if crop_size <= 0 or crop_left < 0 or crop_size > h or crop_left + crop_size > w:
        raise ValueError(
            f"Crop [x={crop_left}:{crop_left + crop_size}, y=0:{crop_size}] "
            f"does not fit image {w}x{h}"
        )

    cropped = image_rgb[0:crop_size, crop_left : crop_left + crop_size]
    if cropped.shape != (crop_size, crop_size, 3):
        raise RuntimeError(f"Unexpected cropped image shape: {cropped.shape}")

    pil_img = Image.fromarray(cropped)
    pil_resized = pil_img.resize((target_size, target_size), Image.LANCZOS)
    return np.ascontiguousarray(pil_resized, dtype=np.uint8)


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
        wrist_crop_left: int,
        external_crop_left: int,
        crop_size: int,
        debug_image_dir: str,
        debug_image_every: int,
        log_action_chunks: bool,
    ):
        self.client = client
        self.cam_wrist = cam_wrist
        self.cam_external = cam_external
        self.prompt = prompt
        self.arm = arm
        self.overlap_k = overlap_k
        self.num_open_loop_steps = num_open_loop_steps
        self.proprio_dim = proprio_dim
        self.wrist_crop_left = wrist_crop_left
        self.external_crop_left = external_crop_left
        self.crop_size = crop_size
        self.debug_image_dir = Path(debug_image_dir).expanduser() if debug_image_dir else None
        self.debug_image_every = max(1, debug_image_every)
        self.log_action_chunks = log_action_chunks
        self._previous_action_chunk = None
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
        return raw, crop_and_resize(
            raw,
            crop_left=self.wrist_crop_left,
            crop_size=self.crop_size,
        )

    def _capture_ext(self) -> tuple[np.ndarray, np.ndarray]:
        raw = self.cam_external.get_frame()
        return raw, crop_and_resize(
            raw,
            crop_left=self.external_crop_left,
            crop_size=self.crop_size,
        )

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
            self.debug_image_dir
            / f"{prefix}_external_x{self.external_crop_left}_{POLICY_IMAGE_SIZE}.jpg",
            img_ext,
        )
        save_debug_image(self.debug_image_dir / f"{prefix}_wrist_raw.jpg", raw_wrist)
        save_debug_image(
            self.debug_image_dir
            / f"{prefix}_wrist_x{self.wrist_crop_left}_{POLICY_IMAGE_SIZE}.jpg",
            img_wrist,
        )

    def _log_action_chunk(self, actions: np.ndarray, state: np.ndarray) -> None:
        if not self.log_action_chunks:
            return

        chunk = np.asarray(actions, dtype=np.float64)
        summary_parts = [
            f"first={np.array2string(chunk[0], precision=4, suppress_small=False)}",
            f"mean={np.array2string(chunk.mean(axis=0), precision=4, suppress_small=False)}",
            f"std={np.array2string(chunk.std(axis=0), precision=4, suppress_small=False)}",
            f"state={np.array2string(state, precision=4, suppress_small=False)}",
        ]

        if self._previous_action_chunk is not None and self._previous_action_chunk.shape == chunk.shape:
            delta = np.abs(chunk - self._previous_action_chunk)
            summary_parts.append(f"diff_prev_mean={delta.mean():.5f}")
            summary_parts.append(f"diff_prev_max={delta.max():.5f}")
        else:
            summary_parts.append("diff_prev=NA")

        self._previous_action_chunk = chunk.copy()
        print("  ACTION CHUNK | " + " | ".join(summary_parts), flush=True)

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
        self._log_action_chunk(actions, state)
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


def interruptible_sleep(duration: float, should_stop=None, poll_dt: float = 0.02) -> bool:
    end_time = time.perf_counter() + duration
    while time.perf_counter() < end_time:
        if should_stop is not None and should_stop():
            return True
        remaining = end_time - time.perf_counter()
        if remaining > 0:
            time.sleep(min(poll_dt, remaining))
    return False


def servo_hold(
    arm: XArmAPI,
    tracked_pose: np.ndarray,
    servo_dt: float,
    duration: float,
    should_stop=None,
) -> bool:
    pose_list = tracked_pose.tolist()
    end_time = time.perf_counter() + duration
    while time.perf_counter() < end_time:
        if should_stop is not None and should_stop():
            return True
        t_h = time.perf_counter()
        arm.set_servo_cartesian(pose_list, is_radian=True)
        elapsed = time.perf_counter() - t_h
        remaining = servo_dt - elapsed
        if remaining > 0 and interruptible_sleep(remaining, should_stop):
            return True
    return False


def init_gripper(arm: XArmAPI, open_pos: int, speed: int) -> float:
    arm.set_gripper_enable(True)
    arm.set_gripper_mode(0)
    arm.set_gripper_speed(speed)
    arm.set_gripper_position(open_pos, wait=True)
    return -1.0


def startup_reset_to_home(
    arm: XArmAPI,
    reset_angles_deg: list[float],
    *,
    reset_speed: float,
    reset_pause: float,
    reset_timeout: float,
    dry_run: bool,
) -> np.ndarray:
    """Reset before cameras and the first model query, matching PAIR startup."""
    print("\n  [STARTUP RESET] Returning to the configured collection pose...")
    if dry_run:
        print(f"  [STARTUP RESET] dry-run: would move joints to {reset_angles_deg}")
    else:
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.5)
        code = arm.set_servo_angle(
            angle=reset_angles_deg,
            speed=reset_speed,
            is_radian=False,
            wait=True,
            timeout=reset_timeout,
        )
        if code != 0:
            try:
                err_warn = arm.get_err_warn_code()
            except Exception:
                err_warn = None
            raise RuntimeError(
                "automatic startup reset failed: "
                f"code={code}, arm_error={arm.error_code}, arm_state={arm.state}, err_warn={err_warn}"
            )
        print(f"  [STARTUP RESET] Reached joint pose: {reset_angles_deg}")

    if reset_pause > 0:
        print(f"  [STARTUP RESET] Pausing {reset_pause:.1f}s...")
        time.sleep(reset_pause)

    code, pose = arm.get_position(is_radian=True)
    if code != 0 or pose is None or len(pose) < 6:
        raise RuntimeError(f"get_position failed after automatic startup reset: code={code}")
    tracked_pose = np.asarray(pose[:6], dtype=np.float64)
    print(
        "  [STARTUP RESET] Complete. TCP: "
        f"[{', '.join(f'{value:.2f}' for value in tracked_pose)}]\n"
    )
    return tracked_pose


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
    reset_timeout: float,
    reset_gripper_pos: int | None,
    servo_dt: float,
    dry_run: bool,
    async_requery: bool,
    gripper_enabled: bool,
    gripper_open_pos: int,
    gripper_speed: int,
    reason: str = "reset requested",
) -> np.ndarray:
    """Move to the configured reset joint pose and return a re-synced TCP pose."""
    print(f"\n  [RESET] {reason}: stopping policy actions and moving to reset pose...")

    if async_requery:
        worker.stop()

    stale = flush_action_queue(worker)
    print(f"  [RESET] Cleared {stale} queued actions")

    # All demos start with an open gripper, so a trial must too. Open in place
    # before moving home so anything still held is dropped where it is instead
    # of being carried across the workspace.
    if gripper_enabled:
        if dry_run:
            print(f"  [RESET] dry-run: would open gripper in place to {gripper_open_pos}")
        else:
            print("  [RESET] Opening gripper in place...")
            arm.set_gripper_speed(gripper_speed)
            arm.set_gripper_position(gripper_open_pos, wait=True)

    if dry_run:
        print(f"  [RESET] dry-run: would move joints to {reset_angles_deg}")
    else:
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(enable=True)
        arm.set_mode(0)
        arm.set_state(0)
        time.sleep(0.5)
        code = arm.set_servo_angle(
            angle=reset_angles_deg,
            speed=reset_speed,
            is_radian=False,
            wait=True,
            timeout=reset_timeout,
        )
        if code != 0:
            try:
                err_warn = arm.get_err_warn_code()
            except Exception:
                err_warn = None
            raise RuntimeError(
                "set_servo_angle reset failed: "
                f"code={code}, arm_error={arm.error_code}, arm_state={arm.state}, err_warn={err_warn}"
            )
        print(f"  [RESET] Reached reset joint pose: {reset_angles_deg}")

        if reset_gripper_pos is not None:
            arm.set_gripper_position(reset_gripper_pos, wait=True)
            print(f"  [RESET] Reset gripper position: {reset_gripper_pos}")

    if dry_run and reset_gripper_pos is not None:
        print(f"  [RESET] dry-run: would move gripper to {reset_gripper_pos}")

    if reset_pause > 0:
        print(f"  [RESET] Pausing {reset_pause:.1f}s...")
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
        "--task",
        choices=["custom", *sorted(TASK_PRESETS)],
        default="custom",
        help="Optional preset for default instruction/reset pose. Use custom for explicit --instruction and --reset-position-deg.",
    )
    parser.add_argument(
        "--prompt",
        "--instruction",
        dest="prompt",
        default=None,
        help="Raw task instruction. Do not include the OpenVLA 'In: ... Out:' wrapper.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Safety cap in 10 Hz policy steps; defaults to the selected task preset.",
    )
    parser.add_argument("--action-hz", type=float, default=10.0)
    parser.add_argument("--servo-hz", type=float, default=100.0)
    parser.add_argument("--num-open-loop-steps", type=int, default=8)
    parser.add_argument("--async-requery", action="store_true", help="Use legacy overlap/blending requery")
    parser.add_argument("--overlap-k", type=int, default=5)

    parser.add_argument("--host", default="127.0.0.1", help="OFT server host")
    parser.add_argument("--port", type=int, default=8777, help="OFT server port")
    parser.add_argument("--server-url", default="", help="Full OFT /act URL; overrides host/port")
    parser.add_argument("--request-timeout", type=float, default=120.0)

    parser.add_argument("--proprio-dim", type=int, default=6, help="xArm OFT proprio dimension; use 6 or legacy padded 8")
    parser.add_argument("--external-cam-serial", default=DEFAULT_EXTERNAL_CAM_SERIAL)
    parser.add_argument("--wrist-cam-serial", default=DEFAULT_WRIST_CAM_SERIAL)
    parser.add_argument("--camera-width", type=int, default=TRAIN_CAMERA_WIDTH)
    parser.add_argument("--camera-height", type=int, default=TRAIN_CAMERA_HEIGHT)
    parser.add_argument("--camera-fps", type=int, default=TRAIN_CAMERA_FPS)
    parser.add_argument("--camera-warmup-frames", type=int, default=30)
    parser.add_argument(
        "--external-crop-left",
        type=int,
        default=TRAIN_EXTERNAL_CROP_LEFT,
        help="External cam_1 square-crop left edge; training used x=270.",
    )
    parser.add_argument(
        "--wrist-crop-left",
        type=int,
        default=TRAIN_WRIST_CROP_LEFT,
        help="Wrist cam_0 square-crop left edge; training used x=380.",
    )
    parser.add_argument("--crop-size", type=int, default=TRAIN_CROP_SIZE)
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
    parser.add_argument("--record-video", action="store_true")
    parser.add_argument("--video-dir", default="outputs/videos")
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument(
        "--save-video-frames",
        action="store_true",
        help="Save every recorded external/wrist video frame as an aligned JPEG pair.",
    )
    parser.add_argument("--frame-jpeg-quality", type=int, default=95)

    parser.add_argument("--speed-scale", type=float, default=1.0)
    parser.add_argument("--max-delta-mm", type=float, default=200.0)
    parser.add_argument("--max-delta-rad", type=float, default=1.0)
    parser.add_argument("--dry-run", action="store_true", help="Run inference and timing without moving the arm")
    parser.add_argument("--verbose-actions", action="store_true")
    parser.add_argument(
        "--log-action-chunks",
        action="store_true",
        help="Log each model action chunk summary, including change from the previous chunk.",
    )
    parser.add_argument("--disable-keyboard-reset", action="store_true", help="Disable press-R reset to home pose")
    parser.add_argument(
        "--reset-position-deg",
        type=float,
        nargs=6,
        default=None,
        help="Joint angles (deg) for startup/press-R reset.",
    )
    parser.add_argument("--reset-speed", type=float, default=30.0)
    parser.add_argument("--reset-pause", type=float, default=2.0)
    parser.add_argument("--reset-timeout", type=float, default=15.0)
    parser.add_argument(
        "--reset-gripper-pos",
        type=int,
        default=None,
        help="If set and gripper is enabled, move gripper to this position on reset.",
    )
    parser.add_argument(
        "--reset-trigger-file",
        default="/tmp/oft_xarm_reset",
        help="If this file appears during inference, delete it and reset the arm. Set empty string to disable.",
    )

    parser.add_argument("--disable-gripper", action="store_true", help="Ignore action[6] and do not command gripper")
    parser.add_argument(
        "--gripper-open-pos",
        type=int,
        default=850,
        help="Open position used at init, on release, and on press-R reset (demos start fully open).",
    )
    parser.add_argument("--gripper-close-pos", type=int, default=0)
    parser.add_argument("--gripper-init-speed", type=int, default=1000)
    parser.add_argument("--gripper-close-speed", type=int, default=5000)
    parser.add_argument("--gripper-open-hold", type=float, default=2.8)
    parser.add_argument("--gripper-close-hold", type=float, default=1.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    preset = TASK_PRESETS.get(args.task)
    if args.prompt is None:
        if preset is None:
            raise ValueError("--instruction is required when --task custom")
        args.prompt = preset["instruction"]
    if args.reset_position_deg is None:
        if preset is None:
            raise ValueError("--reset-position-deg is required when --task custom")
        args.reset_position_deg = list(preset["reset_position_deg"])
    if args.max_steps is None:
        args.max_steps = int(preset["max_steps"]) if preset is not None else 30000
    args.prompt = normalize_prompt(args.prompt)
    warn_if_full_openvla_prompt(args.prompt)

    if args.max_steps <= 0:
        raise ValueError(f"--max-steps must be positive, got {args.max_steps}")
    if args.action_hz <= 0 or args.servo_hz < args.action_hz:
        raise ValueError("Require 0 < action_hz <= servo_hz")
    if args.camera_width <= 0 or args.camera_height <= 0 or args.camera_fps <= 0:
        raise ValueError("Camera width, height, and FPS must be positive")
    if args.camera_warmup_frames < 0:
        raise ValueError("--camera-warmup-frames must be non-negative")
    if args.crop_size <= 0:
        raise ValueError("--crop-size must be positive")
    if args.record_video:
        if args.video_fps <= 0 or args.video_fps > args.camera_fps:
            raise ValueError("Require 0 < video_fps <= camera_fps")
        if not args.video_dir.strip():
            raise ValueError("--video-dir cannot be empty when recording video")
        if not 1 <= args.frame_jpeg_quality <= 100:
            raise ValueError("--frame-jpeg-quality must be in [1,100]")
    elif args.save_video_frames:
        raise ValueError("--save-video-frames requires --record-video")

    if preset is not None:
        actual_geometry = (
            args.camera_width,
            args.camera_height,
            args.camera_fps,
            args.crop_size,
            args.external_crop_left,
            args.wrist_crop_left,
        )
        training_geometry = (
            TRAIN_CAMERA_WIDTH,
            TRAIN_CAMERA_HEIGHT,
            TRAIN_CAMERA_FPS,
            TRAIN_CROP_SIZE,
            TRAIN_EXTERNAL_CROP_LEFT,
            TRAIN_WRIST_CROP_LEFT,
        )
        if actual_geometry != training_geometry:
            raise ValueError(
                "New UF850 tasks require the exact training camera geometry "
                f"(width,height,fps,crop_size,external_left,wrist_left)={training_geometry}; "
                f"got {actual_geometry}"
            )

    # Validate both crops before opening hardware. This catches negative or
    # truncated numpy slices instead of silently resizing the wrong view.
    geometry_probe = np.zeros((args.camera_height, args.camera_width, 3), dtype=np.uint8)
    crop_and_resize(geometry_probe, args.external_crop_left, args.crop_size)
    crop_and_resize(geometry_probe, args.wrist_crop_left, args.crop_size)

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
    video_recorder = None
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
        if video_recorder is not None:
            try:
                video_recorder.stop()
            except Exception as exc:
                print(f"Video cleanup error: {exc}")
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

        print(f"Task preset: {args.task}")
        print(f"Task instruction: {args.prompt!r}")
        print(f"Reset pose (deg): {args.reset_position_deg}")
        print(f"Video recording: {args.record_video}")
        print(
            "Training camera geometry: "
            f"{args.camera_width}x{args.camera_height} @ {args.camera_fps} FPS; "
            f"external=[{args.external_crop_left},0,"
            f"{args.external_crop_left + args.crop_size},{args.crop_size}], "
            f"wrist=[{args.wrist_crop_left},0,"
            f"{args.wrist_crop_left + args.crop_size},{args.crop_size}] -> "
            f"{POLICY_IMAGE_SIZE}x{POLICY_IMAGE_SIZE}"
        )
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
                    f"(would open to {args.gripper_open_pos})"
                )
            else:
                # Init to the same open position used on release so every trial
                # starts with an identical gripper aperture (demos start open).
                current_gripper_state = init_gripper(
                    arm,
                    open_pos=args.gripper_open_pos,
                    speed=args.gripper_init_speed,
                )
                print(f"xArm gripper initialized open at position {args.gripper_open_pos}")

        print("xArm initialized")

        tracked_pose = startup_reset_to_home(
            arm,
            list(args.reset_position_deg),
            reset_speed=args.reset_speed,
            reset_pause=args.reset_pause,
            reset_timeout=args.reset_timeout,
            dry_run=args.dry_run,
        )

        print("Starting cameras...")
        cam_external = RealsenseCapture(
            args.external_cam_serial,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            warmup_frames=args.camera_warmup_frames,
        )
        cam_wrist = RealsenseCapture(
            args.wrist_cam_serial,
            width=args.camera_width,
            height=args.camera_height,
            fps=args.camera_fps,
            warmup_frames=args.camera_warmup_frames,
        )

        if args.record_video:
            video_recorder = InferenceVideoRecorder(
                args.video_dir,
                cam_external,
                cam_wrist,
                args.video_fps,
                args.task,
                args.prompt,
                save_frames=args.save_video_frames,
                frame_jpeg_quality=args.frame_jpeg_quality,
            )
            video_recorder.start()

        worker = AsyncInferenceWorker(
            client=client,
            cam_wrist=cam_wrist,
            cam_external=cam_external,
            prompt=args.prompt,
            arm=arm,
            overlap_k=args.overlap_k,
            num_open_loop_steps=args.num_open_loop_steps,
            proprio_dim=args.proprio_dim,
            wrist_crop_left=args.wrist_crop_left,
            external_crop_left=args.external_crop_left,
            crop_size=args.crop_size,
            debug_image_dir=args.debug_image_dir,
            debug_image_every=args.debug_image_every,
            log_action_chunks=args.log_action_chunks,
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
        reset_trigger_path = Path(args.reset_trigger_file).expanduser() if args.reset_trigger_file else None
        if reset_trigger_path is not None:
            if reset_trigger_path.exists():
                try:
                    reset_trigger_path.unlink()
                    print(f"  Removed stale reset trigger file: {reset_trigger_path}")
                except OSError as exc:
                    print(f"  WARNING: could not remove stale reset trigger file {reset_trigger_path}: {exc}")
            print(f"  Or run this in another terminal to reset: touch {reset_trigger_path}\n")

        move_ok = 0
        hold_steps = 0
        reset_count = 0

        def reset_requested() -> tuple[bool, str]:
            if key_listener is not None and key_listener.check_and_clear():
                return True, "'R' pressed"
            if reset_trigger_path is not None and reset_trigger_path.exists():
                try:
                    reset_trigger_path.unlink()
                except OSError:
                    pass
                return True, f"trigger file {reset_trigger_path}"
            return False, ""

        def perform_reset(step_idx: int, reason: str) -> None:
            nonlocal tracked_pose, reset_count, current_gripper_state
            reset_count += 1
            tracked_pose = reset_to_home(
                arm,
                worker,
                list(args.reset_position_deg),
                reset_speed=args.reset_speed,
                reset_pause=args.reset_pause,
                reset_timeout=args.reset_timeout,
                reset_gripper_pos=args.reset_gripper_pos if gripper_enabled else None,
                servo_dt=servo_dt,
                dry_run=args.dry_run,
                async_requery=args.async_requery,
                gripper_enabled=gripper_enabled,
                gripper_open_pos=args.gripper_open_pos,
                gripper_speed=args.gripper_init_speed,
                reason=reason,
            )
            if gripper_enabled:
                if args.reset_gripper_pos is not None:
                    current_gripper_state = 1.0 if args.reset_gripper_pos < 400 else -1.0
                else:
                    current_gripper_state = -1.0
            print(f"  Reset #{reset_count} complete. Continuing from step {step_idx}.")

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

            do_reset, reset_reason = reset_requested()
            if do_reset:
                perform_reset(step, reset_reason)
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
                        do_reset = interruptible_sleep(action_dt, lambda: reset_requested()[0])
                    else:
                        do_reset = servo_hold(arm, tracked_pose, servo_dt, action_dt, lambda: reset_requested()[0])
                    if do_reset:
                        perform_reset(step, "reset requested during hold")
                        continue
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
                if interruptible_sleep(action_dt, lambda: reset_requested()[0]):
                    perform_reset(step, "reset requested during dry-run step")
                    continue
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

                    current_gripper_state = gripper_cmd
                    if hold_dur > 0 and servo_hold(
                        arm,
                        tracked_pose,
                        servo_dt,
                        hold_dur,
                        lambda: reset_requested()[0],
                    ):
                        perform_reset(step, "reset requested during gripper hold")
                        continue

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

                reset_during_action = False
                for _ in range(substeps):
                    do_reset, reset_reason = reset_requested()
                    if do_reset:
                        reset_during_action = True
                        break
                    t_sub = time.perf_counter()
                    tracked_pose[:3] += sub_pos
                    tracked_pose[3:6] += sub_rot
                    arm.set_servo_cartesian(tracked_pose.tolist(), is_radian=True)

                    remaining = servo_dt - (time.perf_counter() - t_sub)
                    if remaining > 0 and interruptible_sleep(remaining, lambda: reset_requested()[0]):
                        reset_during_action = True
                        break

                if reset_during_action:
                    perform_reset(step, "reset requested during action")
                    continue
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
