import importlib.util
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "oft_xarm" / "inference_oft_xarm.py"
)
SPEC = importlib.util.spec_from_file_location("inference_oft_xarm", MODULE_PATH)
OFT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(OFT)


class TaskPresetTests(unittest.TestCase):
    def test_new_task_prompts_and_limits(self):
        expected = {
            "put-blue-bowl-in-second-drawer": (
                "put the blue bowl in the second drawer",
                650,
            ),
            "erase-circle-from-whiteboard": (
                "erase the circle from the whiteboard",
                1950,
            ),
        }
        self.assertEqual(set(OFT.TASK_PRESETS), set(expected))
        for task, (instruction, max_steps) in expected.items():
            self.assertEqual(OFT.TASK_PRESETS[task]["instruction"], instruction)
            self.assertEqual(OFT.TASK_PRESETS[task]["max_steps"], max_steps)
            self.assertEqual(
                OFT.TASK_PRESETS[task]["reset_position_deg"],
                OFT.COLLECTION_RESET_POSITION_DEG,
            )


class CameraCropTests(unittest.TestCase):
    def setUp(self):
        rows = np.arange(OFT.TRAIN_CAMERA_HEIGHT, dtype=np.uint16)[:, None]
        cols = np.arange(OFT.TRAIN_CAMERA_WIDTH, dtype=np.uint16)[None, :]
        self.image = np.empty(
            (OFT.TRAIN_CAMERA_HEIGHT, OFT.TRAIN_CAMERA_WIDTH, 3),
            dtype=np.uint8,
        )
        self.image[..., 0] = cols % 256
        self.image[..., 1] = rows % 256
        self.image[..., 2] = (cols + rows) % 256

    def test_external_crop_matches_training_manifest(self):
        result = OFT.crop_and_resize(
            self.image,
            OFT.TRAIN_EXTERNAL_CROP_LEFT,
            OFT.TRAIN_CROP_SIZE,
            target_size=OFT.TRAIN_CROP_SIZE,
        )
        expected = self.image[:, 270:810]
        np.testing.assert_array_equal(result, expected)

    def test_wrist_crop_matches_training_manifest(self):
        result = OFT.crop_and_resize(
            self.image,
            OFT.TRAIN_WRIST_CROP_LEFT,
            OFT.TRAIN_CROP_SIZE,
            target_size=OFT.TRAIN_CROP_SIZE,
        )
        expected = self.image[:, 380:920]
        np.testing.assert_array_equal(result, expected)

    def test_policy_outputs_are_224_square_rgb(self):
        for left in (
            OFT.TRAIN_EXTERNAL_CROP_LEFT,
            OFT.TRAIN_WRIST_CROP_LEFT,
        ):
            result = OFT.crop_and_resize(self.image, left)
            self.assertEqual(result.shape, (224, 224, 3))
            self.assertEqual(result.dtype, np.uint8)

    def test_out_of_bounds_crop_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not fit"):
            OFT.crop_and_resize(self.image, crop_left=540, crop_size=540)

    def test_non_rgb_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Expected RGB"):
            OFT.crop_and_resize(np.zeros((540, 960), dtype=np.uint8), crop_left=270)


class VideoRecorderTests(unittest.TestCase):
    def test_records_both_streams_and_writes_metadata(self):
        class FakeCamera:
            def __init__(self, serial):
                self.serial = serial
                self.width = 16
                self.height = 12

            def get_frame(self):
                return np.zeros((self.height, self.width, 3), dtype=np.uint8)

        class FakeWriter:
            def __init__(self, path):
                self.path = path
                self.frames = 0
                self.released = False

            def isOpened(self):
                return True

            def write(self, _frame):
                self.frames += 1

            def release(self):
                self.released = True

        writers = []
        written_frames = []

        def make_writer(path, *_args):
            writer = FakeWriter(path)
            writers.append(writer)
            return writer

        def write_frame(path, *_args):
            written_frames.append(path)
            return True

        with tempfile.TemporaryDirectory() as tmpdir, mock.patch.object(
            OFT.cv2, "VideoWriter", side_effect=make_writer
        ), mock.patch.object(
            OFT.cv2, "imwrite", side_effect=write_frame
        ):
            recorder = OFT.InferenceVideoRecorder(
                tmpdir,
                FakeCamera("external-serial"),
                FakeCamera("wrist-serial"),
                fps=50,
                task="erase-circle-from-whiteboard",
                instruction="erase the circle from the whiteboard",
                save_frames=True,
                frame_jpeg_quality=95,
            )
            recorder.start()
            time.sleep(0.08)
            recorder.stop()
            recorder.stop()

            self.assertEqual(len(writers), 2)
            self.assertTrue(all(writer.frames > 0 for writer in writers))
            self.assertTrue(all(writer.released for writer in writers))
            self.assertTrue(writers[0].path.endswith("external.mp4"))
            self.assertTrue(writers[1].path.endswith("wrist.mp4"))
            self.assertGreater(len(written_frames), 0)
            self.assertEqual(len(written_frames) % 2, 0)
            self.assertTrue(any("external_frames/frame_000000.jpg" in path for path in written_frames))
            self.assertTrue(any("wrist_frames/frame_000000.jpg" in path for path in written_frames))

            metadata = json.loads((recorder.run_dir / "run_meta.json").read_text())
            self.assertEqual(metadata["task"], "erase-circle-from-whiteboard")
            self.assertEqual(metadata["fps"], 50)
            self.assertEqual(metadata["resolution"], [16, 12])
            self.assertTrue(metadata["save_frames"])
            self.assertEqual(metadata["frame_jpeg_quality"], 95)
            self.assertEqual(metadata["external_camera_serial"], "external-serial")
            self.assertEqual(metadata["wrist_camera_serial"], "wrist-serial")


class StartupResetTests(unittest.TestCase):
    def test_startup_reset_moves_to_collection_pose_and_resyncs_tcp(self):
        class FakeArm:
            error_code = 0
            state = 0

            def __init__(self):
                self.calls = []

            def clean_error(self):
                self.calls.append(("clean_error",))

            def clean_warn(self):
                self.calls.append(("clean_warn",))

            def motion_enable(self, **kwargs):
                self.calls.append(("motion_enable", kwargs))

            def set_mode(self, mode):
                self.calls.append(("set_mode", mode))

            def set_state(self, state):
                self.calls.append(("set_state", state))

            def set_servo_angle(self, **kwargs):
                self.calls.append(("set_servo_angle", kwargs))
                return 0

            def get_position(self, **kwargs):
                self.calls.append(("get_position", kwargs))
                return 0, [200.0, 330.0, 550.0, -3.0, 0.0, 0.8]

        arm = FakeArm()
        with mock.patch.object(OFT.time, "sleep"):
            pose = OFT.startup_reset_to_home(
                arm,
                list(OFT.COLLECTION_RESET_POSITION_DEG),
                reset_speed=20,
                reset_pause=2,
                reset_timeout=30,
                dry_run=False,
            )

        reset_call = next(call for call in arm.calls if call[0] == "set_servo_angle")
        self.assertEqual(reset_call[1]["angle"], OFT.COLLECTION_RESET_POSITION_DEG)
        self.assertEqual(reset_call[1]["speed"], 20)
        self.assertEqual(reset_call[1]["timeout"], 30)
        self.assertTrue(reset_call[1]["wait"])
        np.testing.assert_array_equal(
            pose,
            np.array([200.0, 330.0, 550.0, -3.0, 0.0, 0.8]),
        )


if __name__ == "__main__":
    unittest.main()
