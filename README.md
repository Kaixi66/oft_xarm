# RealWorld LeRobot to RLDS

## UF850 real-robot evaluation

The real-robot launchers are in `oft_xarm/`. For normal use, edit only this
line near the top of `oft_xarm/serve_oft_xarm.sh`:

```bash
OFT_TASK_SELECTION="erase-circle-from-whiteboard"
```

Available tasks:

| Task | Instruction | Checkpoint |
| --- | --- | --- |
| `put-blue-bowl-in-second-drawer` | `put the blue bowl in the second drawer` | local 10k OFT checkpoint |
| `erase-circle-from-whiteboard` | `erase the circle from the whiteboard` | local 30k OFT checkpoint |

Start the selected model in terminal 1:

```bash
cd /home/zheyu/kaixi/oft_xarm/oft_xarm
./serve_oft_xarm.sh
```

After the server is ready, start the robot client in terminal 2:

```bash
cd /home/zheyu/kaixi/oft_xarm/oft_xarm
./run_inference_oft_xarm.sh
```

Every inference run automatically opens the gripper and returns the UF850 to
the collection reset pose before starting the cameras or querying the model.

You can also request the same reset manually:

```bash
cd /home/zheyu/kaixi/oft_xarm/oft_xarm
./reset_oft_arm
```

If inference is running, this asks the running client to stop policy actions
and reset safely. If inference is not running, it uses PAIR's guarded
standalone reset, verifies the joint pose, and leaves the arm in STOP.

Video recording is enabled by default. Every inference run saves both full
camera streams and its task metadata here:

```text
/home/zheyu/kaixi/oft_xarm/oft_xarm/outputs/videos/<task>/<timestamp_pid>/
  external.mp4
  wrist.mp4
  external_frames/frame_000000.jpg, frame_000001.jpg, ...
  wrist_frames/frame_000000.jpg, frame_000001.jpg, ...
  run_meta.json
```

The two frame directories contain aligned full-resolution JPEG pairs for every
recorded video frame. Frame saving defaults to 30 FPS with JPEG quality 95.

The client reads the task selected by the running server. Press `R` during
inference to stop policy motion, open the gripper, return to the collection
reset pose, and continue from a fresh observation.

The new task checkpoints require the exact training camera geometry. The
client enforces it and exits instead of silently using a mismatched crop:

```text
RealSense: 960x540 YUYV @ 60 FPS
external cam_1: [270, 0, 810, 540] -> 224x224
wrist    cam_0: [380, 0, 920, 540] -> 224x224
policy rate: 10 Hz
model action chunk: 25 steps
executed before each re-query: 8 steps
```

The checkpoints and runtime paths are configured directly in the launchers:

```text
/home/zheyu/kaixi/OFT-UF850-checkpoints
/home/zheyu/kaixi/openvla-oft
/home/zheyu/miniforge3/envs/openvla-oft-thor/bin/python
```

Single-script conversion for the current local LeRobot datasets:

```bash
/root/angli/hf_cache/lerobot/lab/xarm_setting1_51
/root/angli/hf_cache/lerobot/lab/xarm_setting2_51
```

Output defaults to:

```bash
/workspace/kaixi/RealWorld/rlds_data/utokyo_xarm_pick_and_place_converted_externally_to_rlds
```

Smoke test:

```bash
python /workspace/kaixi/RealWorld/lerobot_to_rlds.py \
  --overwrite \
  --max-episodes 2 \
  --max-frames-per-episode 16
```

Full conversion. By default `--val-ratio=0.0`, so all episodes go into the
`train` split and no `val` split is written:

```bash
python /workspace/kaixi/RealWorld/lerobot_to_rlds.py --overwrite
```

Split conversion for training the two settings separately. `--filter-noops`
drops idle frames (~30% of the raw data: near-zero motion, unchanged gripper),
matching the `remove_zero` filtering used for the pi0 checkpoints; without it
the policy learns to stand still at episode starts. These commands also keep
all episodes in `train` by default:

```bash
python /workspace/kaixi/RealWorld/lerobot_to_rlds.py \
  --overwrite --filter-noops \
  --dataset-root /root/angli/hf_cache/lerobot/lab/xarm_setting1_51 \
  --tfds-data-dir /workspace/kaixi/RealWorld/rlds_data_setting1

python /workspace/kaixi/RealWorld/lerobot_to_rlds.py \
  --overwrite --filter-noops \
  --dataset-root /root/angli/hf_cache/lerobot/lab/xarm_setting2_51 \
  --tfds-data-dir /workspace/kaixi/RealWorld/rlds_data_setting2
```

Thresholds are tunable via `--noop-pos-thresh` (cm/step, default 0.02) and
`--noop-rot-thresh` (rad/step, default 0.002). Gripper open/close transition
frames are always kept.

Train OpenVLA-OFT on one setting:

```bash
cd /workspace/kaixi/RealWorld
TASK=setting1 ./train_oft_realworld.sh
TASK=setting2 ./train_oft_realworld.sh

TASK=setting1 RUN_NAME=xarm_setting1_test01 ./train_oft_realworld.sh
```

Without an explicit `RUN_NAME`, each `TASK` uses its own default run name:

```text
oft_setting1_paper
oft_setting2_paper
oft_merged_paper
```

After training, merge the LoRA checkpoint before serving:

```bash
python /workspace/kaixi/RealWorld/merge_oft_lora_to_base.py \
  --checkpoint-dir /workspace/kaixi/RealWorld/openvla_oft_runs/checkpoints/oft_setting1_paper \
  --output-dir /workspace/kaixi/RealWorld/openvla_oft_runs/merged_public_checkpoints/oft_setting1_paper
```

Train OpenVLA-OFT with:

```bash
--data_root_dir /workspace/kaixi/RealWorld/rlds_data
--dataset_name utokyo_xarm_pick_and_place_converted_externally_to_rlds
```
