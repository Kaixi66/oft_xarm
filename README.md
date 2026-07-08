# RealWorld LeRobot to RLDS

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
