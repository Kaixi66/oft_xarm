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

Full conversion:

```bash
python /workspace/kaixi/RealWorld/lerobot_to_rlds.py --overwrite
```

Split conversion for training the two settings separately:

```bash
python /workspace/kaixi/RealWorld/lerobot_to_rlds.py \
  --overwrite \
  --dataset-root /root/angli/hf_cache/lerobot/lab/xarm_setting1_51 \
  --tfds-data-dir /workspace/kaixi/RealWorld/rlds_data_setting1

python /workspace/kaixi/RealWorld/lerobot_to_rlds.py \
  --overwrite \
  --dataset-root /root/angli/hf_cache/lerobot/lab/xarm_setting2_51 \
  --tfds-data-dir /workspace/kaixi/RealWorld/rlds_data_setting2
```

Train OpenVLA-OFT on one setting:

```bash
cd /workspace/kaixi/RealWorld
TASK=setting1 ./train_oft_realworld.sh
TASK=setting2 ./train_oft_realworld.sh

TASK=setting1 RUN_NAME=xarm_setting1_test01 ./train_oft_realworld.sh
```

Train OpenVLA-OFT with:

```bash
--data_root_dir /workspace/kaixi/RealWorld/rlds_data
--dataset_name utokyo_xarm_pick_and_place_converted_externally_to_rlds
```
