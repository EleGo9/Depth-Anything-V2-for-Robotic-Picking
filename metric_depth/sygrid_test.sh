#!/bin/bash
now=$(date +"%Y%m%d_%H%M%S")

bs=1
gpus=1
encoder=vits
dataset=sygrid # vkitti
img_size=518
min_depth=0.001 #1 millimeters
max_depth=0.80 # NOTE: # 80 m for virtual kitti
pretrained_from=../checkpoints/depth_anything_v2_${encoder}.pth
load_from=exp/sygrid_lr0_000005_maxd0_8m_ep120/latest.pth
save_path=exp/sygrid_lr0_000005_maxd0_8m_ep120 # exp/vkitti remember to change the name into 80cm

mkdir -p $save_path

export OPENCV_IO_ENABLE_OPENEXR=1


python3 -m torch.distributed.launch \
    --nproc_per_node=$gpus \
    --nnodes 1 \
    --node_rank=0 \
    --master_addr=localhost \
    --master_port=20596 \
    val.py --encoder $encoder --bs $bs --dataset $dataset --load_from $load_from\
    --img-size $img_size --min-depth $min_depth --max-depth $max_depth --pretrained-from $pretrained_from \
    --port 20596 2>&1 | tee -a $save_path/$now.log
