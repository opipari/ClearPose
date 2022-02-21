#!/bin/bash
n_gpu=1  # number of gpu to use
checkpoint=train_log/ycb/checkpoints/FFB6D_baseline.pth.tar
# python /home/huijie/research/progresslabeller/FFB6D/ffb6d/datasets/ycb/dataset_config/generate_list.py train primesense train_scene* 0.1
# python /home/huijie/research/progresslabeller/FFB6D/ffb6d/datasets/ycb/dataset_config/generate_list.py test primesense train_scene11 0.1
python3 -m torch.distributed.launch --nproc_per_node=$n_gpu train_ycb.py --gpus=$n_gpu --opt_level O2 -checkpoint $checkpoint
