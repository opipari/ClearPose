# FFB6D

## Install

The source code of [FFB6D](https://github.com/ethnhe/FFB6D) is copied in ``clearpose/he_ffb6d/.`` We make some modification to train on clearpose dataset. Please follow the [FFB6D repo](https://github.com/ethnhe/FFB6D) for installation.
## Dataset

Please create a soft link for the dataset to the target path:
```bash
ln -s <path/to/dataset> <path/to/clearpose>/clearpose/he_ffb6d/ffb6d/datasets/clearpose
```

## Train & Test

Training

```bash
cd <path/to/clearpose>/clearpose/he_ffb6d/ffb6d/
python -m torch.distributed.launch --nproc_per_node=1 train_clearpose.py --gpu 0 --gpus=1 --opt_level O2 -train_depth_type GT -test_depth_type GT
## train_depth_type in ["GT", "raw"]  test_depth_type in ["GT", "raw"]
```

Testing

```bash
cd <path/to/clearpose>/clearpose/he_ffb6d/ffb6d/
python -m torch.distributed.launch --nproc_per_node=1 train_clearpose_test.py --gpu 0 -eval_net -checkpoint <path/to/checkpoint> -test_type wou -test_depth_type GT -test -test_pose -debug
## test_type in ["wou", "occlusion", "non-planner", "covered", "color", "standard"]  test_depth_type in ["GT", "raw"]
```