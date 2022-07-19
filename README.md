# Xu et al. 6DoF Pose Estimation of Transparent Object from a Single RGB-D Image

This branch contains our reimplementation of the model presented by [Xu et al.](https://www.mdpi.com/1424-8220/20/23/6790).

## Install

Setup virtual environment 
** Note that since development, pytorch3d 

```bash
python3.8 -m venv .venv/xu-6dof-env
source .venv/xu-6dof-env/bin/activate
pip install --upgrade pip
pip install -r clearpose/xu_6dof/requirements.txt
pip install -e .
```

** Note that in the time between development and release of clearpose, the pytorch3d dependency no longer supports prebuilt binaries for the required version 0.6.1 and will need to be installed from [source available on github](https://github.com/facebookresearch/pytorch3d/releases/tag/v0.6.1).

Compile the ransac voting layer:

```bash
cd clearpose/xu_6dof/networks/references/posenet/ransac_voting
python setup.py install
```


## Dataset

Please create a soft link for the dataset to the target path:
```bash
ln -s <path/to/dataset> <path/to/clearpose>/data/clearpose
```

Next create the train/test split files
```bash
python data/preprocess.py --root <path/to/clearpose>/data/clearpose
```

## Train & Test

Training Stage One

 - Mask R-CNN
 `python clearpose/xu_6dof/networks/stage1/transparent_segmentation/train_mask_rcnn.py`

 - DeepLabV3
 `python clearpose/xu_6dof/networks/stage1/surface_normals/train_deeplabv3.py`


Training Stage Two
 - `python clearpose/xu_6dof/networks/stage2/train_stage2.py`

Testing

<!-- ```bash
cd <path/to/clearpose>/clearpose/he_ffb6d/ffb6d/
python -m torch.distributed.launch --nproc_per_node=1 train_clearpose_test.py --gpu 0 -eval_net -checkpoint <path/to/checkpoint> -test_type wou -test_depth_type GT -test -test_pose -debug
## test_type in ["wou", "occlusion", "non-planner", "covered", "color", "standard"]  test_depth_type in ["GT", "raw"]
``` -->
