# PoseClarified



## Setup


### Dataset


### Dependencies 

```bash
python3 -m venv .venv/clearpose-env
source .venv/clearpose-env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## Model Training

### Stage One

 - Mask R-CNN
 `python clearpose/networks/transparent6dofpose/stage1/transparent_segmentation/train_mask_rcnn.py`

 - DeepLabV3
 `python clearpose/networks/transparent6dofpose/stage1/surface_normals/train_deeplabv3.py`


## Stage Two