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

Compile the ransac voting layer:

```bash
cd path_to_object-posenet/lib/ransac_voting
python setup.py install --user
```

Compile the gpu version of knn:

```bash
cd path_to_object-posenet/lib/knn
python setup.py install --user
```

## Model Training

### Stage One

 - Mask R-CNN
 `python clearpose/networks/transparent6dofpose/stage1/transparent_segmentation/train_mask_rcnn.py`

 - DeepLabV3
 `python clearpose/networks/transparent6dofpose/stage1/surface_normals/train_deeplabv3.py`


## Stage Two