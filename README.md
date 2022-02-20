# ClearPose



## Dataset


## Models 


### (RGB) Peng et al. PVNet

<details>
<summary><strong>Setup</strong></summary>

Setup virtual environment 

```bash
python3 -m venv .venv/peng-pvnet-env
source .venv/peng-pvnet-env/bin/activate
pip install --upgrade pip
pip install -r clearpose/peng_pvnet/requirements.txt
pip install -e .
```

</details>

### (RGB-D) He et al. FFB6D

<details>
<summary><strong>Setup</strong></summary>

Setup virtual environment 

```bash
python3 -m venv .venv/he-ffb6d-env
source .venv/he-ffb6d-env/bin/activate
pip install --upgrade pip
pip install -r clearpose/he_ffb6d/requirements.txt
pip install -e .
```

</details>


### (RGB-D) Xu et al. 6DoF Transparent

<details>
<summary><strong>Setup</strong></summary>

Setup virtual environment 

```bash
python3 -m venv .venv/xu-6dof-env
source .venv/xu-6dof-env/bin/activate
pip install --upgrade pip
pip install -r clearpose/xu_6dof/requirements.txt
pip install -e .
```

Compile the ransac voting layer:

```bash
cd clearpose/xu_6dof/networks/references/posenet/ransac_voting
python setup.py install
```

</details>



<details>
<summary><strong>Training</strong></summary>

Stage One

 - Mask R-CNN
 `python clearpose/xu_6dof/networks/stage1/transparent_segmentation/train_mask_rcnn.py`

 - DeepLabV3
 `python clearpose/xu_6dof/networks/stage1/surface_normals/train_deeplabv3.py`


Stage Two
 - `python clearpose/xu_6dof/networks/stage2/train_stage2.py`

</details>
