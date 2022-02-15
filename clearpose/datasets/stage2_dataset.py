import os
import cv2
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.ops import roi_align

import matplotlib.pyplot as plt




class Stage2Dataset(Dataset):
	def __init__(self, image_list="./data/images.csv", 
					   object_list="./data/objects.csv",
					   transforms=None):
		self.image_list = [ln.strip().split(',') for ln in open(image_list,'r').readlines()]
		self.object_list = [ln.strip().split(',') for ln in open(object_list,'r').readlines()]

		self.object_lookup_id = {obj[1]: int(obj[0]) for obj in self.object_list}
		self.object_lookup_name = {int(obj[0]): obj[1] for obj in self.object_list}

		self.transforms = transforms

		self.intrinsic = np.array([[902.187744140625, 0.0, 662.3499755859375],
						  [0.0, 902.391357421875, 372.2278747558594], 
						  [0.0, 0.0, 1.0]])
		
	def __len__(self):
		return len(self.image_list)


	def __getitem__(self, idx):
		_, scene_path, intid = self.image_list[idx]

		color_path = os.path.join(scene_path, intid+'-color.png')
		mask_path = os.path.join(scene_path, intid+'-label.png')
		normal_path = os.path.join(scene_path, intid+'-normal_true.png')
		plane_path = os.path.join(scene_path, intid+'-tableplane_depth.png')
		depth_path = os.path.join(scene_path, intid+'-depth_true.png')
		meta_path = os.path.join(scene_path, intid+'-meta.mat')

		color = Image.open(color_path).convert("RGB")
		mask = Image.open(mask_path)
		normal = Image.open(normal_path)
		plane = np.array(Image.open(plane_path))
		depth = np.array(Image.open(depth_path))
		meta = loadmat(meta_path)
		
		depth = Image.fromarray(depth/meta['factor_depth'].item())
		plane = Image.fromarray(plane/meta['factor_depth'].item())


		mask = np.array(mask)
		obj_ids = np.unique(mask)
		obj_ids = obj_ids[1:]
		
		masks = mask == obj_ids[:, None, None]

		num_objs = len(obj_ids)
		boxes = []
		for i in range(num_objs):
			pos = np.where(masks[i])
			xmin = np.min(pos[1])
			xmax = np.max(pos[1])
			ymin = np.min(pos[0])
			ymax = np.max(pos[0])
			boxes.append([xmin, ymin, xmax, ymax])
		
		boxes = torch.as_tensor(boxes, dtype=torch.float32)
		labels = torch.from_numpy(obj_ids, dtype=torch.int64)
		masks = torch.as_tensor(masks, dtype=torch.uint8)


		image_id = torch.tensor([idx])
		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
		iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
		
		min_area = 100
		valid_area = area>min_area
		boxes = boxes[valid_area]
		area = area[valid_area]
		labels = labels[valid_area]
		masks = masks[valid_area]
		iscrowd = iscrowd[valid_area]

		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["masks"] = masks
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd

		if self.transforms is not None:
			(color,normal,plane,depth), target = self.transforms((color,normal,plane,depth), target)
		
		
		H, W = depth.shape[1:]
		U, V = np.tile(np.arange(W), (H, 1)), np.tile(np.arange(H), (W, 1)).T
		fx, fy, cx, cy = self.intrinsic[0, 0], self.intrinsic[1, 1], self.intrinsic[0, 2], self.intrinsic[1, 2]
		X, Y = depth * (U - cx) / fx, depth * (V - cy) / fy

		X = F.convert_image_dtype(X)
		Y = F.convert_image_dtype(Y)
		Z = F.convert_image_dtype(depth)
		uv = torch.stack([X,Y,Z]).squeeze(1)

		boxes_per_image = len(target['boxes'])
		crops_color = roi_align(color.unsqueeze(0), [target['boxes']], (80,80), 1, 1)

		crops_normal = roi_align(normal.unsqueeze(0), [target['boxes']], (80,80), 1, 1)
		crops_plane = roi_align(plane.unsqueeze(0), [target['boxes']], (80,80), 1, 1)
		crops_uvz = roi_align(uv.unsqueeze(0), [target['boxes']], (80,80), 1, 1)
		crops_masks = roi_align(target["masks"].unsqueeze(1).type(torch.float32), [box.unsqueeze(0) for box in target['boxes']], (80,80), 1, 1)
		# crops_depth = roi_align(depth.unsqueeze(0), [target['boxes']], (80,80), 1, 1)
		crops_geometry_per_image = torch.cat([crops_normal, crops_plane, crops_uvz], dim=1)

		poses = np.transpose(meta['poses'], (2,0,1))
		target["poses"] = torch.from_numpy(poses)
		target["quats"] = torch.stack([torch.from_numpy(R.from_matrix(rmat[:, :3]).as_quat()) for rmat in poses])
		target["trans"] = torch.stack([torch.from_numpy(rmat[:, 3]) for rmat in poses])

		return color, crops_color, crops_geometry_per_image, crops_masks, target
