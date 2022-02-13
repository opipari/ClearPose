import os
from scipy.io import loadmat
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

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
		plane = Image.open(plane_path)
		depth = Image.open(depth_path)
		meta = loadmat(meta_path)

		


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
		labels = torch.ones((num_objs,), dtype=torch.int64)
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
		
		depth = depth/meta['factor_depth'].item()
		plane = plane/meta['factor_depth'].item()

		H, W = depth.shape[1:]
		U, V = np.tile(np.arange(W), (H, 1)), np.tile(np.arange(H), (W, 1)).T
		fx, fy, cx, cy = self.intrinsic[0, 0], self.intrinsic[1, 1], self.intrinsic[0, 2], self.intrinsic[1, 2]
		X, Y = depth * (U - cx) / fx, depth * (V - cy) / fy

		X = F.convert_image_dtype(X)
		Y = F.convert_image_dtype(Y)
		uv = torch.stack([X,Y])



		from torchvision.ops import roi_align
		color, normal, plane, uv, targets = next(iter(data_loader))
		color, normal, plane, uv = torch.stack(color), torch.stack(normal), torch.stack(plane), torch.stack(uv)

		
		boxes_per_image = [len(t['boxes']) for t in targets]
		crops_color = roi_align(color, [t['boxes'] for t in targets], (80,80), 1, 1)
		crops_color_per_image = [crops_color[sum(boxes_per_image[:i]):sum(boxes_per_image[:i+1])] for i in range(len(boxes_per_image))]

		crops_normal = roi_align(normal, [t['boxes'] for t in targets], (80,80), 1, 1)
		crops_plane = roi_align(plane, [t['boxes'] for t in targets], (80,80), 1, 1)
		crops_uv = roi_align(uv, [t['boxes'] for t in targets], (80,80), 1, 1)
		crops_geometry_per_image = []
		for i in range(len(boxes_per_image)):
			crops_geometry.append(torch.cat([crops_normal[sum(boxes_per_image[:i]):sum(boxes_per_image[:i+1])],
											crops_plane[sum(boxes_per_image[:i]):sum(boxes_per_image[:i+1])],
											crops_uv[sum(boxes_per_image[:i]):sum(boxes_per_image[:i+1])]],
											dim=1))
		
		return color, normal, plane, uv, target
