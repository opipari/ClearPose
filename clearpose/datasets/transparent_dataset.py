import os
from scipy.io import loadmat
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt




class TransparentDataset(Dataset):
	def __init__(self, image_list="./data/images.csv", 
					   object_list="./data/objects.csv",
					   transforms=None):
		self.image_list = [ln.strip().split(',') for ln in open(image_list,'r').readlines()]
		self.object_list = [ln.strip().split(',') for ln in open(object_list,'r').readlines()]

		self.object_lookup_id = {obj[1]: int(obj[0]) for obj in self.object_list}
		self.object_lookup_name = {int(obj[0]): obj[1] for obj in self.object_list}

		self.transforms = transforms

		
	def __len__(self):
		return len(self.image_list)


	def __getitem__(self, idx):
		_, scene_path, intid = self.image_list[idx]

		color_path = os.path.join(scene_path, intid+'-color.png')
		mask_path = os.path.join(scene_path, intid+'-label.png')
		depth_path = os.path.join(scene_path, intid+'-depth.png')
		# meta_path = os.path.join(scene_path, intid+'-meta.mat')
		# box_path = os.path.join(scene_path, intid+'-box.txt')

		color = Image.open(color_path).convert("RGB")
		mask = Image.open(mask_path)
		# depth = Image.open(depth_path)
		# boxes = [ln.strip().split() for ln in open(box_path,'r').readlines()]
		# Bounding Boxes in [tlx, tly, brx, bry]
		# boxes = [[box[0], self.object_lookup_id[box[0]], [int(i) for i in box[1:5]]] for box in boxes]

		# mat = loadmat(meta_path)
		# cls_indexes = mat['cls_indexes']
		# center = mat['center']
		# factor_depth = mat['factor_depth']
		# intrinsic_matrix = mat['intrinsic_matrix']
		# poses = mat['poses']
		# rotation_translation_matrix = mat['rotation_translation_matrix']

		width, height = mask.size

		mask = np.array(mask)
		obj_ids = np.unique(mask)

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

		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["masks"] = masks
		target["image_id"] = image_id
		target["area"] = area
		target["iscrowd"] = iscrowd


		if self.transforms is not None:
			color, target = self.transforms(color, target)

		return color, target
