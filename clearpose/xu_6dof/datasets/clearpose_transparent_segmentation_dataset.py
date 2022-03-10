import os
from scipy.io import loadmat
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt




class TransparentSegmentationDataset(Dataset):
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
		_, scene_path, intid, num_objs_ = self.image_list[idx]

		color_path = os.path.join(scene_path, intid+'-color.png')
		mask_path = os.path.join(scene_path, intid+'-label.png')
		meta_path = os.path.join(scene_path, 'metadata.mat')

		meta = loadmat(meta_path)[intid][0,0]

		color = Image.open(color_path).convert("RGB")
		mask = Image.open(mask_path)

		mask = np.array(mask)
		cls_indexes = meta[0]
		obj_ids = cls_indexes.flatten()
		ign_ids = obj_ids<200
		obj_ids = obj_ids[ign_ids]
		masks = (mask == obj_ids[:, None, None])
		obj_ids = obj_ids-1
		num_objs = len(obj_ids)
		assert num_objs==int(num_objs_), "{} {}".format(num_objs, num_objs_)

		box_list = []
		for idx_offset, label in enumerate(obj_ids):
			pos = np.where(masks[idx_offset])
			xmin = np.min(pos[1])
			xmax = np.max(pos[1])
			ymin = np.min(pos[0])
			ymax = np.max(pos[0])
			box_list.append([xmin, ymin, xmax, ymax])

		boxes = torch.as_tensor(box_list, dtype=torch.float32)
		labels = torch.as_tensor(np.array(obj_ids), dtype=torch.int64)
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
			color, target = self.transforms(color, target)

		return color, target
