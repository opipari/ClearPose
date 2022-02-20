import os
from scipy.io import loadmat
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt




class SurfaceNormalDataset(Dataset):
	def __init__(self, image_list="./data/images.csv", 
					   transforms=None):
		self.image_list = [ln.strip().split(',') for ln in open(image_list,'r').readlines()]

		self.transforms = transforms

		
	def __len__(self):
		return len(self.image_list)


	def __getitem__(self, idx):
		_, scene_path, intid = self.image_list[idx]

		color_path = os.path.join(scene_path, intid+'-color.png')
		normal_path = os.path.join(scene_path, intid+'-normal_true.png')

		color = Image.open(color_path).convert("RGB")
		normal = Image.open(normal_path).convert("RGB")

		# print((color.size[0]//1.5, color.size[1]//1.5))
		color = color.resize((int(color.size[0]//1.5), int(color.size[1]//1.5)))
		normal = normal.resize((int(normal.size[0]//1.5), int(normal.size[1]//1.5)))

		if self.transforms is not None:
			color, normal = self.transforms(color, normal)
		
		normal = normal.type(torch.float32)/255
		norm = torch.linalg.vector_norm(normal, dim=0, keepdim=True)
		norm_nonzero = torch.nonzero(norm, as_tuple=True)
		normal[norm_nonzero] = normal[norm_nonzero] / norm[norm_nonzero]
		return color, normal
