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
		normal_path = os.path.join(scene_path, intid+'-color.png')

		color = Image.open(color_path).convert("RGB")
		normal = Image.open(normal_path).convert("RGB")

		if self.transforms is not None:
			color, normal = self.transforms(color, normal)

		return color, normal
