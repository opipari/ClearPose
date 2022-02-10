import random
import numpy as np
import torch
import torch.nn as nn

from clearpose.networks.transparent6dofpose.stage1 import build_model



class Transparent6DoFPose(nn.Module):
	def __init__(self):
		super(Transparent6DoFPose, self).__init__()



	def forward(self, x):
		return x




def build_model(config):

	model = Transparent6DoFPose()

	return model