import torch
import torch.nn as nn

from clearpose.xu_6dof.networks.stage1.transparent_segmentation.mask_rcnn import build_model as build_mask_rcnn_model
from clearpose.xu_6dof.networks.stage1.surface_normals.deeplabv3 import build_model as build_deeplabv3_model
from clearpose.xu_6dof.networks.stage2.build_model import build_model as build_stage2_model





class xu6dof_model(nn.Module):
	def __init__(self):
		super(xu6dof_model, self).__init__()

		self.mask_rcnn_model = build_mask_rcnn_model({"num_classes": 63})
		self.deeplabv3_model = build_deeplabv3_model()
		self.stage2_model = build_stage2_model({'num_obj': 63})


	def forward(self, color_norm, image_list):

		segmentation = self.mask_rcnn_model(image_list)
		normals = self.deeplabv3_model(color_norm)
		return segmentation, normals



def build_model(config):
	model = xu6dof_model()
	model.mask_rcnn_model.load_state_dict(torch.load(config["mask_rcnn_model"])['model_state_dict'], strict=False)
	model.deeplabv3_model.load_state_dict(torch.load(config["deeplabv3_model"])['model_state_dict'], strict=False)
	model.stage2_model.load_state_dict(torch.load(config["stage2_model"])['model_state_dict'], strict=False)

	return model
