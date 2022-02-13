import torch.nn as nn
from clearpose.networks.transparent6dofpose.stage1.transparent_segmentation import mask_rcnn
from clearpose.networks.transparent6dofpose.stage1.surface_normals import deeplabv3





class stage1_model(nn.Module):
	def __init__(self):
		super(stage1_model, self).__init__()

		self.mask_rcnn_model = mask_rcnn.build_model({"num_classes": 63})
		self.deeplabv3_model = deeplabv3.build_model()


	def forward(self, x):


		segmentation = self.mask_rcnn_model(x)
		normals = self.deeplabv3_model(x)



def build_model(config):
	model = stage1_model()
	model.mask_rcnn_model.load_state_dict(torch.load(config["mask_rcnn_model"]), strict=False)
	model.deeplabv3_model.load_state_dict(torch.load(config["deeplabv3_model"]), strict=False)

	return model
