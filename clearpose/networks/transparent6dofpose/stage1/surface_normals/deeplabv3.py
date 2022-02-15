import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class deeplabv3_resnet50_normals(nn.Module):
	def __init__(self):
		super(deeplabv3_resnet50_normals, self).__init__()

		self.deeplabv3 = deeplabv3_resnet50(pretrained=True, 
									  progress=True, 
									  num_classes=21, 
									  aux_loss=None,
									  pretrained_backbone=True)

		num_classes = 3
		in_channels = self.deeplabv3.classifier[-1].in_channels
		kernel_size = self.deeplabv3.classifier[-1].kernel_size
		stride = self.deeplabv3.classifier[-1].stride
		self.deeplabv3.classifier[-1] = nn.Conv2d(in_channels, num_classes, kernel_size, stride)
		
	def forward(self, x):
		normals = self.deeplabv3(x)['out']
		normals = normals / torch.linalg.vector_norm(normals, dim=1, keepdim=True)
		return normals


def build_model():
	deeplabv3 = deeplabv3_resnet50_normals()
	return deeplabv3
