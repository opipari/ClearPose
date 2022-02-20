import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F

from clearpose.xu_6dof.networks.references.segmentation.train import train_one_epoch
import clearpose.xu_6dof.networks.references.segmentation.transforms as T

from clearpose.xu_6dof.networks.stage1.surface_normals.deeplabv3 import build_model
from clearpose.xu_6dof.datasets.surface_normal_dataset import SurfaceNormalDataset



class NormalCriterion(nn.Module):
	def __init__(self):
		super(NormalCriterion, self).__init__()
		self.cosine_similarity = torch.nn.CosineSimilarity(dim=1)

	def forward(self, output, target):
		norm = torch.linalg.vector_norm(target, dim=1)
		norm_nonzero = torch.nonzero(norm, as_tuple=True)
		output_nonzero = output[norm_nonzero[0],:,norm_nonzero[1],norm_nonzero[2]]
		target_nonzero = target[norm_nonzero[0],:,norm_nonzero[1],norm_nonzero[2]]
		
		return torch.mean(1-self.cosine_similarity(output_nonzero, target_nonzero))


def get_transform(train):
	transforms = []
	transforms.append(T.PILToTensor())
	transforms.append(T.ConvertImageDtype(torch.float))
	transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
	if train:
		transforms.append(T.RandomHorizontalFlip(0.5))
	return T.Compose(transforms)


def show(imgs):
	if not isinstance(imgs, list):
		imgs = [imgs]
	fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
	for i, img in enumerate(imgs):
		img = img.detach()
		img = F.to_pil_image(img)
		axs[0, i].imshow(np.asarray(img))
		axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
	plt.show()

def main(save_dir=os.path.join("experiments","xu_6dof","stage1","surface_normals","models")):
	# train on the GPU or on the CPU, if a GPU is not available
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# use our dataset and defined transformations
	dataset = SurfaceNormalDataset(transforms=get_transform(train=True))
	dataset_test = SurfaceNormalDataset(transforms=get_transform(train=False))

	# split the dataset in train and test set
	indices = torch.randperm(len(dataset)).tolist()
	dataset = torch.utils.data.Subset(dataset, indices[:-50])
	dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

	# define training and validation data loaders
	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=2, drop_last=True, shuffle=True, num_workers=4)

	data_loader_test = torch.utils.data.DataLoader(
		dataset_test, batch_size=1, shuffle=False, num_workers=4)

	criterion = NormalCriterion()

	# get the model using our helper function
	model = build_model()


	# move model to the right device
	model.to(device)

	# construct an optimizer
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.SGD(params, lr=0.005,
								momentum=0.9, weight_decay=0.0005)
	# and a learning rate scheduler
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
												   step_size=3,
												   gamma=0.1)

	# let's train it for 10 epochs
	num_epochs = 10

	torch.save(model.state_dict(), os.path.join(save_dir,"deeplabv3_0.pt"))
	for epoch in range(num_epochs):
		# train for one epoch, printing every 10 iterations
		train_one_epoch(model, criterion, optimizer, data_loader, lr_scheduler, device, epoch, print_freq=100)
		# update the learning rate
		torch.save(model.state_dict(), os.path.join(save_dir,"deeplabv3_"+str(epoch)+".pt"))
		lr_scheduler.step()


if __name__=="__main__":
	main()