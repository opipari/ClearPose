import os
import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F

from clearpose.xu_6dof.networks.references.detection.engine import train_one_epoch, evaluate
import clearpose.xu_6dof.networks.references.detection.utils as utils
import clearpose.xu_6dof.networks.references.detection.transforms as T

from clearpose.xu_6dof.networks.stage1.transparent_segmentation.mask_rcnn import build_model
from clearpose.xu_6dof.datasets.transparent_segmentation_dataset import TransparentSegmentationDataset



def get_transform(train):
	transforms = []
	transforms.append(T.ToTensor())
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

def main(config={"num_classes": 63}, save_dir=os.path.join("experiments","transparent_segmentation","models")):
	# train on the GPU or on the CPU, if a GPU is not available
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# use our dataset and defined transformations
	dataset = TransparentSegmentationDataset(transforms=get_transform(train=True))
	dataset_test = TransparentSegmentationDataset(transforms=get_transform(train=False))

	# split the dataset in train and test set
	indices = torch.randperm(len(dataset)).tolist()
	dataset = torch.utils.data.Subset(dataset, indices[:-50])
	dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

	# define training and validation data loaders
	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=2, shuffle=True, num_workers=4,
		collate_fn=utils.collate_fn)

	data_loader_test = torch.utils.data.DataLoader(
		dataset_test, batch_size=1, shuffle=False, num_workers=4,
		collate_fn=utils.collate_fn)

	# get the model using our helper function
	model = build_model(config)


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
	num_epochs = 100

	
	torch.save(model.state_dict(), os.path.join(save_dir,"mask_rcnn_0.pt"))
	for epoch in range(num_epochs):
		# train for one epoch, printing every 10 iterations
		train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=100)
		# update the learning rate
		lr_scheduler.step()
		# evaluate on the test dataset
		evaluate(model, data_loader_test, device=device)
		torch.save(model.state_dict(), os.path.join(save_dir,"mask_rcnn_"+str(epoch)+".pt"))
	


if __name__=="__main__":
	main()