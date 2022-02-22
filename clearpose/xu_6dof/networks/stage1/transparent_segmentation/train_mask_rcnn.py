import os
import random
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


def main(config={"num_classes": 63}, save_dir=os.path.join("experiments","xu_6dof","stage1","transparent_segmentation","models")):
	# train on the GPU or on the CPU, if a GPU is not available
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# use our dataset and defined transformations
	dataset = TransparentSegmentationDataset(image_list="./data/train_images.csv", transforms=get_transform(train=True))
	dataset_test = TransparentSegmentationDataset(image_list="./data/test_images.csv", transforms=get_transform(train=False))

	# define training and validation data loaders
	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=5, shuffle=True, num_workers=4,
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
	optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0001)
	
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
											   step_size=120000,
											   gamma=0.1)

	logfile = open(os.path.join(save_dir,"mask_rcnn_log.txt"), 'w')

	# let's train it for 10 epochs
	num_epochs = 100
	
	torch.save({'epoch': -1,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'scheduler_state_dict': lr_scheduler.state_dict()},
				os.path.join(save_dir,"mask_rcnn_0.pt"))

	for epoch in range(num_epochs):
		train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler_=lr_scheduler, print_freq=100, logfile=logfile)
		# evaluate(model, data_loader_test, device=device)
		torch.save({'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict(),
				'scheduler_state_dict': lr_scheduler.state_dict()},
				os.path.join(save_dir,"mask_rcnn_"+str(epoch)+".pt"))
	
	logfile.close()


if __name__=="__main__":
	torch.manual_seed(0)
	random.seed(0)
	np.random.seed(0)
		
	main()