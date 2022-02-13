import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F

# from clearpose.networks.references.detection.engine import train_one_epoch, evaluate
import clearpose.networks.references.detection.utils as utils
import clearpose.networks.references.detection.transforms as T

from clearpose.networks.transparent6dofpose.stage1.transparent_segmentation.mask_rcnn import build_model
from clearpose.datasets.stage2_dataset import Stage2Dataset

from torch.utils.data._utils.collate import default_collate

def get_transform(train):
	transforms = []
	transforms.append(T.ToTensorSet())
	if train:
		transforms.append(T.RandomHorizontalFlipSet(0.5))
	return T.Compose(transforms)


def show(imgs,titles,subimgs):
	if not isinstance(imgs, list):
		imgs = [imgs]


	fig = plt.figure()
	print(1+int(math.ceil(subimgs.shape[0]/4)))
	gs = gridspec.GridSpec(1+int(math.ceil(subimgs.shape[0]/4)), 4, figure=fig)
	ax1 = fig.add_subplot(gs[0, :2])
	ax2 = fig.add_subplot(gs[0, 2:])
	for i, (img,title,ax) in enumerate(zip(imgs,titles,[ax1,ax2])):
		img = img.detach()
		img = F.to_pil_image(img)
		ax.imshow(np.asarray(img))
		ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
		ax.set_title(title)

	for i, img in enumerate(subimgs):
		ax = fig.add_subplot(gs[1+i//4, i%4])
		img = img.detach()
		img = F.to_pil_image(img)
		ax.imshow(np.asarray(img))
		ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

	plt.show()

def main():
	# train on the GPU or on the CPU, if a GPU is not available
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# use our dataset and defined transformations
	dataset = Stage2Dataset(transforms=get_transform(train=True))
	dataset_test = Stage2Dataset(transforms=get_transform(train=False))

	# split the dataset in train and test set
	indices = torch.randperm(len(dataset)).tolist()
	dataset = torch.utils.data.Subset(dataset, indices[:-50])
	dataset_test = torch.utils.data.Subset(dataset_test, indices[-50:])

	# define training and validation data loaders
	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=4, shuffle=True, num_workers=4,
		collate_fn=utils.collate_fn)

	data_loader_test = torch.utils.data.DataLoader(
		dataset_test, batch_size=1, shuffle=False, num_workers=4,
		collate_fn=utils.collate_fn)

	target_plot = draw_bounding_boxes((255*color[0]).type(torch.uint8).cpu(), targets[0]['boxes'], width=1)
	masks = targets[0]['masks'][targets[0]['masks'].sum(1).sum(1)<100000]
	masks = (masks.sum(0)>0.5)
	alpha_plot = draw_bounding_boxes((255*color[0]).type(torch.uint8).cpu()*masks.detach().cpu().type(torch.uint8), targets[0]['boxes'], width=1)
	
	show([target_plot, alpha_plot],['Ground Truth','Predictions'], crops_per_image[0])


	plt.imshow(crops_per_image[0][0].detach().cpu().permute(1,2,0))
	plt.show()


if __name__=="__main__":
	main()