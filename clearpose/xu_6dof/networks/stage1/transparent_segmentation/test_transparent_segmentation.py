import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F

from clearpose.networks.references.detection.engine import train_one_epoch, evaluate
import clearpose.networks.references.detection.utils as utils
import clearpose.networks.references.detection.transforms as T

from clearpose.networks.transparent6dofpose.stage1.transparent_segmentation.mask_rcnn import build_model
from clearpose.datasets.transparent_segmentation_dataset import TransparentSegmentationDataset



def get_transform(train):
	transforms = []
	transforms.append(T.ToTensor())
	if train:
		transforms.append(T.RandomHorizontalFlip(0.5))
	return T.Compose(transforms)


def show(imgs,titles):
	if not isinstance(imgs, list):
		imgs = [imgs]
	fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
	for i, (img,title) in enumerate(zip(imgs,titles)):
		img = img.detach()
		img = F.to_pil_image(img)
		axs[0, i].imshow(np.asarray(img))
		axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
		axs[0, i].set_title(title)
	plt.show()

def main():
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
		dataset, batch_size=4, shuffle=True, num_workers=4,
		collate_fn=utils.collate_fn)

	data_loader_test = torch.utils.data.DataLoader(
		dataset_test, batch_size=1, shuffle=False, num_workers=4,
		collate_fn=utils.collate_fn)

	# get the model using our helper function
	model = build_model({"num_classes": 63})

	model.load_state_dict(torch.load("./mask_rcnn_14.pt"), strict=False)
	model.to(device)

	# evaluate(model, data_loader_test, device=device)




	model.eval()
	cpu_device = torch.device("cpu")

	for images, targets in data_loader_test:
		images = list(img.to(device) for img in images)

		if torch.cuda.is_available():
			torch.cuda.synchronize()
		outputs = model(images)

		outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]


		target_plot = draw_bounding_boxes((255*images[0]).type(torch.uint8).cpu(), targets[0]['boxes'], width=1)
		box_plot = draw_bounding_boxes((255*images[0]).type(torch.uint8).cpu(), outputs[0]['boxes'], width=1)
		masks = outputs[0]['masks'][outputs[0]['masks'].sum(1).sum(1).sum(1)<100000]
		masks = (masks.sum(0)>0.5)
		mask_plot = draw_bounding_boxes(255*masks.detach().cpu().type(torch.uint8), outputs[0]['boxes'], width=1)
		alpha_plot = draw_bounding_boxes((255*images[0]).type(torch.uint8).cpu()*masks.detach().cpu().type(torch.uint8), outputs[0]['boxes'], width=1)
		show([target_plot, alpha_plot],['Ground Truth','Predictions'])
	


if __name__=="__main__":
	main()