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

from clearpose.networks.transparent6dofpose.stage2.build_model import build_model
from clearpose.datasets.stage2_dataset import Stage2Dataset

from torch.utils.data._utils.collate import default_collate

def get_transform(train):
	transforms = []
	transforms.append(T.ToTensorSet())
	# if train:
	# 	transforms.append(T.RandomHorizontalFlipSet(0.5))
	return T.Compose(transforms)


def loss(pred_r, pred_t, pred_c, target_r, target_t):
	print(pred_r.shape, pred_t.shape, pred_c.shape)
	print(target_r.shape, target_t.shape)
	# rot_anchors = torch.from_numpy(rot_anchors).float().cuda()
	# rot_anchors = rot_anchors.unsqueeze(0).repeat(bs, 1, 1).permute(0, 2, 1)
	# cos_dist = torch.bmm(pred_r, rot_anchors)   # bs x num_rot x num_rot
	# loss_reg = F.threshold((torch.max(cos_dist, 2)[0] - torch.diagonal(cos_dist, dim1=1, dim2=2)), 0.001, 0)
	# loss_reg = torch.mean(loss_reg)



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

	color, color_crops, geom_crops, masks_crops, targets = next(iter(data_loader_test))
	boxes_per_image = [c.shape[0] for c in color_crops]
	color, color_crops, geom_crops, masks_crops, obj_ids = torch.stack(color), torch.cat(color_crops), torch.cat(geom_crops), torch.cat(masks_crops), torch.cat([t["labels"] for t in targets]) 

	target_plot = draw_bounding_boxes((255*color[0]).type(torch.uint8).cpu(), targets[0]['boxes'], width=1)
	masks = (targets[0]['masks'].sum(0)>0.5)
	alpha_plot = draw_bounding_boxes((255*color[0]).type(torch.uint8).cpu()*masks.detach().cpu().type(torch.uint8), targets[0]['boxes'], width=1)

	# show([target_plot, alpha_plot],['Ground Truth','color'], color_crops[sum(boxes_per_image[:0]):sum(boxes_per_image[:0+1])])
	# show([target_plot, alpha_plot],['Ground Truth',' masks'], masks_crops[sum(boxes_per_image[:0]):sum(boxes_per_image[:0+1])])
	# show([target_plot, alpha_plot],['Ground Truth',' depth'], depth_crops[sum(boxes_per_image[:0]):sum(boxes_per_image[:0+1])])
	# show([target_plot, alpha_plot],['Ground Truth','normals'], geom_crops[sum(boxes_per_image[:0]):sum(boxes_per_image[:0+1])][:,:3])
	# show([target_plot, alpha_plot],['Ground Truth','plane'], geom_crops[sum(boxes_per_image[:0]):sum(boxes_per_image[:0+1])][:,3])
	# show([target_plot, alpha_plot],['Ground Truth','x'], geom_crops[sum(boxes_per_image[:0]):sum(boxes_per_image[:0+1])][:,4])
	# show([target_plot, alpha_plot],['Ground Truth','y'], geom_crops[sum(boxes_per_image[:0]):sum(boxes_per_image[:0+1])][:,5])

	print(color_crops.shape, geom_crops.shape, masks_crops.shape, obj_ids.shape)

	model = build_model({'num_obj': 63})
	model.cuda()
	tx, rx, cx = model(color_crops.cuda(), geom_crops.permute(0,2,3,1).cuda(), masks_crops.permute(0,2,3,1).cuda(), obj_ids.cuda())

	print(tx.shape, rx.shape, cx.shape)
	print([t["quats"] for t in targets])
	loss(rx, tx, cx, targets["quats"], targets["trans"])

if __name__=="__main__":
	main()