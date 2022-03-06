import os
import math
import numpy as np
import torch
from torchvision.ops import roi_align
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes

import clearpose.xu_6dof.networks.references.detection.utils as utils
import clearpose.xu_6dof.networks.references.detection.transforms as T

from clearpose.xu_6dof.networks.build_model import build_model
from clearpose.xu_6dof.datasets.clear_pose_dataset import ClearPoseDataset

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec




def get_transform(train):
	transforms = []
	transforms.append(T.ToTensorSet())
	transforms.append(T.NormalizeSet(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
	# if train:
	# 	transforms.append(T.RandomHorizontalFlipSet(0.5))
	return T.Compose(transforms)


def show(imgs,titles,subimgs=None):
	if not isinstance(imgs, list):
		imgs = [imgs]

	if subimgs is None:
		fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
		for i, (img,title) in enumerate(zip(imgs,titles)):
			img = img.detach()
			img = F.to_pil_image(img)
			axs[0, i].imshow(np.asarray(img))
			axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
			axs[0, i].set_title(title)
	else:
		fig = plt.figure()
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



if __name__=="__main__":

	verbose_plots = True

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	dataset_test = ClearPoseDataset(image_list="./data/train_images.csv", transforms=get_transform(train=False))


	model_config = {"mask_rcnn_model": os.path.join("experiments","xu_6dof","stage1","transparent_segmentation","models","mask_rcnn_2.pt"),
					"deeplabv3_model": os.path.join("experiments","xu_6dof","stage1","surface_normals","models","deeplabv3_1.pt"),
					"stage2_model": os.path.join("experiments","xu_6dof","stage2","models","predepth","stage2_15.pt")}
	model = build_model(model_config)
	model.eval()
	model.to(device)

	with torch.no_grad():
		for color_norm, color, uvz, color_crops_norm, color_crops, geom_crops, masks_crops, targets in dataset_test:
			color_norm = color_norm.unsqueeze(0).to(device)
			color = color.to(device)
			uvz = uvz.to(device)
			plane = uvz[2]
			targets = targets = [{k: v.to(device) for k, v in targets.items()}]
			images = [color]

			segmentation_output, normal_output = model(color_norm, images)


			segmentation_output = [{k: v.to('cpu') for k, v in t.items()} for t in segmentation_output]

			if verbose_plots:
				normal = (255*(normal_output[0]+1)/2).type(torch.uint8).cpu().detach()
				color_plot = (255*images[0]).type(torch.uint8).cpu().detach()
				show([color_plot, normal],['Color Input','DeepLabv3 Predictions'])


				target_plot = draw_bounding_boxes((255*images[0]).type(torch.uint8).cpu(), targets[0]['boxes'], width=1)
				box_plot = draw_bounding_boxes((255*images[0]).type(torch.uint8).cpu(), segmentation_output[0]['boxes'], width=1)
				masks = segmentation_output[0]['masks'][segmentation_output[0]['masks'].sum(1).sum(1).sum(1)<100000]
				masks = (masks.sum(0)>0.5)
				mask_plot = draw_bounding_boxes(255*masks.detach().cpu().type(torch.uint8), segmentation_output[0]['boxes'], width=1)
				alpha_plot = draw_bounding_boxes((255*images[0]).type(torch.uint8).cpu()*masks.detach().cpu().type(torch.uint8), segmentation_output[0]['boxes'], width=1)
				show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','Mask RCNN Predictions'])
				plt.close()


			pred_crops_color = roi_align(color.unsqueeze(0), [segmentation_output[0]['boxes'].to(device)], (80,80), 1, 1)
			pred_crops_color_normal = roi_align(color_norm, [segmentation_output[0]['boxes'].to(device)], (80,80), 1, 1)
			pred_crops_normal = roi_align(normal_output, [segmentation_output[0]['boxes'].to(device)], (80,80), 1, 1)
			pred_crops_plane = roi_align(plane.unsqueeze(0).unsqueeze(0), [segmentation_output[0]['boxes'].to(device)], (80,80), 1, 1)
			pred_crops_uvz = roi_align(uvz.unsqueeze(0), [segmentation_output[0]['boxes'].to(device)], (80,80), 1, 1)
			pred_crops_masks = roi_align(segmentation_output[0]['masks'].to(device).type(torch.float32), [bx.unsqueeze(0) for bx in segmentation_output[0]['boxes'].to(device)], (80,80), 1, 1)
			pred_crops_geom = torch.cat([pred_crops_normal, pred_crops_plane, pred_crops_uvz], dim=1)	
		

			if verbose_plots:
				target_plot = draw_bounding_boxes((255*images[0]).type(torch.uint8).cpu(), targets[0]['boxes'], width=1)
				masks = segmentation_output[0]['masks'][segmentation_output[0]['masks'].sum(1).sum(1).sum(1)<100000]
				masks = (masks.sum(0)>0.5)
				alpha_plot = draw_bounding_boxes((255*images[0]).type(torch.uint8).cpu()*masks.detach().cpu().type(torch.uint8), segmentation_output[0]['boxes'], width=1)

				# Ground Truth Crops
				show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','color'], color_crops)
				show([target_plot, alpha_plot],['Ground Truth Bounding Boxes',' masks'], masks_crops)
				show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','normals'], geom_crops[:,:3])
				show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','plane'], geom_crops[:,3])
				show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','x'], geom_crops[:,4])
				show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','y'], geom_crops[:,5])

				# Predicted Crops
				show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','color (pred)'], pred_crops_color)
				show([target_plot, alpha_plot],['Ground Truth Bounding Boxes',' masks (pred)'], pred_crops_masks)
				show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','normals (pred)'], pred_crops_geom[:,:3])
				show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','plane'], pred_crops_geom[:,3])
				show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','x'], pred_crops_geom[:,4])
				show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','y'], pred_crops_geom[:,5])


			geom_crops = geom_crops.to(device)
			masks_crops = masks_crops.to(device)
			obj_ids = obj_ids.to(device)
			tx, rx, cx, choose = model(pred_crops_color_normal, pred_geom_crops, masks_crops.permute(0,2,3,1), obj_ids)
