import os
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as TF
import torch.nn.functional as F


# from clearpose.xu_6dof.networks.references.detection.engine import train_one_epoch, evaluate
import clearpose.xu_6dof.networks.references.detection.utils as utils
import clearpose.xu_6dof.networks.references.detection.transforms as T

from clearpose.xu_6dof.networks.stage2.build_model import build_model
from clearpose.xu_6dof.datasets.stage2_dataset import Stage2Dataset

from clearpose.xu_6dof.networks.references.posenet.ransac_voting.ransac_voting_gpu import ransac_voting_layer

from torch.utils.data._utils.collate import default_collate

from pytorch3d.renderer import (
    look_at_view_transform,
    PerspectiveCameras, 
    PointLights, 
    DirectionalLights,
    AmbientLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.transforms import quaternion_apply, Translate

def get_transform(train):
	transforms = []
	transforms.append(T.ToTensorSet())
	# if train:
	# 	transforms.append(T.RandomHorizontalFlipSet(0.5))
	return T.Compose(transforms)


class Loss(nn.Module):
	def __init__(self, rot_anchors):
		super(Loss, self).__init__()
		self.rot_anchors = rot_anchors

	def forward(self, pred_t, pred_r, pred_c, target_r, target_t, model_points, choose, symmetric, diameters):
		bs = pred_r.shape[0]
		num_rot = self.rot_anchors.shape[0]

		rot_anchors = torch.from_numpy(self.rot_anchors).float().to(pred_r.device)
		rot_anchors = rot_anchors.unsqueeze(0).repeat(bs, 1, 1).permute(0, 2, 1)

		cos_dist = torch.bmm(pred_r, rot_anchors)   # bs x num_rot x num_rot
		loss_reg = F.threshold((torch.max(cos_dist, 2)[0] - torch.diagonal(cos_dist, dim1=1, dim2=2)), 0.001, 0)
		loss_reg = torch.mean(loss_reg)

		# rotation loss
		rotations = torch.cat(((1.0 - 2.0*(pred_r[:, :, 2]**2 + pred_r[:, :, 3]**2)).view(bs, num_rot, 1),\
							   (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] - 2.0*pred_r[:, :, 0]*pred_r[:, :, 3]).view(bs, num_rot, 1), \
							   (2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_rot, 1), \
							   (2.0*pred_r[:, :, 1]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 3]*pred_r[:, :, 0]).view(bs, num_rot, 1), \
							   (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 3]**2)).view(bs, num_rot, 1), \
							   (-2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_rot, 1), \
							   (-2.0*pred_r[:, :, 0]*pred_r[:, :, 2] + 2.0*pred_r[:, :, 1]*pred_r[:, :, 3]).view(bs, num_rot, 1), \
							   (2.0*pred_r[:, :, 0]*pred_r[:, :, 1] + 2.0*pred_r[:, :, 2]*pred_r[:, :, 3]).view(bs, num_rot, 1), \
							   (1.0 - 2.0*(pred_r[:, :, 1]**2 + pred_r[:, :, 2]**2)).view(bs, num_rot, 1)), dim=2).contiguous().view(bs,num_rot, 3, 3)
		rotations = rotations.contiguous().transpose(3, 2).contiguous()

		loss_r = 0
		for i in range(len(model_points)):
			obj_diameter = diameters[i]
			model_points_ = model_points[i].view(1, 1, model_points[i].shape[1], 3).repeat(1, num_rot, 1, 1).view(1*num_rot, model_points[i].shape[1], 3)
			model_points_ = model_points_.to(rotations.device)
			pred_r = torch.bmm(model_points_, rotations[i])
			if symmetric[i]:
				dist = torch.linalg.vector_norm(pred_r.unsqueeze(2) - target_r[i].unsqueeze(1).to(pred_r.device), dim=3)
				loss_r += torch.mean(torch.mean(dist.min(2)[0], dim=1) / (obj_diameter*pred_c[i]) + torch.log(pred_c[i]), dim=0)
			else:
				dist = torch.linalg.vector_norm(pred_r - target_r[i].to(pred_r.device), dim=2)
				loss_r += torch.mean(torch.mean(dist, dim=1) / (obj_diameter*pred_c[i]) + torch.log(pred_c[i]), dim=0)

		target_t_samples = target_t.permute(0,2,3,1).flatten(start_dim=1, end_dim=2)[choose].view(pred_t.shape).to(pred_t.device)
		loss_t = F.smooth_l1_loss(pred_t, target_t_samples, reduction='mean')
		return loss_r + 2*loss_reg + 5*loss_t, loss_r, loss_reg, loss_t #loss_r + 2*loss_reg + 5*loss_t


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
		img = TF.to_pil_image(img)
		ax.imshow(np.asarray(img))
		ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
		ax.set_title(title)

	for i, img in enumerate(subimgs):
		ax = fig.add_subplot(gs[1+i//4, i%4])
		img = img.detach()
		img = TF.to_pil_image(img)
		ax.imshow(np.asarray(img))
		ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

	plt.show()

def train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq, logfile, scaler=None):
	model.train()
	metric_logger = utils.MetricLogger(delimiter="  ", logfile=logfile)
	metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
	header = f"Epoch: [{epoch}]"

	i=0
	for color, color_crops, geom_crops, masks_crops, targets in metric_logger.log_every(data_loader, print_freq, header):
		# boxes_per_image = [c.shape[0] for c in color_crops]
		color, color_crops, geom_crops, masks_crops, obj_ids = torch.stack(color), torch.cat(color_crops), torch.cat(geom_crops), torch.cat(masks_crops), torch.cat([t["labels"] for t in targets]) 
		
		target_quats = torch.cat([t["quats"] for t in targets])
		target_trans = torch.cat([t["trans"] for t in targets])
		target_diameters = torch.stack([t["diameter"] for t in targets])
		target_meshes = [t["mesh"] for t in targets]
		target_mesh_rots = [t["mesh_rot"] for t in targets]
		target_symmetric = [t["symmetric"] for t in targets]

		color_crops = color_crops.to(device)
		geom_crops = geom_crops.to(device)
		masks_crops = masks_crops.to(device)
		obj_ids = obj_ids.to(device)
		with torch.cuda.amp.autocast(enabled=scaler is not None):
			tx, rx, cx, choose = model(color_crops, geom_crops.permute(0,2,3,1), masks_crops.permute(0,2,3,1), obj_ids)
			loss, loss_r, loss_reg, loss_t = criterion(tx, rx, cx, target_mesh_rots, target_trans, target_meshes, choose, target_symmetric, target_diameters)


		optimizer.zero_grad()
		if scaler is not None:
			scaler.scale(loss).backward()
			scaler.step(optimizer)
			scaler.update()
		else:
			loss.backward()
			optimizer.step()


		metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"], loss_r=loss_r, loss_reg=loss_reg, loss_t=loss_t)
		
		if i>20000:
			break
		i+=1

def test_quant(model, criterion, optimizer, data_loader, device, print_freq, logfile, scaler=None, e=1):
	model.eval()
	metric_logger = utils.MetricLogger(delimiter="  ", logfile=logfile)
	header = f"Test"

	success_count = [0 for i in range(model.num_obj)]
	num_count = [0 for i in range(model.num_obj)]

	b=0
	for color, color_crops, geom_crops, masks_crops, targets in metric_logger.log_every(data_loader, print_freq, header):
		# boxes_per_image = [c.shape[0] for c in color_crops]
		color, color_crops, geom_crops, masks_crops, obj_ids = torch.stack(color), torch.cat(color_crops), torch.cat(geom_crops), torch.cat(masks_crops), torch.cat([t["labels"] for t in targets]) 
		
		target_quats = torch.cat([t["quats"] for t in targets])
		target_trans = torch.cat([t["trans"] for t in targets])
		target_diameters = torch.stack([t["diameter"] for t in targets])
		target_meshes = [t["mesh"] for t in targets]
		target_mesh_rots = [t["mesh_rot"] for t in targets]
		target_symmetric = [t["symmetric"] for t in targets]

		color_crops = color_crops.to(device)
		geom_crops = geom_crops.to(device).permute(0,2,3,1)
		masks_crops = masks_crops.to(device).permute(0,2,3,1)
		obj_ids = obj_ids.to(device)
		with torch.cuda.amp.autocast(enabled=scaler is not None):
			pred_t, pred_r, pred_c, choose = model(color_crops, geom_crops, masks_crops, obj_ids)
			loss, loss_r, loss_reg, loss_t = criterion(pred_t, pred_r, pred_c, target_mesh_rots, target_trans, target_meshes, choose, target_symmetric, target_diameters)

		
		how_min, which_min = torch.min(pred_c, 1)
		points = geom_crops[:,:,:,4:].flatten(start_dim=1, end_dim=2)[choose].view(geom_crops.shape[0], model.N, 3)
		pred_t, pred_mask = ransac_voting_layer(points, pred_t, inlier_thresh=-100000)
		
		
		pred_t = pred_t.cpu().data.numpy()
		for obj_i in range(pred_r.shape[0]):
			pred_r_ = pred_r[obj_i][which_min[obj_i]].view(-1).cpu().data.numpy()
			pred_r_ = R.from_quat(pred_r_[[1,2,3,0]]).as_matrix()
			pred_t_ = pred_t[obj_i]
			model_points_ = targets[obj_i]["mesh"][0].cpu().detach().numpy()
			pred = np.dot(model_points_, pred_r_.T) + pred_t_
			target = targets[obj_i]["mesh_rot"][0].cpu().detach().numpy() + targets[obj_i]["trans_gt"][0].cpu().data.numpy()
			dis = torch.linalg.vector_norm(torch.from_numpy(pred).type(torch.float32).unsqueeze(1) - torch.from_numpy(target).type(torch.float32).unsqueeze(0), dim=2)
			dis = torch.mean(torch.min(dis, dim=1).values)
			
			if dis < 0.1 * target_diameters[obj_i]:
				success_count[targets[obj_i]["labels"][0].item()] += 1
			num_count[targets[obj_i]["labels"][0].item()] += 1

		
		metric_logger.update(loss=loss.item(), loss_r=loss_r, loss_reg=loss_reg, loss_t=loss_t)
		if b>200:
			break
		b+=1

	logfile.write('Test Results:\n')
	for i in range(len(success_count)):
		if num_count[i]>0:
			logfile.write(str(i)+' '+str(success_count[i]/num_count[i])+'\n')


def test_qual(model, criterion, optimizer, dataset, device, print_freq, save_dir, scaler=None, e=1):
	model.eval()
	metric_logger = utils.MetricLogger(delimiter="  ")
	header = f"Test"

	with torch.no_grad():
		i = 0#np.random.randint(len(dataset.object_counts))
		start_idx = dataset.object_counts[i-1] if (i>0) else 0
		data = utils.collate_fn([dataset[idx] for idx in range(start_idx, dataset.object_counts[i])])
		color, color_crops, geom_crops, masks_crops, targets = data
		color, color_crops, geom_crops, masks_crops, obj_ids = torch.stack(color), torch.cat(color_crops), torch.cat(geom_crops), torch.cat(masks_crops), torch.cat([t["labels"] for t in targets]) 

		target_quats = torch.cat([t["quats"] for t in targets])
		target_trans = torch.cat([t["trans"] for t in targets])
		target_trans_gt = torch.cat([t["trans_gt"] for t in targets])
		target_diameters = torch.stack([t["diameter"] for t in targets])
		target_meshes = [t["mesh"] for t in targets]
		target_mesh_rots = [t["mesh_rot"] for t in targets]
		labels = [t["labels"].item() for t in targets]
		target_symmetric = [t["symmetric"] for t in targets]

		color_crops = color_crops.to(device)
		geom_crops = geom_crops.to(device).permute(0,2,3,1)
		masks_crops = masks_crops.to(device).permute(0,2,3,1)
		obj_ids = obj_ids.to(device)
		with torch.cuda.amp.autocast(enabled=scaler is not None):
			pred_t, pred_r, pred_c, choose = model(color_crops, geom_crops, masks_crops, obj_ids)
			loss, loss_r, loss_reg, loss_t = criterion(pred_t, pred_r, pred_c, target_mesh_rots, target_trans, target_meshes, choose, target_symmetric, target_diameters)

		how_min, which_min = torch.min(pred_c, 1)
		points = geom_crops[:,:,:,4:].flatten(start_dim=1, end_dim=2)[choose].view(geom_crops.shape[0], model.N, 3)
		pred_t, pred_mask = ransac_voting_layer(points, pred_t, inlier_thresh=-100000)

		# print("True Trans:", target_trans_gt)
		# print("Estimated Trans:", pred_t)
		# print("Difference:", target_trans_gt.cuda()-pred_t)
		# print("Predicted Quat:", target_quats)
		# print("Estimated Quat:", torch.diagonal(pred_r[:,which_min]).permute(1,0))
		# print("Estimated Quat:", target_quats.cuda()-torch.diagonal(pred_r[:,which_min]).permute(1,0))
		# print()

		plot_test(color[0], dataset, torch.diagonal(pred_r[:,which_min]).permute(1,0), pred_t, target_quats, target_trans_gt, labels, save_dir=save_dir, e=e)



def plot_test(color, dataset, quats, trans, quats_true, trans_true, labels, save_dir, e=1):
	H, W = color.shape[1:]
	device = torch.device('cuda:0')

	cameras = PerspectiveCameras(device=device, focal_length=((dataset.intrinsic[0,0], dataset.intrinsic[1,1]),), 
								principal_point=((dataset.intrinsic[0,2], dataset.intrinsic[1,2]),), 
								in_ndc=False, image_size=torch.tensor([[H,W]]),
								R=torch.tensor([[[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]]))

	raster_settings = RasterizationSettings(
	    image_size=(H,W), 
	    blur_radius=0.0, 
	    faces_per_pixel=1, 
	)

	
	color = color.permute(1,2,0).to(device)

	
	quats = quats.unsqueeze(1)
	# trans = trans.unsqueeze(1)
	quats_true = quats_true.unsqueeze(1)
	# trans_true = trans_true.unsqueeze(1)



	# Load obj file
	meshes = dataset.meshes[labels].clone().to(device)

	meshes = meshes.update_padded(quaternion_apply(quats.to(device).type(torch.float32), meshes.verts_padded()))
	meshes = meshes.update_padded(Translate(trans.to(device).type(torch.float32)).transform_points(meshes.verts_padded()))

	meshes_true = dataset.meshes[labels].clone().to(device)
	meshes_true = meshes_true.update_padded(quaternion_apply(quats_true.to(device).type(torch.float32), meshes_true.verts_padded()))
	meshes_true = meshes_true.update_padded(Translate(trans_true.to(device).type(torch.float32)).transform_points(meshes_true.verts_padded()))

	# Initialize each vertex to be white in color.
	verts_rgb = torch.ones_like(meshes.verts_padded())  # (1, V, 3)
	verts_rgb[:,:,:] = torch.FloatTensor([[[1, 0.984, 0.047]]])
	meshes.textures = TexturesVertex(verts_features=verts_rgb.to(device))

	# Initialize each vertex to be white in color.
	verts_rgb_true = torch.ones_like(meshes_true.verts_padded())  # (1, V, 3)
	verts_rgb_true[:,:,:] = torch.FloatTensor([[[0.047, 0.718, 0]]])
	meshes_true.textures = TexturesVertex(verts_features=verts_rgb_true.to(device))
	
	
	lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])
	# lights = AmbientLights(device=device)

	# Create a Phong renderer by composing a rasterizer and a shader. The textured Phong shader will 
	# interpolate the texture uv coordinates for each vertex, sample from a texture image and 
	# apply the Phong lighting model
	renderer = MeshRenderer(
	    rasterizer=MeshRasterizer(
	        cameras=cameras, 
	        raster_settings=raster_settings
	    ),
	    shader=SoftPhongShader(
	        device=device, 
	        cameras=cameras,
	        lights=lights
	    )
	)

	images = renderer(meshes)
	images_true = renderer(meshes_true)
	images[images==images.max()] = 0
	images = images.max(0)[0]

	images_true[images_true==images_true.max()] = 0
	images_true = images_true.max(0)[0]

	a=0.9
	color_alpha = a*images[:,:,3].unsqueeze(2)
	color_alpha_true = a*images_true[:,:,3].unsqueeze(2)
	images_alpha = 1-(color_alpha+color_alpha_true)

	color_ = color*images_alpha + images[:,:,:3]*color_alpha + images_true[:,:,:3]*color_alpha_true
	# for i in range(len(images)):
	fig, ax = plt.subplots(1,2)
	#.figure(figsize=(10, 10))
	# ax[0].imshow(color.cpu().detach())
	ax[0].imshow(color_.detach().cpu().numpy())

	color_ = color*(1-2*color_alpha) + images[:,:,:3]*(2*color_alpha)

	ax[1].imshow(color_.detach().cpu().numpy())
	fig.tight_layout()
	fig.savefig(os.path.join(save_dir,'stage2_output'+str(e)+'.png'), dpi=300)






def main(save_dir=os.path.join("experiments","xu_6dof","stage2","models")):
	# train on the GPU or on the CPU, if a GPU is not available
	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	# use our dataset and defined transformations
	dataset = Stage2Dataset(image_list="./data/train_images.csv", transforms=get_transform(train=True))
	dataset_test = Stage2Dataset(image_list="./data/val_images.csv", transforms=get_transform(train=False))

	# define training and validation data loaders
	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=4, shuffle=True, num_workers=4, drop_last=True,
		collate_fn=utils.collate_fn)

	data_loader_test = torch.utils.data.DataLoader(
		dataset_test, batch_size=4, shuffle=True, num_workers=4,
		collate_fn=utils.collate_fn)

	

	logfile = open(os.path.join(save_dir,"stage2_log.txt"), 'w')

	model = build_model({'num_obj': 63})
	model.to(device)

	criterion = Loss(model.rot_anchors)


	# construct an optimizer
	params = [p for p in model.parameters() if p.requires_grad]
	optimizer = torch.optim.Adam(params, lr=0.0005)

	# let's train it for 10 epochs
	num_epochs = 100
	#model.load_state_dict(torch.load(os.path.join(save_dir,"stage2_0.pt"))['model_state_dict'])
	torch.save({'epoch': -1,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()},
				os.path.join(save_dir,"stage2_0.pt"))
	for epoch in range(num_epochs):
		train_one_epoch(model, criterion, optimizer, data_loader, device, epoch, print_freq=100, logfile=logfile)
		torch.save({'epoch': epoch,
				'model_state_dict': model.state_dict(),
				'optimizer_state_dict': optimizer.state_dict()},
				os.path.join(save_dir,"stage2_"+str(epoch)+".pt"))
		#test_qual(model, criterion, optimizer, dataset_test, device, print_freq=100, save_dir=save_dir, e=epoch)
		#test_quant(model, criterion, optimizer, data_loader_test, device, print_freq=100, logfile=logfile)
		

if __name__=="__main__":
	torch.manual_seed(0)
	random.seed(0)
	np.random.seed(0)

	main()