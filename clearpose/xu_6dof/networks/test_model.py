import os
import math
from signal import pthread_sigmask
import numpy as np
import torch
from torchvision.ops import roi_align
import torchvision.transforms.functional as F
from torchvision.utils import draw_bounding_boxes

import clearpose.xu_6dof.networks.references.detection.utils as utils
import clearpose.xu_6dof.networks.references.detection.transforms as T

from clearpose.xu_6dof.networks.build_model import build_model
from clearpose.xu_6dof.datasets.clear_pose_dataset import ClearPoseDataset
from clearpose.xu_6dof.networks.stage2.train_stage2 import Loss

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from PIL import Image

from clearpose.xu_6dof.networks.references.posenet.ransac_voting.ransac_voting_gpu import ransac_voting_layer

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
import cv2
from scipy.spatial.transform import Rotation as R


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

def plot_test(color, dataset, quats, trans, labels, save_dir):
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


	# Load obj file
	meshes = dataset.meshes[labels].clone().to(device)

	meshes = meshes.update_padded(quaternion_apply(quats.to(device).type(torch.float32), meshes.verts_padded()))
	meshes = meshes.update_padded(Translate(trans.to(device).type(torch.float32)).transform_points(meshes.verts_padded()))


	# Initialize each vertex to be white in color.
	verts_rgb = torch.ones_like(meshes.verts_padded())  # (1, V, 3)
	verts_rgb[:,:,:] = torch.FloatTensor([[[1, 0.984, 0.047]]])
	meshes.textures = TexturesVertex(verts_features=verts_rgb.to(device))


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
	images[images==images.max()] = 0
	images = images.max(0)[0]


	a=0.9
	color_alpha = a*images[:,:,3].unsqueeze(2)

	# for i in range(len(images)):
	fig, ax = plt.subplots(1,2)
	#.figure(figsize=(10, 10))
	# ax[0].imshow(color.cpu().detach())
	ax[0].imshow(color.detach().cpu().numpy())

	color_ = color*(1-2*color_alpha) + images[:,:,:3]*(2*color_alpha)

	ax[1].imshow(color_.detach().cpu().numpy())
	fig.tight_layout()
	fig.savefig(os.path.join(save_dir,'stage2_output.png'), dpi=300)


def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

# match by bbox, for every gt box, if there is one pred box with largest iou > threshold, assign gt label to this box
def match_pred_gt_bbox(pred_boxes, gt_boxes, gt_label, iou_thres=0.5):
	pred_label = []
	pred_i = []
	gt_j = []
	for j, (box, label) in enumerate(zip(gt_boxes.cpu().numpy(), gt_label[0].cpu().numpy())):
		max_iou = 0
		max_label, max_i = None, None
		for i, pbox in enumerate(pred_boxes.cpu().numpy()):
			if (box[2] - box[0]) * (box[3] - box[1]) < 100:
				continue
			iou = get_iou({'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}, {'x1': pbox[0], 'y1': pbox[1], 'x2': pbox[2], 'y2': pbox[3]})
			if iou > max_iou:
				max_iou = iou
				max_label = label
				max_i = i
		if max_iou > iou_thres:
			pred_label.append(max_label)
			pred_i.append(max_i)
			gt_j.append(j)
	return torch.tensor(pred_label, device='cuda'), pred_i, gt_j

# match by label, for every gt label, if there are multiple detections, select one with largest box iou, if not, save as false negative, ignore other detections
def match_pred_gt(pred_boxes, pred_label, gt_boxes, gt_label):
	pred_i = []
	gt_j = []
	gt_fn = []
	for j, (box, label) in enumerate(zip(gt_boxes.cpu().numpy(), gt_label[0].cpu().numpy())):
		max_iou = 0
		max_i = None
		ind = torch.where(pred_label == label)[0].cpu().numpy()
		if len(ind) == 0:
			gt_fn.append(label)
			continue
		for i in ind:
			pbox = pred_boxes[i].cpu().numpy()
			iou = get_iou({'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}, {'x1': pbox[0], 'y1': pbox[1], 'x2': pbox[2], 'y2': pbox[3]})
			if iou >= max_iou:
				max_iou = iou
				max_i = i
		pred_i.append(max_i)
		gt_j.append(j)
	return pred_i, gt_j, torch.tensor(pred_label[pred_i], device='cuda', dtype=torch.int), gt_fn



def cal_add_cuda(
	pred_RT, gt_RT, p3ds
):
	pred_p3ds = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
	gt_p3ds = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
	dis = torch.norm(pred_p3ds - gt_p3ds, dim=1)
	return torch.mean(dis)

def cal_adds_cuda(
	pred_RT, gt_RT, p3ds
):
	N, _ = p3ds.size()
	pd = torch.mm(p3ds, pred_RT[:, :3].transpose(1, 0)) + pred_RT[:, 3]
	pd = pd.view(1, N, 3).repeat(N, 1, 1)
	gt = torch.mm(p3ds, gt_RT[:, :3].transpose(1, 0)) + gt_RT[:, 3]
	gt = gt.view(N, 1, 3).repeat(1, N, 1)
	dis = torch.norm(pd - gt, dim=2)
	mdis = torch.min(dis, dim=1)[0]
	return torch.mean(mdis)

def calculate_add_adds(pred_t, pred_r, pred_obj_ids, gt_t, gt_r, obj_pts):
	result = []
	for i in range(pred_obj_ids.shape[0]):
		p_r = R.from_quat(pred_r[i].cpu().numpy()[[1, 2, 3, 0]]).as_matrix() # wxyz -> xyzw
		p_RT = np.zeros((3, 4))
		p_RT[:, :3] = p_r
		p_RT[:, 3] = pred_t[i].cpu().numpy()
		g_r = R.from_quat(gt_r[i].cpu().numpy()[[1, 2, 3, 0]]).as_matrix() # wxyz -> xyzw
		g_RT = np.zeros((3, 4))
		g_RT[:, :3] = g_r
		g_RT[:, 3] = gt_t[i].cpu().numpy()
		p_RT = torch.tensor(p_RT, device='cuda', dtype=torch.float32)
		g_RT = torch.tensor(g_RT, device='cuda', dtype=torch.float32)
		add = cal_add_cuda(p_RT, g_RT, obj_pts[i])
		adds = cal_adds_cuda(p_RT, g_RT, obj_pts[i])
		result.append([pred_obj_ids[i].item(), add.item(), adds.item()])
	return result

def calculate_add_s_accuracy(result):
	result = np.array(result)
	obj_ids = np.unique(result[:, 0])
	aucs, accs = [], []
	for id in obj_ids:
		lines = np.where(result[:, 0] == id)[0]
		add = result[lines, 1]
		add_s = result[lines, 2]
		auc = cal_auc(add_s) / 100
		acc = np.sum(add < 0.1) / len(add)
		print(id, auc, acc)
		aucs.append(auc)
		accs.append(acc)
	print('mean', sum(aucs) / len(aucs), sum(accs) / len(accs))

def cal_auc(add_dis, max_dis=0.1):
	D = np.array(add_dis)
	D[np.where(D > max_dis)] = np.inf
	D = np.sort(D)
	n = len(add_dis)
	acc = np.cumsum(np.ones((1,n)), dtype=np.float32) / n
	aps = VOCap(D, acc)
	return aps * 100

def VOCap(rec, prec):
    idx = np.where(rec != np.inf)
    if len(idx[0]) == 0:
        return 0
    rec = rec[idx]
    prec = prec[idx]
    mrec = np.array([0.0]+list(rec)+[0.1])
    mpre = np.array([0.0]+list(prec)+[prec[-1]])
    for i in range(1, prec.shape[0]):
        mpre[i] = max(mpre[i], mpre[i-1])
    i = np.where(mrec[1:] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[i] - mrec[i-1]) * mpre[i]) * 10
    return ap

def project_p3d(p3d, cam_scale, K):
	p3d = p3d * cam_scale
	p2d = np.dot(p3d, K.T)
	p2d_3 = p2d[:, 2]
	p2d_3[np.where(p2d_3 < 1e-8)] = 1.0
	p2d[:, 2] = p2d_3
	p2d = np.around((p2d[:, :2] / p2d[:, 2:])).astype(np.int32)
	return p2d

def draw_p2ds(img, p2ds, r=1, color=[(255, 0, 0)]):
	if type(color) == tuple:
		color = [color]
	if len(color) != p2ds.shape[0]:
		color = [color[0] for i in range(p2ds.shape[0])]
	h, w = img.shape[0], img.shape[1]
	for pt_2d, c in zip(p2ds, color):
		pt_2d[0] = np.clip(pt_2d[0], 0, w)
		pt_2d[1] = np.clip(pt_2d[1], 0, h)
		img = cv2.circle(img.copy(), (pt_2d[0], pt_2d[1]), r, c, -1)
	return img

def get_label_color(cls_id, n_obj=63, mode=0):
	if mode == 0:
		cls_color = [
			255, 255, 255,  # 0
			180, 105, 255,   # 194, 194, 0,    # 1 # 194, 194, 0
			0, 255, 0,      # 2
			0, 0, 255,      # 3
			0, 255, 255,    # 4
			255, 0, 255,    # 5
			180, 105, 255,  # 128, 128, 0,    # 6
			128, 0, 0,      # 7
			0, 128, 0,      # 8
			0, 165, 255,    # 0, 0, 128,      # 9
			128, 128, 0,    # 10
			0, 0, 255,      # 11
			255, 0, 0,      # 12
			0, 194, 0,      # 13
			0, 194, 0,      # 14
			255, 255, 0,    # 15 # 0, 194, 194
			64, 64, 0,      # 16
			64, 0, 64,      # 17
			185, 218, 255,  # 0, 0, 64,       # 18
			0, 0, 255,      # 19
			0, 64, 0,       # 20
			0, 0, 192       # 21
		]
		cls_color = np.array(cls_color).reshape(-1, 3)
		color = cls_color[cls_id]
		bgr = (int(color[0]), int(color[1]), int(color[2]))
	elif mode == 1:
		cls_color = [
			255, 255, 255,  # 0
			0, 127, 255,    # 180, 105, 255,   # 194, 194, 0,    # 1 # 194, 194, 0
			0, 255, 0,      # 2
			255, 0, 0,      # 3
			180, 105, 255, # 0, 255, 255,    # 4
			255, 0, 255,    # 5
			180, 105, 255,  # 128, 128, 0,    # 6
			128, 0, 0,      # 7
			0, 128, 0,      # 8
			185, 218, 255,# 0, 0, 255, # 0, 165, 255,    # 0, 0, 128,      # 9
			128, 128, 0,    # 10
			0, 0, 255,      # 11
			255, 0, 0,      # 12
			0, 194, 0,      # 13
			0, 194, 0,      # 14
			255, 255, 0,    # 15 # 0, 194, 194
			0, 0, 255, # 64, 64, 0,      # 16
			64, 0, 64,      # 17
			185, 218, 255,  # 0, 0, 64,       # 18
			0, 0, 255,      # 19
			0, 0, 255, # 0, 64, 0,       # 20
			0, 255, 255,# 0, 0, 192       # 21
		]
		cls_color = np.array(cls_color).reshape(-1, 3)
		color = cls_color[cls_id]
		bgr = (int(color[0]), int(color[1]), int(color[2]))
	else:
		mul_col = 255 * 255 * 255 // n_obj * cls_id
		r, g, b= mul_col // 255 // 255, (mul_col // 255) % 255, mul_col % 255
		bgr = (int(r), int(g) , int(b))
	return bgr

def draw(idx, rgb, pred_t, pred_r, pred_obj_ids, gt_t, gt_r, obj_pts):
	K = np.array([
		[601, 0, 334],
		[0, 601, 248],
		[0, 0, 1]
	])
	rgb_cpu = np.array((rgb * 255).to(torch.uint8).to("cpu"))
	img = rgb_cpu.transpose(1, 2, 0)[:, :, ::-1]
	for i in range(pred_obj_ids.shape[0]):
		p_r = R.from_quat(pred_r[i].cpu().numpy()[[1, 2, 3, 0]]).as_matrix() # wxyz -> xyzw
		p_RT = np.zeros((3, 4))
		p_RT[:, :3] = p_r
		p_RT[:, 3] = pred_t[i].cpu().numpy()
		g_r = R.from_quat(gt_r[i].cpu().numpy()[[1, 2, 3, 0]]).as_matrix() # wxyz -> xyzw
		g_RT = np.zeros((3, 4))
		g_RT[:, :3] = g_r
		g_RT[:, 3] = gt_t[i].cpu().numpy()
		p_RT = torch.tensor(p_RT, device='cuda', dtype=torch.float32)
		g_RT = torch.tensor(g_RT, device='cuda', dtype=torch.float32)
		p3d = np.array((p_RT[:3, :3] @ obj_pts[i].T + p_RT[:3, [3]]).to('cpu'))
		p2d = project_p3d(p3d.T, 1.0, K)
		color = get_label_color(pred_obj_ids[i], n_obj=63, mode=2)
		img = draw_p2ds(img, p2d, r=1, color=[color])
	cv2.imwrite(f"{idx}.png", img)

if __name__=="__main__":

	verbose_plots = False

	device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

	dataset_test = ClearPoseDataset(image_list="data/non-planner_test.csv", transforms=get_transform(train=False))


	model_config = {"mask_rcnn_model": os.path.join("experiments","xu_6dof","stage1","transparent_segmentation","models","finetune","mask_rcnn_8.pt"),
					"deeplabv3_model": os.path.join("experiments","xu_6dof","stage1","surface_normals","models","deeplabv3_1.pt"),
					"stage2_model": os.path.join("experiments","xu_6dof","stage2","models","stage2_18.pt")}
	model = build_model(model_config)
	model.eval()
	model.to(device)


	criterion = Loss(model.stage2_model.rot_anchors)
	result_all = []#np.load('wou.npy')
	i = 0
	with torch.no_grad():
		for color_norm, color, uvz, color_crops_norm, color_crops, geom_crops, masks_crops, targets in dataset_test:
			# if i < 1315:
			# 	continue
			# pose_dict = {}
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
				#show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','color'], color_crops)
				#show([target_plot, alpha_plot],['Ground Truth Bounding Boxes',' masks'], masks_crops)
				#show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','normals'], geom_crops[:,:3])
				#show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','plane'], geom_crops[:,3])
				#show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','x'], geom_crops[:,4])
				#show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','y'], geom_crops[:,5])

				# Predicted Crops
				show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','color (pred)'], pred_crops_color)
				show([target_plot, alpha_plot],['Ground Truth Bounding Boxes',' masks (pred)'], pred_crops_masks)
				show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','normals (pred)'], pred_crops_geom[:,:3])
				#show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','plane'], pred_crops_geom[:,3])
				#show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','x'], pred_crops_geom[:,4])
				#show([target_plot, alpha_plot],['Ground Truth Bounding Boxes','y'], pred_crops_geom[:,5])


			# pred_obj_ids, index, index_gt = match_pred_gt_bbox(segmentation_output[0]['boxes'], targets[0]['boxes'], targets[0]['labels'])
			index, index_gt, pred_obj_ids, index_gt_false_negative = match_pred_gt(segmentation_output[0]['boxes'], segmentation_output[0]['labels'], targets[0]['boxes'], targets[0]['labels'])
			
			for ind in index_gt_false_negative:
				result_all.append([ind, 100, 100])
			if len(index) == 0:
				print('no matched bbox')
				continue
			pred_crops_color_normal = pred_crops_color_normal.to(device)[index]
			pred_crops_geom = pred_crops_geom.to(device).permute(0,2,3,1)[index]
			pred_crops_masks = pred_crops_masks.to(device).permute(0,2,3,1)[index]
			# pred_obj_ids = segmentation_output[0]['labels'].to(device)
			pred_t, pred_r, pred_c, choose = model.stage2_model(pred_crops_color_normal, pred_crops_geom, pred_crops_masks, pred_obj_ids)

			target_boxes = targets[0]["boxes"][index_gt]
			target_labels = torch.cat([t for t in targets[0]["labels"]])[index_gt]
			target_quats = torch.cat([t for t in targets[0]["quats"]])[index_gt]
			target_trans = torch.cat([t for t in targets[0]["trans"]])[index_gt]
			target_trans_gt = torch.cat([t for t in targets[0]["trans_gt"]])[index_gt]
			# target_diameters = torch.stack([t for t in targets[0]["diameter"]])[index_gt]
			# target_meshes = [t.unsqueeze(0) for t in targets[0]["mesh"]][index_gt]
			# target_mesh_rots = [t.unsqueeze(0) for t in targets[0]["mesh_rot"]][index_gt]
			# target_symmetric = [t.unsqueeze(0) for t in targets[0]["symmetric"]][index_gt]
			# pred_diameters = torch.stack([dataset_test.diameters[t.item()] for t in pred_obj_ids])

		
			#loss, loss_r, loss_reg, loss_t = criterion(pred_t, pred_r, pred_c, target_mesh_rots, target_trans, target_meshes, choose, target_symmetric, target_diameters)
			#print('loss',loss)
			
			how_min, which_min = torch.min(pred_c, 1)
			points = pred_crops_geom[:,:,:,4:].flatten(start_dim=1, end_dim=2)[choose].view(pred_crops_geom.shape[0], model.stage2_model.N, 3)
			try:
				pred_t, pred_mask = ransac_voting_layer(points, pred_t, inlier_thresh=0.1)

				pred_r = torch.diagonal(pred_r[:,which_min]).permute(1,0)
				# pred_boxes = segmentation_output[0]['boxes'][index]
				# plot_test(color, dataset_test, pred_r, pred_t, pred_obj_ids, save_dir='.')
				res = calculate_add_adds(pred_t, pred_r, pred_obj_ids, target_trans_gt, target_quats, targets[0]["mesh"][index_gt])
				# draw(i, color, pred_t, pred_r, pred_obj_ids, target_trans_gt, target_quats, targets[0]["mesh"][index_gt])
				result_all += res
				
			except Exception as e:
				print(e)
			i += 1
			# for (id, r, t) in zip(pred_obj_ids, pred_r, pred_t):
			# 	p_r = R.from_quat(r.cpu().numpy()[[1, 2, 3, 0]]).as_matrix() # wxyz -> xyzw
			# 	p_RT = np.zeros((3, 4))
			# 	p_RT[:, :3] = p_r
			# 	p_RT[:, 3] = t.cpu().numpy()
			# 	pose_dict[id] = p_RT
			# import pickle as pkl
			# with open('{}.pkl'.format(i), 'wb') as f:
			# 	pkl.dump(pose_dict, f)
			# break
			# print(i)
		calculate_add_s_accuracy(result_all)
		np.save('occlusion.npy', result_all)