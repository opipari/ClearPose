import os
from scipy.io import loadmat
from scipy.spatial.transform import Rotation as R
from PIL import Image
import numpy as np

import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from torchvision.ops import roi_align

import matplotlib.pyplot as plt

from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.transforms import quaternion_apply, Translate

class Stage2Dataset(Dataset):
	def __init__(self, image_list="./data/images.csv", 
					   object_list="./data/objects.csv",
					   model_dir="./data",
					   transforms=None):
		self.N = 500
		self.N_mesh = 500
		self.image_list = [ln.strip().split(',') for ln in open(image_list,'r').readlines()]
		self.object_counts = np.cumsum(np.array([int(ln[-1]) for ln in self.image_list]))
		self.object_list = [ln.strip().split(',') for ln in open(object_list,'r').readlines()]

		self.object_lookup_id = {obj[1]: int(obj[0]) for obj in self.object_list}
		self.object_lookup_name = {int(obj[0]): obj[1] for obj in self.object_list}

		self.transforms = transforms

		self.intrinsic = np.array([[902.187744140625, 0.0, 662.3499755859375],
						  [0.0, 902.391357421875, 372.2278747558594], 
						  [0.0, 0.0, 1.0]])


		
		self.obj_filenames = []
		for label_ in sorted(self.object_lookup_name.keys()):
			obj_filename = os.path.join(model_dir, "model", self.object_lookup_name[label_], self.object_lookup_name[label_]+".obj")
			self.obj_filenames.append(obj_filename)


		# Load obj file
		self.meshes = load_objs_as_meshes(self.obj_filenames, device=torch.device('cpu'))
		
		self.diameters = []
		for mesh in self.meshes:
			obj_cld = mesh.verts_padded()[0]
			obj_center = (obj_cld.min(0).values + obj_cld.max(0).values) / 2.0
			obj_cld = obj_cld - obj_center
			obj_diameter = torch.max(torch.linalg.norm(obj_cld, axis=1)) * 2
			self.diameters.append(obj_diameter)


	def __len__(self):
		return self.object_counts[-1]


	def __getitem__(self, idx):
		idx = idx % 12
		img_idx = np.argmax(idx<self.object_counts)
		_, scene_path, imgid, _ = self.image_list[img_idx]

		idx_offset = idx
		if img_idx>0:
			idx_offset = idx-self.object_counts[img_idx-1]

		color_path = os.path.join(scene_path, imgid+'-color.png')
		mask_path = os.path.join(scene_path, imgid+'-label.png')
		normal_path = os.path.join(scene_path, imgid+'-normal_true.png')
		plane_path = os.path.join(scene_path, imgid+'-tableplane_depth.png')
		depth_path = os.path.join(scene_path, imgid+'-depth_true.png')
		meta_path = os.path.join(scene_path, imgid+'-meta.mat')

		color = Image.open(color_path).convert("RGB")
		mask = Image.open(mask_path)
		normal = Image.open(normal_path)
		plane = np.array(Image.open(plane_path))
		depth = np.array(Image.open(depth_path))
		meta = loadmat(meta_path)
		
		depth = Image.fromarray(depth/meta['factor_depth'].item())
		plane = Image.fromarray(plane/meta['factor_depth'].item())

		mask = np.array(mask)
		obj_ids = meta['cls_indexes'].flatten()
		masks = (mask == obj_ids[:, None, None])[idx_offset]
		obj_ids = obj_ids[idx_offset]

		pos = np.where(masks)
		xmin = np.min(pos[1])
		xmax = np.max(pos[1])
		ymin = np.min(pos[0])
		ymax = np.max(pos[0])
		
		boxes = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)
		labels = torch.as_tensor([obj_ids], dtype=torch.int64)
		masks = torch.as_tensor(masks, dtype=torch.uint8)

		area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

		target = {}
		target["boxes"] = boxes
		target["labels"] = labels
		target["masks"] = masks
		target["image_id"] = torch.tensor([img_idx])
		target["area"] = area
		target["iscrowd"] = torch.zeros((1,), dtype=torch.int64)

		if self.transforms is not None:
			(color,normal,plane,depth), target = self.transforms((color,normal,plane,depth), target)

		H, W = depth.shape[1:]
		U, V = np.tile(np.arange(W), (H, 1)), np.tile(np.arange(H), (W, 1)).T
		fx, fy, cx, cy = self.intrinsic[0, 0], self.intrinsic[1, 1], self.intrinsic[0, 2], self.intrinsic[1, 2]
		X, Y = depth * (U - cx) / fx, depth * (V - cy) / fy
		
		X = F.convert_image_dtype(X)
		Y = F.convert_image_dtype(Y)
		Z = F.convert_image_dtype(depth)
		uvz = torch.stack([X,Y,Z]).squeeze(1)

		crops_color = roi_align(color.unsqueeze(0), [target['boxes']], (80,80), 1, 1)
		crops_normal = roi_align(normal.unsqueeze(0), [target['boxes']], (80,80), 1, 1)
		crops_plane = roi_align(plane.unsqueeze(0), [target['boxes']], (80,80), 1, 1)
		crops_uvz = roi_align(uvz.unsqueeze(0), [target['boxes']], (80,80), 1, 1)
		crops_masks = roi_align(masks.unsqueeze(0).unsqueeze(0).type(torch.float32), [target['boxes']], (80,80), 1, 1)
		crops_geometry_per_image = torch.cat([crops_normal, crops_plane, crops_uvz], dim=1)

		poses = np.transpose(meta['poses'], (2,0,1))
		target["poses"] = torch.from_numpy(poses)[idx_offset].unsqueeze(0)
		target["quats"] = torch.stack([torch.from_numpy(R.from_matrix(rmat[:, :3]).as_quat()) for rmat in poses])[:,[3,0,1,2]][idx_offset].unsqueeze(0)
		target["trans_gt"] = torch.stack([torch.from_numpy(rmat[:, 3]) for rmat in poses])[idx_offset].unsqueeze(0)
		target["trans"] = target["trans_gt"].view(1,3,1,1) - crops_uvz
		target["trans"] = target["trans"] / torch.linalg.vector_norm(target["trans"], dim=1, keepdim=True)

		mesh = self.meshes[list(target['labels']-1)].clone()
		mesh_pre_rot = mesh.verts_padded()
		mesh = mesh.update_padded(quaternion_apply(target['quats'].unsqueeze(1).type(torch.float32), mesh.verts_padded()))
		mesh_post_rot = mesh.verts_padded()

		mesh_samples = np.random.choice(range(mesh_pre_rot.shape[1]), size=self.N_mesh)

		target["mesh"] = mesh_pre_rot[:,mesh_samples]
		target["mesh_rot"] = mesh_post_rot[:,mesh_samples]
		target["diameter"] = self.diameters[obj_ids-1]

		return color, crops_color, crops_geometry_per_image, crops_masks, target




