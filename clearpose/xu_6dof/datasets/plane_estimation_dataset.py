import os
from scipy.io import loadmat
from PIL import Image
import numpy as np
import open3d as o3d

import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt




class PlaneEstimationDataset(Dataset):
	def __init__(self, image_list="./data/images.csv", 
					   transforms=None):
		self.image_list = [ln.strip().split(',') for ln in open(image_list,'r').readlines()]

		self.transforms = transforms

		self.intrinsic = np.array([[902.187744140625, 0.0, 662.3499755859375],
						  [0.0, 902.391357421875, 372.2278747558594], 
						  [0.0, 0.0, 1.0]])

		
	def __len__(self):
		return len(self.image_list)


	def __getitem__(self, idx):
		_, scene_path, intid = self.image_list[idx]

		depth_path = os.path.join(scene_path, intid+'-depth.png')

		depth_raw = o3d.io.read_image(depth_path)
		height, width = np.array(depth_raw).shape
		pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_raw),
															  o3d.camera.PinholeCameraIntrinsic(width, 
																								height, 
																								self.intrinsic[0, 0], 
																								self.intrinsic[1, 1], 
																								self.intrinsic[0, 2], 
																								self.intrinsic[1, 2]),
															  )
		pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))
		normal = ((np.array(pcd.normals) + 1)*255/2).astype(np.uint8)
		points = np.array(pcd.points)

		plt.imshow(depth_raw)
		plt.show()
		# if self.transforms is not None:
		# 	color, normal = self.transforms(color, normal)
		o3d.visualization.draw_geometries([pcd]
								  )
		plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
												 ransac_n=3,
												 num_iterations=1000)
		[a, b, c, d] = plane_model
		print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

		inlier_cloud = pcd.select_by_index(inliers)
		inlier_cloud.paint_uniform_color([1.0, 0, 0])
		outlier_cloud = pcd.select_by_index(inliers, invert=True)
		o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])

		#fit(pts, thresh=0.05, minPoints=100, maxIteration=1000)
		# return color, normal
