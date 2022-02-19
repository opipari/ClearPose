import os
import torch
import matplotlib.pyplot as plt

# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
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

# add path for demo utils functions 
import sys
import os
sys.path.append(os.path.abspath(''))

import clearpose.networks.references.detection.transforms as T
from clearpose.datasets.stage2_dataset import Stage2Dataset


def get_transform(train):
	transforms = []
	transforms.append(T.ToTensorSet())
	# if train:
	# 	transforms.append(T.RandomHorizontalFlipSet(0.5))
	return T.Compose(transforms)



def main():
	# use our dataset and defined transformations
	dataset = Stage2Dataset(transforms=get_transform(train=True))
	color, crops_color, crops_geometry_per_image, crops_masks, target = dataset[100]

	# Setup
	if torch.cuda.is_available():
	    device = torch.device("cuda:0")
	    torch.cuda.set_device(device)
	else:
	    device = torch.device("cpu")

	# Set paths
	DATA_DIR = "./data"
	
	H, W = color.shape[1:]
	cameras = PerspectiveCameras(device=device, focal_length=((dataset.intrinsic[0,0], dataset.intrinsic[1,1]),), 
								principal_point=((dataset.intrinsic[0,2], dataset.intrinsic[1,2]),), 
								in_ndc=False, image_size=torch.tensor([[H,W]]),
								R=torch.tensor([[[-1., 0., 0.], [0., -1., 0.], [0., 0., 1.]]]))

	raster_settings = RasterizationSettings(
	    image_size=(H,W), 
	    blur_radius=0.0, 
	    faces_per_pixel=1, 
	)




	
	color = color.permute(1,2,0).cpu().detach()
	
	obj_filenames = []
	for i, (label, quat, trans) in enumerate(zip(target['labels'], target['quats'], target['trans'])):
		obj_filename = os.path.join(DATA_DIR, "model", dataset.object_lookup_name[label.item()], dataset.object_lookup_name[label.item()]+".obj")
		obj_filenames.append(obj_filename)


	# Load obj file
	print(target['quats'].shape, target['trans_gt'].shape)
	mesh = load_objs_as_meshes(obj_filenames, device=device)
	mesh = mesh.update_padded(quaternion_apply(target['quats'].to(device).type(torch.float32), mesh.verts_padded()))
	mesh = mesh.update_padded(Translate(target['trans_gt'].to(device).type(torch.float32)).transform_points(mesh.verts_padded()))

	# Initialize each vertex to be white in color.
	verts_rgb = torch.ones_like(mesh.verts_padded())  # (1, V, 3)
	verts_rgb[:,:,0] = 0

	mesh.textures = TexturesVertex(verts_features=verts_rgb.to(device))
	
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



	images = renderer(mesh)

	images[images==images.max()] = 0
	images = images.sum(0)
	# for i in range(len(images)):
	fig, ax = plt.subplots(1,2)
	#.figure(figsize=(10, 10))
	ax[0].imshow(color)
	ax[1].imshow(images.cpu().numpy())
	plt.axis("off");
	plt.show()
			


if __name__=="__main__":
	main()