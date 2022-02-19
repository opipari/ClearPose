import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch_geometric.nn import EdgeConv

from clearpose.networks.references.posenet.pspnet import PSPNet
from clearpose.networks.references.posenet.utils import sample_rotations_60

psp_models = {
	'resnet18': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet18'),
	'resnet34': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=512, deep_features_size=256, backend='resnet34'),
	'resnet50': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet50'),
	'resnet101': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet101'),
	'resnet152': lambda: PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024, backend='resnet152')
}


class stage2_model(nn.Module):
	def __init__(self, num_obj):
		super(stage2_model, self).__init__()
		self.N = 500
		self.k = 16
		self.num_rot = 60
		self.rot_anchors = sample_rotations_60()

		self.num_obj = num_obj

		self.color_cnn = psp_models['resnet18']()

		self.edge_conv1 = torch.nn.Conv2d(12, 64, 1)
		# self.edge_conv2 = torch.nn.Conv2d(128, 64, 1)
		self.edge_conv2 = torch.nn.Conv2d(128, 128, 1)
		# ec = EdgeConv(nn: Callable, aggr: str = 'max')

		self.conv1 = torch.nn.Conv1d(32, 64, 1)
		self.conv2 = torch.nn.Conv1d(64, 128, 1)

		# self.conv1_t = torch.nn.Conv1d(256, 256, 1)
		# self.conv2_t = torch.nn.Conv1d(256, 1024, 1)
		self.translation_pool = nn.Sequential(nn.Conv1d(256, 256, 1), 
												nn.ReLU(), 
												nn.Conv1d(256, 1024, 1), 
												nn.ReLU(), 
												nn.AdaptiveAvgPool1d(1))

		# self.conv1_r = torch.nn.Conv1d(256, 256, 1)
		# self.conv2_r = torch.nn.Conv1d(256, 1024, 1)
		self.rotation_pool = nn.Sequential(nn.Conv1d(256, 256, 1), 
												nn.ReLU(), 
												nn.Conv1d(256, 1024, 1), 
												nn.ReLU(), 
												nn.AdaptiveAvgPool1d(1))


		self.translation_head = nn.Sequential(nn.Conv1d(1152, 512, 1), 
												nn.ReLU(), 
												nn.Conv1d(512, 256, 1),
												nn.ReLU(),  
												nn.Conv1d(256, 128, 1),
												nn.ReLU(), 
												nn.Conv1d(128, self.num_obj*3, 1))


		self.rotation_head = nn.Sequential(nn.Conv1d(1024, 512, 1), 
												nn.ReLU(), 
												nn.Conv1d(512, 256, 1),
												nn.ReLU(),  
												nn.Conv1d(256, 128, 1),
												nn.ReLU(), 
												nn.Conv1d(128, self.num_obj*self.num_rot*4, 1))


		self.confidence_head = nn.Sequential(nn.Conv1d(1024, 512, 1), 
												nn.ReLU(), 
												nn.Conv1d(512, 256, 1),
												nn.ReLU(),  
												nn.Conv1d(256, 128, 1),
												nn.ReLU(), 
												nn.Conv1d(128, self.num_obj*self.num_rot*1, 1),
												nn.Sigmoid())


	def get_edge_feature(self, x, nn_idx):
		""" Construct edge feature.
		Args:
			x: bs x c x n_p
			nn_idx: bs x n_p x k
		Returns:
			edge_feature: bs x 2c x n_p x k
		"""
		bs, c, n_p = x.size()
		nn_idx = torch.unsqueeze(nn_idx, 1).repeat(1, c, 1, 1).view(bs, c, n_p*self.k)
		neighbors = torch.gather(x, 2, nn_idx).view(bs, c, n_p, self.k)
		central = torch.unsqueeze(x, 3).repeat(1, 1, 1, self.k)
		edge_feature = torch.cat((central, neighbors - central), dim=1)
		return edge_feature

	def forward(self, color, geometry, masks, obj):
		choose = masks.flatten(start_dim=1).nonzero(as_tuple=True)
		nonzeros = torch.count_nonzero(masks.flatten(start_dim=1),dim=1)
		nonzeros_cumsum = torch.cumsum(nonzeros,0)
		nonzero_samples = torch.cat([torch.randint(high=nonzeros_cumsum[0], size=(self.N,))] + [torch.randint(low=nonzeros_cumsum[i], high=nonzeros_cumsum[i+1], size=(self.N,)) for i in range(nonzeros_cumsum.shape[0]-1)])
		choose = tuple(c[nonzero_samples] for c in choose)
			
		color_emb = self.color_cnn(color).permute(0,2,3,1)
		color_emb_samples = color_emb.flatten(start_dim=1, end_dim=2)[choose].view(nonzeros_cumsum.shape[0], self.N, color_emb.shape[-1])
		geometry_samples = geometry.flatten(start_dim=1, end_dim=2)[choose].view(nonzeros_cumsum.shape[0], self.N, 7)
		point_samples = geometry_samples[:,:,4:]
		geometry_samples = geometry_samples[:,:,:6].permute(0,2,1)
		color_emb_samples = color_emb_samples.permute(0,2,1)

		dist = torch.sqrt(torch.bmm(point_samples, point_samples.permute(0,2,1)))
		nn_idx = torch.topk(dist, self.k, largest=False)[1]

		geometry_feats64 = F.relu(self.edge_conv1(self.get_edge_feature(geometry_samples, nn_idx)))
		geometry_feats64, _ = torch.max(geometry_feats64, dim=3, keepdim=False)
		geometry_feats128 = F.relu(self.edge_conv2(self.get_edge_feature(geometry_feats64, nn_idx)))
		geometry_feats128, _ = torch.max(geometry_feats128, dim=3, keepdim=False)

		color_emb64 = F.relu(self.conv1(color_emb_samples))
		point_feats = torch.cat((color_emb64,geometry_feats64),dim=1)
		color_emb128 = F.relu(self.conv2(color_emb64))
		
		fusion = torch.cat((color_emb128, geometry_feats128), dim=1)
		
		t_emb = self.translation_pool(fusion)
		t_emb = torch.cat((point_feats, t_emb.repeat(1,1,self.N)), dim=1)

		r_emb = self.rotation_pool(fusion)
		
		trans = self.translation_head(t_emb).view(t_emb.shape[0], self.num_obj, 3, self.N)
		rots = self.rotation_head(r_emb).view(r_emb.shape[0], self.num_obj, self.num_rot, 4)
		confs = self.confidence_head(r_emb).view(r_emb.shape[0], self.num_obj, self.num_rot)

		out_tx = torch.diagonal(torch.index_select(trans, 1, obj)).permute(2,1,0)
		out_cx = torch.diagonal(torch.index_select(confs, 1, obj)).permute(1,0)
		out_rx = torch.diagonal(torch.index_select(rots, 1, obj)).permute(2,0,1)
		out_rx = F.normalize(out_rx, p=2, dim=2)    # 1 x num_rot x 4

		rot_anchors = torch.from_numpy(self.rot_anchors).float().cuda()
		rot_anchors = torch.unsqueeze(torch.unsqueeze(rot_anchors, dim=0), dim=3)     # 1 x num_rot x 4 x 1
		out_rx = torch.unsqueeze(out_rx, 2)     # 1 x num_rot x 1 x 4
		out_rx = torch.cat((out_rx[:, :, :, 0], -out_rx[:, :, :, 1], -out_rx[:, :, :, 2], -out_rx[:, :, :, 3], \
							out_rx[:, :, :, 1],  out_rx[:, :, :, 0],  out_rx[:, :, :, 3], -out_rx[:, :, :, 2], \
							out_rx[:, :, :, 2], -out_rx[:, :, :, 3],  out_rx[:, :, :, 0],  out_rx[:, :, :, 1], \
							out_rx[:, :, :, 3],  out_rx[:, :, :, 2], -out_rx[:, :, :, 1],  out_rx[:, :, :, 0], \
							), dim=2).contiguous().view(out_rx.shape[0], self.num_rot, 4, 4)
		out_rx = torch.squeeze(torch.matmul(out_rx, rot_anchors), dim=3)
		return out_tx, out_rx, out_cx, choose

def build_model(config):
	model = stage2_model(config["num_obj"])
	# model.mask_rcnn_model.load_state_dict(torch.load(config["mask_rcnn_model"]), strict=False)
	# model.deeplabv3_model.load_state_dict(torch.load(config["deeplabv3_model"]), strict=False)

	return model
