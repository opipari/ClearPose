import torch
import numpy as np
from common import Config
from utils.basic_utils import Basic_Utils

config = Config(ds_name='ycb')
bs_utils = Basic_Utils(config)
config = Config(ds_name='ycb')
cls_lst = config.ycb_cls_lst

def eval_metric(
    cls_ids, pred_pose_lst, pred_cls_ids, RTs, pred_kpc_lst
):
    n_cls = config.n_classes
    cls_add_dis = [list() for i in range(n_cls)]
    cls_adds_dis = [list() for i in range(n_cls)]
    for icls, cls_id in enumerate(cls_ids):
        if cls_id == 0:
            break

        cls_idx = np.where(pred_cls_ids == cls_id[0].item())[0]
        if len(cls_idx) == 0:
            pred_RT = torch.zeros(3, 4).cuda()
        else:
            pred_RT = pred_pose_lst[cls_idx[0]]
            pred_kp = pred_kpc_lst[cls_idx[0]][:-1, :]
            pred_RT = torch.from_numpy(pred_RT.astype(np.float32)).cuda()

        gt_RT = RTs[icls]
        mesh_pts = bs_utils.get_pointxyz_cuda(cls_lst[cls_id-1]).clone()
        add = bs_utils.cal_add_cuda(pred_RT, gt_RT, mesh_pts)
        adds = bs_utils.cal_adds_cuda(pred_RT, gt_RT, mesh_pts)
        cls_add_dis[cls_id].append(add.item())
        cls_adds_dis[cls_id].append(adds.item())
        cls_add_dis[0].append(add.item())
        cls_adds_dis[0].append(adds.item())

    return cls_add_dis, cls_adds_dis