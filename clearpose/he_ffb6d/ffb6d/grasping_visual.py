#!/usr/bin/env python3
from __future__ import (
    division,
    absolute_import,
    with_statement,
    print_function,
    unicode_literals,
)
import os
import tqdm
import cv2
import torch
import argparse
import torch.nn as nn
import numpy as np
import pickle as pkl
from common import Config, ConfigRandLA
from models.ffb6d import FFB6D
from datasets.ycb.ycb_dataset import Dataset as YCB_Dataset
from datasets.linemod.linemod_dataset import Dataset as LM_Dataset
from utils.pvn3d_eval_utils_kpls import cal_frame_poses, cal_frame_poses_lm
from utils.basic_utils import Basic_Utils
import normalSpeed
from PIL import Image
from models.RandLA.helper_tool import DataProcessing as DP
import json 
from time import sleep
from paramiko import SSHClient
from scp import SCPClient


try:
    from neupeak.utils.webcv2 import imshow, waitKey
except ImportError:
    from cv2 import imshow, waitKey


parser = argparse.ArgumentParser(description="Arg parser")
parser.add_argument(
    "-checkpoint", type=str, default=None, help="Checkpoint to eval"
)
parser.add_argument(
    "-dataset", type=str, default="linemod",
    help="Target dataset, ycb or linemod. (linemod as default)."
)
parser.add_argument(
    "-cls", type=str, default="ape",
    help="Target object to eval in LineMOD dataset. (ape, benchvise, cam, can," +
    "cat, driller, duck, eggbox, glue, holepuncher, iron, lamp, phone)"
)
parser.add_argument(
    "-show", action='store_true', help="View from imshow or not."
)
args = parser.parse_args()

if args.dataset == "ycb":
    config = Config(ds_name=args.dataset)
else:
    config = Config(ds_name=args.dataset, cls_type=args.cls)
bs_utils = Basic_Utils(config)


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))


def load_checkpoint(model=None, optimizer=None, filename="checkpoint"):
    filename = "{}.pth.tar".format(filename)

    assert os.path.isfile(filename), "==> Checkpoint '{}' not found".format(filename)
    print("==> Loading from checkpoint '{}'".format(filename))
    try:
        checkpoint = torch.load(filename)
    except Exception:
        checkpoint = pkl.load(open(filename, "rb"))
    epoch = checkpoint.get("epoch", 0)
    it = checkpoint.get("it", 0.0)
    best_prec = checkpoint.get("best_prec", None)
    if model is not None and checkpoint["model_state"] is not None:
        ck_st = checkpoint['model_state']
        if 'module' in list(ck_st.keys())[0]:
            tmp_ck_st = {}
            for k, v in ck_st.items():
                tmp_ck_st[k.replace("module.", "")] = v
            ck_st = tmp_ck_st
        model.load_state_dict(ck_st)
    if optimizer is not None and checkpoint["optimizer_state"] is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    print("==> Done")
    return it, epoch, best_prec


def cal_view_pred_pose(model, data, epoch=0, obj_id=-1, visual = True):
    model.eval()
    with torch.set_grad_enabled(False):
        cu_dt = {}
        
        # device = torch.device('cuda:{}'.format(args.local_rank))
        for key in data.keys():
            if data[key].dtype in [np.float32, np.uint8]:
                cu_dt[key] = torch.from_numpy(data[key].astype(np.float32)).cuda()
            elif data[key].dtype in [np.int32, np.uint32]:
                cu_dt[key] = torch.LongTensor(data[key].astype(np.int32)).cuda()
            elif data[key].dtype in [torch.uint8, torch.float32]:
                cu_dt[key] = data[key].float().cuda()
            elif data[key].dtype in [torch.int32, torch.int16]:
                cu_dt[key] = data[key].long().cuda()
        end_points = model(cu_dt)

        cu_dt['cls_ids'] = torch.tensor([1, 2, 3, 4, 5, 6, 8, 9, 14, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])[None, :, None]
        _, classes_rgbd = torch.max(end_points['pred_rgbd_segs'], 1)

        pcld = cu_dt['cld_rgb_nrm'][:, :3, :].permute(0, 2, 1).contiguous()
        if args.dataset == "ycb":
            pred_cls_ids, pred_pose_lst, _ = cal_frame_poses(
                pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0],
                end_points['pred_kp_ofs'][0], True, config.n_objects, True,
                None, None
            )
        else:
            pred_pose_lst = cal_frame_poses_lm(
                pcld[0], classes_rgbd[0], end_points['pred_ctr_ofs'][0],
                end_points['pred_kp_ofs'][0], True, config.n_objects, False, obj_id
            )
            pred_cls_ids = np.array([[1]])

        if visual:
            np_rgb = cu_dt['rgb'].cpu().numpy().astype("uint8")[0].transpose(1, 2, 0).copy()
            if args.dataset == "ycb":
                np_rgb = np_rgb[:, :, ::-1].copy()
            ori_rgb = np_rgb.copy()
            for cls_id in cu_dt['cls_ids'][0].cpu().numpy():
                idx = np.where(pred_cls_ids == cls_id)[0]
                if len(idx) == 0:
                    continue
                pose = pred_pose_lst[idx[0]]
                if args.dataset == "ycb":
                    obj_id = int(cls_id[0])
                mesh_pts = bs_utils.get_pointxyz(obj_id, ds_type=args.dataset).copy()
                mesh_pts = np.dot(mesh_pts, pose[:, :3].T) + pose[:, 3]
                # if args.dataset == "ycb":
                #     K = config.intrinsic_matrix["ycb_K1"]
                # else:
                #     K = config.intrinsic_matrix["linemod"]
                K = np.squeeze(data['K'])
                mesh_p2ds = bs_utils.project_p3d(mesh_pts, 1.0, K)
                color = bs_utils.get_label_color(obj_id, n_obj=22, mode=2)
                np_rgb = bs_utils.draw_p2ds(np_rgb, mesh_p2ds, color=color)
            vis_dir = "/home/huijie/Desktop/grasping_visual/visual"
            ensure_fd(vis_dir)
            f_pth = os.path.join(vis_dir, "{}.jpg".format(epoch))
            if args.dataset == 'ycb':
                bgr = np_rgb
                ori_bgr = ori_rgb
            else:
                bgr = np_rgb[:, :, ::-1]
                ori_bgr = ori_rgb[:, :, ::-1]
            cv2.imwrite(f_pth, bgr)
            if args.show:
                imshow("projected_pose_rgb", bgr)
                imshow("original_rgb", ori_bgr)
                waitKey()
        
        return pred_cls_ids, pred_pose_lst

def dpt_2_pcld(dpt, cam_scale, K):
    if len(dpt.shape) > 2:
        dpt = dpt[:, :, 0]
    dpt = dpt.astype(np.float32) / cam_scale
    msk = (dpt > 1e-8).astype(np.float32)
    xmap = np.array([[j for i in range(640)] for j in range(480)])
    ymap = np.array([[i for i in range(640)] for j in range(480)])
    row = (ymap - K[0][2]) * dpt / K[0][0]
    col = (xmap - K[1][2]) * dpt / K[1][1]
    dpt_3d = np.concatenate(
        (row[..., None], col[..., None], dpt[..., None]), axis=2
    )
    dpt_3d = dpt_3d * msk[:, :, None]
    return dpt_3d

def get_item(path, idx):
    with Image.open(os.path.join(path, "{0:06d}-depth.png".format(idx))) as di:
        dpt_um= np.array(di)
    with Image.open(os.path.join(path, "{0:06d}-color.png".format(idx))) as ri:
        rgb = np.array(ri)

    # K = np.array([
    #     [536.0, 0, 324.0],
    #     [0, 538.0, 224.0],
    #     [0, 0, 1],
    # ])
    K = np.array([
        [527.0, 0, 324.0],
        [0, 526.0, 227.0],
        [0, 0, 1],
    ])
    # K = np.array([
    #     [534.520, 0, 327.7614],
    #     [0, 524.5708, 235.9347],
    #     [0, 0, 1],
    # ])
    cam_scale = 1000
    dpt_um = bs_utils.fill_missing(dpt_um, cam_scale, 1)
    msk_dp = dpt_um > 1e-6
    dpt_mm = (dpt_um.copy()/10).astype(np.uint16)
    nrm_map = normalSpeed.depth_normal(
        dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False
    )
    dpt_m = dpt_um.astype(np.float32) / cam_scale
    dpt_xyz = dpt_2_pcld(dpt_m, 1.0, K)
    choose = msk_dp.flatten().nonzero()[0].astype(np.uint32)
    if len(choose) < 400:
        return None
    choose_2 = np.array([i for i in range(len(choose))])
    if len(choose_2) < 400:
        return None
    if len(choose_2) > config.n_sample_points:
        c_mask = np.zeros(len(choose_2), dtype=int)
        c_mask[:config.n_sample_points] = 1
        np.random.shuffle(c_mask)
        choose_2 = choose_2[c_mask.nonzero()]
    else:
        choose_2 = np.pad(choose_2, (0, config.n_sample_points-len(choose_2)), 'wrap')
    choose = np.array(choose)[choose_2]

    sf_idx = np.arange(choose.shape[0])
    np.random.shuffle(sf_idx)
    choose = choose[sf_idx]

    cld = dpt_xyz.reshape(-1, 3)[choose, :]
    rgb_pt = rgb.reshape(-1, 3)[choose, :].astype(np.float32)
    nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
    choose = np.array([choose])
    cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1).transpose(1, 0)
    
    dpt_6c = np.concatenate((dpt_xyz, nrm_map[:, :, :3]), axis=2).transpose(2, 0, 1)
    rgb = np.transpose(rgb, (2, 0, 1)) # hwc2chw

    xyz_lst = [dpt_xyz.transpose(2, 0, 1)] # c, h, w
    msk_lst = [dpt_xyz[2, :, :] > 1e-8]

    for i in range(3):
        scale = pow(2, i+1)
        nh, nw = 480 // pow(2, i+1), 640 // pow(2, i+1)
        ys, xs = np.mgrid[:nh, :nw]
        xyz_lst.append(xyz_lst[0][:, ys*scale, xs*scale])
        msk_lst.append(xyz_lst[-1][2, :, :] > 1e-8)
    sr2dptxyz = {
        pow(2, ii): item.reshape(3, -1).transpose(1, 0) for ii, item in enumerate(xyz_lst)
    }
    sr2msk = {
        pow(2, ii): item.reshape(-1) for ii, item in enumerate(msk_lst)
    }

    rgb_ds_sr = [4, 8, 8, 8]
    n_ds_layers = 4
    pcld_sub_s_r = [4, 4, 4, 4]
    inputs = {}
    # DownSample stage
    for i in range(n_ds_layers):
        nei_idx = DP.knn_search(
            cld[None, ...], cld[None, ...], 16
        ).astype(np.int32).squeeze(0)
        sub_pts = cld[:cld.shape[0] // pcld_sub_s_r[i], :]
        pool_i = nei_idx[:cld.shape[0] // pcld_sub_s_r[i], :]
        up_i = DP.knn_search(
            sub_pts[None, ...], cld[None, ...], 1
        ).astype(np.int32).squeeze(0)
        inputs['cld_xyz%d'%i] = cld.astype(np.float32).copy()
        inputs['cld_nei_idx%d'%i] = nei_idx.astype(np.int32).copy()
        inputs['cld_sub_idx%d'%i] = pool_i.astype(np.int32).copy()
        inputs['cld_interp_idx%d'%i] = up_i.astype(np.int32).copy()
        nei_r2p = DP.knn_search(
            sr2dptxyz[rgb_ds_sr[i]][None, ...], sub_pts[None, ...], 16
        ).astype(np.int32).squeeze(0)
        inputs['r2p_ds_nei_idx%d'%i] = nei_r2p.copy()
        nei_p2r = DP.knn_search(
            sub_pts[None, ...], sr2dptxyz[rgb_ds_sr[i]][None, ...], 1
        ).astype(np.int32).squeeze(0)
        inputs['p2r_ds_nei_idx%d'%i] = nei_p2r.copy()
        cld = sub_pts

    n_up_layers = 3
    rgb_up_sr = [4, 2, 2]
    for i in range(n_up_layers):
        r2p_nei = DP.knn_search(
            sr2dptxyz[rgb_up_sr[i]][None, ...],
            inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...], 16
        ).astype(np.int32).squeeze(0)
        inputs['r2p_up_nei_idx%d'%i] = r2p_nei.copy()
        p2r_nei = DP.knn_search(
            inputs['cld_xyz%d'%(n_ds_layers-i-1)][None, ...],
            sr2dptxyz[rgb_up_sr[i]][None, ...], 1
        ).astype(np.int32).squeeze(0)
        inputs['p2r_up_nei_idx%d'%i] = p2r_nei.copy()

    show_rgb = rgb.transpose(1, 2, 0).copy()[:, :, ::-1]    
    
    # dict ={}
    # dict['rgb'] = torch.from_numpy(rgb).to("cuda")
    # dict['rgb'] = torch.permute(dict['rgb'], (2, 0, 1))
    # dict['rgb'] = dict['rgb'][None, :]
    item_dict = dict(
        rgb=rgb.astype(np.uint8),  # [bs, c, h, w]
        cld_rgb_nrm=cld_rgb_nrm.astype(np.float32),  # [bs, 9, npts]
        choose=choose.astype(np.int32),  # [bs, 1, npts]
        dpt_map_m=dpt_m.astype(np.float32),  # [bs, h, w]
        K = K.astype(np.float32),
    )
    item_dict.update(inputs)
    for item in item_dict:
        item_dict[item] = item_dict[item][np.newaxis, :]
    return item_dict


def main():
    if args.dataset == "ycb":
        test_ds = YCB_Dataset('test')
        obj_id = -1
    else:
        test_ds = LM_Dataset('test', cls_type=args.cls)
        obj_id = config.lm_obj_dict[args.cls]
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config.test_mini_batch_size, shuffle=False,
        num_workers=20
    )

    rndla_cfg = ConfigRandLA
    model = FFB6D(
        n_classes=config.n_objects, n_pts=config.n_sample_points, rndla_cfg=rndla_cfg,
        n_kps=config.n_keypoints
    )
    model.cuda()

    # load status from checkpoint
    if args.checkpoint is not None:
        load_checkpoint(
            model, None, filename=args.checkpoint[:-8]
        )
    object_label = { # according to 21 objects in ycbv
        "002_master_chef_can" : 1,
        "003_cracker_box" : 2,
        "004_sugar_box" : 3,
        "005_tomato_soup_can" : 4,
        "006_mustard_bottle" : 5,
        "007_tuna_fish_can" : 6,
        "009_gelatin_box" : 8,
        "010_potted_meat_can" : 9,
        "025_mug" : 14,
        "040_large_marker" : 18
    }
    # ssh = SSHClient()
    # ssh.load_system_host_keys()
    # ssh.connect(hostname='fetch7',
    #             username='fetch',
    #             password='wolverine')


    # SCPCLient takes a paramiko transport as its only argument
    # scp = SCPClient(ssh.get_transport())



    while True:
        files = os.listdir("/home/huijie/Desktop/grasping_visual")
        if "{0:06d}-color.png".format(0) in files and "{0:06d}-depth.png".format(0) in files :
            print("#################################")
            print("Staring inference the coming data")
            print("#################################")
            label_object = {v: k for k, v in object_label.items()}
            data = get_item("/home/huijie/Desktop/grasping_visual", 0)
            pred_cls_ids, pred_pose_lst = cal_view_pred_pose(model, data, epoch=1, obj_id=obj_id)
            pose_dict = {}
            for i in range(len(pred_cls_ids)):
                idx = pred_cls_ids[i]
                if idx in label_object:
                    pose_dict[label_object[idx]] = pred_pose_lst[i].tolist()
            with open("/home/huijie/Desktop/grasping_visual/pose/pose.json", 'w') as outfile:
                json.dump(pose_dict, outfile, indent=True)
            # scp.put('/home/huijie/Desktop/grasping_visual/visual/1.jpg', '~/Desktop/grasping_visual/visual')
            # scp.put('/home/huijie/Desktop/grasping_visual/pose/pose.json', '~/Desktop/grasping_visual/pose')
            # scp.get('file_path_on_remote_machine', 'file_path_on_local_machine')
            # p = subprocess.Popen(['scp', '/home/huijie/Desktop/grasping_visual/visual/1.png', 'fecth@fetch18:~/Desktop/grasping_visual/visual'], 
            #          stdin=subprocess.PIPE, stdout=subprocess.PIPE, universal_newlines=True)
            # p.communicate('wolverine\n')
            os.system("rm /home/huijie/Desktop/grasping_visual/000000-color.png")
            os.system("rm /home/huijie/Desktop/grasping_visual/000000-depth.png")
            print("\n\n\n")
            print("Detecting object:")
            print(pose_dict.keys())
            print("\n\n\n")
            print("#################################")
            print("Finishing inference the coming data")
            print("#################################")
        sleep(0.5)
    # scp.close()

if __name__ == "__main__":
    main()

# vim: ts=4 sw=4 sts=4 expandtab
