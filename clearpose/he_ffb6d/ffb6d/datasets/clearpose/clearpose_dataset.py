#!/usr/bin/env python3
import os
from turtle import color
import cv2
import torch
import os.path
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from common import Config
import pickle as pkl
from utils.basic_utils import Basic_Utils
import scipy.io as scio
import scipy.misc
try:
    from neupeak.utils.webcv2 import imshow, waitKey
except:
    from cv2 import imshow, waitKey
import normalSpeed
from models.RandLA.helper_tool import DataProcessing as DP
import glob
import random
import time
import yaml
import colorsys


config = Config(ds_name='clearpose')
bs_utils = Basic_Utils(config)


class Dataset():

    def __init__(self, dataset_name, DEBUG=False):
        self.dataset_name = dataset_name
        self.debug = DEBUG
        self.xmap = np.array([[j for i in range(640)] for j in range(480)])
        self.ymap = np.array([[i for i in range(640)] for j in range(480)])
        self.diameters = {}
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.224])
        self.cls_lst = config.clearpose_obj_dict
        self.obj_dict = {}
        for cls_id, cls in enumerate(self.cls_lst, start=1):
            self.obj_dict[cls] = cls_id
        self.rng = np.random
        self.root = config.clearpose_root
        self.model_config = self.parsemodel(self.root)
        if dataset_name == 'train':
            self.add_noise = True
            self.all_lst = self.dataset_list(self.root, config.train_list, config.train_ratio)
            self.minibatch_per_epoch = len(self.all_lst) // config.mini_batch_size
            self.meta_data = self.loadmeta(self.root, config.train_list)
            self.depth_type = config.depth_train
        else:
            self.pp_data = None
            self.add_noise = False
            self.all_lst = self.dataset_list(self.root, config.test_list, config.test_ratio)
            self.meta_data = self.loadmeta(self.root, config.test_list)
            self.depth_type = config.depth_test
        print("{}_dataset_size: ".format(dataset_name), len(self.all_lst))
        
    @staticmethod
    def dataset_list(root_path, dataset_list, ratio):
        datalst = []
        for set_idx in dataset_list:
            for scene_idx in dataset_list[set_idx]:
                file_lst = glob.glob(os.path.join(root_path, set_idx, f"scene{scene_idx}", "*-color.png"))
                datalst += [(set_idx, f"scene{scene_idx}", os.path.basename(f).split("-color.png")[0]) for f in file_lst]
        random.shuffle(datalst)
        return datalst[:int(len(datalst)*ratio)]

    @staticmethod
    def loadmeta(root_path, dataset_list):
        metalist = {}
        for set_idx in dataset_list:
            metalist[set_idx] = {}
            for scene_idx in dataset_list[set_idx]:
                metalist[set_idx][f"scene{scene_idx}"] = scio.loadmat(os.path.join(root_path, set_idx, f"scene{scene_idx}", "metadata.mat"))

        return metalist
    
    @staticmethod
    def parsemodel(root_path):
        model = {}
        models_pkg = os.listdir(os.path.join(root_path, "model"))
        for model_pkg in models_pkg:
            if model_pkg in config.clearpose_obj_dict:
                f = open(os.path.join(root_path, "model", model_pkg, f"{model_pkg}_description.txt"))
                model[config.clearpose_obj_dict[model_pkg]] = yaml.load(f)
                model[config.clearpose_obj_dict[model_pkg]]['name'] = model_pkg
                model[config.clearpose_obj_dict[model_pkg]]['path'] = os.path.join(root_path, "model", model_pkg, f"{model_pkg}.obj")
        return model

    def gaussian_noise(self, rng, img, sigma):
        """add gaussian noise of given sigma to image"""
        img = img + rng.randn(*img.shape) * sigma
        img = np.clip(img, 0, 255).astype('uint8')
        return img

    def linear_motion_blur(self, img, angle, length):
        """:param angle: in degree"""
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        a = int(max(list(map(abs, (dx, dy)))) * length * 2)
        if a <= 0:
            return img
        kern = np.zeros((a, a))
        cx, cy = a // 2, a // 2
        dx, dy = list(map(int, (dx * length + cx, dy * length + cy)))
        cv2.line(kern, (cx, cy), (dx, dy), 1.0)
        s = kern.sum()
        if s == 0:
            kern[cx, cy] = 1.0
        else:
            kern /= s
        return cv2.filter2D(img, -1, kern)

    def rgb_add_noise(self, img):
        rng = self.rng
        # apply HSV augmentor
        if rng.rand() > 0:
            hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.uint16)
            hsv_img[:, :, 1] = hsv_img[:, :, 1] * self.rand_range(rng, 1.25, 1.45)
            hsv_img[:, :, 2] = hsv_img[:, :, 2] * self.rand_range(rng, 1.15, 1.35)
            hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1], 0, 255)
            hsv_img[:, :, 2] = np.clip(hsv_img[:, :, 2], 0, 255)
            img = cv2.cvtColor(hsv_img.astype(np.uint8), cv2.COLOR_HSV2BGR)

        if rng.rand() > .8:  # sharpen
            kernel = -np.ones((3, 3))
            kernel[1, 1] = rng.rand() * 3 + 9
            kernel /= kernel.sum()
            img = cv2.filter2D(img, -1, kernel)

        if rng.rand() > 0.8:  # motion blur
            r_angle = int(rng.rand() * 360)
            r_len = int(rng.rand() * 15) + 1
            img = self.linear_motion_blur(img, r_angle, r_len)

        if rng.rand() > 0.8:
            if rng.rand() > 0.2:
                img = cv2.GaussianBlur(img, (3, 3), rng.rand())
            else:
                img = cv2.GaussianBlur(img, (5, 5), rng.rand())

        if rng.rand() > 0.2:
            img = self.gaussian_noise(rng, img, rng.randint(15))
        else:
            img = self.gaussian_noise(rng, img, rng.randint(25))

        if rng.rand() > 0.8:
            img = img + np.random.normal(loc=0.0, scale=7.0, size=img.shape)

        return np.clip(img, 0, 255).astype(np.uint8)

    def add_real_back(self, rgb, labels, dpt, dpt_msk):
        real_item = self.real_gen()
        with Image.open(os.path.join(self.root, real_item+'-depth.png')) as di:
            real_dpt = np.array(di)
        with Image.open(os.path.join(self.root, real_item+'-label.png')) as li:
            bk_label = np.array(li)
        bk_label = (bk_label <= 0).astype(rgb.dtype)
        bk_label_3c = np.repeat(bk_label[:, :, None], 3, 2)
        with Image.open(os.path.join(self.root, real_item+'-color.png')) as ri:
            back = np.array(ri)[:, :, :3] * bk_label_3c
        dpt_back = real_dpt.astype(np.float32) * bk_label.astype(np.float32)

        msk_back = (labels <= 0).astype(rgb.dtype)
        msk_back = np.repeat(msk_back[:, :, None], 3, 2)
        rgb = rgb * (msk_back == 0).astype(rgb.dtype) + back * msk_back

        dpt = dpt * (dpt_msk > 0).astype(dpt.dtype) + \
            dpt_back * (dpt_msk <= 0).astype(dpt.dtype)
        return rgb, dpt

    def dpt_2_pcld(self, dpt, cam_scale, K):
        if len(dpt.shape) > 2:
            dpt = dpt[:, :, 0]
        dpt = dpt.astype(np.float32) / cam_scale
        msk = (dpt > 1e-8).astype(np.float32)
        row = (self.ymap - K[0][2]) * dpt / K[0][0]
        col = (self.xmap - K[1][2]) * dpt / K[1][1]
        dpt_3d = np.concatenate(
            (row[..., None], col[..., None], dpt[..., None]), axis=2
        )
        dpt_3d = dpt_3d * msk[:, :, None]
        return dpt_3d

    def get_item(self, item_name):
        scene_idx, set_idx, data_idx = item_name
        if self.depth_type == "GT":
            with Image.open(os.path.join(self.root, scene_idx, set_idx, data_idx+'-depth_true.png')) as di:
                dpt_um = np.array(di)
        elif self.depth_type == "raw":
            with Image.open(os.path.join(self.root, scene_idx, set_idx, data_idx+'-depth.png')) as di:
                dpt_um = np.array(di)                       
        with Image.open(os.path.join(self.root, scene_idx, set_idx, data_idx+'-label.png')) as li:
            labels = np.array(li)
        
        rgb_labels = labels.copy()
        # meta = scio.loadmat(os.path.join(self.root, item_name+'-meta.mat'))
        # if item_name[:8] != 'data_syn' and int(item_name[5:9]) >= 60:
        #     K = config.intrinsic_matrix['ycb_K2']
        # else:
        #     K = config.intrinsic_matrix['ycb_K1']

        K = self.meta_data[scene_idx][set_idx][data_idx]['intrinsic_matrix'][0][0]

        with Image.open(os.path.join(self.root, scene_idx, set_idx, data_idx+'-color.png')) as ri:
            if self.add_noise:
                ri = self.trancolor(ri)
            rgb = np.array(ri)[:, :, :3]
        rnd_typ = 'real'
        cam_scale = self.meta_data[scene_idx][set_idx][data_idx]['factor_depth'].astype(np.float32)[0][0]
        msk_dp = dpt_um > 1e-6


        # dpt_um = bs_utils.fill_missing(dpt_um, cam_scale, 1)
        msk_dp = dpt_um > 1e-6

        dpt_mm = dpt_um.copy().astype(np.uint16)

        # nrm_map = normalSpeed.depth_normal(
        #     dpt_mm, K[0][0], K[1][1], 5, 2000, 20, False
        # )
        nrm_map = normalSpeed.depth_normal(
            dpt_mm, K[0][0], K[1][1], 1, 100000, 100000, True
        )
        if self.debug:
            show_nrm_map = ((nrm_map + 1.0) * 127).astype(np.uint8)
            cv2.imwrite("nrm_map.png", show_nrm_map[:, :, ::-1])
            im_depth = (dpt_um * 255/dpt_um.max()).astype(np.uint8)
            cv2.imwrite("depth_map.png", im_depth)

        dpt_m = dpt_um.astype(np.float32) / cam_scale
        dpt_xyz = self.dpt_2_pcld(dpt_m, 1.0, K)

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

        ## to remove objects from YCB and Hope
        cls_id_lst = self.meta_data[scene_idx][set_idx][data_idx]['cls_indexes'][0][0].flatten().astype(np.uint32)
        labels[labels > config.n_classes - 1] = 0
        np.delete(cls_id_lst, np.where(cls_id_lst > config.n_classes - 1))
        for cls in cls_id_lst:
            if np.sum(labels == cls) <= config.mask_ignore:
                np.delete(cls_id_lst, np.where(cls_id_lst == cls))
                labels[labels == cls] = 0

        cld = dpt_xyz.reshape(-1, 3)[choose, :]
        rgb_pt = rgb.reshape(-1, 3)[choose, :].astype(np.float32)
        nrm_pt = nrm_map[:, :, :3].reshape(-1, 3)[choose, :]
        labels_pt = labels.flatten()[choose]
        choose = np.array([choose])
        cld_rgb_nrm = np.concatenate((cld, rgb_pt, nrm_pt), axis=1).transpose(1, 0)

        
        RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst = self.get_pose_gt_info(
            cld, labels_pt, cls_id_lst, self.meta_data[scene_idx][set_idx][data_idx]
        )

        h, w = rgb_labels.shape
        dpt_6c = np.concatenate((dpt_xyz, nrm_map[:, :, :3]), axis=2).transpose(2, 0, 1)
        rgb = np.transpose(rgb, (2, 0, 1)) # hwc2chw

        xyz_lst = [dpt_xyz.transpose(2, 0, 1)] # c, h, w
        msk_lst = [dpt_xyz[2, :, :] > 1e-8]

        for i in range(3):
            scale = pow(2, i+1)
            nh, nw = h // pow(2, i+1), w // pow(2, i+1)
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
        if self.debug:
            for ip, xyz in enumerate(xyz_lst):
                pcld = xyz.reshape(3, -1).transpose(1, 0)
                p2ds = bs_utils.project_p3d(pcld, cam_scale, K)
                print(show_rgb.shape, pcld.shape)
                srgb = bs_utils.paste_p2ds(show_rgb.copy(), p2ds, (0, 0, 255))
                imshow("rz_pcld_%d" % ip, srgb)
                p2ds = bs_utils.project_p3d(inputs['cld_xyz%d'%ip], cam_scale, K)
                srgb1 = bs_utils.paste_p2ds(show_rgb.copy(), p2ds, (0, 0, 255))
                imshow("rz_pcld_%d_rnd" % ip, srgb1)

        item_dict = dict(
            rgb=rgb.astype(np.uint8),  # [c, h, w]
            cld_rgb_nrm=cld_rgb_nrm.astype(np.float32),  # [9, npts]
            choose=choose.astype(np.int32),  # [1, npts]
            labels=labels_pt.astype(np.int32),  # [npts]
            rgb_labels=rgb_labels.astype(np.int32),  # [h, w]
            dpt_map_m=dpt_m.astype(np.float32),  # [h, w]
            RTs=RTs.astype(np.float32),
            kp_targ_ofst=kp_targ_ofst.astype(np.float32),
            ctr_targ_ofst=ctr_targ_ofst.astype(np.float32),
            cls_ids=cls_ids.astype(np.int32),
            ctr_3ds=ctr3ds.astype(np.float32),
            kp_3ds=kp3ds.astype(np.float32),
            K = K.astype(np.float32),
        )
        item_dict.update(inputs)
        if self.debug:
            extra_d = dict(
                dpt_xyz_nrm=dpt_6c.astype(np.float32),  # [6, h, w]
                cam_scale=np.array([cam_scale]).astype(np.float32),
                K=K.astype(np.float32),
            )
            item_dict.update(extra_d)
            item_dict['normal_map'] = nrm_map[:, :, :3].astype(np.float32)
        return item_dict

    def get_pose_gt_info(self, cld, labels, cls_id_lst, meta):
        RTs = np.zeros((config.n_objects, 3, 4))
        kp3ds = np.zeros((config.n_objects, config.n_keypoints, 3))
        ctr3ds = np.zeros((config.n_objects, 3))
        cls_ids = np.zeros((config.n_objects, 1))
        kp_targ_ofst = np.zeros((config.n_sample_points, config.n_keypoints, 3))
        ctr_targ_ofst = np.zeros((config.n_sample_points, 3))
        for i, cls_id in enumerate(cls_id_lst):
            if cls_id in self.model_config:
                r = meta['poses'][0][0][:, :, i][:, 0:3]
                t = np.array(meta['poses'][0][0][:, :, i][:, 3:4].flatten()[:, None])
                RT = np.concatenate((r, t), axis=1)
                RTs[i] = RT

                ctr = np.array(self.model_config[cls_id]['center'])
                ctr = np.dot(ctr.T, r.T) + t[:, 0]
                ctr3ds[i, :] = ctr
                msk_idx = np.where(labels == cls_id)[0]

                target_offset = np.array(np.add(cld, -1.0*ctr3ds[i, :]))
                ctr_targ_ofst[msk_idx,:] = target_offset[msk_idx, :]
                cls_ids[i, :] = np.array([cls_id])

                # key_kpts = ''
                # if config.n_keypoints == 8:
                #     kp_type = 'farthest'
                # else:
                #     kp_type = 'farthest{}'.format(config.n_keypoints)
                # kps = bs_utils.get_kps(
                #     self.cls_lst[cls_id-1], kp_type=kp_type, ds_type='ycb'
                # ).copy()
                kps = np.array(self.model_config[cls_id]['keypoints'])
                kps = np.dot(kps, r.T) + t[:, 0]
                kp3ds[i] = kps

                target = []
                for kp in kps:
                    target.append(np.add(cld, -1.0*kp))
                target_offset = np.array(target).transpose(1, 0, 2)  # [npts, nkps, c]
                kp_targ_ofst[msk_idx, :, :] = target_offset[msk_idx, :, :]
        return RTs, kp3ds, ctr3ds, cls_ids, kp_targ_ofst, ctr_targ_ofst

    def __len__(self):
        return len(self.all_lst)

    def __getitem__(self, idx):
        item_name = self.all_lst[idx]
        data = self.get_item(item_name)
        return data

    def verify(self, idx):
        item_name = self.all_lst[idx]
        scene_idx, set_idx, data_idx = item_name
        cls_id_lst = self.meta_data[scene_idx][set_idx][data_idx]['cls_indexes'][0][0].flatten().astype(np.uint32)
        with Image.open(os.path.join(self.root, scene_idx, set_idx, data_idx+'-label.png')) as li:
            labels = np.array(li)
        for cls in cls_id_lst:
            if np.sum(labels == cls) <= config.mask_ignore:
                np.delete(cls_id_lst, np.where(cls_id_lst == cls))
                labels[labels == cls] = 0
        data = self.get_item(item_name)
        image = data['rgb'].transpose(1, 2, 0)[...,::-1]
        
        N = len(cls_id_lst)
        HSV_tuples = [(x*1.0/N, 0.5, 0.5) for x in range(N)]
        RGB_tuples = [tuple((np.array(colorsys.hsv_to_rgb(*x))*255).astype(np.uint8).tolist()) for x in HSV_tuples]

        intrinsic = data['K']
        for i, cls_id in enumerate(cls_id_lst):
            # RT = data['RTs'][cls_id]
            kps = data['kp_3ds'][i]
            kps_px = intrinsic.dot(kps.T)
            kps_px = np.around(kps_px[:2]/kps_px[2]).astype(np.uint16)
            kplist = kps_px.T.tolist()
            for kp in kplist:
                image = cv2.circle(image, tuple(kp), radius=0, color=RGB_tuples[i], thickness=2)
        cv2.imwrite(f"verify_{scene_idx}_{set_idx}_{data_idx}.png", image)
        print(f"verify_{scene_idx}_{set_idx}_{data_idx}")


def main():
    # config.mini_batch_size = 1
    global DEBUG
    DEBUG = True
    ds = {}
    ds = Dataset('test', DEBUG=False)
    while True:
        index = np.random.randint(0, len(ds))
        ds.__getitem__(index)
        # input("Press Enter to continue...")

if __name__ == "__main__":
    main()
# vim: ts=4 sw=4 sts=4 expandtab
