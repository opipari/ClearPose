#!/usr/bin/env python3
import os
import yaml
import numpy as np


def ensure_fd(fd):
    if not os.path.exists(fd):
        os.system('mkdir -p {}'.format(fd))


class ConfigRandLA:
    k_n = 16  # KNN
    num_layers = 4  # Number of layers
    num_points = 480 * 640 // 24  # Number of input points
    num_classes = 22  # Number of valid classes
    sub_grid_size = 0.06  # preprocess_parameter

    batch_size = 4  # batch_size during training
    val_batch_size = 4  # batch_size during validation and test
    train_steps = 500  # Number of steps per epochs
    val_steps = 100  # Number of validation steps per epoch
    in_c = 9

    sub_sampling_ratio = [4, 4, 4, 4]  # sampling ratio of random sampling at each layer
    d_out = [32, 64, 128, 256]  # feature dimension
    num_sub_points = [num_points // 4, num_points // 16, num_points // 64, num_points // 256]


class Config:
    def __init__(self, ds_name='ycb', cls_type='', test_type = 'GT'):
        self.dataset_name = ds_name
        # self.exp_dir = os.path.dirname(__file__)
        self.exp_dir = "/home/huijie/research/transparentposeestimation/ClearPose/experiments/he_ffb6d"
        self.exp_name = os.path.basename(self.exp_dir)
        
        self.resnet_ptr_mdl_p = os.path.abspath(
            os.path.join(
                self.exp_dir,
                'models/cnn/ResNet_pretrained_mdl'
            )
        )
        ensure_fd(self.resnet_ptr_mdl_p)

        # log folder
        self.cls_type = cls_type
        self.log_dir = os.path.abspath(
            os.path.join(self.exp_dir, 'train_log', self.dataset_name)
        )
        ensure_fd(self.log_dir)
        self.log_model_dir = os.path.join(self.log_dir, 'checkpoints', self.cls_type)
        ensure_fd(self.log_model_dir)
        self.log_eval_dir = os.path.join(self.log_dir, 'eval_results', self.cls_type)
        ensure_fd(self.log_eval_dir)
        self.log_traininfo_dir = os.path.join(self.log_dir, 'train_info', self.cls_type)
        ensure_fd(self.log_traininfo_dir)

        self.n_total_epoch = 20
        self.mini_batch_size = 6
        self.val_mini_batch_size = 4
        self.test_mini_batch_size = 1

        self.n_sample_points = 480 * 640 // 24  # Number of input points
        self.n_keypoints = 8
        self.n_min_points = 400

        self.noise_trans = 0.05  # range of the random noise of translation added to the training data

        self.preprocessed_testset_pth = ''
        if self.dataset_name == 'ycb':
            self.n_objects = 21 + 1  # 21 objects + background
            self.n_classes = self.n_objects
            self.use_orbfps = True
            self.kp_orbfps_dir = 'datasets/ycb/ycb_kps/'
            self.kp_orbfps_ptn = os.path.join(self.kp_orbfps_dir, '%s_%d_kps.txt')
            self.ycb_cls_lst_p = os.path.abspath(
                os.path.join(
                    self.exp_dir, 'datasets/ycb/dataset_config/classes.txt'
                )
            )
            self.ycb_root = os.path.abspath(
                os.path.join(
                    self.exp_dir, 'datasets/ycb/YCB_Video_Dataset'
                )
            )
            self.ycb_kps_dir = os.path.abspath(
                os.path.join(
                    self.exp_dir, 'datasets/ycb/ycb_kps/'
                )
            )
            ycb_r_lst_p = os.path.abspath(
                os.path.join(
                    self.exp_dir, 'datasets/ycb/dataset_config/radius.txt'
                )
            )
            self.ycb_r_lst = list(np.loadtxt(ycb_r_lst_p))
            self.ycb_cls_lst = self.read_lines(self.ycb_cls_lst_p)
            self.ycb_sym_cls_ids = [13, 16, 19, 20, 21]
        elif self.dataset_name == 'clearpose':
            self.n_objects = 63 + 1  # 21 objects + background

            self.n_classes = self.n_objects
            self.use_orbfps = False
            self.clearpose_root = "/home/huijie/research/transparentposeestimation/ClearPose/clearpose/he_ffb6d/ffb6d/datasets/clearpose/dataset"

            self.clearpose_obj_dict = {
            "beaker_1": 1,
            "dropper_1": 2,
            "dropper_2": 3,
            "flask_1": 4,
            "funnel_1": 5,
            "graduated_cylinder_1": 6,
            "graduated_cylinder_2": 7,
            "pan_1": 8,
            "pan_2": 9,
            "pan_3": 10,
            "reagent_bottle_1": 11,
            "reagent_bottle_2": 12,
            "stick_1": 13,
            "syringe_1": 14,
            "bottle_1": 15,
            "bottle_2": 16,
            "bottle_3": 17,
            "bottle_4": 18,
            "bottle_5": 19,
            "bowl_1": 20,
            "bowl_2": 21,
            "bowl_3": 22,
            "bowl_4": 23,
            "bowl_5": 24,
            "bowl_6": 25,
            "container_1": 26,
            "container_2": 27,
            "container_3": 28,
            "container_4": 29,
            "container_5": 30,
            "fork_1": 31,
            "knife_1": 32,
            "knife_2": 33,
            "mug_1": 34,
            "mug_2": 35,
            "pitcher_1": 36,
            "plate_1": 37,
            "plate_2": 38,
            "spoon_1": 39,
            "spoon_2": 40,
            "water_cup_1": 41,
            "water_cup_2": 42,
            "water_cup_3": 43,
            "water_cup_4": 44,
            "water_cup_5": 45,
            "water_cup_6": 46,
            "water_cup_7": 47,
            "water_cup_8": 48,
            "water_cup_9": 49,
            "water_cup_10": 50,
            "water_cup_11": 51,
            "water_cup_12": 52,
            "water_cup_13": 53,
            "water_cup_14": 54,
            "wine_cup_1": 55,
            "wine_cup_2": 56,
            "wine_cup_3": 57,
            "wine_cup_4": 58,
            "wine_cup_5": 59,
            "wine_cup_6": 60,
            "wine_cup_7": 61,
            "wine_cup_8": 62,
            "wine_cup_9": 63
            }
            self.clearpose_normal_cls = [1, 6, 7, 12, 14, 31, 32, 33, 34, 35, 36, 39, 40]

            self.clearpose_sym_cls_ids = [i for i in range(1, 64) if i not in self.clearpose_normal_cls]
            # self.train_list = {
            #     "set1": [1, 2, 3, 4],
            #     "set4": [1, 2, 3, 4, 5],
            #     "set5": [1, 2, 3, 4, 5],
            #     "set6": [1, 2, 3, 4, 5],
            #     "set7": [1, 2, 3, 4, 5]
            # }

            # self.test_list = {
            #     "set1": [5],
            #     "set2": [1, 3, 4, 5, 6],
            #     "set3": [1, 3, 4, 8, 11],
            #     "set4": [6],
            #     "set5": [6],
            #     "set6": [6],
            #     "set7": [6],
            #     "set8": [1, 2, 3, 4, 5, 6]          
            # }
            self.train_list = {}
            self.test_list = {}

            self.train_ratio = 1.0
            self.test_ratio = 1.0

            ## train could be ["GT", "raw", "depthcomplete"]
            self.depth_train = "raw"
            assert test_type in ["GT", "raw", "depthcomplete_transcg_raw", "depthcomplete_transcg_mask", "depthcomplete_id_mask"]
            self.depth_test = test_type

            self.mask_ignore = 0

        self.intrinsic_matrix = {
            'linemod': np.array([[572.4114, 0.,         325.2611],
                                [0.,        573.57043,  242.04899],
                                [0.,        0.,         1.]]),
            'blender': np.array([[700.,     0.,     320.],
                                 [0.,       700.,   240.],
                                 [0.,       0.,     1.]]),
            'ycb_K1': np.array([[1066.778, 0.        , 312.9869],
                                [0.      , 1067.487  , 241.3109],
                                [0.      , 0.        , 1.0]], np.float32),
            'ycb_K2': np.array([[1077.836, 0.        , 323.7872],
                                [0.      , 1078.189  , 279.6921],
                                [0.      , 0.        , 1.0]], np.float32),
            'clearpose': np.array([[601.3, 0.        , 334.7],
                                   [0.      , 601.3  , 248.0],
                                   [0.      , 0.        , 1.0]], np.float32)
        }

    def read_lines(self, p):
        with open(p, 'r') as f:
            return [
                line.strip() for line in f.readlines()
            ]


# config = Config()
# vim: ts=4 sw=4 sts=4 expandtab
