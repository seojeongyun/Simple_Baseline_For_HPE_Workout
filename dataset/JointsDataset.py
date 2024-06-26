# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import random
import os

import cv2
import numpy as np
import torch
import glob
import json

from tqdm import tqdm

from torch.utils.data import Dataset
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints


logger = logging.getLogger(__name__)


class JointsDataset(Dataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        self.cfg = cfg
        self.num_joints = 24
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.is_train = is_train
        self.root = root
        self.image_set = image_set                  # whether the dataset is train or validation

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale_factor = cfg.DATASET.SCALE_FACTOR
        self.rotation_factor = cfg.DATASET.ROT_FACTOR
        self.flip = cfg.DATASET.FLIP

        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        self.sigma = cfg.MODEL.EXTRA.SIGMA

        self.transform = transform
        # self.json_files = self.get_json_files()
        # self.make_db = self.make_db()
        self.key, self.db = self.get_db()

    def get_json_files(self):
        label_path = []
        json_list = []

        for path in os.listdir(self.cfg.DATASET.ROOT_LABEL):
            if 'json' not in path and 'furniture' in path or 'new' in path:
                label_path.append(self.cfg.DATASET.ROOT_LABEL + '/' + path)

        for fitness_type_idx, _ in enumerate(tqdm(label_path, desc="collect every json file in json_list  ")):
            for _, num in enumerate(os.listdir(label_path[fitness_type_idx])):
                path = os.path.join(label_path[fitness_type_idx] + '/' + num)
                path = path + '/' + os.listdir(path)[0]
                json_files = glob.glob(path + '/' + '*.json')
                for _, json_file in enumerate(json_files):
                    if '3d' not in (json_file):
                        json_list.append(json_file)

        return json_list

    def make_db(self):
        # data_dict = {'pts' : [] ,
        #              'img_path' : [],
        #              'joints' : [],
        #              'joints_vis' : []}
        data_dict = {}

        for json_idx in tqdm(range(len(self.json_files)), leave=True, desc="extracting img path and pts from json file  "):
            json_file = self.json_files[json_idx]
            exercise_path = self.json_files[json_idx].replace('label', 'image').split('/')
            if 'barbell_dumbbell' in exercise_path[8]:
                num = exercise_path[8].split('_')[2]
                exercise_path[8] = 'babel'
                exercise_path[8] = '_'.join([exercise_path[8], num])
            exercise_path = os.path.join('/'.join(exercise_path[0:7]), '/'.join(exercise_path[8:-2]))
            with open(json_file, 'r') as f:
                data = json.load(f)
                for frame_idx in range(len(data['frames'])):
                    for idx, view_idx in enumerate(data['frames'][frame_idx].keys()):
                        # PTS
                        img_path = os.path.join(exercise_path, data['frames'][frame_idx][view_idx]['img_key'])
                        # cnt += 1
                        if img_path not in data_dict.keys():
                            # jnt += 1
                            # data_dict[img_path]['pts'] = data['frames'][frame_idx][view_idx]['pts']
                            # data_dict['pts'].append(data['frames'][frame_idx][view_idx]['pts'])
                            # data_dict['img_path'].append(os.path.join(exercise_path, data['frames'][frame_idx][view_idx]['img_key']))

                            # JOINTS
                            joints = []
                            joints_vis = []
                            #
                            data_dict.setdefault(img_path, {'joints': None})
                            data_dict.setdefault(img_path, {'joints_vis': None})
                            # data_dict[img_path].setdefault('joints', None)
                            # data_dict[img_path].setdefault('joints_vis', None)
                            #
                            for joint_idx, joint in enumerate(data['frames'][frame_idx][view_idx]['pts'].keys()):
                                joint_pts = [data['frames'][frame_idx][view_idx]['pts'][joint]['x'],
                                             data['frames'][frame_idx][view_idx]['pts'][joint]['y']]
                                joints.append(joint_pts)

                                joint_visibility = [1, 1]
                                joints_vis.append(joint_visibility)

                            data_dict[img_path]['joints'] = joints
                            data_dict[img_path]['joints_vis'] = joints_vis

        return data_dict
    def get_db(self):
        if self.image_set == 'train':
            with open('/storage/jysuh/Simple_Baseline_For_HPE_Workout/data_pts_del.json', 'r') as f:
                db = json.load(f)

        elif self.image_set == 'validation':
            with open('/storage/jysuh/Simple_Baseline_For_HPE_Workout/data_pts_del.json', 'r') as f:
                db = json.load(f)

        dict_key_list = []

        self.cfg.MODEL.IMAGE_SIZE
        for _, key in enumerate(tqdm(db.keys(), desc="get key of dict from db")):
            dict_key_list.append(key)

        return dict_key_list, db

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        # db_rec = copy.deepcopy(self.db[idx])
        # db_rec.keys() = ['image', 'center', 'scale', 'joints_3d', 'joints_3d_vis', 'filename', 'imgnum']
        image_file = self.key[idx]

        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        w, h = data_numpy.shape[1], data_numpy.shape[0]
        #
        joints = np.array(self.db[image_file]['joints'])
        joints[:,0], joints[:,1] = joints[:,0] / w * self.cfg.MODEL.IMAGE_SIZE[0], joints[:,1] / h * self.cfg.MODEL.IMAGE_SIZE[0]
        #
        data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)
        data_numpy = cv2.resize(data_numpy, (256,256))
        #
        data_numpy = data_numpy.transpose(2,0,1)
        # c = db_rec['center']
        # s = db_rec['scale']
        # score = db_rec['score'] if 'score' in db_rec else 1
        # r = 0 # what is r ? rotate ?

        # if self.is_train:
        #     sf = self.scale_factor
        #     rf = self.rotation_factor
        #     s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
        #     # np.random.randn() -> gaussian distribution, mean = 0, standard deviation = 1
        #     # s * np.clip( 0 ~ 2, 0.75, 1.25 )
        #     # min = 0.75 ~ max = 1.25
        #     r = np.clip(np.random.randn()*rf, -rf*2, rf*2) if random.random() <= 0.6 else 0
        #     # min = -60 ~ max = 60
        #
        #     if self.flip and random.random() <= 0.5:
        #         data_numpy = data_numpy[:, ::-1, :] # reverse for x axis
        #         # (H,W,C)
        #         joints, joints_vis = fliplr_joints(
        #             joints, joints_vis, data_numpy.shape[1], self.flip_pairs)
        #         c[0] = data_numpy.shape[1] - c[0] - 1 # because of reverse for x axis
        #
        # trans = get_affine_transform(c, s, r, self.image_size)
        # input = cv2.warpAffine(
        #     data_numpy,
        #     trans,
        #     (int(self.image_size[0]), int(self.image_size[1])),
        #     flags=cv2.INTER_LINEAR)
        # # extract a person image in a picture. input.shape = [4, 3, 256, 256]

        # input_rgb = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        # if self.transform:
        #     input = self.transform(input)
        #     # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        #
        # for i in range(self.num_joints):
        #     if joints_vis[i, 0] > 0.0:
        #         joints[i, 0:2] = affine_transform(joints[i, 0:2], trans)

        target, target_weight = self.generate_target(joints)
        data_numpy = torch.from_numpy(data_numpy).float()
        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'image': image_file,
            'joints': joints,
        }

        return data_numpy, target, target_weight, meta

    def select_data(self, db):
        db_selected = []
        for rec in db:
            num_vis = 0
            joints_x = 0.0
            joints_y = 0.0
            for joint, joint_vis in zip(
                    rec['joints_3d'], rec['joints_3d_vis']):
                if joint_vis[0] <= 0:
                    continue
                num_vis += 1

                joints_x += joint[0]
                joints_y += joint[1]
            if num_vis == 0:
                continue

            joints_x, joints_y = joints_x / num_vis, joints_y / num_vis

            area = rec['scale'][0] * rec['scale'][1] * (self.pixel_std**2)
            joints_center = np.array([joints_x, joints_y])
            bbox_center = np.array(rec['center'])
            diff_norm2 = np.linalg.norm((joints_center-bbox_center), 2)
            ks = np.exp(-1.0*(diff_norm2**2) / ((0.2)**2*2.0*area))

            metric = (0.2 / 16) * num_vis + 0.45 - 0.2 / 16
            if ks > metric:
                db_selected.append(rec)

        logger.info('=> num db: {}'.format(len(db)))
        logger.info('=> num selected db: {}'.format(len(db_selected)))
        return db_selected

    def generate_target(self, joints):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)

        assert self.target_type == 'gaussian', \
            'Only support gaussian map now!'

        if self.target_type == 'gaussian':
            target = np.zeros((self.num_joints,
                               self.heatmap_size[1],
                               self.heatmap_size[0]),
                              dtype=np.float32)

            tmp_size = self.sigma * 3

            for joint_id in range(self.num_joints):
                feat_stride = np.array(self.image_size) / np.array(self.heatmap_size)
                mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
                mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                # # Generate gaussian
                size = 2 * tmp_size + 1  # heatmap shape is (size, size)
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                # The gaussian is not normalized, we want the center value to equal 1
                g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, target_weight

        # all = np.zeros_like(target[0])
        # for i in range(17):
        #     all += target[i]

        # import matplotlib.pyplot as plt
        # plt.matshow(target[4])

if __name__ == '__main__':
    JointsDataset
    print("fuck you")