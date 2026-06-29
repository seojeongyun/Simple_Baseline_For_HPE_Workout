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
    def __init__(self, cfg, root, task, transform=None):
        self.cfg = cfg
        self.num_joints = 24
        self.pixel_std = 200
        self.flip_pairs = []
        self.parent_ids = []

        self.task = task
        self.root = root                 # whether the dataset is train or validation

        self.output_path = cfg.OUTPUT_DIR
        self.data_format = cfg.DATASET.DATA_FORMAT

        self.scale = cfg.DATASET.SCALE
        self.rotate = cfg.DATASET.ROTATE

        self.scale_min = cfg.DATASET.SCALE_MIN
        self.scale_max = cfg.DATASET.SCALE_MAX
        self.rotation_factor = cfg.DATASET.ROT_FACTOR_MAX

        self.flip = cfg.DATASET.FLIP
        self.flip_prob = cfg.DATASET.FLIP_PROB
        self.flip_pairs = cfg.DATASET.FLIP_JOINTS_PAIRS

        self.image_size = cfg.MODEL.IMAGE_SIZE
        self.target_type = cfg.MODEL.EXTRA.TARGET_TYPE
        self.heatmap_size = cfg.MODEL.EXTRA.HEATMAP_SIZE
        #
        # self.sigma = cfg.MODEL.EXTRA.SIGMA
        self.transform = transform
        self.debugging = True

        if self.task == 'get_sequences_for_tf':
            self.img_paths, self.workout_conditions, self.video_idx_list, self.view_idx_list, self.db, self.max_frame = self.get_db()
        else:
            self.img_paths, self.db = self.get_db()
        # DEBUG
        self.out_of_scene = 0

    def get_db(self):
        if self.task == 'train':
            with open(self.cfg.DATASET.TRAIN_SET_PATH, 'r', encoding="utf-8") as f:
                db = json.load(f)
                # the number of img path in db: 2696768
            img_path_list = []

            for _, key in enumerate(tqdm(db.keys(), desc="get train data from train_img_paths.json", leave=True)):
                img_path_list.append(key)

            return img_path_list, db

        elif self.task == 'validation':
            with open(self.cfg.DATASET.VALID_SET_PATH, 'r', encoding="utf-8") as f:
                db = json.load(f)

            img_path_list = []

            for _, key in enumerate(tqdm(db.keys(), desc="get valid data from valid_img_paths.json", leave=True)):
                img_path_list.append(key)

            return img_path_list, db

        elif self.task == 'get_sequences_for_tf':
            with open(self.cfg.DATASET.GET_SEQUENCES_SET_PATH, 'r', encoding="utf-8") as f:
                db = json.load(f)
                # exercise_dict['오버 헤드 프레스']['4']['view1']['img_path']
                # exercise_dict['what_exer']['seq_num']['view_num / type_info']['img_path']
            max_frame = db['max_frame']
            img_path_list = []
            workout_condition_list = []
            video_idx_list = []
            view_idx_list = []
            #
            for what_exer in tqdm(db.keys(), desc="get sequence data", leave=True):
                if what_exer != 'max_frame':
                    for video_idx in db[what_exer].keys():
                        for view_idx in db[what_exer][video_idx].keys():
                            if 'view' in view_idx:
                                for img_path in db[what_exer][video_idx][view_idx]['img_path']:
                                    img_path_list.append(img_path)
                                    workout_condition_list.append(db[what_exer][video_idx]['type_info'])
                                    video_idx_list.append(video_idx)
                                    view_idx_list.append(view_idx)

            return img_path_list, workout_condition_list, video_idx_list, view_idx_list, db, max_frame

            # delete exer type lower than threshold
            # threshold = 500
            # for what_exer, _ in db.items():
            #     if(len(db[what_exer]['view1'].keys()) < threshold):
            #         del db[what_exer]
            #     print("{} : {}".format(what_exer, len(db[what_exer]['view1'].keys()))

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image_file = self.img_paths[idx]

        if self.task == 'get_sequences_for_tf':
            condition = self.workout_conditions[idx]
            video_idx = self.video_idx_list[idx]
            view_idx = self.view_idx_list[idx]
            max_frame = self.max_frame

        if self.data_format == 'zip':  # in this case, data_format is jpg
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        if data_numpy is None:
            logger.error('=> fail to read {}'.format(image_file))
            raise ValueError('Fail to read {}'.format(image_file))

        H, W = data_numpy.shape[:2]  # H=1080, W=1920
        cx, cy = W / 2.0, H / 2.0  # (960, 540)
        CROP = 1080
        s_min, s_max, r_max = self.scale_min, self.scale_max, self.rotation_factor

        scale = np.random.uniform(s_min, s_max) if self.scale else 1.0
        angle = np.random.uniform(-r_max, r_max) if self.rotate else 0.0

        if (self.scale or self.rotate) and self.task != 'validation':
            M = cv2.getRotationMatrix2D((cx, cy), angle, scale)
            # Apply Scaling or Rotation
            transformed_img = cv2.warpAffine(
                data_numpy, M, (W, H),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0
            )

        # Calc UpLeft and BottomRight point
        x0, x1 = int(round(cx - CROP / 2)), int(round(cx + CROP / 2))
        y0, y1 = int(round(cy - CROP / 2)), int(round(cy + CROP / 2))
        cropped = transformed_img[y0:y1, x0:x1] if (self.scale or self.rotate) and self.task != 'validation' else data_numpy[y0:y1, x0:x1]

        # map original coord to cropped scale coord
        if self.task != 'get_sequences_for_tf':
            joints = np.array(self.db[image_file]['joints'], dtype=np.float32)
            joints_xy = joints[:, :2].copy()

            if (self.scale or self.rotate) and self.task != 'validation':
                ones = np.ones((joints_xy.shape[0], 1), dtype=np.float32)
                hom = np.hstack([joints_xy, ones])  # (N,3)
                transformed_xy = (M @ hom.T).T  # (N,2)
            else:
                transformed_xy = joints_xy.copy()

            transformed_xy -= np.array([x0, y0], dtype=np.float32)

            # Flip
            do_flip = np.random.rand() < self.flip_prob if self.flip else False
            if do_flip and self.task != 'validation':
                cropped = cv2.flip(cropped, 1)
                transformed_xy[:, 0] = (CROP - 1) - transformed_xy[:, 0]

                for a, b in self.flip_pairs:
                    transformed_xy[[a, b]] = transformed_xy[[b, a]]

            valid = (
                    (transformed_xy[:, 0] >= 0) & (transformed_xy[:, 0] < CROP) &
                    (transformed_xy[:, 1] >= 0) & (transformed_xy[:, 1] < CROP)
            )

        data_numpy = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)

        # DEBUG
        # for (x, y) in transformed_xy:
        #     cv2.circle(data_numpy, (int(x), int(y)), radius=5, color=(0, 255, 0), thickness=-1)
        #
        # cv2.imshow("Keypoints", data_numpy)
        # key = cv2.waitKey(0)
        # if key == 27:  # Esc
        #     cv2.destroyAllWindows()
        #

        out_w, out_h = self.cfg.MODEL.IMAGE_SIZE
        if (out_w, out_h) != (CROP, CROP):
            data_numpy = cv2.resize(data_numpy, (out_w, out_h))
            scale_x = out_w / float(CROP)
            scale_y = out_h / float(CROP)

        else:
            scale_x = scale_y = 1.0

        # Normalize
        if self.transform:
            data_numpy = self.transform(data_numpy)
        else:
            data_numpy = data_numpy.transpose(2, 0, 1)
            data_numpy = torch.from_numpy(data_numpy).float()

        if self.task != 'get_sequences_for_tf':
            transformed_xy_scaled = transformed_xy * np.array([scale_x, scale_y], dtype=np.float32)

            joints_out = joints.copy()
            joints_out[:, :2] = transformed_xy_scaled

            target, target_weight = self.generate_target(joints_out)
            if target_weight.shape[0] == valid.shape[0]:
                target_weight *= valid[:, None].astype(np.float32)

            target = torch.from_numpy(target)
            target_weight = torch.from_numpy(target_weight)

            meta = {
                'image': image_file,
                'joints': joints_out,
                'scale_used': float(scale),
                'angle_used': float(angle),
                'crop_box': (x0, y0, x1, y1)
            }

        if self.task == 'get_sequences_for_tf':
            return data_numpy, condition, image_file, video_idx, view_idx, max_frame
        else:
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
        self.sigma = np.random.uniform(1.0, 2.0)
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
                # g_x and g_y is translated to tuple.
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
    print("")