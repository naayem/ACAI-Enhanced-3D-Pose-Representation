# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import os
import json
import copy
import random

import numpy as np
import matplotlib.pyplot as plt

# import cv2
import torch as th 
from torch.utils.data import Dataset

from ..utils.transforms import get_affine_transform, affine_transform, fliplr_joints

from .datasets_common import TRANSFORMS

ROOT_IMG = '/cvlabsrc1/cvlab/dataset_davydov/mpii'

class MPII(Dataset):
    def __init__(self, mode, transform=TRANSFORMS['W/ NORM'], get_heatmaps=False, scale_factor=0.25, rotation_factor=30, flip=True):
        self.root = ROOT_IMG
        self.num_joints = 16
        self.flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
        self.mode = mode

        # create train/val split
        file_name = os.path.join(self.root, 'annot', self.mode +'.json')
        with open(file_name, 'r') as anno_file:
            self.db = json.load(anno_file)

        self.db = self._get_db()

        print('MPII, {} mode'.format(mode) + '=> load {} samples'.format(len(self.db)))

        self.scale_factor = scale_factor
        self.rotation_factor = rotation_factor
        self.flip = flip

        self.image_size = np.array([256,256])
        self.heatmap_size = np.array([64,64]) 
        self.sigma = 2. 

        self.transform = transform
        self.get_heatmaps = get_heatmaps

    def __len__(self):
        return len(self.db)

    def _get_db(self):
        
        gt_db = []
        for a in self.db:
            image_name = os.path.join(self.root, 'images', a['image'])
            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            gt_db.append({
                    'image_path': image_name,
                    'center': c,
                    'scale': s,
                    'joints': a['joints'],
                    'joints_mask': a['joints_vis'],
                    })

        return gt_db


    def __getitem__(self, idx):
        sample = self.db[idx]

        image_path = sample['image_path']
        img = plt.imread(image_path)

        joints = np.array(sample['joints'])
        joints_mask = np.array(sample['joints_mask']).astype(np.bool8)

        c = np.array(sample['center'])
        s = np.array(sample['scale'])
        r = 0

        if self.mode == 'train':
            sf = self.scale_factor
            rf = self.rotation_factor
            s = s * np.clip(np.random.randn()*sf + 1, 1 - sf, 1 + sf)
            r = np.clip(np.random.randn()*rf, -rf*2, rf*2) \
                if random.random() <= 0.6 else 0

            if self.flip and random.random() <= 0.5:
                img = img[:, ::-1, :]
                joints, joints_mask = fliplr_joints(joints, joints_mask, img.shape[1], self.flip_pairs)
                c[0] = img.shape[1] - c[0] - 1

        trans = get_affine_transform(c, s, r, self.image_size)

        # img = cv2.warpAffine(img, trans, 
        #     (int(self.image_size[0]), int(self.image_size[1])), flags=cv2.INTER_LINEAR)

        for i in range(self.num_joints):
            if joints_mask[i] > 0.0:
                joints[i] = affine_transform(joints[i], trans)

        if self.transform is not None:
            img = self.transform(img)

        meta = {
            'image_path': image_path,
            'center': c, 'scale': s, 'rotation': r,}

        joints = th.from_numpy(joints)
        output = {'img':img, 'meta':meta, 'joints':joints}

        if self.get_heatmaps:
            heatmaps, joints_mask = self.generate_heatmaps(joints, joints_mask) 
            heatmaps = th.from_numpy(heatmaps)
            output['heatmaps'] = heatmaps

        joints_mask = th.from_numpy(joints_mask)
        output['joints_mask'] = joints_mask

        return output


    def generate_heatmaps(self, joints, joints_mask):
        '''
        :param joints:  [num_joints, 3]
        :param joints_mask: [num_joints]
        :return: heatmaps, joints_mask(1: visible, 0: invisible)
        '''

        heatmaps = np.zeros((self.num_joints, self.heatmap_size[1], self.heatmap_size[0]), 
                            dtype=np.float32)
        mask = joints_mask.copy()
        tmp_size = self.sigma * 3

        for joint_id, (joint, joint_mask) in enumerate(zip(joints, joints_mask)):

            if not joint_mask: # joint is invisible
                continue

            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joint[0] / feat_stride[0] + 0.5)
            mu_y = int(joint[1] / feat_stride[1] + 0.5)
            # Check that any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)] # up-left
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)] # bottom-right

            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                # If not, just return the image as is
                mask[joint_id] = False
                continue

            # # Generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * self.sigma ** 2))

            # Usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            # Image range
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            heatmaps[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return heatmaps, mask


    # def evaluate(self, preds, output_dir, *args, **kwargs):
    #     # convert 0-based index to 1-based index
    #     preds = preds[:, :, 0:2] + 1.0

    #     if output_dir:
    #         pred_file = os.path.join(output_dir, 'pred.mat')
    #         savemat(pred_file, mdict={'preds': preds})

    #     SC_BIAS = 0.6
    #     threshold = 0.5

    #     gt_file = os.path.join('data', 'mpii', 'annot',
    #                            'gt_valid_samejts.mat')
    #     gt_dict = loadmat(gt_file)
    #     dataset_joints = gt_dict['dataset_joints']
    #     jnt_missing = gt_dict['jnt_missing']
    #     pos_gt_src = gt_dict['pos_gt_src']
    #     headboxes_src = gt_dict['headboxes_src']

    #     pos_pred_src = np.transpose(preds, [1, 2, 0])

    #     # these joint indices below correspond to "same_jts" strategy
    #     head = 0 # np.where(dataset_joints == 'head')[1][0]

    #     lsho = 5 # np.where(dataset_joints == 'lsho')[1][0]
    #     lelb = 6 # np.where(dataset_joints == 'lelb')[1][0]
    #     lwri = 7 # np.where(dataset_joints == 'lwri')[1][0]
    #     lhip = 12 # np.where(dataset_joints == 'lhip')[1][0]
    #     lkne = 13 # np.where(dataset_joints == 'lkne')[1][0]
    #     lank = 14 # np.where(dataset_joints == 'lank')[1][0]

    #     rsho = 3 # np.where(dataset_joints == 'rsho')[1][0]
    #     relb = 2 # np.where(dataset_joints == 'relb')[1][0]
    #     rwri = 1 # np.where(dataset_joints == 'rwri')[1][0]
    #     rkne = 9 # np.where(dataset_joints == 'rkne')[1][0]
    #     rank = 8 # np.where(dataset_joints == 'rank')[1][0]
    #     rhip = 10 # np.where(dataset_joints == 'rhip')[1][0]

    #     jnt_visible = 1 - jnt_missing
    #     uv_error = pos_pred_src - pos_gt_src
    #     uv_err = np.linalg.norm(uv_error, axis=1)
    #     headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
    #     headsizes = np.linalg.norm(headsizes, axis=0)
    #     headsizes *= SC_BIAS
    #     scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
    #     scaled_uv_err = np.divide(uv_err, scale)
    #     scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
    #     jnt_count = np.sum(jnt_visible, axis=1)
    #     less_than_threshold = np.multiply((scaled_uv_err <= threshold),
    #                                       jnt_visible)
    #     PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

    #     rng = np.arange(0, 0.5+0.01, 0.01)
    #     pckAll = np.zeros((len(rng), 16))

    #     for r in range(len(rng)):
    #         threshold = rng[r]
    #         less_than_threshold = np.multiply(scaled_uv_err <= threshold,
    #                                           jnt_visible)
    #         pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
    #                                  jnt_count)

    #     PCKh = np.ma.array(PCKh, mask=False)
    #     PCKh.mask[6:8] = True

    #     jnt_count = np.ma.array(jnt_count, mask=False)
    #     jnt_count.mask[6:8] = True
    #     jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

    #     name_value = [
    #         ('Head', PCKh[head]),
    #         ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
    #         ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
    #         ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
    #         ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
    #         ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
    #         ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
    #         ('Mean', np.sum(PCKh * jnt_ratio)),
    #         ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
    #     ]
    #     name_value = OrderedDict(name_value)

    #     return name_value, name_value['Mean']
