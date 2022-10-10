import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch as th

import pickle
import json

from torchvision.transforms import Compose, ToTensor, Normalize

p = os.path.abspath('.')
sys.path.insert(3, p)

from ..functional.image_augmentations_3d import Augment_3d

import logging
logger = logging.getLogger(__name__)


NORMALIZE = Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
TRANSFORM_DEFAULT = Compose([ToTensor(),NORMALIZE])
IMAGE_SIZE = 256

SUBJECTS = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
SCENARIOS = ['Directions','Discussion','Eating','Greeting',
             'Phoning','Posing','Purchases','Sitting','SittingDown',
             'Smoking','Waiting','WalkTogether','Walking','WalkingDog',]
CAMERAS_CODES = ['54138969', '55011271', '58860488', '60457274']
CAMERAS = [f'cam{cam}' for cam in range(4)]

BONES_H36M_3D_17 = [
        [[0,4],[4,5],[5,6]], ### left leg
        [[12,13],[11,12],[8,11]], ### left hand
        [[0,1],[1,2],[2,3]], ### right leg
        [[8,14],[14,15],[15,16]], ### right hand
        [[0,7],[7,8]], ### spine
        [[8,9],[9,10],[8,10]] ### head
    ] 

ROOT_JOINT = 0

dirname=os.path.dirname
project_root = dirname(dirname(dirname(dirname(__file__))))
ANNOT_PATH = f'/data/facades/h36m/dict_h36m_desc.pkl'

with open(ANNOT_PATH, 'rb') as f:
    DICT_H36M_DESC = pickle.load(f)

# KEYPOINTS_PATH = os.path.join(os.path.dirname(__file__), '../../data/h36m/dict_h36m_keypoints.pkl')
# with open(KEYPOINTS_PATH, 'rb') as f:
#     DICT_H36M_KEYPOINTS = pickle.load(f)

KEYPOINTS_WORLD_PATH = f'/data/facades/h36m/dict_h36m_keypoints_world.pkl'
with open(KEYPOINTS_WORLD_PATH, 'rb') as f:
    DICT_H36M_KEYPOINTS_WORLD = pickle.load(f)

CAMERA_PARAMS_PATH = f'/data/facades/h36m/camera-parameters.json'
with open(CAMERA_PARAMS_PATH) as f:
    CAMERA_PARAMS = json.load(f)


logger.info('=> H36M database is loaded.')

OUT_TYPES = ['img', 'pose2d', 'pose3d', 'pose_mask', 'meta', 'pose3d_world', 'intrinsics', 'extrinsics', 'pelvis3d', 'crop_transform']


class H36M(th.utils.data.Dataset):
    def __init__(self, out_types=[], # what kind of data to return
                subjects=[], scenarios=[], cameras=[], # particular frames to get
                shuffle=False, num_images=999_999_999, skip_frames=1, seed=999,
                verbose=False, use_m=True, augmentations=None
                    ):

        # self.num_joints = 17 # TODO
        ### left-right correspondencies in 3D pose
        self.flip_pairs3d = [[11, 14], [16, 13], [15, 12], [1, 4], [2, 5], [3, 6]]

        self.augment = Augment_3d(augmentations, flip_pairs3d=self.flip_pairs3d)

        self.use_m = use_m

        self.skip_frames = skip_frames
        self.seed = seed
        self.shuffle = shuffle
        self.num_images = num_images

        # given initialization specifications, one needs to specify required frame indices

        # get all subjects if input subjects is empty
        self.subjects = SUBJECTS if subjects == [] else subjects
        # get all scenarios if input scenarios is empty
        self.scenarios = SCENARIOS if scenarios == [] else scenarios
        # get all cameras if input cameras is empty
        self.cameras = CAMERAS if cameras == [] else cameras

        # get all data_types if input data_types is empty
        self.out_types = OUT_TYPES if out_types == [] else out_types

        self.ANNOT_PATH = ANNOT_PATH
        self.KEYPOINTS_WORLD_PATH = KEYPOINTS_WORLD_PATH

        self.dict_h36m_desc = DICT_H36M_DESC
        # self.dict_h36m_keypoints = DICT_H36M_KEYPOINTS
        self.dict_h36m_keypoints_world = DICT_H36M_KEYPOINTS_WORLD

        self.camera_params = CAMERA_PARAMS

        index_subj_scen_seq_cam = []
        for subj in self.subjects:
            for scen in self.scenarios:
                for seq in self.dict_h36m_desc[subj][scen]:
                    for cam in self.cameras:
                        for frame in self.dict_h36m_desc[subj][scen][seq][cam]:
                            index_subj_scen_seq_cam.append([subj, scen, seq, cam, int(frame)])

        self.index_subj_scen_seq_cam = index_subj_scen_seq_cam
        self.init_db()

        if verbose:
            logger.info(f'H36M dset loaded => \n {self.__repr__()}')

    def init_db(self):

        # skip frames
        self.index_subj_scen_seq_cam = self.index_subj_scen_seq_cam[::self.skip_frames]

        # shuffle frames
        if self.shuffle:
            np.random.seed(self.seed)
            np.random.shuffle(self.index_subj_scen_seq_cam)

        # take only "num_images" frames
        self.index_subj_scen_seq_cam = self.index_subj_scen_seq_cam[:self.num_images]

    def __len__(self):
        return len(self.index_subj_scen_seq_cam)

    def __repr__(self):
        s = f'Total number of frames: {self.__len__()}\n'
        s += f'do crop: {self.do_crop}, skip_frames: {self.skip_frames}\n'
        s += f'Subjects: {self.subjects}\n'
        s += f'Scenarios: {self.scenarios}\n'
        s += f'Cameras: {self.cameras}\n'
        return s

    def get_projected_pose(self, subj, scen, seq, cam, frame):
        """
        This function returns the following information for a certain pose:
        3d pose:
        """
        pose3d_w = self.dict_h36m_keypoints_world[subj][scen][seq][frame].copy()

        cam_code = CAMERAS_CODES[int(cam[-1])]
        calibration_matrix = self.camera_params['intrinsics'][cam_code]['calibration_matrix']
        intrinsics = np.array(calibration_matrix)
        intrinsics_inv = np.linalg.inv(intrinsics)

        R = self.camera_params['extrinsics'][subj][cam_code]['R']
        t = self.camera_params['extrinsics'][subj][cam_code]['t']
        extrinsics = np.hstack([R, t])

        P = calibration_matrix @ extrinsics

        pose3d_homo = np.hstack([pose3d_w, np.ones((len(pose3d_w),1))])
        pose3d_cam = np.dot(extrinsics, pose3d_homo.T).T

        pelvis3d = pose3d_cam[ROOT_JOINT]

        return pose3d_w.copy(), pose3d_cam.copy(), pelvis3d.copy(), intrinsics.copy(), intrinsics_inv.copy(), extrinsics.copy()

    def __getitem__(self, idx):
        subj, scen, seq, cam, frame = self.index_subj_scen_seq_cam[idx]

        ### load 2d and 3d poses
        pose3d_world, pose3d_cam, pelvis3d, intrinsics, intrinsics_inv, extrinsics = self.get_projected_pose(subj, scen, seq, cam, frame)

        pose3d = pose3d_cam - pelvis3d # pose3d in camera coordinates, pelvis-centered
        if self.use_m:
            pose3d /= 1000.
        
        ### image augmentation: rotate, flip, colorjitter
        pose3d = self.augment(pose3d)

        out = {}
        out['pose3d'] = th.from_numpy(pose3d)
        meta = {'subj':subj, 'scen':scen, 'seq':seq, 'cam':cam, 'frame':frame}
        out['meta'] = meta

        if 'pose3d_world' in self.out_types:
            out['pose3d_world'] = th.from_numpy(pose3d_world)
        if 'intrinsics' in self.out_types:
            out['intrinsics'] = th.from_numpy(intrinsics)
        if 'intrinsics_inv' in self.out_types:
            out['intrinsics_inv'] = th.from_numpy(intrinsics_inv)
        if 'extrinsics' in self.out_types:
            out['extrinsics'] = th.from_numpy(extrinsics)
        if 'pelvis3d' in self.out_types:
            out['pelvis3d'] = th.from_numpy(pelvis3d)
        
        return out