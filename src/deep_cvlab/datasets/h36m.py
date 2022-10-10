import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch as th

import pickle
import json

from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, Normalize

p = os.path.abspath('.')
sys.path.insert(3, p)

from ..functional.image_augmentations import Augment

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

# Let's preload the KEYPOINTS database to not loading it each time reinstantiating the dataset.
IMAGES_PATH = '/cvlabdata2/cvlab/Human36m/OpenPose/'

dirname=os.path.dirname
project_root = dirname(dirname(dirname(dirname(__file__))))
ANNOT_PATH = f'{project_root}/data/h36m/dict_h36m_desc.pkl'
with open(ANNOT_PATH, 'rb') as f:
    DICT_H36M_DESC = pickle.load(f)

# KEYPOINTS_PATH = os.path.join(os.path.dirname(__file__), '../../data/h36m/dict_h36m_keypoints.pkl')
# with open(KEYPOINTS_PATH, 'rb') as f:
#     DICT_H36M_KEYPOINTS = pickle.load(f)

KEYPOINTS_WORLD_PATH = f'{project_root}/data/h36m/dict_h36m_keypoints_world.pkl'
with open(KEYPOINTS_WORLD_PATH, 'rb') as f:
    DICT_H36M_KEYPOINTS_WORLD = pickle.load(f)

CAMERA_PARAMS_PATH = f'{project_root}/data/h36m/camera-parameters.json'
with open(CAMERA_PARAMS_PATH) as f:
    CAMERA_PARAMS = json.load(f)


logger.info('=> H36M database is loaded.')

OUT_TYPES = ['img', 'pose2d', 'pose3d', 'pose_mask', 'meta', 'pose3d_world', 'intrinsics', 'extrinsics', 'pelvis3d', 'crop_transform']


class H36M(th.utils.data.Dataset):
    def __init__(self, image_transform=TRANSFORM_DEFAULT, do_crop=True,
                out_types=[], # what kind of data to return
                subjects=[], scenarios=[], cameras=[], # particular frames to get
                shuffle=False, num_images=999_999_999, skip_frames=1, seed=999,
                verbose=False, use_m=True, augmentations=None
                    ):

        # self.num_joints = 17 # TODO
        ### left-right correspondencies in 3D pose
        self.flip_pairs3d = [[11, 14], [16, 13], [15, 12], [1, 4], [2, 5], [3, 6]]

        # ### left-right correspondencies in 2D pose
        # self.flip_pairs2d = [[2, 5], [3,6], [4, 7], [8, 11], [9, 12], [10, 13]]

        self.augment = Augment(augmentations, flip_pairs3d=self.flip_pairs3d, flip_pairs2d=self.flip_pairs3d)

        self.use_m = use_m
        self.do_crop = do_crop

        self.skip_frames = skip_frames
        self.seed = seed
        self.shuffle = shuffle
        self.num_images = num_images

        self.image_transform = image_transform

        # given initialization specifications, one needs to specify required frame indices

        # get all subjects if input subjects is empty
        self.subjects = SUBJECTS if subjects == [] else subjects
        # get all scenarios if input scenarios is empty
        self.scenarios = SCENARIOS if scenarios == [] else scenarios
        # get all cameras if input cameras is empty
        self.cameras = CAMERAS if cameras == [] else cameras

        # get all data_types if input data_types is empty
        self.out_types = OUT_TYPES if out_types == [] else out_types

        if self.do_crop:
            if not ('img' in self.out_types and 'pose2d' in self.out_types):
                raise ValueError('To do crop both image and 2d pose must be specified as out types!')

        self.IMAGES_PATH = IMAGES_PATH
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

    def get_image(self, subj, scen, seq, cam, frame):
        scenseq = scen if seq == 'seq0' else scen+'_'+str(seq[-1])
        cam = CAMERAS_CODES[CAMERAS.index(cam)]
        imgpath = os.path.join(self.IMAGES_PATH, subj, 'Images', scenseq+'.'+cam+'_'+f'{frame:012d}'+'.jpg')
        img = plt.imread(imgpath)
        return img

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
        pose_im = np.dot(P, pose3d_homo.T)
        pose_im = pose_im[:2] / pose_im[2]
        pose_im = pose_im.T

        pelvis3d = pose3d_cam[ROOT_JOINT]

        return pose3d_w.copy(), pose3d_cam.copy(), pose_im.copy(), pelvis3d.copy(), intrinsics.copy(), intrinsics_inv.copy(), extrinsics.copy()

    def __getitem__(self, idx):
        subj, scen, seq, cam, frame = self.index_subj_scen_seq_cam[idx]

        ### load image
        img = self.get_image(subj, scen, seq, cam, frame)

        ### load 2d and 3d poses
        pose3d_world, pose3d_cam, pose2d, pelvis3d, intrinsics, intrinsics_inv, extrinsics = self.get_projected_pose(subj, scen, seq, cam, frame)

        pose3d = pose3d_cam - pelvis3d # pose3d in camera coordinates, pelvis-centered
        if self.use_m:
            pose3d /= 1000.

        ### crop image and update pose2d in correspondence
        if self.do_crop:
            img, pose2d, _, crop_transform = _get_cropped_img(img, pose2d)
        
        ### image augmentation: rotate, flip, colorjitter
        img, pose3d, pose2d = self.augment(img, pose3d, pose2d)
        pose_mask = _get_pose_mask(pose2d) # if pose_mask[idx], then joint is inside the image

#        if self.image_transform is not None:
#            img = self.image_transform(img)

        out = {}
        out['img'] = img
        out['pose2d'] = th.from_numpy(pose2d)
        out['pose_mask'] = th.from_numpy(pose_mask)
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
        if 'crop_transform' in self.out_types:
            out['crop_transform'] = th.from_numpy(crop_transform)
        
        return out

    
def _resize(img, _2d_arr, output_shape=(IMAGE_SIZE,IMAGE_SIZE)): # numpy arrays as inputs and outputs
    img = img.astype(np.uint8)
    img_shape = img.shape[:2]
    resize_transform = Compose([
            ToPILImage(),
            Resize((output_shape[0], output_shape[1])),
            ToTensor(),
        ])
    img = resize_transform(img)
    img = img.permute(1,2,0).numpy()
    
    pose_scale = np.array([output_shape[1]/img_shape[1], output_shape[0]/img_shape[0]]).reshape(1,2)
    _2d_arr *= pose_scale
    
    return img, _2d_arr, pose_scale
    

def _get_pose_mask(pose2d, img_size=IMAGE_SIZE):
    pose_mask = np.concatenate([
        np.any(pose2d[:,0].reshape(-1,1) >= 0, axis=1, keepdims=True),
        np.any(pose2d[:,0].reshape(-1,1) < img_size, axis=1, keepdims=True),
        np.any(pose2d[:,1].reshape(-1,1) >= 0, axis=1, keepdims=True),
        np.any(pose2d[:,1].reshape(-1,1) < img_size, axis=1, keepdims=True),
    ], axis=1)
    pose_mask = np.all(pose_mask, axis=1) # shape: (B, )
    return pose_mask


def _get_cropped_img(img, pose2d, pose2d_mask=None): # resize in IMAGE_SIZE x IMAGE_SIZE automatically

    img, pose2d = img.copy(), pose2d.copy()
    if pose2d_mask is None:
        pose2d_mask = np.ones((len(pose2d)), dtype='bool')

    img_height, img_width = img.shape[:2]
    _2d_min_w, _2d_max_w = np.min(pose2d[pose2d_mask][:,0]), np.max(pose2d[pose2d_mask][:,0])
    _2d_min_h, _2d_max_h = np.min(pose2d[pose2d_mask][:,1]), np.max(pose2d[pose2d_mask][:,1])

    scale = 2.
    center_w, center_h = (_2d_max_w + _2d_min_w)//2, (_2d_max_h + _2d_min_h)//2

    gap = max(center_w-_2d_min_w, _2d_max_w-center_w, center_h-_2d_min_h, _2d_max_h-center_h)
    gap = int(gap * scale)
    new_img = np.zeros((2 * gap, 2 * gap, 3))

    min_w = int(np.maximum(center_w - gap, 0))
    max_w = int(np.minimum(center_w + gap, img_width))
    min_h = int(np.maximum(center_h - gap, 0))
    max_h = int(np.minimum(center_h + gap, img_height))

    ### always match top left corner of the bounding box and the crop image
    new_img[:max_h-min_h,:max_w-min_w] = img[min_h:max_h, min_w:max_w]

    pose_shift = - np.array([min_w, min_h]).reshape(1, 2)

    pose2d += pose_shift
    pose2d[~pose2d_mask] = -1.
    new_img, pose2d[pose2d_mask], pose_scale = _resize(new_img, pose2d[pose2d_mask])
    pose_transform = np.stack((pose_shift, pose_scale))

    return new_img, pose2d, pose2d_mask, pose_transform

def decrop_pose(pose2d, crop_transform):
    '''
    size of crop_transform : 2 x 1 x 2
    size of pose2d : J x 2
    '''
    shift, scale = crop_transform[0], crop_transform[1] # 1 x 2 each
    pose2d = pose2d / scale - shift
    return pose2d

def crop_pose(pose2d, crop_transform):
    '''
    size of crop_transform : 2 x 1 x 2
    size of pose2d : J x 2
    '''
    shift, scale = crop_transform[0], crop_transform[1] # 1 x 2 each
    pose2d = (pose2d + shift) * scale
    return pose2d

# ### TODO
# def crop_pose_batch(pose2d, crop_transform):
#     '''
#     size of crop_transform : B x 2 x 1 x 2
#     size of pose2d : B x J x 2
#     '''
#     shift, scale = crop_transform[:,0], crop_transform[:,1] # B x 1 x 2 each
#     pose2d = (pose2d + shift) * scale
#     return pose2d


def main():

    augmentations = {
        'rotate' : {
            'max_angle' : 30
        },
        'flip' : True,
        'colorjitter' : {
            'brightness' : 0.8,
            'saturation' : 0.8,
            'contrast' : 0.5,
            'hue' : 0.05
        }
    }

    seed = 1234
    kwargs = dict(do_crop=True, augmentations=augmentations, shuffle=True, seed=seed,
    image_transform=None, skip_frames=20)
    th.manual_seed(seed)
    ds = H36M(**kwargs)
    
    if True:
        _check_poses(ds)


def _check_poses(ds):

    annotate = True

    from h36m_visual_utils import plot_pose3d
    rows, cols = 10, 10
    fig_3d, ax_3d = plt.subplots(rows, cols, figsize=(4*cols,4*rows))
    for i in range(rows):
        for j in range(cols):
            
            print(i,j, end=' ')

            sample = ds[i*cols + j]

            ### pose3d
            pose3d = sample['pose3d']
            plot_pose3d(ax_3d[i,j], pose3d, skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
            
            print('Done!')

    name = ['3d']
    fig_3d.subplots_adjust(wspace=0.01, hspace=0.01)
    fig_3d.patch.set_facecolor('white')
    fig_3d.tight_layout()
    fig_3d.savefig(f'./SAMPLE_h36m_{name}.png')


if __name__ == '__main__':
    main()