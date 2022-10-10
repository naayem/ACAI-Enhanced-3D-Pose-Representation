import torchvision.transforms.functional as TF
import torchvision.transforms as TT
import numpy as np
import random

class Augment_3d(object):
    '''
    Augments:
        - correspoinding pose in 3D coordinates (pelvis-centered, camera coordinates)
    '''
    def __init__(self, augmentations=None, flip_pairs3d=None):

        self.augmentations = augmentations
        self.current_augmentations = ['rotate', 'flip', 'colorjitter'] 
        
        # more possible options: noise, scale, translation TODO

        if self.augmentations is not None:
            for aug_type in self.augmentations:
                assert aug_type in self.current_augmentations, f'{aug_type} is not implemented!'

            if 'flip' in self.augmentations and self.augmentations['flip']:
                assert flip_pairs3d is not None, \
                    'Necessary to specify left-right pairs of indices to flip!'
                self.flip_pairs3d = flip_pairs3d

    def _set_cur_aug_params(self):
        '''
        Initialize parameters for augmentations of a particular sample.
        '''
        if 'rotate' in self.augmentations:
            max_angle = self.augmentations['rotate']['max_angle']
            angle = np.clip( ( random.random() * 2 - 1 ) * max_angle, -max_angle, max_angle)
            self.angle = angle.astype(np.float64)

            angle = - np.radians(angle) # "-" allows to perform inverse transform without inverting a matrix
            c, s = np.cos(angle), np.sin(angle)
            self.R = np.array(((c, -s), (s, c)))

    def __call__(self, pose3d):
        '''
        pose3d: numpy array, Jx3 
        '''
        if self.augmentations is None:
            return pose3d

        self._set_cur_aug_params()
        
        if 'rotate' in self.augmentations:
            ### rotate pose3d
            pose3d = pose3d.copy()
            pose_xy = pose3d[:,:2].T # (2 x J)
            pose_xy = self.R.dot(pose_xy).T # (J x 2)
            pose3d[:,:2] = pose_xy

        if 'flip' in self.augmentations:
            if (self.augmentations['flip'] and random.random() > 0.5):
                pose3d = flip_joints(pose3d, dim=3, flip_pairs=self.flip_pairs3d)

        return pose3d

    def __repr__(self):
        return self.augmentations.__repr__()


def flip_joints(x, dim, flip_pairs=None, img_width=None):
    y = x.copy()

    if dim == 3: # 3D keypoints
        y[:,0] = -y[:,0] # change the coordinates NOTE valid only for pelvis_centered poses!

        z = y.copy()
        for pair in flip_pairs: # flip indices (to keep left-right consistency with the image)
            z[ pair[0] ] = y[ pair[1] ]
            z[ pair[1] ] = y[ pair[0] ]
        return z
    
    else: # 2D keypoints
        y[:,0] = (img_width - 1) - y[:,0]

        z = y.copy()
        for pair in flip_pairs:
            z[ pair[0] ] = y[ pair[1] ]
            z[ pair[1] ] = y[ pair[0] ]

        return z