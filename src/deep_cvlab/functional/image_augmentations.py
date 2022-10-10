import torchvision.transforms.functional as TF
import torchvision.transforms as TT
import numpy as np
import random

class Augment(object):
    '''
    Augments:
        - image 
        - correspoinding pose in 3D coordinates (pelvis-centered, camera coordinates)
        - corresponding pose in 2D coordinates
    '''
    def __init__(self, augmentations=None, flip_pairs2d=None, flip_pairs3d=None):

        self.augmentations = augmentations
        self.current_augmentations = ['rotate', 'flip', 'colorjitter'] 
        
        # more possible options: noise, scale, translation TODO

        if self.augmentations is not None:
            for aug_type in self.augmentations:
                assert aug_type in self.current_augmentations, f'{aug_type} is not implemented!'

            if 'colorjitter' in self.augmentations:
                self.colorjitter = TT.ColorJitter(**self.augmentations['colorjitter'])

            if 'flip' in self.augmentations and self.augmentations['flip']:
                assert flip_pairs2d is not None or flip_pairs3d is not None, \
                    'Necessary to specify left-right pairs of indices to flip!'
                self.flip_pairs2d = flip_pairs2d
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

    def __call__(self, img, pose3d, pose2d):
        '''
        img: numpy array, HxWx3
        pose3d: numpy array, Jx3 
        pose2d: numpy array, Jx2 
        '''
        if self.augmentations is None:
            return img, pose3d, pose2d

        self._set_cur_aug_params()

        H, W = img.shape[:2]
        img = TT.ToPILImage()(TT.ToTensor()(img.copy())) # TF works only with PIL Image
        
        if 'rotate' in self.augmentations:
            ### rotate image
            img = TF.rotate(img, self.angle)

            ### rotate pose3d
            pose3d = pose3d.copy()
            pose_xy = pose3d[:,:2].T # (2 x J)
            pose_xy = self.R.dot(pose_xy).T # (J x 2)
            pose3d[:,:2] = pose_xy

            ### rotate pose2d
            pose2d = pose2d.copy()
            img_center = np.array([[W/2, H/2]])
            pose2d -= img_center
            pose_xy = pose2d[:,:2].T # (2 x J)
            pose_xy = self.R.dot(pose_xy).T # (J x 2)
            pose2d[:,:2] = pose_xy
            pose2d += img_center

        if 'flip' in self.augmentations:
            if (self.augmentations['flip'] and random.random() > 0.5):
                img = TF.hflip(img)
                pose3d = flip_joints(pose3d, dim=3, flip_pairs=self.flip_pairs3d)
                pose2d = flip_joints(pose2d, dim=2, flip_pairs=self.flip_pairs2d, img_width=W)

        if 'colorjitter' in self.augmentations:
                img = self.colorjitter(img)

        img = np.array(img) # cast back to numpy array
        return img, pose3d, pose2d

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