import matplotlib.pyplot as plt
import torch as th

from deep_cvlab.datasets import h36m

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
    ds = h36m.H36M(**kwargs)
    
    if True:
        h36m._check_poses(ds)

if __name__ == '__main__':
    main()








