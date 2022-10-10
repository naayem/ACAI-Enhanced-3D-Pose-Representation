import os

import torch
from time import time
import matplotlib.pyplot as plt
from ..procedures_common import status_msg
from ...utils.h36m_visual_utils import _check_poses
from pathlib import Path
from ...utils import metrics
import numpy as np
from ...utils.h36m_visual_utils import plot_pose3d, BONES_H36M_3D_17
from ...utils.visu_utils import loss_visu
from ...functional.interpolation import random_interpolation, uniform_interpolation, uniform__smoothness, uniform_smoothness_faster, uniform_smoothness_perm, uniform_smoothness_random
from ...utils.interpolation_visu import uniform_interpolation_batchS, uniform_interpolation_randomS, uniform_smoothness_batchS, uniform_smoothness_randomS

'''
"proc_ae_interp_visu" - visualisation procedure for an autoencoder with the following notations:
    - one model, which is called "ae"
    - dataloaders, called "train" and "valid" provide datasamples of the dict-like type
    - the loss function, called "ae_loss", computes the error between ae output and ground truth, 
            in both training and validation modes

    This procedure considers the simplest case, where network does simple inference img -> gt and
    loss is computed as loss(out, gt). 
'''
def train(trainer):
    return 0

def valid(trainer): 
    loss_visu(trainer.cfg)
    visualization(trainer)
    
    return 0


def visualization(trainer):
    train_loader = trainer.dataload.train
    test_loader = trainer.dataload.valid
    visu_output_dir = Path(os.path.abspath('.'))/trainer.cfg.OUTPUT_DIR / trainer.cfg.EXP_NAME
    latent_dim =trainer.cfg.MODELS.ae.PARAMS.latent_dim

    #random_interpolation(trainer, train_loader, savepath= f'{visu_output_dir}/Train_RandomInterpolation_{trainer.cur_epoch}.png')
    #random_interpolation(trainer, test_loader, savepath= f'{visu_output_dir}/Test_RandomInterpolation_{trainer.cur_epoch}.png')
    
    #uniform_interpolation(trainer, train_loader, N=5, savepath= f'{visu_output_dir}/Train_UniformInterpolation_{trainer.cur_epoch}.png')
    uniform_interpolation(trainer, test_loader, N=5, savepath= f'{visu_output_dir}/Test_UniformInterpolation_{trainer.cur_epoch}.png')
    #uniform__smoothness(trainer, test_loader, N=1000, savepath= f'{visu_output_dir}/Test_smoothness_{trainer.cur_epoch}_new.png')
    uniform_smoothness_faster(trainer, test_loader, N=1000, savepath= f'{visu_output_dir}/Test_smoothness_{trainer.cur_epoch}_faster.png')
    uniform_smoothness_perm(trainer, test_loader, N=1000, savepath= f'{visu_output_dir}/Test_smoothness_{trainer.cur_epoch}_perm.png')
    uniform_interpolation_randomS(trainer, space_dim=latent_dim, savepath= f'{visu_output_dir}/uniform_interpolation_random_{trainer.cur_epoch}.png')
    uniform_interpolation_batchS(trainer, test_loader, space_dim=latent_dim, savepath= f'{visu_output_dir}/uniform_interpolation_batch_{trainer.cur_epoch}.png')
    uniform_smoothness_batchS(trainer, test_loader, space_dim=latent_dim, B=1000, steps=101, seq_type="S", slerp_type="in", savepath= f'{visu_output_dir}/uniform_smoothness_batchS_{trainer.cur_epoch}.png')
    uniform_smoothness_randomS(trainer, test_loader, space_dim=latent_dim, B=1000, steps=101, seq_type="S", slerp_type="in", savepath= f'{visu_output_dir}/uniform_smoothness_randomS_{trainer.cur_epoch}.png')