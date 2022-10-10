import os
from time import time

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from ..utils.h36m_visual_utils import plot_pose3d, BONES_H36M_3D_17
from mpl_toolkits.mplot3d import Axes3D
from ..losses.joints_loss import JointsLoss, MPJPE
from ..functional.interpolation_s import create_sequence
from ..functional.sampling import generate_random_input as _generate_random_input


def uniform_interpolation_randomS(trainer, space_dim, B=5, steps=11, seq_type="S", slerp_type="in", savepath=None):
    cdn = lambda x: x.cpu().detach().numpy()
    # Get the device on which AE is located
    device = list(trainer.models.ae.parameters())[0].device

    seed = 0
    torch.manual_seed(seed)

    x = _generate_random_input(B, latent_space_dim=space_dim, latent_space_type='S')
    y = _generate_random_input(B, latent_space_dim=space_dim, latent_space_type='S')

    z, _ = create_sequence(x, y, steps, seq_type, slerp_type)
    '''For each interpolation N pair of ground truths (i and i-1 of batch)
    Get M=11 'equidistant' latent interpolations and their reconstruction between each pair of ground truth images
    Alpha varies from 0 to 1 included
    '''

    zs = z.size()
    z_expand = torch.reshape(z, (-1,space_dim))
    reconstruction_interp = trainer.models.ae(z_expand, 'decoder')['reconstructed']

    reconstruction_interp = cdn(torch.reshape(reconstruction_interp, (zs[0], zs[1], 17,3)))

    alphas = cdn(torch.arange(0, 1.1, step=(1/(steps-1)), dtype=torch.float32).reshape(-1, 1))

    annotate = False
    fig_3d, ax_3d = plt.subplots(ncols=(steps), nrows=B, figsize=(25, int(25//(11/B))))
    for row in range(B):
        for col in range(steps):
            #Plot the interpolations alpha from 0 to 1.
            plot_pose3d(ax_3d[row,col], reconstruction_interp[row][col], skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
            if (row == 0) : ax_3d[row, col].set_title(f'interpolation \n alpha={alphas[col][0]:.2f}', size=10)

    plt.tight_layout(pad=0.22)
    for fig, name in zip([fig_3d], ['3d']):
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        fig.patch.set_facecolor('white')
        fig.tight_layout()
        fig.savefig(savepath)

def uniform_interpolation_batchS(trainer, loader, space_dim, B=5, steps=11, seq_type="S", slerp_type="in", savepath=None):
    '''For each interpolation N pair of ground truths (i and i-1 of batch)
    Get M=11 'equidistant' latent interpolations and their reconstruction between each pair of ground truth images
    Alpha varies from 0 to 1 included
    '''
    cdn = lambda x: x.cpu().detach().numpy()
    # Get the device on which AE is located
    device = list(trainer.models.ae.parameters())[0].device
    
    #Get batch from loader and batch size
    batch= next(iter(loader))['pose3d']
    batch = batch.to(device)
    bs = batch.shape[0]
    random_index = torch.randperm(bs)

    latent_code, reconstruction = trainer.models.ae(batch).values()

    seed = 0
    torch.manual_seed(seed)

    x = latent_code
    y = latent_code[random_index]

    z, _ = create_sequence(x, y, steps, seq_type, slerp_type)
    '''For each interpolation N pair of ground truths (i and i-1 of batch)
    Get M=11 'equidistant' latent interpolations and their reconstruction between each pair of ground truth images
    Alpha varies from 0 to 1 included
    '''
    zs = z.size()
    z_expand = torch.reshape(z, (-1,space_dim))
    reconstruction_interp = trainer.models.ae(z_expand, 'decoder')['reconstructed']

    reconstruction_interp = cdn(torch.reshape(reconstruction_interp, (zs[0], zs[1], 17,3)))

    alphas = np.arange(0, 1.1, (1/(steps-1)) ) #For printing only

    annotate = False
    fig_3d, ax_3d = plt.subplots(ncols=(steps+2), nrows=B, figsize=(25, int(25//(11/B))))
    ax_3d[0, 0].set_title(f'Ground Truth A', size=10)
    ax_3d[0, steps+1].set_title(f'Ground Truth B', size=10)
    for row in range(B):
        #Plot ground truths pair at each extremity
        plot_pose3d(ax_3d[row,0], cdn(batch[row]), skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
        plot_pose3d(ax_3d[row, steps+1], cdn(batch[random_index][row]), skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
        for col in range(steps):
            #Plot the interpolations alpha from 0 to 1.
            plot_pose3d(ax_3d[row,col+1], reconstruction_interp[row][col], skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
            if (row == 0) : ax_3d[row, col+1].set_title(f'interpolation \n alpha={alphas[col]:.2f}', size=10)

    plt.tight_layout(pad=0.22)
    for fig, name in zip([fig_3d], ['3d']):
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        fig.patch.set_facecolor('white')
        fig.tight_layout()
        fig.savefig(savepath)

def uniform_smoothness_batchS(trainer, loader, space_dim, B=5, steps=101, seq_type="S", slerp_type="in", savepath=None):
    absolute_start = time()
    
    cdn = lambda x: x.cpu().detach().numpy()
    # Get the device on which AE is located
    device = list(trainer.models.ae.parameters())[0].device
    
    #Get batch from loader and batch size
    batch= next(iter(loader))['pose3d']
    batch = batch.to(device)[:B]
    bs = batch.shape[0]
    random_index = torch.randperm(bs)

    latent_code, reconstruction = trainer.models.ae(batch).values()

    seed = 0
    torch.manual_seed(seed)

    x = latent_code
    y = latent_code[random_index]

    z, _ = create_sequence(x, y, steps, seq_type, slerp_type)
    zs = z.size()
    z_expand = torch.reshape(z, (-1,space_dim))
    reconstruction_interp = trainer.models.ae(z_expand, 'decoder')['reconstructed']

    reconstruction_interp = torch.reshape(reconstruction_interp, (zs[0], zs[1], -1,3))

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    pair_smoothness = []
    differences_computations = MPJPE()
    x_arr = torch.arange(1, steps)
    swapped = torch.swapaxes(reconstruction_interp, 0,1)
    for step in torch.arange(1, steps):
        pair_smoothness.append(cdn(differences_computations(swapped[step],swapped[step-1])).item())
    print(pair_smoothness)
    ax.plot(x_arr, pair_smoothness)
    
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(savepath)
    total_time = time() - absolute_start
    print(total_time)

def uniform_smoothness_randomS(trainer, loader, space_dim, B=5, steps=101, seq_type="S", slerp_type="in", savepath=None):
    absolute_start = time()

    cdn = lambda x: x.cpu().detach().numpy()
    # Get the device on which AE is located
    device = list(trainer.models.ae.parameters())[0].device
    
    seed = 0
    torch.manual_seed(seed)

    x = _generate_random_input(B, latent_space_dim=space_dim, latent_space_type='S')
    y = _generate_random_input(B, latent_space_dim=space_dim, latent_space_type='S')

    z, _ = create_sequence(x, y, steps, seq_type, slerp_type)
    zs = z.size()
    z_expand = torch.reshape(z, (-1,space_dim))
    reconstruction_interp = trainer.models.ae(z_expand, 'decoder')['reconstructed']

    reconstruction_interp = torch.reshape(reconstruction_interp, (zs[0], zs[1], -1,3))

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    pair_smoothness = []
    differences_computations = MPJPE()
    x_arr = torch.arange(1, steps)
    swapped = torch.swapaxes(reconstruction_interp, 0,1)
    for step in torch.arange(1, steps):
        pair_smoothness.append(cdn(differences_computations(swapped[step],swapped[step-1])).item())
    print(pair_smoothness)
    ax.plot(x_arr, pair_smoothness)
    
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(savepath)
    total_time = time() - absolute_start
    print(total_time)