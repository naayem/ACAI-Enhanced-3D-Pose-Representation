import os
from time import time

import torch
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
from ..utils.h36m_visual_utils import plot_pose3d, BONES_H36M_3D_17
from mpl_toolkits.mplot3d import Axes3D
from ..losses.joints_loss import JointsLoss, MPJPE


def random_interpolation(trainer, loader, savepath=None):
    '''Each column represent a pair of ground truths, 
    their respective recontruction, and the reconstruction of an interpolation with alpha random.'''
    cdn = lambda x: x.cpu().detach().numpy()
    # Get the device on which AE is located
    device = list(trainer.models.ae.parameters())[0].device
    
    #Get batch from loader and batch size
    batch = next(iter(loader))['pose3d']
    batch = batch.to(device)
    bs = batch.shape[0]
    
    #Get the random alphas, get the latent_code and reconstruction from bacth through ae
    alphas = 0.5 * torch.rand(bs, 1).to(device)
    latent_code, reconstruction = trainer.models.ae(batch).values()
    
    #shift the index by -1, interpolate between i and i-1, get reconstruction of this latent interpolation
    shifted_index = torch.arange(0, bs) - 1
    interpolated_code = alphas * latent_code[shifted_index] + (1 - alphas) * latent_code 
    interpolation = cdn(trainer.models.ae(interpolated_code, 'decoder')['reconstructed'])

    #Get ground truths for top bottom and their respective reconstruction to display    
    top, bottom = cdn(batch[:5]), cdn(batch[shifted_index][:5])
    top_reconstruction, bottom_reconstruction = cdn(reconstruction[:5]), cdn(reconstruction[shifted_index][:5]) 

    #Each column represent a pair of ground truths, their respective recontruction, and the reconstruction of an interpolation with alpha random
    #Titles explicit content and order for each column
    annotate = True
    rows, cols = 5, 5
    fig_3d, ax_3d = plt.subplots(rows, cols, figsize=(5*cols,5*rows))
    for col in range(cols):
        alpha = np.round(alphas[col].cpu().numpy().item(), 3)
        
        ax_3d[0, col].set_title(f'Ground Truth {col} A', size=15)
        ax_3d[1, col].set_title(f'Reconstruction {col} A', size=15)
        ax_3d[2, col].set_title(f'Interpolation {col} Alpha: {alpha}', size=15)
        ax_3d[3, col].set_title(f'Reconstruction {col} B', size=15)
        ax_3d[4, col].set_title(f'Ground Truth {col} B', size=15)
        
        for v in range(5):
            ax_3d[v, col].axis('off')

        ### pose3d
        plot_pose3d(ax_3d[0,col], top[col], skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
        plot_pose3d(ax_3d[1,col], top_reconstruction[col], skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
        plot_pose3d(ax_3d[2,col], interpolation[col], skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
        plot_pose3d(ax_3d[3,col], bottom_reconstruction[col], skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
        plot_pose3d(ax_3d[4,col], bottom[col], skeleton_bones=BONES_H36M_3D_17, annotate=annotate)

    plt.tight_layout(pad=0.22)

    for fig, name in zip([fig_3d], ['3d']):
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        fig.patch.set_facecolor('white')
        fig.tight_layout()
        fig.savefig(savepath)
        
def uniform_interpolation(trainer, loader, N=11, savepath=None):
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
    
    # Grid of 11 uniform alphas [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] by default
    alphas = torch.arange(0, 1.1, step=0.1, dtype=torch.float32).reshape(-1, 1).to(device)

    latent_code, reconstruction = trainer.models.ae(batch).values()

    shifted_index = torch.arange(0, bs) - 1
    
    #For each interpolation a pair of ground truths (i and i-1 of batch)
    #Get M=11 'equidistant' latent interpolations and their reconstruction between each pair of ground truth images
    #Interpolation and Reconstruction from alpha 0 to alpha 1 included
    pair_interpolations = []
    for pair in range(N):
        pair_code = torch.cat([ (1- alpha.reshape(1, 1)) * latent_code[pair] + \
                               alpha.reshape(1, 1) * (latent_code[shifted_index][pair]) for alpha in alphas])
        pair_interpolations.append(cdn(trainer.models.ae(pair_code, 'decoder')['reconstructed']))
            
    annotate = False
    fig_3d, ax_3d = plt.subplots(ncols=(len(alphas)+2), nrows=N, figsize=(25, int(25//(11/N))))
    ax_3d[0, 0].set_title(f'Ground Truth A', size=10)
    ax_3d[0, len(alphas)+1].set_title(f'Ground Truth B', size=10)
    for row in range(N):
        #Plot ground truths pair at each extremity
        plot_pose3d(ax_3d[row,0], cdn(batch[row]), skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
        plot_pose3d(ax_3d[row,len(alphas)+1], cdn(batch[shifted_index][row]), skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
        for col in range(len(alphas)):
            #Plot the interpolations alpha from 0 to 1.
            plot_pose3d(ax_3d[row,col+1], pair_interpolations[row][col], skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
            if (row == 0) : ax_3d[row, col+1].set_title(f'interpolation \n alpha={alphas[col][0]:.2f}', size=10)

    plt.tight_layout(pad=0.22)
    for fig, name in zip([fig_3d], ['3d']):
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        fig.patch.set_facecolor('white')
        fig.tight_layout()
        fig.savefig(savepath)

def uniform__smoothness(trainer, loader, N=11, savepath=None):
    absolute_start = time()
    '''For each interpolation N pair of ground truths (i and i-1 of batch)
    From M=11 'equidistant' latent interpolations and their reconstruction between each pair of ground truth images
    Compute the differences delta between two consecutive interpolation' reconstruction
    '''
    cdn = lambda x: x.cpu().detach().numpy()
    # Get the device on which AE is located
    device = list(trainer.models.ae.parameters())[0].device
    
    #Get batch from loader and batch size
    batch= next(iter(loader))['pose3d']
    batch = batch.to(device)
    bs = batch.shape[0]
    
    # Grid of 11 uniform alphas [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] by default
    alphas = torch.arange(0, 1.01, step=0.01, dtype=torch.float32).reshape(-1, 1).to(device)

    latent_code, reconstruction = trainer.models.ae(batch).values()

    shifted_index = torch.arange(0, bs) - 1
    
    #For each interpolation a pair of ground truths (i and i-1 of batch)
    #Get M=11 'equidistant' latent interpolations and their reconstruction between each pair of ground truth images
    #Interpolation and Reconstruction from alpha 0 to alpha 1 included
    pair_interpolations = []
    for pair in range(N):
        pair_code = torch.cat([ (1- alpha.reshape(1, 1)) * latent_code[pair] + \
                               alpha.reshape(1, 1) * (latent_code[shifted_index][pair]) for alpha in alphas])
        pair_interpolations.append(trainer.models.ae(pair_code, 'decoder')['reconstructed'])

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    pair_smoothness = []
    differences_computations = MPJPE()
    shifted_alpha_index = torch.arange(0, len(alphas)) - 1
    x_arr = torch.arange(1, len(alphas))
    for row in range(N):
        deltas = []
        for col in torch.arange(1, len(alphas)):
            deltas.append( cdn(differences_computations(pair_interpolations[row][col], pair_interpolations[row][shifted_alpha_index][col])).item() )
        pair_smoothness.append(deltas)
    pair_smoothness = np.asarray(pair_smoothness)
    total_deltas = pair_smoothness.mean(axis=0)
    ax.plot(x_arr, total_deltas)
    
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(savepath)
    total_time = time() - absolute_start
    print(total_time)

def uniform_smoothness_faster(trainer, loader, N=11, savepath=None):
    absolute_start = time()
    '''For each interpolation N pair of ground truths (i and i-1 of batch)
    From M=11 'equidistant' latent interpolations and their reconstruction between each pair of ground truth images
    Compute the differences delta between two consecutive interpolation' reconstruction
    '''
    cdn = lambda x: x.cpu().detach().numpy()
    # Get the device on which AE is located
    device = list(trainer.models.ae.parameters())[0].device
    
    #Get batch from loader and batch size
    batch= next(iter(loader))['pose3d']
    batch = batch.to(device)
    bs = batch.shape[0]
    
    # Grid of 11 uniform alphas [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] by default
    alphas = torch.arange(0, 1.01, step=0.01, dtype=torch.float32).reshape(-1, 1).to(device)

    latent_code, reconstruction = trainer.models.ae(batch).values()
    print(latent_code.size())

    shifted_index = torch.arange(0, bs) - 1
    
    #For each interpolation a pair of ground truths (i and i-1 of batch)
    #Get M=11 'equidistant' latent interpolations and their reconstruction between each pair of ground truth images
    #Interpolation and Reconstruction from alpha 0 to alpha 1 included
    pair_interpolations = []
    for pair in range(N):
        pair_code = torch.cat([ (1- alpha.reshape(1, 1)) * latent_code[pair] + \
                               alpha.reshape(1, 1) * (latent_code[shifted_index][pair]) for alpha in alphas])
        pair_interpolations.append(trainer.models.ae(pair_code, 'decoder')['reconstructed'])

    stack = torch.stack(pair_interpolations)
    swapped = torch.swapaxes(stack, 0,1)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    pair_smoothness = []
    differences_computations = MPJPE()
    shifted_alpha_index = torch.arange(0, len(alphas)) - 1
    x_arr = torch.arange(1, len(alphas))
    for step in torch.arange(1, len(alphas)):
        pair_smoothness.append(cdn(differences_computations(swapped[step],swapped[shifted_alpha_index][step])).item())

    ax.plot(x_arr, pair_smoothness)
    
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(savepath)
    total_time = time() - absolute_start
    print(total_time)

def uniform_smoothness_random(trainer, batch_size = 1024, savepath=None):
    absolute_start = time()
    '''For each interpolation N pair of ground truths (i and i-1 of batch)
    From M=11 'equidistant' latent interpolations and their reconstruction between each pair of ground truth images
    Compute the differences delta between two consecutive interpolation' reconstruction
    '''
    cdn = lambda x: x.cpu().detach().numpy()
    # Get the device on which AE is located
    device = list(trainer.models.ae.parameters())[0].device
    
    
    # Grid of 11 uniform alphas [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] by default
    alphas = torch.arange(0, 1.01, step=0.01, dtype=torch.float32).reshape(-1, 1).to(device)

    #latent_code, reconstruction = trainer.models.ae(batch).values()
    latent_code = torch.normal(mean=0, std=1 , size=[1024,64]).to(device)

    shifted_index = torch.arange(0, batch_size) - 1
    
    #For each interpolation a pair of ground truths (i and i-1 of batch)
    #Get M=11 'equidistant' latent interpolations and their reconstruction between each pair of ground truth images
    #Interpolation and Reconstruction from alpha 0 to alpha 1 included
    pair_interpolations = []
    for pair in range(batch_size):
        pair_code = torch.cat([ (1- alpha.reshape(1, 1)) * latent_code[pair] + \
                               alpha.reshape(1, 1) * (latent_code[shifted_index][pair]) for alpha in alphas])
        pair_interpolations.append(trainer.models.ae(pair_code, 'decoder')['reconstructed'])

    stack = torch.stack(pair_interpolations)
    swapped = torch.swapaxes(stack, 0,1)


    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    pair_smoothness = []
    differences_computations = MPJPE()
    shifted_alpha_index = torch.arange(0, len(alphas)) - 1
    x_arr = torch.arange(1, len(alphas))
    for step in torch.arange(1, len(alphas)):
        pair_smoothness.append(cdn(differences_computations(swapped[step],swapped[shifted_alpha_index][step])).item())

    ax.set_yscale('log')
    #ax.set_ylim([0, 1])
    ax.plot(x_arr, pair_smoothness)
    
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(savepath)
    total_time = time() - absolute_start
    print(total_time)

def uniform_smoothness_perm(trainer, loader, N=11, savepath=None):
    absolute_start = time()
    '''For each interpolation N pair of ground truths (i and i-1 of batch)
    From M=11 'equidistant' latent interpolations and their reconstruction between each pair of ground truth images
    Compute the differences delta between two consecutive interpolation' reconstruction
    '''
    cdn = lambda x: x.cpu().detach().numpy()
    # Get the device on which AE is located
    device = list(trainer.models.ae.parameters())[0].device
    
    #Get batch from loader and batch size
    batch= next(iter(loader))['pose3d']
    batch = batch.to(device)
    bs = batch.shape[0]
    
    # Grid of 11 uniform alphas [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] by default
    alphas = torch.arange(0, 1.01, step=0.01, dtype=torch.float32).reshape(-1, 1).to(device)

    latent_code, reconstruction = trainer.models.ae(batch).values()
    print(latent_code.size())

    shifted_index = torch.arange(0, bs) - 1
    random_index = torch.randperm(bs)
    print(shifted_index)
    print(len(shifted_index))
    print(random_index)
    print(len(random_index))

    #For each interpolation a pair of ground truths (i and i-1 of batch)
    #Get M=11 'equidistant' latent interpolations and their reconstruction between each pair of ground truth images
    #Interpolation and Reconstruction from alpha 0 to alpha 1 included
    pair_interpolations = []
    for pair in range(N):
        pair_code = torch.cat([ (1- alpha.reshape(1, 1)) * latent_code[pair] + \
                               alpha.reshape(1, 1) * (latent_code[random_index][pair]) for alpha in alphas])
        pair_interpolations.append(trainer.models.ae(pair_code, 'decoder')['reconstructed'])

    stack = torch.stack(pair_interpolations)
    swapped = torch.swapaxes(stack, 0,1)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    pair_smoothness = []
    differences_computations = MPJPE()
    shifted_alpha_index = torch.arange(0, len(alphas)) - 1
    x_arr = torch.arange(1, len(alphas))
    for step in torch.arange(1, len(alphas)):
        pair_smoothness.append(cdn(differences_computations(swapped[step],swapped[step-1])).item())

    ax.plot(x_arr, pair_smoothness)
    
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(savepath)
    total_time = time() - absolute_start
    print(total_time)