import os

import torch as th
import matplotlib.pyplot as plt
from pathlib import Path
from ...utils.h36m_visual_utils import plot_pose3d, BONES_H36M_3D_17

from ...metrics.coverage.coverage import BestNeighbors, update_best_nn
from tqdm import tqdm
from ...functional.sampling import generate_random_input
from ...functional.interpolation_s import create_sequence

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
    #visualization(trainer)
    coverage(trainer)
    
    return 0

def sample_from_latent(trainer, num_samples, latent_space_dim, seed=2021_10_29):
    th.manual_seed(seed)
    #dl = th.utils.data.DataLoader(dset, batch_size=1024, shuffle=True, num_workers=10)

    z = generate_random_input(num_samples, latent_space_dim, latent_space_type='S', device=None)
    fake_poses = trainer.models.ae(z, 'decoder')

    return fake_poses['reconstructed'].flatten(start_dim=1)

def sample_from_real(train_loader, num_samples, seed=2021_10_29):
    th.manual_seed(seed)
    #dl = th.utils.data.DataLoader(dset, batch_size=1024, shuffle=True, num_workers=10)

    arr_x_pose = th.empty((0))
    for sample in train_loader:
        pose = sample['pose3d']
        arr_x_pose = th.cat((arr_x_pose, pose))

        if len(arr_x_pose) >= num_samples:
            arr_x_pose = arr_x_pose[:num_samples]
            break
    return arr_x_pose.flatten(start_dim=1)


def coverage(trainer):
    steps = 10
    seq_type = "S"
    slerp_type = "in"

    len_datasets = 189322

    cfg = trainer.cfg
    coverage_type = cfg.COVERAGE_TYPE

    train_loader = trainer.dataload.train
    test_loader = trainer.dataload.valid
    visu_output_dir = Path(os.path.abspath('.'))/trainer.cfg.OUTPUT_DIR / trainer.cfg.EXP_NAME
    latent_dim =trainer.cfg.MODELS.ae.PARAMS.latent_dim

    th.set_grad_enabled(False)
    device = th.device(cfg.DEVICE)
    best_obj = BestNeighbors(cfg.TARGET_BATCH_SIZE, device)

    if coverage_type == 'recall' or 'real_on_real': # init batch from data
        poses_target = sample_from_real(train_loader, cfg.TARGET_BATCH_SIZE, seed=2021_10_29).to(device=device)
    if coverage_type == 'precision': # init batch from data
        poses_target = sample_from_latent(trainer, cfg.TARGET_BATCH_SIZE, latent_dim).double().to(device=device)
    if coverage_type == 'mean_distance': # init batch from data
        nb_for_pairs = (cfg.TARGET_BATCH_SIZE*2) #check for batch_size
        nb_for_steps = (cfg.TARGET_BATCH_SIZE*steps)

        best_obj = BestNeighbors(nb_for_steps, device)
        poses_target = sample_from_real(train_loader, nb_for_pairs, seed=2021_10_29).to(device=device)

        latent_code, reconstruction = trainer.models.ae(poses_target).values()
        x, y = latent_code.split(cfg.TARGET_BATCH_SIZE)
        z, _ = create_sequence(x, y, steps, seq_type, slerp_type)
        z = th.reshape(z, (-1,latent_dim))
        poses_target = trainer.models.ae(z, 'decoder')['reconstructed'].flatten(start_dim=1).double().to(device=device)

    if coverage_type == 'real_on_real' or coverage_type == 'precision' or coverage_type == 'mean_distance': # go across data
        #dl = th.utils.data.DataLoader(dset, batch_size=cfg.ITER_BATCH_SIZE, num_workers=10, drop_last=True)
        for sample in tqdm(test_loader):
            poses_real = sample['pose3d'].to(device=device)
            update_best_nn(poses_real, poses_target, best_obj)

    elif coverage_type == 'recall': # go across latent (fixed number of samples)
        num_iterations = len_datasets * 2 // cfg.ITER_BATCH_SIZE + 1
        for idx in tqdm(range(num_iterations)):
            poses_fake = sample_from_latent(trainer, cfg.ITER_BATCH_SIZE, latent_dim).double().to(device=device)
            update_best_nn(poses_fake, poses_target, best_obj, dist_type=cfg.DIST_TYPE)

    save_path = os.path.join(visu_output_dir, f'{coverage_type}_dataset_type_network_type')
    os.makedirs(save_path, exist_ok=True)

    print(th.mean(best_obj.best_neighbors_distances, 1))
    print(th.var(best_obj.best_neighbors_distances, unbiased=True))
    print(th.median(best_obj.best_neighbors_distances))

    th.save(poses_target.detach().cpu(), os.path.join(save_path, 'meshes_target.pth'))
    th.save(best_obj.best_neighbors_meshes.detach().cpu().view_as(poses_target), os.path.join(save_path, 'meshes_nn_best.pth'))
    th.save(best_obj.best_neighbors_distances.detach().cpu()[0], os.path.join(save_path, 'distances_best.pth'))

    visualise_target(poses_target, save_path)

    latent_dim = cfg.MODELS.ae.PARAMS.latent_dim
    Lambda = cfg.LOSSES.ae_loss.PARAMS.ae_reg_coef
    coveragedict = {"model": latent_dim,
                f"{coverage_type}":{
                    "Lambda": Lambda,
                    "mean": best_obj.best_neighbors_distances.mean().item(),
                    "var": best_obj.best_neighbors_distances.var().item(),
                    'median': best_obj.best_neighbors_distances.median().item(),
                    }
                }
    import json
    with open(os.path.join(save_path, 'coverage_'+cfg.EXP_NAME+'.json'), 'w', encoding ='utf8') as json_file:
        json.dump(coveragedict, json_file, indent=4)


def visualise_target(poses_target, save_path):
    annotate = False
    fig_3d, ax_3d = plt.subplots(ncols=(10), nrows=10, figsize=(25, int(25//(11/10))))
    for row in range(10):
        rowTen = row*10
        for col in range(10):
            #Plot the interpolations alpha from 0 to 1.
            plot_pose3d(ax_3d[row,col], poses_target.detach().cpu().view(-1,17,3)[rowTen+col], skeleton_bones=BONES_H36M_3D_17, annotate=annotate)


    plt.tight_layout(pad=0.22)
    for fig, name in zip([fig_3d], ['3d']):
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        fig.patch.set_facecolor('white')
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, 'sample_from_latent'))