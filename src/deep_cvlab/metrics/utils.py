import torch as th
import sys
import os
import numpy as np
sys.path.append('/cvlabdata2/home/davydov/smplprior/smplprior')
import lib
from lib.functional.renderer.render import init_smpl
from lib.functional.renderer.utils import get_renderer
from lib.datasets.amass import AMASS
from lib.models.vposer_gan import VPoserGenerator

import pytorch3d.structures
import pytorch3d.renderer

sys.path.append('/cvlabdata2/home/davydov/smplprior/vposer_v1_0')
from vposer_smpl import VPoser


def init_network(network_type):
    if network_type == 'vae':
        net = VPoser(num_neurons=512, latentD=32, data_shape=[1,21,3], use_cont_repr=True)
        ckpt = th.load('/cvlabdata2/home/davydov/smplprior/vposer_v1_0/snapshots/TR00_E096.pt', map_location='cpu')
        net.load_state_dict(ckpt)
        net.eval()
    elif network_type.startswith('gan'):
        net = VPoserGenerator(latentD=32, num_joints=21)

        latent_type = network_type[-1] 
        assert latent_type in ['S', 'U', 'N']
        # ckpt = f'/cvlabdata2/home/davydov/smplprior/smplprior/output/vposer_gan/vposer_gan_{latent_type}_amass/ckpt_0060.pth' # first version of VPoser_GAN
        ckpt = f'/cvlabdata2/home/davydov/smplprior/smplprior/output/vposer_gan/vposer_gan_{latent_type}_amass_ORIG_TRAIN/ckpt_0060.pth' # trained with original training split 3.11.2021
        ckpt = th.load(ckpt, map_location='cpu')['generator_state_dict']
        net.load_state_dict(ckpt)
        net.eval()
    return net


def sample_from_gan(gen, num_samples, latent_space_type, latent_space_dim=32):
    from lib.functional.sampling import generate_random_input as generate_random_input_
    generate_random_input = lambda batch_size : generate_random_input_(batch_size, latent_space_dim, latent_space_type)
    z = generate_random_input(num_samples)
    with th.no_grad():
        x = gen(z, output_type='matrot')
        pose = VPoser.matrot2aa(x.view([-1, 1, x.shape[-2], 9])).squeeze(1).cpu()
    return pose

def net_inference(z, net, network_type):
    with th.no_grad():
        if network_type == 'vae':
            print(network_type)
            x = net.decode(z, output_type='matrot')        
        else:
            x = net(z, output_type='matrot')
        pose = VPoser.matrot2aa(x.view([-1, 1, x.shape[-2], 9])).squeeze(1).cpu()
        return pose

def net_inference_with_grad(z, net, network_type):
            
    if network_type == 'vae':
        # print(network_type)
        x = net.decode(z, output_type='matrot')        
    else:
        x = net(z, output_type='matrot')
    pose = VPoser.matrot2aa(x.view([-1, 1, x.shape[-2], 9])).squeeze(1) #.cpu() wtf???
    return pose

def sample_from_net(num_samples, net, network_type):
    if network_type == 'vae':
        poses_fake = net.sample_poses(num_poses=num_samples) # B x 1 x 21 x 3
    else: # gan network, network_type in ['ganS', 'ganU', 'ganN']
        poses_fake = sample_from_gan(net, num_samples, latent_space_type=network_type[-1], latent_space_dim=32)
    poses_fake = poses_fake.flatten(start_dim=1)
    return poses_fake


def init_dataset(dataset_type, use_original_splits=True):

    if use_original_splits: ### use original splits
        dset = AMASS(mode=dataset_type, num_betas=10, num_joints=21)

    else: ### custom datasets for each dataset_type
        if dataset_type == 'train':
            datasets = ['CMU', 'MPILimits']
        elif dataset_type == 'test':
            datasets = ['HumanEva', 'MPIHDM05', 'SFU', 'MPImosh', 'BMLhandball', 'BMLmovi', 'TotalCapture', 'Transitionsmocap']
        else:
            raise ValueError

        dset = AMASS(datasets=datasets, num_betas=10, num_joints=21)
    return dset


def sample_from_real(dset, num_samples, seed=2021_10_29):
    th.manual_seed(seed)
    dl = th.utils.data.DataLoader(dset, batch_size=1024, shuffle=True, num_workers=10)

    arr_x_pose = th.empty((0))
    for sample in dl:
        pose = sample['pose']
        arr_x_pose = th.cat((arr_x_pose, pose))

        if len(arr_x_pose) >= num_samples:
            arr_x_pose = arr_x_pose[:num_samples]
            break
    return arr_x_pose.flatten(start_dim=1)


def render_meshes(poses, smpl_model):
    poses = th.cat([poses, th.zeros(len(poses), 6).to(device=poses.device)], dim=1) # there must be 63 = 21*3 parameters!
    output = smpl_model(body_pose=poses, expression=None, return_verts=True)
    vertices = output.vertices
    vertices = vertices.reshape(len(poses), vertices.shape[-2], 3)
    return vertices

def dists_in_batches(poses, device):


    print(poses.shape) # 8128 x 100 x 63
    num_steps = 100
    max_size = 64

    cpu = th.device('cpu')

    poses_split = th.split(poses, max_size)
    
    smpl_model, _ = init_smpl(max_size*num_steps, device=device)

    dists_pervertex_arr = th.empty(0)
    dists_pervertex_start_end_arr = th.empty(0)
    from tqdm import tqdm
    for pose in tqdm(poses_split):
        # print(f'{i} in {len(poses_split)}...')
        # print(pose.shape) # 64 x 100 x 63
        meshes_interp = render_meshes(pose.view(-1,63).to(device), smpl_model)
        # print(meshes_interp.shape)
        meshes_interp = meshes_interp.reshape(-1, num_steps, 6890, 3)
        # print(meshes_interp.shape)
        delta = meshes_interp[:, 1:] - meshes_interp[:, :-1]
        dists_pervertex = delta.norm(dim=-1).mean(dim=-1).to(cpu)
        dists_pervertex_arr = th.cat([dists_pervertex_arr, dists_pervertex])

        delta_start_end = meshes_interp[:, :1] - meshes_interp[:, -1:]
        dists_pervertex_start_end = delta_start_end.norm(dim=-1).mean(dim=-1).to(cpu)
        dists_pervertex_start_end_arr = th.cat([dists_pervertex_start_end_arr, dists_pervertex_start_end])

        # print(dists_pervertex_arr.shape, dists_pervertex_start_end_arr.shape)

    return dists_pervertex_arr, dists_pervertex_start_end_arr


def init_renderer(batch_size, device, img_size=256):
    renderer = get_renderer(batch_size, img_size=img_size, cam_distance=2.4, device=device)

    ### initialize SMPL model
    smpl_model, faces = init_smpl(batch_size, device, model_type='smpl', model_folder='/cvlabdata2/home/davydov/smplprior/smpl/smpl/new_models/smpl_neutral_lbs_10_207_0_v1.0.0.pkl')
    return renderer, faces


def render(vertices, faces, renderer, device):
    faces = th.tensor(faces.copy()).unsqueeze(0).repeat(vertices.shape[0],1,1).to(device)
    meshes = pytorch3d.structures.Meshes(vertices, faces)

    verts_shape = meshes.verts_packed().shape
    verts_rgb = th.full([1, verts_shape[0], 3], 0.5, device=device, requires_grad=False)
    meshes.textures = pytorch3d.renderer.TexturesVertex(verts_features=verts_rgb)

    with th.no_grad():
        normal_out, silhouette_out = renderer(meshes)

    images = (255*normal_out.cpu().numpy()).astype(np.uint8)
    return images

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Training Launch')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default='experiments/exp.yaml',
                        required=False,
                        type=str)
    args, rest = parser.parse_known_args()
    args = parser.parse_args()
    return args