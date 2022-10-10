import os
import torch as th
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

from ...utils.h36m_visual_utils import plot_pose3d, BONES_H36M_3D_17
from ...functional.interpolation_s import create_sequence
from ...losses.joints_loss import MPJPE, MPJPE_PERPOSE

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
    visu_output_dir = Path(os.path.abspath('.'))/trainer.cfg.OUTPUT_DIR / trainer.cfg.EXP_NAME
    smoothness(trainer, visu_output_dir)
    return 0

def sample_from_real(train_loader, num_samples, seed=2021_10_29):
    th.manual_seed(seed)

    arr_x_pose = th.empty((0))
    for sample in train_loader:
        pose = sample['pose3d']
        arr_x_pose = th.cat((arr_x_pose, pose))

        if len(arr_x_pose) >= num_samples:
            arr_x_pose = arr_x_pose[:num_samples]
            break
    return arr_x_pose.flatten(start_dim=1)

def mpjpe_along_axis(function, x, axis: int = 0):
    # 
    return th.stack([function(x_i,x_j) for x_i, x_j in zip(th.unbind(x, dim=axis)[:-1], th.unbind(x, dim=axis)[1:] )], dim=axis)

def expected_average_transformation(swapped):
    mpjpe_perpose = MPJPE_PERPOSE()
    p_zero = swapped[0]
    p_last = swapped[-1]
    nb_steps = len(swapped)
    return mpjpe_perpose(p_zero, p_last)/nb_steps

def normalize_deltas(deltas, eat):
    return deltas/eat

def smoothness_ratios(deltas):
    sratios = [th.max(seq)/th.min(seq) for seq in deltas]
    return th.tensor(sratios)

def smoothness(trainer, visu_output_dir):
    cfg = trainer.cfg
    steps = cfg.STEPS
    coverage_type = cfg.COVERAGE_TYPE
    latent_dim = cfg.MODELS.ae.PARAMS.latent_dim
    device = th.device(cfg.DEVICE)

    save_path = os.path.join(visu_output_dir, f'{coverage_type}_dataset_type_network_type')
    os.makedirs(save_path, exist_ok=True)
    train_loader = trainer.dataload.train
    th.set_grad_enabled(False)
    mpjpe = MPJPE_PERPOSE()

    # Sample 2B poses and get 2B codes
    nb_for_pairs = (cfg.TARGET_BATCH_SIZE*2) #check for batch_size
    poses_target = sample_from_real(train_loader, nb_for_pairs, seed=2021_10_29).to(device=device)
    latent_code, _ = trainer.models.ae(poses_target).values()

    # Split in 2xB codes
    x, y = latent_code.split(cfg.TARGET_BATCH_SIZE)

    # Create Interpolation sequence of Bxstepsxlatent_dim
    z, _ = create_sequence(x, y, steps)
    z = th.reshape(z, (-1,latent_dim))
    # Get reconstructed poses Bxstepsx17x3
    poses_target = trainer.models.ae(z, 'decoder')['reconstructed'].double().to(device=device)
    poses_target = poses_target.view(-1,steps,17,3)

    # Get Batches per steps stepsxBx17x3
    seq_poses_target = th.swapaxes(poses_target, 0,1)
    # Get eat=MPJPE(B0,BT)/steps for each batch,  Bx
    eat = expected_average_transformation(seq_poses_target)
 
    # Compte deltas MPJPE(Bi,Bj), 100xB
    seq_deltas = mpjpe_along_axis(mpjpe, seq_poses_target, 0)

    seq_deltas_norm = normalize_deltas(seq_deltas, eat)

    # Compute smoothness ratios for each seq in batch
    deltas_norm = th.swapaxes(seq_deltas_norm, 0,1)
    sratios = smoothness_ratios(deltas_norm)
    print('sratios')
    print(sratios.shape)
    print(sratios.mean())
    print(sratios.var())
    print(sratios.median())

    # For Plot: Compute mean for each step
    aggregated_deltas = seq_deltas_norm.mean(axis=1)
    
    # Plot aggregated smoothness
    plot_smoothness(save_path, aggregated_deltas.cpu().detach().numpy(), 0)

    # Plot individual smoothness for 5 first seq
    for i in range(6):
        plot_smoothness(save_path, deltas_norm[i].cpu().detach().numpy(), i+1)
    
    # Plot poses interpolation (10 steps) for 5 first seq
    visualise_target(poses_target, save_path)
    print(aggregated_deltas.tolist())
    msg = (
        f'\naggregated_deltas: { {*aggregated_deltas.tolist()} }\n'
        f'\nsmoothness ratios: { {*sratios.tolist()} }\n'
        f'smoothness ratios mean: {sratios.mean():12.5e}\t'
        f'smoothness ratios var: {sratios.var():12.5e}\t'
        f'smoothness ratios median: {sratios.median():12.5e}\n'
            )


    trainer.logger.info(msg)

    Lambda = cfg.LOSSES.ae_loss.PARAMS.ae_reg_coef
    smoothnessdict = {"model": latent_dim,
                "smoothness":{
                    "Lambda": Lambda,
                    "aggregated_deltas": aggregated_deltas.tolist(),
                    "aggregated_deltas_mean": aggregated_deltas.mean().item(),
                    "aggregated_deltas_var": aggregated_deltas.var().item(),
                    'aggregated_deltas_median': aggregated_deltas.median().item(),
                    'ratios_mean': sratios.mean().item(),
                    'ratios_var': sratios.var().item(),
                    'ratios_median': sratios.median().item()
                    }
                }
    import json
    with open(os.path.join(save_path, 'smoothness_'+cfg.EXP_NAME+'.json'), 'w', encoding ='utf8') as json_file:
        json.dump(smoothnessdict, json_file, indent=4)



    #th.save(poses_target.detach().cpu(), os.path.join(save_path, 'poses_target.pth'))

def plot_smoothness(save_path, deltas, id):
    name = f'plot_smoothness_seq_{id}'
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    x_arr = th.arange(1, len(deltas)+1)
    ax.plot(x_arr, deltas)
    
    fig.subplots_adjust(wspace=0.01, hspace=0.01)
    fig.patch.set_facecolor('white')
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, name))

def visualise_target(poses_target, save_path):
    annotate = False
    fig_3d, ax_3d = plt.subplots(ncols=(11), nrows=6, figsize=(25, int(25//(11/10))))
    for row in range(6):
        for col in range(11):
            step = col*10
            # Plot the interpolations alpha from 0 to 1.
            plot_pose3d(ax_3d[row,col], poses_target[row][step].detach().cpu(), skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
            if (row == 0) : ax_3d[row, col].set_title(f's alpha={(step/len(poses_target[0])):.1f}', size=20)

    plt.tight_layout(pad=0.22)
    for fig, name in zip([fig_3d], ['3d']):
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        fig.patch.set_facecolor('white')
        fig.tight_layout()
        fig.savefig(os.path.join(save_path, 'sample_from_latent'))