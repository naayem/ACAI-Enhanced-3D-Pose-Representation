import torch
from time import time
import matplotlib.pyplot as plt
from ..procedures_common import status_msg
from ...utils.h36m_visual_utils import _check_poses
from pathlib import Path
from ...utils import metrics
import numpy as np
from ...utils.h36m_visual_utils import plot_pose3d, BONES_H36M_3D_17

'''
"proc_basic" - basic training procedure with the following notations:
    - one model, which is called "net"
    - dataloaders, called "train" and "valid" provide datasamples of the dict-like type
    - the loss function, called "loss", computes the error between net output and ground truth, 
            in both training and validation modes

    This procedure considers the simplest case, where network does simple inference img -> gt and
    loss is computed as loss(out, gt). 
'''
def loss_visu(cfg):
    visu_output_dir = Path(cfg.OUTPUT_DIR) / cfg.EXP_NAME

    arr_name = list(cfg.LOSSES.keys())[0]

    title = cfg.EXP_NAME.split('/')[-1]
    meter = metrics.AvgMeter()

    for loss in cfg.VISU:
        fig_train, ax_train = plt.subplots(1, 1, figsize=(7, 5))

        print(f'{cfg.VISU[loss].OUTPUT_PATH}{cfg.VISU[loss].TRAIN_LOSS_PATH}')
        #meter.load_state(f'{final_output_dir}/metrics/{arr_name}_train.pth')
        meter.load_state(f'{cfg.VISU[loss].OUTPUT_PATH}{cfg.VISU[loss].TRAIN_LOSS_PATH}')
        vals = meter.prev_vals
        x_arr = torch.arange(1, len(vals)+1)
        ax_train.plot(x_arr, vals, label=arr_name)

        ax_train.legend(fontsize='x-large')
        plt.suptitle(title+' training')

        fig_valid, ax_valid = plt.subplots(1, 1, figsize=(7, 5))

        #meter.load_state(f'{final_output_dir}/metrics/{arr_name}_valid.pth')
        meter.load_state(f'{cfg.VISU[loss].OUTPUT_PATH}{cfg.VISU[loss].VALID_LOSS_PATH}')
        vals = meter.prev_vals
        x_arr = torch.arange(1, len(vals)+1)
        ax_valid.plot(x_arr, vals, label=arr_name)

        ax_valid.legend(fontsize='x-large')
        plt.suptitle(title+' validation')

        for fig, name in zip([fig_train, fig_valid], [f'{loss}+_train', f'{loss}+_valid']):
            fig.subplots_adjust(wspace=0.01, hspace=0.01)
            fig.patch.set_facecolor('white')
            fig.tight_layout()
            fig.savefig(f'{visu_output_dir}/plot_{name}_{cfg.EXP_NAME}.png')

def train(trainer):
    return 0
    """
    absolute_start = time()
    dl_len = len(trainer.dataload.train)
    with torch.no_grad():
        for batch_idx, sample in enumerate(trainer.dataload.train, start=1):
            gt = sample['pose3d']
            gt = gt.to(device=trainer.device0, non_blocking=True)

            p = trainer.models.net(gt)
            reconstruction = p["reconstructed"]

            _check_poses(gt, reconstruction, trainer.cfg.EXP_NAME)
     
            total_time = time() - absolute_start
            status_msg(trainer, batch_idx, dl_len, trainer.meters.train.p_loss, total_time) 
    """

def valid(trainer): 
    absolute_start = time()

    dl_len = len(trainer.dataload.valid)
    with torch.no_grad():
        for batch_idx, sample in enumerate(trainer.dataload.valid, start=1):

            gt = sample['pose3d']
            gt = gt.to(device=trainer.device0, non_blocking=True)
            batch_size = gt.size(0)

            #losses = train_on_batch(gt, trainer, batch_size)
            p = trainer.models.ae(gt)
            reconstruction = p["reconstructed"]
     
            #_check_poses(gt, reconstruction, trainer.cfg)

            total_time = time() - absolute_start
            status_msg(trainer, batch_idx, dl_len, trainer.meters.valid.ae_loss, total_time)
            status_msg(trainer, batch_idx, dl_len, trainer.meters.valid.c_loss, total_time)
            
    loss_visu(trainer.cfg)
    perf_indicator = trainer.meters.valid.ae_loss.cur_avg
    perf_indicator = trainer.meters.valid.c_loss.cur_avg 
    visualization(trainer)
    
    return 1



def random_interpolation(trainer, loader, savepath=None):
    cdn = lambda x: x.cpu().detach().numpy()
    
    # Get the device on which AE is located
    device = list(trainer.models.ae.parameters())[0].device
    
    batch = next(iter(loader))['pose3d']

    batch = batch.to(device)
    bs = batch.shape[0]
    
    alphas = 0.5 * torch.rand(bs, 1).to(device)
    latent_code, reconstruction = trainer.models.ae(batch).values()
    
    shifted_index = torch.arange(0, bs) - 1
    interpolated_code = latent_code + alphas * (latent_code[shifted_index] - latent_code)
        
    interpolation = cdn(trainer.models.ae(interpolated_code, 'decoder')['reconstructed'])
    
    def img_reshape(img):
        h, w = img.shape

        return np.transpose(img, (1,0))
        
    top, bottom = cdn(batch[:5]), cdn(batch[shifted_index][:5])
    top_reconstruction, bottom_reconstruction = cdn(reconstruction[:5]), cdn(reconstruction[shifted_index][:5]) 
    '''
        
    fig, ax = plt.subplots(ncols=5, nrows=5, figsize=(25, 25))
    
    for col in range(5):
        alpha = np.round(alphas[col].cpu().numpy().item(), 3)
        
        ax[0, col].set_title(f'Alpha: {alpha}', size=30)
        
        for v in range(5):
            ax[v, col].axis('off')

        ax[0, col].imshow(img_reshape(top[col]), cmap=plt.cm.gray)
        ax[1, col].imshow(img_reshape(top_reconstruction[col]), cmap=plt.cm.gray)
        
        ax[2, col].imshow(img_reshape(interpolation[col]), cmap=plt.cm.gray)
        
        ax[3, col].imshow(img_reshape(bottom_reconstruction[col]), cmap=plt.cm.gray)
        ax[4, col].imshow(img_reshape(bottom[col]), cmap=plt.cm.gray)
        
    plt.tight_layout(pad=0.22)
    
    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()
    '''

    annotate = True
    rows, cols = 5, 5
    fig_3d, ax_3d = plt.subplots(rows, cols, figsize=(5*cols,5*rows))

    for col in range(cols):
        alpha = np.round(alphas[col].cpu().numpy().item(), 3)
        
        ax_3d[0, col].set_title(f'Alpha: {alpha}', size=30)
        
        
        print(col, end=' ')
        for v in range(5):
            ax_3d[v, col].axis('off')

        ### pose3d
        plot_pose3d(ax_3d[0,col], top[col], skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
        plot_pose3d(ax_3d[1,col], top_reconstruction[col], skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
        plot_pose3d(ax_3d[2,col], interpolation[col], skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
        plot_pose3d(ax_3d[3,col], bottom_reconstruction[col], skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
        plot_pose3d(ax_3d[4,col], bottom[col], skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
        
        print('Done!')

    plt.tight_layout(pad=0.22)

    for fig, name in zip([fig_3d], ['3d']):
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        fig.patch.set_facecolor('white')
        fig.tight_layout()
        fig.savefig(savepath)
        
def uniform_interpolation(trainer, loader, N=11, savepath=None):
    cdn = lambda x: x.cpu().detach().numpy()
    
    # Get the device on which AE is located
    device = list(trainer.models.ae.parameters())[0].device
    
    batch= next(iter(loader))['pose3d']
    batch = batch.to(device)
    bs = batch.shape[0]
    
    # Grid of 11 uniform alphas [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] by default
    alphas = torch.arange(0, 1.1, step=0.1, dtype=torch.float32).reshape(-1, 1).to(device)

    latent_code, reconstruction = trainer.models.ae(batch).values()
    
    shifted_index = torch.arange(0, bs) - 1
    
    pair_interpolations = []
    for pair in range(N):
        pair_code = torch.cat([latent_code[pair] + alpha.reshape(1, 1) * \
                              (latent_code[shifted_index][pair] - latent_code[pair]) for alpha in alphas])
        pair_interpolations.append(cdn(trainer.models.ae(pair_code, 'decoder')['reconstructed']))
            
    def img_reshape(img):
        h, w = img.shape
        return np.transpose(img, (1,0))
            
    fig, ax = plt.subplots(ncols=len(alphas), nrows=N, figsize=(25, int(25//(11/N))))
    annotate = True
    rows, cols = len(alphas), N
    fig_3d, ax_3d = plt.subplots(ncols=len(alphas), nrows=N, figsize=(25, int(25//(11/N))))
    
    for row in range(N):
        for col in range(len(alphas)):
            
            ax[row, col].axis('off')
            plot_pose3d(ax_3d[row,col], pair_interpolations[row][col], skeleton_bones=BONES_H36M_3D_17, annotate=annotate)

        
    plt.tight_layout(pad=0.22)

    for fig, name in zip([fig_3d], ['3d']):
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        fig.patch.set_facecolor('white')
        fig.tight_layout()
        fig.savefig(savepath)
        
        
def visualization(trainer):
    train_loader = trainer.dataload.train
    test_loader = trainer.dataload.valid
    visu_output_dir = Path(trainer.cfg.ROOT_DIR)/trainer.cfg.OUTPUT_DIR / trainer.cfg.EXP_NAME

    print(f'{visu_output_dir}/Train_RandomInterpolation_{trainer.cur_epoch}.png')
    #random_interpolation(trainer, train_loader, savepath= f'{visu_output_dir}/Train_RandomInterpolation_{trainer.cur_epoch}.png')
    #random_interpolation(trainer, test_loader, savepath= f'{visu_output_dir}/Test_RandomInterpolation_{trainer.cur_epoch}.png')
    
    uniform_interpolation(trainer, train_loader, N=11, savepath= f'{visu_output_dir}/Train_UniformInterpolation_{trainer.cur_epoch}.png')
    uniform_interpolation(trainer, test_loader, N=11, savepath= f'{visu_output_dir}/Test_UniformInterpolation_{trainer.cur_epoch}.png')