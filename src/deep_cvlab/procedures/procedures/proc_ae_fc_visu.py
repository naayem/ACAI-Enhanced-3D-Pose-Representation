import torch
from time import time
import matplotlib.pyplot as plt
from ..procedures_common import status_msg
from ...utils.h36m_visual_utils import _check_poses
from pathlib import Path
from ...utils import metrics

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

    fig_train, ax_train = plt.subplots(1, 1, figsize=(7, 5))

    #meter.load_state(f'{final_output_dir}/metrics/{arr_name}_train.pth')
    meter.load_state(f'{cfg.VISU.OUTPUT_PATH}{cfg.VISU.TRAIN_LOSS_PATH}')
    vals = meter.prev_vals
    x_arr = torch.arange(1, len(vals)+1)
    ax_train.plot(x_arr, vals, label=arr_name)

    ax_train.legend(fontsize='x-large')
    plt.suptitle(title+' training')

    fig_valid, ax_valid = plt.subplots(1, 1, figsize=(7, 5))

    #meter.load_state(f'{final_output_dir}/metrics/{arr_name}_valid.pth')
    meter.load_state(f'{cfg.VISU.OUTPUT_PATH}{cfg.VISU.VALID_LOSS_PATH}')
    vals = meter.prev_vals
    x_arr = torch.arange(1, len(vals)+1)
    ax_valid.plot(x_arr, vals, label=arr_name)

    ax_valid.legend(fontsize='x-large')
    plt.suptitle(title+' validation')

    for fig, name in zip([fig_train, fig_valid], ['p_loss_train', 'p_loss_valid']):
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

            p = trainer.models.net(gt)
            reconstruction = p["reconstructed"]

            _check_poses(gt, reconstruction, trainer.cfg)
     
            total_time = time() - absolute_start
            status_msg(trainer, batch_idx, dl_len, trainer.meters.valid.p_loss, total_time)
    
    loss_visu(trainer.cfg)
    perf_indicator = trainer.meters.valid.p_loss.cur_avg
    return 1