import torch
from time import time
import matplotlib.pyplot as plt
from pathlib import Path
from ..utils import metrics

#Plot the train and validation losses of each type of losses used for the model
def loss_visu(cfg):
    visu_output_dir = Path(cfg.OUTPUT_DIR) / cfg.EXP_NAME
    title = cfg.EXP_NAME.split('/')[-1]

    for loss in cfg.VISU:
        #name of loss for titles
        arr_name = f'{loss}_loss'

        #get values fot training loss
        meter = metrics.AvgMeter()
        meter.load_state(f'{cfg.VISU[loss].OUTPUT_PATH}{cfg.VISU[loss].TRAIN_LOSS_PATH}')
        vals = meter.prev_vals
        x_arr = torch.arange(1, len(vals)+1)

        #get values fot validation loss
        meter2 = metrics.AvgMeter()
        meter2.load_state(f'{cfg.VISU[loss].OUTPUT_PATH}{cfg.VISU[loss].VALID_LOSS_PATH}')
        vals2 = meter2.prev_vals
        x_arr2 = torch.arange(1, len(vals2)+1)

        #plot for training only
        fig_train, ax_train = plt.subplots(1, 1, figsize=(7, 5))  
        ax_train.plot(x_arr, vals, label='training loss')
        ax_train.legend(fontsize='x-large')
        plt.suptitle(title+' '+arr_name+' training')

        #plot for validation only
        fig_valid, ax_valid = plt.subplots(1, 1, figsize=(7, 5))
        ax_valid.plot(x_arr2, vals2, label='validation loss')
        ax_valid.legend(fontsize='x-large')
        plt.suptitle(title+' '+arr_name+' validation')

        #plot for training and validation
        fig_both, ax_both = plt.subplots(1, 1, figsize=(7, 5))
        ax_both.plot(x_arr, vals, label='training loss')
        ax_both.plot(x_arr2, vals2, label='validation loss')
        plt.legend()
        plt.suptitle(title+' '+arr_name)

        #final adjustments and save
        for fig, name in zip([fig_train, fig_valid, fig_both], [f'{loss}_train', f'{loss}_valid', f'{loss}_train_valid']):
            fig.subplots_adjust(wspace=0.01, hspace=0.01)
            fig.patch.set_facecolor('white')
            fig.tight_layout()
            fig.savefig(f'{visu_output_dir}/plot_{name}.png')