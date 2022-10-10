from deep_cvlab import utils, core
import matplotlib.pyplot as plt
import pprint
from pathlib import Path
import torch


def loss_visu(cfg):
    output_dir = cfg.OUTPUT_DIR
    exp_foldername = cfg.EXP_NAME

    root_output_dir = Path(output_dir)
    final_output_dir = root_output_dir / exp_foldername

    arr_name = list(cfg.LOSSES.keys())[0]

    title = cfg.EXP_NAME.split('/')[-1]
    meter = utils.metrics.AvgMeter()

    fig_train, ax_train = plt.subplots(1, 1, figsize=(7, 5))

    #meter.load_state(f'{final_output_dir}/metrics/{arr_name}_train.pth')
    meter.load_state(f'{final_output_dir}/metrics/full_train.pth')
    vals = meter.prev_vals
    x_arr = torch.arange(1, len(vals)+1)
    ax_train.plot(x_arr, vals, label=arr_name)

    ax_train.legend(fontsize='x-large')
    plt.suptitle(title+' training')

    fig_valid, ax_valid = plt.subplots(1, 1, figsize=(7, 5))

    #meter.load_state(f'{final_output_dir}/metrics/{arr_name}_valid.pth')
    meter.load_state(f'{final_output_dir}/metrics/full_valid.pth')
    vals = meter.prev_vals
    x_arr = torch.arange(1, len(vals)+1)
    ax_valid.plot(x_arr, vals, label=arr_name)

    ax_valid.legend(fontsize='x-large')
    plt.suptitle(title+' validation')

    for fig, name in zip([fig_train, fig_valid], ['p_loss_train', 'p_loss_valid']):
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        fig.patch.set_facecolor('white')
        fig.tight_layout()
        fig.savefig(f'./plot_{name}_{exp_foldername}.png')


def main():
    # initialize Trainer
    cfg = utils.utils.parse_args().cfg
    trainer = core.trainer.Trainer(cfg)

    # copy yaml description file to the save folder
    #utils.utils.copy_exp_file(trainer)

    # copy proc.py file to the save folder
    #utils.utils.copy_proc_file(trainer)
    print(trainer.cfg.LOSSES.keys())

    for loss_type in trainer.cfg.LOSSES:
        print(trainer.cfg.LOSSES[loss_type])
        print(loss_type)

    loss_visu(trainer.cfg)

    #trainer.logger.info(pprint.pformat(trainer.cfg))
    trainer.logger.info('#'*100)
    


if __name__ == '__main__':
    main()
