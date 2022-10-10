import torch
from time import time

from ..procedures_common import status_msg, status_msg_acai
from ...losses.acai_loss import AcaiOutputs, AcaiOutputsRandom
from ...core.trainer import Trainer

'''
"proc_ae_acai" - training procedure for autoencoder with acai regularisation with the following notations:
    - two model, which are called "ae" and "critic"
    - model "ae" outputs "internals" dict composed of "code" and "reconstructed"
    - dataloaders, called "train" and "valid" provide datasamples of the dict-like type
    - the loss function, called "loss", computes the error between net output and ground truth, 
            in both training and validation modes

    This procedure considers the simplest case, where network does simple inference img -> gt and
    loss is computed as loss(out, gt). 
'''

def train(trainer: Trainer):
    absolute_start = time() 
    acai_outputs = AcaiOutputsRandom()
    dl_len = len(trainer.dataload.train)
    for batch_idx, sample in enumerate(trainer.dataload.train, start=1):
        gt = sample['pose3d'] 
        gt = gt.to(device=trainer.device0, non_blocking=True)
        batch_size = gt.size(0)
        gt1, gt2 =torch.split(gt, int(batch_size/2))
        assert gt1.shape == gt2.shape

        reconstruction, alpha, reconstruction_interpolated = acai_outputs(gt1, gt2, trainer)
        
        ae_loss, reconstruction_loss, critic_fooling_loss, ae_reg_coef = trainer.losses.ae_loss(gt1, reconstruction, reconstruction_interpolated, trainer)

        # AE's parameters update
        trainer.optim.opts.ae.zero_grad()
        ae_loss.backward(retain_graph=True)
        trainer.optim.opts.ae.step()

        reconstruction = reconstruction.detach()
        reconstruction_interpolated = reconstruction_interpolated.detach()

        critic_loss, alpha_guessing_loss, realistic_loss = trainer.losses.c_loss(gt1, reconstruction, alpha, reconstruction_interpolated, trainer)

        # Critic's parameters update
        trainer.optim.opts.critic.zero_grad()
        critic_loss.backward()
        trainer.optim.opts.critic.step()

        ### measure accuracy and record loss             
        trainer.meters.train.ae_loss.update(ae_loss.item(), n=batch_size)
        trainer.meters.train.c_loss.update(critic_loss.item(), n=batch_size)
        trainer.meters.train.aeSplitTrain_loss.update(reconstruction_loss.item(), n=batch_size)
        trainer.meters.valid.aeSplitTrain_loss.update((critic_fooling_loss).item(), n=batch_size)
        total_time = time() - absolute_start
        status_msg(trainer, batch_idx, dl_len, trainer.meters.train.ae_loss, total_time)
        status_msg_acai(trainer, batch_idx, dl_len, trainer.meters.train.ae_loss, trainer.meters.train.aeSplitTrain_loss, trainer.meters.valid.aeSplitTrain_loss)
        status_msg(trainer, batch_idx, dl_len, trainer.meters.train.c_loss, total_time)


def valid(trainer): 
    absolute_start = time()
    acai_outputs = AcaiOutputsRandom()

    dl_len = len(trainer.dataload.valid)
    with torch.no_grad():
        for batch_idx, sample in enumerate(trainer.dataload.valid, start=1):
            gt = sample['pose3d'] 
            gt = gt.to(device=trainer.device0, non_blocking=True) 
            batch_size = gt.size(0)
            gt1, gt2 =torch.split(gt, int(batch_size/2))
            assert gt1.shape == gt2.shape                         

            reconstruction, alpha, reconstruction_interpolated = acai_outputs(gt1, gt2, trainer)

            ae_loss, reconstruction_loss, critic_fooling_loss, ae_reg_coef = trainer.losses.ae_loss(gt1, reconstruction, reconstruction_interpolated, trainer)
            critic_loss, alpha_guessing_loss, realistic_loss = trainer.losses.c_loss(gt1, reconstruction, alpha, reconstruction_interpolated, trainer)

            mpjpe_loss = trainer.losses.mpjpe_loss(gt1, reconstruction)

            trainer.meters.valid.ae_loss.update(ae_loss.item(), n=batch_size)
            trainer.meters.valid.c_loss.update(critic_loss.item(), n=batch_size)
            trainer.meters.valid.mpjpe_loss.update(mpjpe_loss.item(), n=batch_size)
            trainer.meters.train.aeSplitValid_loss.update(reconstruction_loss.item(), n=batch_size)
            trainer.meters.valid.aeSplitValid_loss.update((critic_fooling_loss).item(), n=batch_size)
            total_time = time() - absolute_start
            status_msg(trainer, batch_idx, dl_len, trainer.meters.valid.ae_loss, total_time)
            status_msg(trainer, batch_idx, dl_len, trainer.meters.valid.c_loss, total_time)
            status_msg(trainer, batch_idx, dl_len, trainer.meters.valid.mpjpe_loss, total_time)
            status_msg_acai(trainer, batch_idx, dl_len, trainer.meters.valid.ae_loss, trainer.meters.train.aeSplitValid_loss, trainer.meters.valid.aeSplitValid_loss)

    perf_indicator = trainer.meters.valid.ae_loss.cur_avg

    return perf_indicator