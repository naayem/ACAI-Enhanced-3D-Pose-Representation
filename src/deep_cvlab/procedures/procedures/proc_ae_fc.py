import torch
from time import time
from ..procedures_common import status_msg

'''
"proc_ae_fc" -  training procedure for an autoencoder model with the following notations:
    - one model, which is called "ae"
    - model "ae" outputs "internals" dict composed of "code" and "reconstructed"
    - dataloaders, called "train" and "valid" provide datasamples of the dict-like type
    - the loss function, called "ae_loss", computes the error between ae output and ground truth, 
            in both training and validation modes

    This procedure considers the simplest case, where network does simple inference img -> gt and
    loss is computed as loss(out, gt). 
'''

def train(trainer):
    absolute_start = time() 

    dl_len = len(trainer.dataload.train)
    for batch_idx, sample in enumerate(trainer.dataload.train, start=1):

        gt = sample['pose3d'] 
        gt = gt.to(device=trainer.device0, non_blocking=True) 
        batch_size = gt.size(0)                         
                                             
        out = trainer.models.ae(gt)
        reconstruction = out["reconstructed"]
        loss = trainer.losses.ae_loss(reconstruction, gt) 

        trainer.optim.zero_grad()
        loss.backward()
        trainer.optim.step()

        ### measure accuracy and record loss       
        trainer.meters.train.ae_loss.update(loss.item(), n=batch_size)
        total_time = time() - absolute_start
        status_msg(trainer, batch_idx, dl_len, trainer.meters.train.ae_loss, total_time)

def valid(trainer): 
    absolute_start = time()

    dl_len = len(trainer.dataload.valid)
    with torch.no_grad():
        for batch_idx, sample in enumerate(trainer.dataload.valid, start=1):

            gt = sample['pose3d']
            gt = gt.to(device=trainer.device0, non_blocking=True)
            batch_size = gt.size(0)

            p = trainer.models.ae(gt)
            reconstruction = p["reconstructed"]
            loss = trainer.losses.ae_loss(reconstruction, gt)

            mpjpe_loss = trainer.losses.mpjpe_loss(gt, reconstruction)

            trainer.meters.valid.ae_loss.update(loss.item(), n=batch_size)
            trainer.meters.valid.mpjpe_loss.update(mpjpe_loss.item(), n=batch_size)
            total_time = time() - absolute_start
            status_msg(trainer, batch_idx, dl_len, trainer.meters.valid.ae_loss, total_time)
            status_msg(trainer, batch_idx, dl_len, trainer.meters.valid.mpjpe_loss, total_time)

    perf_indicator = trainer.meters.valid.ae_loss.cur_avg

    return perf_indicator