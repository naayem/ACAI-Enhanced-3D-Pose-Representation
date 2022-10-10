import torch as th
from time import time
from ..procedures_common import status_msg

'''
"proc_basic" - basic training procedure with the following notations:
    - one model, which is called "net"
    - dataloaders, called "train" and "valid" provide datasamples of the dict-like type
    - the loss function, called "loss", computes the error between net output and ground truth, 
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
        batch_size = img.size(0)                             
                                             
        out = trainer.models.net(img)        
        loss = trainer.losses.p_loss(out, gt) 

        trainer.optim.zero_grad()
        loss.backward()
        trainer.optim.step()

        ### measure accuracy and record loss       
        trainer.meters.train.p_loss.update(loss.item(), n=batch_size)
        total_time = time() - absolute_start
        status_msg(trainer, batch_idx, dl_len, trainer.meters.train.p_loss, total_time)
        

def valid(trainer): 
    absolute_start = time()

    dl_len = len(trainer.dataload.valid)
    with th.no_grad():
        for batch_idx, sample in enumerate(trainer.dataload.valid, start=1):

            img = sample['img']
            gt = sample['pose3d']
            gt = gt.to(device=trainer.device0, non_blocking=True)
            batch_size = img.size(0)

            p = trainer.models.net(img)
            loss = trainer.losses.mpjpe(p, gt) 
     
            trainer.meters.valid.mpjpe.update(loss.item(), n=batch_size)
            total_time = time() - absolute_start
            status_msg(trainer, batch_idx, dl_len, trainer.meters.valid.mpjpe, total_time)
            
    perf_indicator = trainer.meters.valid.mpjpe.cur_avg 
    
    return perf_indicator    