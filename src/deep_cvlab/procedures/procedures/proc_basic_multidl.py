import torch as th
from time import time
from ..procedures_common import status_msg

def train(trainer):
    absolute_start = time()

    dl_len = len(trainer.dataload.multidl)
    for batch_idx, (annot, unannot) in enumerate(trainer.dataload.multidl, start=1):

        batch_size = annot['pose3d'].size(0)  

        img, p_gt = annot['img'], annot['pose3d']
        p_gt = p_gt.to(device=trainer.device0, non_blocking=True)

        ### 1 annotated loss on Pose Network: L (p, p_gt)
        p = trainer.models.net(img)
        loss = trainer.losses.p_loss(p, p_gt)

        ### 2 dummy inference of unannot sample to update batchnorms, no loss
        img = unannot['img']
        p = trainer.models.net(img)

        trainer.optim.zero_grad()
        loss.backward()
        trainer.optim.step()

        ### measure accuracy and record loss       
        trainer.meters.train.p_loss.update(loss.item(), n=batch_size)
        total_time = time() - absolute_start
        status_msg(trainer, batch_idx, dl_len, trainer.meters.train.p_loss, total_time) 


from .proc_basic import valid