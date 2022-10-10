import torch
import torch.nn as nn


class JointsLoss(nn.Module):

    def __init__(self, criterion='L2'):
        super(JointsLoss, self).__init__()
        self.criterion = criterion

    def forward(self, output, target, target_mask=None):

        batch_size = output.size(0)

        output = output.reshape(batch_size,-1,3)
        target = target.reshape(batch_size,-1,3)

        num_joints = output.size(1)

        if self.criterion == 'L2':
            loss = ((output - target)**2).sum() / batch_size / num_joints
        else:
            assert False # TODO
            
        return loss


class MPJPE(nn.Module):
    def __init__(self, pose_in_m=True):
        super(MPJPE, self).__init__()
        self.pose_in_m = pose_in_m

    def forward(self, output, target, target_mask=None):
        batch_size = output.size(0)

        output = output.reshape(batch_size,-1,3)
        target = target.reshape(batch_size,-1,3)

        num_joints = output.size(1)
        
        if self.pose_in_m:
            output = output * 1000.
            target = target * 1000.
        
        loss = torch.sqrt(((output - target)**2).sum(dim=2)).sum() / num_joints / batch_size
        return loss

class MPJPE_PERPOSE(nn.Module):
    def __init__(self, pose_in_m=True):
        super(MPJPE_PERPOSE, self).__init__()
        self.pose_in_m = pose_in_m

    def forward(self, output, target, target_mask=None):
        batch_size = output.size(0)

        output = output.reshape(batch_size,-1,3)
        target = target.reshape(batch_size,-1,3)

        num_joints = output.size(1)
        
        if self.pose_in_m:
            output = output * 1000.
            target = target * 1000.
        
        loss = torch.sqrt(((output - target)**2).sum(dim=2)).sum(dim=1) / num_joints
        return loss

class GetLoss(nn.Module):

    def __init__(self, criterion='L2'):
        super(GetLoss, self).__init__()

    def forward(self, loss):
            
        return loss