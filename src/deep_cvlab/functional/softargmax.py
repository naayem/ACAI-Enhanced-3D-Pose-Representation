import torch as th


class IntegralSoftargmax(th.nn.Module):
    '''
    Given prediction tensor ("preds") of size B x num_joints x H x W 
    performs soft argmax operation, based on Integral Human Pose Regression paper.
    Output is of a size B x num_joints x 2.
    '''
    def __init__(self, softmax_off=False):
        super(IntegralSoftargmax, self).__init__()
        self.softmax_off = softmax_off

    def forward(self, preds):
        preds = softargmax_function(preds, self.softmax_off)
        return preds


def softargmax_function(preds, softmax_off=False):
    batch_size, num_joints, hm_height, hm_width = preds.size()
    preds = preds.reshape((batch_size, num_joints, hm_height, hm_width))
    preds = preds.reshape((batch_size, num_joints, -1))

    if not softmax_off:
        preds = th.nn.functional.softmax(preds, 2) 

    preds = preds.reshape((batch_size, num_joints, hm_height, hm_width))

    # integrate along every axis
    x = preds.sum(dim=2)
    y = preds.sum(dim=3)

    # compute expectation
    x, y = x * th.arange(hm_width).to(x.device), y * th.arange(hm_height).to(y.device)
    x,y = x.sum(dim=2, keepdim=True), y.sum(dim=2, keepdim=True)

    # normalize predictions to the range [0,1]
    x, y = x / float(hm_width), y / float(hm_height)

    preds = th.cat((x, y), dim=2)
    preds = preds.reshape((batch_size, num_joints, 2))

    return preds