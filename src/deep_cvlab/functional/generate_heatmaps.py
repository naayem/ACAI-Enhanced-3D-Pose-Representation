import torch as th

class GenerateHeatmaps(th.nn.Module):
    '''Provides description and differentiable execution of generating heatmaps given joints tensor and visibility mask.'''
    def __init__(self, image_size=(256,256), heatmap_size=(64,64), sigma=1):
        super(GenerateHeatmaps, self).__init__()
        self.image_size = image_size
        self.heatmap_size = heatmap_size
        self.sigma = sigma

    def forward(self, joints, joints_mask):
        return GenerateHeatmapsFunction.apply(joints, joints_mask, self.image_size, self.heatmap_size, self.sigma)

    def __repr__(self):
        return f'GenerateHeatmaps Module\nimage_size {self.image_size}, heatmap_size {self.heatmap_size}, sigma {self.sigma}'


class GenerateHeatmapsFunction(th.autograd.Function):
    '''Function generates heatmaps by joints tensor and visibility mask.'''
    @staticmethod
    def forward(ctx, joints, joints_mask, image_size, heatmap_size, sigma):
        ''' joints : B x J x 2
            joints_mask : B x J (True/False values)

            return
            heatmaps : B x J x H x W
        '''    
        ctx.heatmap_size = heatmap_size
        ctx.sigma = sigma
        
        heatmaps, mask, joints_h, feat_stride = generate_heatmaps_function(joints, joints_mask, image_size, heatmap_size, sigma)
        ctx.feat_stride = feat_stride
        ctx.save_for_backward(heatmaps, joints_h)

        return heatmaps
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        grad_output : B x J x H x W (size of output tensor of the .forward() method)
        """
        heatmaps, joints_h = ctx.saved_tensors
        grad_joints = None
        
        if ctx.needs_input_grad[0]: # if joints.requires_grad is True
            hmap_height, hmap_width = ctx.heatmap_size[0], ctx.heatmap_size[1]
            device = heatmaps.device
            x_arr = th.arange(0, hmap_width, 1, dtype=th.float32).view(1,1,1,-1).type_as(heatmaps).to(device)
            y_arr = th.arange(0, hmap_height, 1, dtype=th.float32).view(1,1,-1,1).type_as(heatmaps).to(device)

            x_arr = x_arr - joints_h[:,:,0].unsqueeze(-1).unsqueeze(-1) # xi - x, shape is B x J x 1 x hmap_width
            y_arr = y_arr - joints_h[:,:,1].unsqueeze(-1).unsqueeze(-1) # yi - y, shape is B x J x hmap_height x 1

            grads = []
            for arr in [x_arr, y_arr]:
                grad = grad_output * heatmaps * arr / (ctx.sigma**2) # shape is B x J x hmap_height x hmap_width
                grad = grad.sum(dim=-1).sum(dim=-1).unsqueeze(2) # shape is B x J x 1
                grads.append(grad)
            
            grad_joints = th.cat(grads, dim=2)
            grad_joints = grad_joints / ctx.feat_stride # compensate for joints scaling in the forward
            
        return grad_joints, None, None, None, None


def generate_heatmaps_function(joints, joints_mask, image_size=(256,256), heatmap_size=(64,64), sigma=1):
    '''
    :param joints:  [B x num_joints x 2] - 2D keypoints of a pose, coordinates w.r.t. the image size.
    :param joints_mask: [B x num_joints] - visibility mask, True/False array
    :return: 
    heatmaps [B x num_joints x heatmap_height x heatmap_width]
    joints_mask [B x num_joints] - updated mask (1: visible, 0: invisible)
    joints_h [B x num_joints x 2] - heatmap coordinates of joints, centers of gaussians
    '''

    img_height, img_width = image_size[0], image_size[1]
    hmap_height, hmap_width = heatmap_size[0], heatmap_size[1]
    batch_size = joints.size(0)
    num_joints = joints.size(1)
    joints = joints.view(batch_size * num_joints, 2)
    joints_mask = joints_mask.view(batch_size*num_joints)
    heatmaps = th.zeros(batch_size*num_joints, hmap_height, hmap_width).type_as(joints)

    # padded heatmaps
    pad_size = int(sigma * 3)
    heatmaps = th.nn.functional.pad(heatmaps, [pad_size] * 4, mode='constant', value=0)

    # assume that stride is same for height and width
    feat_stride = img_width / hmap_width
    joints_h = th.round((joints / feat_stride)) # shape: [num_joints, 2]

    mask = (joints_h[:,0] >= hmap_width + pad_size) + (joints_h[:,1] >= hmap_height + pad_size) + \
           (joints_h[:,0] < -pad_size) + (joints_h[:,1] < -pad_size)
    mask = ~mask # now if True, then joint is valid
    mask = joints_mask * mask

    # put ones in "joints_h" coordinates on padded heatmaps
    i, x, y = th.arange(len(mask))[mask], joints_h[:,0][mask].long(), joints_h[:,1][mask].long()
    heatmaps[i, y+pad_size, x+pad_size] = 1.
    heatmaps = heatmaps.view(batch_size, num_joints, heatmaps.size(-2), heatmaps.size(-1))

    joints_h = joints_h.view(batch_size, num_joints, 2)
    mask = mask.view(batch_size, num_joints)

    # convolve binary heatmap with gaussian kernel (to smooth heatmaps)
    heatmaps = convolve_gaussian(heatmaps, sigma=sigma)

    return heatmaps, mask, joints_h, feat_stride


def convolve_gaussian(input, sigma=1.):
    '''
    input : input tensor, shape B x C x H x W
    
    return values
    output : output tensor, shape B x C x (H-s) x (W-s), result of convolution input with gaussian kernel.
            (s == kernel_size // 2)
    '''

    channels = input.size(1)
    dtype = input.dtype
    # Generate gaussian kernel
    tmp_size = sigma * 3 
    kernel_size = int(2 * tmp_size + 1)
    g, g_sum = get_gaussian_kernel(kernel_size, sigma, dtype=dtype)
    g = g.type_as(input)
    g = g.view(1, 1, kernel_size, kernel_size)
    g = g.repeat(channels, 1, 1, 1)

    # create a Conv2d filter
    gaussian_filter = th.nn.Conv2d(in_channels=channels, out_channels=channels,
                            kernel_size=kernel_size, groups=channels, bias=False)
    gaussian_filter.weight.data = g
    gaussian_filter.weight.requires_grad = False

    # convolve input with gaussian kernel
    input = gaussian_filter(input)
    return input


def get_gaussian_kernel(kernel_size, sigma, dtype=th.float32):
    ''' Returns a kernel of the size (kernel_size, kernel_size)
    '''
    x = th.arange(0, kernel_size, 1, dtype=dtype).view(-1,1)
    y = th.arange(0, kernel_size, 1, dtype=dtype).view(1,-1)
    x0 = y0 = kernel_size // 2
    g = th.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2)) 
    g_sum = g.sum() 
    g = g / g_sum # the kernel is the normalized distribution now
    return g, g_sum


def main():
    return


if __name__ == '__main__':
    main()
