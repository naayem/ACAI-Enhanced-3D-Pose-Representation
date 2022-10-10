import numpy as np


# Function "generate_heatmaps" is taken from mpii.py MPII class.

def generate_heatmaps(joints, joints_mask, image_size=(256,256), heatmap_size=(64,64), sigma=1):
    '''
    :param joints:  [num_joints, 3]
    :param joints_mask: [num_joints]
    :return: heatmaps [num_joints x heatmap_height x heatmap_width], joints_mask(1: visible, 0: invisible)
    '''

    num_joints = joints.shape[0]
    heatmaps = np.zeros((num_joints, heatmap_size[1], heatmap_size[0]), dtype=np.float32)
    mask = joints_mask.copy()
    tmp_size = sigma * 3

    for joint_id, (joint, joint_mask) in enumerate(zip(joints, joints_mask)):

        if not joint_mask: # joint is invisible
            continue

        feat_stride = (image_size[1] / heatmap_size[1], image_size[0] / heatmap_size[0])
        mu_x = int(joint[0] / feat_stride[0] + 0.5)
        mu_y = int(joint[1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)] # up-left
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)] # bottom-right

        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            mask[joint_id] = False
            continue

        # # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        heatmaps[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
            g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

    return heatmaps, mask