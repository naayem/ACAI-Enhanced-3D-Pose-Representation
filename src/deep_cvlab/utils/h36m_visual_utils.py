'''
Contains functions for poses visualizations.
'''
import matplotlib.pyplot as plt

BONES_COLORS = ['red', 'green', 'yellow', 'magenta', 'grey', 'lightblue']

BOUND_3D_POSE = 800 # in mm

BONES_H36M_3D_17 = [
        [[0,4],[4,5],[5,6]], ### left leg
        [[12,13],[11,12],[8,11]], ### left hand
        [[0,1],[1,2],[2,3]], ### right leg
        [[8,14],[14,15],[15,16]], ### right hand
        [[0,7],[7,8]], ### spine
        [[8,9],[9,10],[8,10]] ### head
    ] 

def _check_poses(gt, reconstruction,cfg):

    annotate = True
    rows, cols = 5, 5
    fig_3d_test, ax_3d_test = plt.subplots(rows, cols, figsize=(4*cols,4*rows))
    fig_3d, ax_3d = plt.subplots(rows, cols, figsize=(4*cols,4*rows))
    for i in range(rows):
        for j in range(cols):
            
            print(i,j, end=' ')

            sample_gt = gt[i*cols + j]
            sample_reconstruction = reconstruction[i*cols + j]

            ### pose3d
            pose3d = sample_gt.data.cpu().numpy()
            plot_pose3d(ax_3d[i,j], pose3d, skeleton_bones=BONES_H36M_3D_17, annotate=annotate)

            ### pose3d_test
            pose3d_test = sample_reconstruction.data.cpu().numpy()
            plot_pose3d(ax_3d_test[i,j], pose3d_test, skeleton_bones=BONES_H36M_3D_17, annotate=annotate)
            
            print('Done!')

    for fig, name in zip([fig_3d, fig_3d_test], ['3d', '3dtest']):
        fig.subplots_adjust(wspace=0.01, hspace=0.01)
        fig.patch.set_facecolor('white')
        fig.tight_layout()
        fig.savefig(f'{cfg.OUTPUT_DIR}/{cfg.EXP_NAME}/SAMPLE_h36m_{name}_{cfg.EXP_NAME}.png')

def plot_pose2d(ax, pose2d, mask=None, **kwargs): 
    _plot_pose(ax, pose2d, mask=mask, **kwargs)

def plot_pose3d(ax, pose3d, use_m=True, **kwargs):
    _plot_pose(ax, pose3d, **kwargs)
    
    bound = BOUND_3D_POSE / 1000 if use_m else BOUND_3D_POSE
    ax.set_ylim(bound,-bound)
    ax.set_xlim(-bound,bound)
    ax.set_aspect('equal')    
    ax.set_axis_off()

def _plot_pose(ax, pose, mask=None, skeleton_bones=None, annotate=True):
    
    if mask is None:
        mask = _init_full_mask(len(pose))

    _plot_points(ax, pose, mask)

    if skeleton_bones is not None:
        _plot_bones(ax, pose, mask, skeleton_bones)    

    if annotate:
        _annotate(ax, pose, mask)


def _init_full_mask(num_joints):
    mask = [True] * num_joints
    return mask

def _plot_points(ax, pose, mask):
    ax.plot(pose[mask][:,0], pose[mask][:,1], 'yo', markersize=3)

def _plot_bones(ax, pose, mask, skeleton_bones, colors=BONES_COLORS):
    for bone_group, color in zip(skeleton_bones, colors):
        for bone in bone_group:
            if mask[bone[0]] == 0 or mask[bone[1]] == 0:
                continue
            ax.plot(
                [ pose[bone[0],0], pose[bone[1],0] ], 
                [ pose[bone[0],1], pose[bone[1],1] ], 
                color=color, linewidth=3)

def _annotate(ax, pose, mask):
    for idx, (joint, flag) in enumerate(zip(pose, mask)):
        if flag:
            ax.annotate(idx, (joint[0], joint[1]), color='red', xytext=(0,5), textcoords='offset points',fontsize=8)