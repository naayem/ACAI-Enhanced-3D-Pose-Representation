'''
Contains functions for poses visualizations.
'''

BONES_COLORS = ['red', 'green', 'yellow', 'magenta', 'grey', 'lightblue']

BOUND_3D_POSE = 800 # in mm

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