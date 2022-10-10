import torch as th

JOINTS_INDICES = {
    'h36m16':list(range(16)),
    'h36m17': list(range(17)),
}

JOINTS_NAMES = {
    'h36m16': 
        ['pelvis','rhip','rknee','rankle','lhip','lknee','lankle','spine','neck',
            'tophead','lshould','lelbow','lwrist','rshould','relbow','rwrist',],
    'h36m17': 
        ['pelvis','rhip','rknee','rankle','lhip','lknee','lankle','spine','neck','frontface',
            'tophead','lshould','lelbow','lwrist','rshould','relbow','rwrist',], 
}

BONES_NAMES = {
    'h36m16': [
        'botspine', 'topspine', 'head', 
        'lpelvis', 'lhip', 'lleg', 'lcollar', 'lshoulder', 'lforearm', 
        'rpelvis', 'rhip', 'rleg', 'rcollar', 'rshoulder', 'rforearm', 
        ],
    'h36m17': [
        'botspine', 'topspine', 'head', 'face',
        'lpelvis', 'lhip', 'lleg', 'lcollar', 'lshoulder', 'lforearm', 
        'rpelvis', 'rhip', 'rleg', 'rcollar', 'rshoulder', 'rforearm', 
        ],
}

ANGLES_NAMES = {
    'h36m16': [
        'botspine-topspine', 'topspine-head', 
        'botspine-lpelvis', 'lpelvis-lhip', 'lhip-lleg', 
        'topspine-lcollar', 'lcollar-lshoulder', 'lshoulder-lforearm',
        'botspine-rpelvis', 'rpelvis-rhip', 'rhip-rleg',
        'topspine-rcollar', 'rcollar-rshoulder', 'rshoulder-rforearm',
               ],
    'h36m17': [
        'botspine-topspine', 'topspine-head', 'head-face',
        'botspine-lpelvis', 'lpelvis-lhip', 'lhip-lleg', 
        'topspine-lcollar', 'lcollar-lshoulder', 'lshoulder-lforearm',
        'botspine-rpelvis', 'rpelvis-rhip', 'rhip-rleg',
        'topspine-rcollar', 'rcollar-rshoulder', 'rshoulder-rforearm',
               ],
}

ROOT_JOINT_NAME = { 'h36m16': 'pelvis', 'h36m17': 'pelvis' }
BASE_BONE_NAME = { 'h36m16': 'botspine', 'h36m17': 'botspine' }

def get_bones_indices(skeleton, joints_indices, joints_names):
    '''
    All bones are directed: they start from the BASE_BONE ("botspine")
    A pattern is the following: { 'bone name' : [start of the bone, end of the bone] }
    '''
    if skeleton == 'h36m16':
        bones_indices = {
            # center skeleton
            'botspine': [joints_indices[joints_names.index('pelvis')], joints_indices[joints_names.index('spine')]],
            'topspine': [joints_indices[joints_names.index('spine')], joints_indices[joints_names.index('neck')]],
            'head':     [joints_indices[joints_names.index('neck')],   joints_indices[joints_names.index('tophead')]],
            # left
            'lcollar':  [joints_indices[joints_names.index('neck')],   joints_indices[joints_names.index('lshould')]],
            'lshoulder':[joints_indices[joints_names.index('lshould')],joints_indices[joints_names.index('lelbow')]],
            'lforearm': [joints_indices[joints_names.index('lelbow')], joints_indices[joints_names.index('lwrist')]],
            'lpelvis':  [joints_indices[joints_names.index('pelvis')], joints_indices[joints_names.index('lhip')]],
            'lhip':     [joints_indices[joints_names.index('lhip')],   joints_indices[joints_names.index('lknee')]],
            'lleg':     [joints_indices[joints_names.index('lknee')],  joints_indices[joints_names.index('lankle')]],
            # right
            'rcollar':  [joints_indices[joints_names.index('neck')],   joints_indices[joints_names.index('rshould')]],
            'rshoulder':[joints_indices[joints_names.index('rshould')],joints_indices[joints_names.index('relbow')]],
            'rforearm': [joints_indices[joints_names.index('relbow')], joints_indices[joints_names.index('rwrist')]],
            'rpelvis':  [joints_indices[joints_names.index('pelvis')], joints_indices[joints_names.index('rhip')]],
            'rhip':     [joints_indices[joints_names.index('rhip')],   joints_indices[joints_names.index('rknee')]],
            'rleg':     [joints_indices[joints_names.index('rknee')],  joints_indices[joints_names.index('rankle')]],
        }   
        
    if skeleton == 'h36m17':
        bones_indices = {
            # center skeleton
            'botspine': [joints_indices[joints_names.index('pelvis')], joints_indices[joints_names.index('spine')]],
            'topspine': [joints_indices[joints_names.index('spine')], joints_indices[joints_names.index('neck')]],
            'head':     [joints_indices[joints_names.index('neck')],   joints_indices[joints_names.index('tophead')]],
            'face':     [joints_indices[joints_names.index('tophead')],   joints_indices[joints_names.index('frontface')]],
            # left
            'lcollar':  [joints_indices[joints_names.index('neck')],   joints_indices[joints_names.index('lshould')]],
            'lshoulder':[joints_indices[joints_names.index('lshould')],joints_indices[joints_names.index('lelbow')]],
            'lforearm': [joints_indices[joints_names.index('lelbow')], joints_indices[joints_names.index('lwrist')]],
            'lpelvis':  [joints_indices[joints_names.index('pelvis')], joints_indices[joints_names.index('lhip')]],
            'lhip':     [joints_indices[joints_names.index('lhip')],   joints_indices[joints_names.index('lknee')]],
            'lleg':     [joints_indices[joints_names.index('lknee')],  joints_indices[joints_names.index('lankle')]],
            # right
            'rcollar':  [joints_indices[joints_names.index('neck')],   joints_indices[joints_names.index('rshould')]],
            'rshoulder':[joints_indices[joints_names.index('rshould')],joints_indices[joints_names.index('relbow')]],
            'rforearm': [joints_indices[joints_names.index('relbow')], joints_indices[joints_names.index('rwrist')]],
            'rpelvis':  [joints_indices[joints_names.index('pelvis')], joints_indices[joints_names.index('rhip')]],
            'rhip':     [joints_indices[joints_names.index('rhip')],   joints_indices[joints_names.index('rknee')]],
            'rleg':     [joints_indices[joints_names.index('rknee')],  joints_indices[joints_names.index('rankle')]],
        }   

    return bones_indices


########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################
########################################################################################################################################################

from math import pi

def getOrthogSpace(v0):
    vx2vy2 = th.nn.functional.relu(1. - v0[:,2]**2).view(-1,1)
    sqrt = th.sqrt(vx2vy2)
    # assert (sqrt > 0).sum() > 0 # always gets this error, let's see how it would behave without this error...

    v1 = th.cat((
        (v0[:,0]*v0[:,2]).view(-1,1), 
        (v0[:,1]*v0[:,2]).view(-1,1), 
        -vx2vy2,
                ), dim=1) / sqrt
    v2 = th.cat((
        -v0[:,1].view(-1,1), 
        v0[:,0].view(-1,1), 
        th.zeros((v0.size(0),1), dtype=v0.dtype, device=v0.device)
                ), dim=1) / sqrt
    
    return v1, v2


def rotateRodrigues(vec_start, axis, angle):
    '''here we apply Rodrigues' rotation formula: 
        https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
        It should be noted that axis is chosen in such a way that it is orthogonal to 
        both vec_start and vec_end, so rotation is being done in the plane, which
        is orthogonal to the axis.
        params:
            vec_start : start of rotation
            angle : angle of rotation (theta), computed acc. to axis-angle notation
    '''
    cross = th.cross(axis, vec_start, dim=1) # shape: B x 3
    angle = angle.view(-1,1)
    sin, cos = th.sin(angle), th.cos(angle)
    vec_end = cos * vec_start + sin * cross
    return vec_end


def getRotation(v0, axisangle, angle):
    ''' 
    params:
        v0 : B x 3,
        axisangle : B x 1,
        angle : B x 1
    '''
    axisangle = axisangle.view(-1,1)
    v1, v2 = getOrthogSpace(v0)
    axis = th.cos(axisangle) * v1 + th.sin(axisangle) * v2
    v = rotateRodrigues(v0, axis, angle)
    return v


def restorePhi(sin, cos, tol=1e-6):
    ''' Shape of sin (cos) is B x 1
    Function restores the angle given its sine and cosine.
    '''
    mask1 = sin >= 0 # 1st and 2nd quadrants
    mask2 = (sin < 0) & (cos >= 0) # 4th quadrant
    mask3 = (sin < 0) & (cos < 0) # 3rd quadrant
    
    phi = th.empty((sin.size(0), 1), dtype=sin.dtype, device=sin.device)
    phi[mask1] = th.acos(cos[mask1]) 
    phi[mask2] = 2*pi + th.asin(sin[mask2]) 
    phi[mask3] = pi - th.asin(sin[mask3]) 

    return phi


def getAxisAngle(vec1, vec2, tol=1e-6):
    ''' vec1 is the start of rotation.
    Shape of vec1 (and vec2) is B x 3.
    Given two vectors, start and end, computes the angle of rotation and the axis of rotation.
    Returns the axis of rotation in the angular representation (phi) and the angle of rotation (theta).

    '''
    vec1 = vec1 / th.norm(vec1, dim=1).view(-1,1)
    vec2 = vec2 / th.norm(vec2, dim=1).view(-1,1)
    cosine = (vec1 * vec2).sum(dim=1) # shape: B
    cosine = th.nn.functional.hardtanh(cosine, -1., 1.)
    theta = th.acos(cosine).view(-1,1)

    axis = th.cross(vec1, vec2, dim=1) # shape: B x 3
    v1_ort, v2_ort = getOrthogSpace(vec1)
    mask = (1. - cosine > tol) & (cosine + 1. > tol) # else the angle is too small/big, no matter where the axis is directed
    axis[mask] = axis[mask] / th.norm(axis[mask], dim=1).view(-1,1) # axis has shape: B x 3
    axis[~mask] = (v1_ort[~mask] + v2_ort[~mask])*float(th.cos(th.tensor([pi/4])))

    cos = (v1_ort * axis).sum(dim=1).view(-1,1)
    sin = (v2_ort * axis).sum(dim=1).view(-1,1)

    cos = th.nn.functional.hardtanh(cos, -1., 1.)
    sin = th.nn.functional.hardtanh(sin, -1., 1.)
    
    phi = restorePhi(sin, cos)

    return phi, theta


def angle2Skeleton(out, skeleton):
    ''' Mapping from axis-angle representation into 3D keypoints.
    
    Shape of input is B x N, 
    N = root (3) + base (3) + bones_unique (#bones) + 
                    axes_angles (#angles) + angles (#angles)

    It is assumed that bones_unique == 3 + #leftBones TODO not true anymore for a new skeleton h36m17
    '''

    joints_names = JOINTS_NAMES[skeleton].copy()
    bones_indices = get_bones_indices(skeleton, JOINTS_INDICES[skeleton], joints_names).copy()
    bones_names = BONES_NAMES[skeleton].copy()
    angles_names = ANGLES_NAMES[skeleton].copy()
    root_joint_name = ROOT_JOINT_NAME[skeleton]
    base_bone_name = BASE_BONE_NAME[skeleton]

    root_joint = out[:,:3]
    base = out[:,3:6]
    base_joint = root_joint + base
    joints = [root_joint, base_joint]
    
    base_joint_name = joints_names[bones_indices[base_bone_name][1]]
    joints_names_attached = [root_joint_name, base_joint_name]
    
    bones_vecs = [base / th.norm(base, dim=1).view(-1,1)]
    bones_names_attached = [base_bone_name]
    
    ### TODO previous version - how to work with UNIQUE bones only?
    ### bone_lengths = out[:,6:14]
    ### bone_lengths = th.cat((bone_lengths, bone_lengths[:,2:]), dim=1).transpose(0,1)
    ### # now the number of bonelengths is equal to number of angles
    ### axes_angles = out[:,14:28].transpose(0,1)
    ### rot_angles = out[:,28:].transpose(0,1)

    bones_num = len(bones_names)-1 ### all bones without a "base_bone" only
    bone_lengths = out[:,6:6+bones_num].transpose(0,1)
    axes_angles = out[:,6+bones_num:-len(angles_names)].transpose(0,1)
    rot_angles = out[:,-len(angles_names):].transpose(0,1)

    for angle_name, b_length, phi, theta in zip(angles_names, bone_lengths, axes_angles, rot_angles): 
        bone1_name, bone2_name = angle_name.split('-')
        joint_start_idx, joint_end_idx = bones_indices[bone2_name]
        joint_start_name = joints_names[joint_start_idx]
        joint_end_name = joints_names[joint_end_idx]
        joint_start = joints[joints_names_attached.index(joint_start_name)]

        bone1_vec = bones_vecs[bones_names_attached.index(bone1_name)]
        bone2_vec = getRotation(bone1_vec, phi, theta)
        
        bones_vecs.append(bone2_vec)
        bones_names_attached.append(bone2_name)
        
        joint = joint_start + bone2_vec * b_length.view(-1,1)
        joints.append(joint)
        joints_names_attached.append(joint_end_name)
    
    joints = [joints[joints_names_attached.index(name)] for name in joints_names]
    joints = th.stack(joints).transpose(0,1)
    
    return joints


def skeleton2Angle(joints, skeleton):
    '''
    Mapping from the 3D keypoints representation into the axis-angle representation.

    Input pose "joints" has shape of B x num_joints x 3.
    '''

    joints_names = JOINTS_NAMES[skeleton].copy()
    bones_names = BONES_NAMES[skeleton].copy()
    bones_indices = get_bones_indices(skeleton, JOINTS_INDICES[skeleton], joints_names).copy()
    angles_names = ANGLES_NAMES[skeleton].copy()
    root_joint_name = ROOT_JOINT_NAME[skeleton]
    base_bone_name = BASE_BONE_NAME[skeleton]

    bone_lengths, bone_vectors = {}, {}
    for bone_name, bone_ids in bones_indices.items():
        b_vec = joints[:, bone_ids[1]] - joints[:, bone_ids[0]]
        bone_vectors[bone_name] = b_vec ### shape: B x 3
        bone_lengths[bone_name] = th.norm(b_vec, dim=1) ### shape: B

    ### only root and base_vec variables depend on the structure of the chosen skeleton
    root = joints[:, joints_names.index(root_joint_name)] ### the only full coordinate in the skeleton # shape: B x 3
    base_vec = bone_vectors[base_bone_name] ### shape: B x 3

    phis, thetas = {}, {}

    for angle_name in angles_names:
        bone1, bone2 = angle_name.split('-')
        phi, theta = getAxisAngle(bone_vectors[bone1], bone_vectors[bone2])
        thetas[angle_name] = theta
        phis[angle_name] = phi

    phis = th.stack(list(phis.values())).transpose(0,1).squeeze(dim=2)
    thetas = th.stack(list(thetas.values())).transpose(0,1).squeeze(dim=2)

    ### TODO how to process unique bones correctly?
    ### ??? bones_unique = bones_names[1:-6] ### unique bones that lie in the output vector # bones_names was for h36m16 only!
    ### ??? bone_lengths = th.stack([bone_lengths[bone_name] for bone_name in bones_unique]).transpose(0,1)

    bones_names.remove(base_bone_name) # as we keep parameters of the base bone in "base_vec" part
    bone_lengths = th.stack([bone_lengths[bone_name] for bone_name in bones_names]).transpose(0,1)

    out = th.cat((root, base_vec, bone_lengths, phis, thetas), dim=1)
    return out


def _check_diff(pose3d, skeleton, tol=1e-10):
    ### compute axis-angle representation from a 3D pose skeleton, transform pose back to 3D and computes the difference
    pose3d = pose3d.double()
    pose_angle = skeleton2Angle(pose3d, skeleton)
    pose3d_restored = angle2Skeleton(pose_angle, skeleton)
    diff_abs = th.abs(pose3d_restored - pose3d)
    diff_rel = diff_abs / (pose3d + tol)
    print(f'{skeleton}: abs.diff {float(diff_abs.sum()):10.3E}, rel.diff {float(diff_rel.sum()):10.3E}')

def main():

    ### skeleton h36m, 16 keypoints (version without a forefront face keypoint)
    skeleton = 'h36m16'

    ### sample from H36M dataset (1 x 16 x 3)
    pose3d = th.tensor([[
         [  0.0000,   0.0000,   0.0000],
         [-119.9711,  -38.4048,   42.5127],
         [-164.4613,  394.9728,  -37.2378],
         [-256.5039,  835.1566,   26.5576],
         [ 119.9713,   38.4048,  -42.5132],
         [  87.5145,  424.0194,  172.9014],
         [ 202.4263,  751.3016,  466.1338],
         [  30.1819, -231.1406,  -13.1982],
         [  43.3224, -484.0849,  -57.1919],
         [  21.6621, -672.1628,  -65.3765],
         [ 170.0854, -410.1992,  -93.0161],
         [ 254.2957, -154.1438,  -21.4624],
         [ 107.5460, -126.3056, -224.0933],
         [ -80.6150, -424.6605,    5.4082],
         [-154.4504, -157.9392,   39.8955],
         [-160.7424, -353.5971, -118.3633]]]) + 10.
    
    _check_diff(pose3d, skeleton)

    ### just another pose from h36m (16)
    pose3d = th.tensor([[
        [ 0.0000,  0.0000,  0.0000],
        [ 0.1174, -0.0069,  0.0620],
        [ 0.0777,  0.3973,  0.2387],
        [ 0.2260,  0.8264,  0.2511],
        [-0.1174,  0.0069, -0.0620],
        [-0.0842,  0.4400,  0.0244],
        [-0.0351,  0.8892,  0.0708],
        [ 0.0551, -0.2065, -0.0940],
        [ 0.0701, -0.4546, -0.1595],
        [ 0.0105, -0.6208, -0.0779],
        [-0.0565, -0.4028, -0.2235],
        [-0.2820, -0.5489, -0.1489],
        [-0.0329, -0.5147, -0.1594],
        [ 0.1987, -0.4130, -0.0920],
        [ 0.2919, -0.5678,  0.1205],
        [ 0.1233, -0.5390, -0.0642]]]) + 10.
    
    _check_diff(pose3d, skeleton)


    ### skeleton h36m, 17 keypoints (original)
    skeleton = 'h36m17'
    
    ### sample from H36M dataset (1 x 17 x 3)
    pose3d = th.tensor([[
            [ 0.0000,  0.0000,  0.0000],
            [ 0.1174, -0.0069,  0.0620],
            [ 0.0777,  0.3973,  0.2387],
            [ 0.2260,  0.8264,  0.2511],
            [-0.1174,  0.0069, -0.0620],
            [-0.0842,  0.4400,  0.0244],
            [-0.0351,  0.8892,  0.0708],
            [ 0.0551, -0.2065, -0.0940],
            [ 0.0701, -0.4546, -0.1595],
            [ 0.0159, -0.5067, -0.0645],
            [ 0.0105, -0.6208, -0.0779],
            [-0.0565, -0.4028, -0.2235],
            [-0.2820, -0.5489, -0.1489],
            [-0.0329, -0.5147, -0.1594],
            [ 0.1987, -0.4130, -0.0920],
            [ 0.2919, -0.5678,  0.1205],
            [ 0.1233, -0.5390, -0.0642]]]) + 10.

    _check_diff(pose3d, skeleton)

    return

if __name__ == '__main__':
    main()
