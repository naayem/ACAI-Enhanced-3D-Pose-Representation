from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import torch as th

class BestNeighbors(object):
    def __init__(self, batch_size, device):
        self.best_neighbors_meshes = float("Inf") * th.ones(batch_size, 17*3).to(device)
        self.best_neighbors_distances = float("Inf") * th.ones(1, batch_size).to(device) # 1 x batch_size 
        

def update_best_nn(meshes_nn, meshes_target, best_obj, dist_type='mesh-to-mesh'):
    ### "nns" - stands for "nearest neighbors"

    #print(best_obj.best_neighbors_distances)
    #print(best_obj.best_neighbors_distances.shape)

    meshes_nn = meshes_nn.flatten(start_dim=1)
    meshes_target = meshes_target.flatten(start_dim=1)
    #print(meshes_nn.shape, meshes_target.shape)

    cur_dists = compute_dist_matrix(meshes_nn, meshes_target, dist_type=dist_type)
    iter_batch_size, target_batch_size = cur_dists.shape
    #print(cur_dists.shape)

    ### 1. Choose those samples which are closer than current_best
    mask_isCloser = cur_dists < best_obj.best_neighbors_distances 
    mask_isCloser = mask_isCloser.to(cur_dists.device)
    #print(mask_isCloser)

    ### 2. Choose best out of those
    cur_dists_indices_best = th.argmin(cur_dists, dim=0, keepdim=True).to(cur_dists.device)
    onehot = th.zeros(iter_batch_size, target_batch_size).to(cur_dists.device)
    onehot.scatter_(dim=0, index=cur_dists_indices_best, value=1.)
    #print(onehot)

    ### 3. mix the two 
    mask_best_to_update = ( mask_isCloser * onehot ).type(th.BoolTensor).to(cur_dists.device)
    #print(mask_best_to_update)

    y_grid, x_grid = th.meshgrid(th.arange(iter_batch_size), th.arange(target_batch_size))
    nns_to_insert = th.masked_select(y_grid.to(cur_dists.device), mask_best_to_update)
    targets_to_update = th.masked_select(x_grid.to(cur_dists.device), mask_best_to_update)

    #print(cur_dists)
    #print('targets: ', targets_to_update)
    #print('nns: ', nns_to_insert)
    cur_dists = cur_dists.float()
    meshes_nn = meshes_nn.float()
    #print(best_obj.best_neighbors_distances[:,targets_to_update].dtype)
    #print(cur_dists[nns_to_insert,targets_to_update].dtype)

    best_obj.best_neighbors_distances[:,targets_to_update] = cur_dists[nns_to_insert,targets_to_update]
    best_obj.best_neighbors_meshes[targets_to_update] = meshes_nn[nns_to_insert]


def compute_dist_matrix(x,y, dist_type='mesh-to-mesh'):
    if dist_type == 'mesh-to-mesh':
        return th.cdist(x.unsqueeze(0), y.unsqueeze(0))[0]
    elif dist_type == 'vertex-to-vertex':
        raise NotImplementedError
    elif dist_type == 'mpjpe':
        raise NotImplementedError