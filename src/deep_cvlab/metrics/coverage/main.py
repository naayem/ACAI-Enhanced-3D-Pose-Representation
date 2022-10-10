import yaml
import os
from easydict import EasyDict as edict

# from coverage import run, plot_distances__dist_coverage, render_frames__dist_coverage

# import sys
# sys.path.append('/cvlabdata2/home/davydov/smplprior/smplprior/output/vposer_vae+gan/src')
# import utils

import sys
sys.path.append('/cvlabdata2/home/davydov/smplprior/smplprior/output/vposer_vae+gan')
import src

from src.coverage.coverage import run, plot_distances__dist_coverage, render_frames__dist_coverage

if __name__ == '__main__':

    cfg_path = src.utils.parse_args().cfg
    with open(cfg_path) as f:
        cfg = edict(yaml.load(f, Loader=yaml.FullLoader))

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    ### run pose coverage experiments
    for dset in cfg.DATASETS:
        for net in cfg.NETS:
            print(dset, net)
            print(f'Experiment "{cfg.COVERAGE_TYPE}, {dset}, {net}" started...')
            run(cfg, dset, net)

    ### plot sorted distances with statistics
    print('Plotting sorted distances with statistics...')
    plot_distances__dist_coverage(cfg.OUTPUT_DIR, cfg.COVERAGE_TYPE, cfg.DATASETS, cfg.NETS)
    
    ### render images with comparsion target bodies <--> NNs
    for dset in cfg.DATASETS:
        for net in cfg.NETS:
            print(f'Render frames {cfg.COVERAGE_TYPE} {dset} {net}...')
            render_frames__dist_coverage(cfg.OUTPUT_DIR, cfg.COVERAGE_TYPE, dset, net, num_frames_to_render=10)
