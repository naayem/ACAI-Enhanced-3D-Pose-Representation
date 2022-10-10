import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def main():
    path = "/home/facades/projects/stud_proj_pose_interp/output_metrics/smoothness/all_smoothness_8.json"
    savep = "/home/facades/projects/stud_proj_pose_interp/smoothness8_final"

    fig = plt.figure(1)
    fig.clf()
    ax = fig.add_subplot(1,1,1)

    # Creating the colar map
    colors = ["blue", "red"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)

    f = open(path)
    data = json.load(f)

    shade = np.arange(0.0, len(data.keys()))/(len(data.keys())-1)
    for idx, k in enumerate(data.keys()):
        lambdas = data[k]['smoothness']['Lambda']
        aggregated_deltas = data[k]['smoothness']['aggregated_deltas']

         # Data used in plot
        t = np.arange(0.0, len(aggregated_deltas))
        print(lambdas)
        print(aggregated_deltas)
        print(t)
        print(shade[idx])

        plt.plot(t, aggregated_deltas, color=cmap1(shade[idx]), label=f'Lambda:{lambdas}')
 
    plt.legend()
    plt.title('mean of normalised interpolations deltas\n latent space dimension 8')
    plt.savefig(savep)

if __name__ == '__main__':
    main()