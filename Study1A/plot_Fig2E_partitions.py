import os
import pathlib

from Study1A.plot_Fig2CD_partitions import get_VD_PA_partitions

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

from Utils.atlas_funcs import get_atlas
import numpy as np
from nilearn import plotting
import matplotlib.pyplot as plt


def plot_quadrant_conn():
    atlas = get_atlas(combine_regions=False)
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_VD_PA_partitions(age='healthy', anat=True,
                             anat_ver=3, combine_regions=False)

    nroi = len(atlas['coords'])
    graph = np.full((nroi, nroi), np.nan)

    p_v_ant = np.array(p_v_ant)
    p_v_ant = p_v_ant[~np.isin(p_v_ant, [78, 79])]

    graph[np.ix_(p_d_ant, p_d_pos)] = -1
    graph[np.ix_(p_v_ant, p_v_pos)] = -1
    graph[np.ix_(p_d_pos, p_v_pos)] = 1
    graph[np.ix_(p_d_ant, p_v_ant)] = 1

    rnd = np.random.uniform(0.8, 1.2, size=graph.shape)
    graph *= rnd

    i_lower = np.tril_indices_from(graph, -1)
    graph[i_lower] = graph.T[i_lower]
    p_all = np.concatenate([p_d_ant, p_d_pos, p_v_ant, p_v_pos])

    graph_tight = graph[np.ix_(p_all, p_all)]

    coords = np.array(atlas['coords'])[p_all]

    vabs = 1.43
    fig = plt.figure(figsize=(3.5, 3.5))
    plotting.plot_connectome(
        graph_tight,
        coords,
        edge_threshold=0.2,
        edge_vmin=-vabs, edge_vmax=vabs,

        edge_cmap='turbo',
        node_size=5,
        node_color='k',
        edge_kwargs={'linewidth': 1., 'alpha': .8},
        display_mode='x',
        figure=fig
    )
    fp = 'result_pics/Fig2/Fig2E_partitions.png'
    plt.savefig(fp, dpi=600)
    plt.show()


if __name__ == '__main__':
    plot_quadrant_conn()
