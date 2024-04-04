from atlas_utils import get_atlas
from vendor_partitioning import get_vendor_partitions
import numpy as np
from nilearn import plotting, image
import matplotlib.pyplot as plt


def plot_quadrant_conn():
    atlas = get_atlas(combine_regions=False)
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_vendor_partitions(age='healthy', flip=True, anat=True, scrub=False,
                              anat_ver=3, combine_regions=False)

    nroi = len(atlas['coords'])
    graph = np.full((nroi, nroi), np.nan)

    # print(f'{p_d_pos=}')
    # print(f'{p_d_ant=}')
    # print(f'{p_v_pos=}')
    # print(f'{p_v_ant=}')
    # quit()
    p_v_ant = np.array(p_v_ant)
    # p_v_ant = p_v_ant[~np.isin(p_v_ant, [78, 79])]
    # p_v_ant = []
    # p_v_ant = [68, 69, 76,  82, 83, 92, 93]
    print(p_v_ant)

    # empty_graph[np.ix_(p_d_pos, p_d_ant)] = -1
    graph[np.ix_(p_d_ant, p_d_pos)] = -1

    # empty_graph[np.ix_(p_v_pos, p_v_ant)] = -1
    graph[np.ix_(p_v_ant, p_v_pos)] = -1
    #
    graph[np.ix_(p_d_pos, p_v_pos)] = 1
    # empty_graph[np.ix_(p_v_pos, p_d_pos)] = 1

    graph[np.ix_(p_d_ant, p_v_ant)] = 1
    # empty_graph[np.ix_(p_v_ant, p_d_ant)] = 1
    # empty_graph[:, :] = 0
    # graph[:, :] = 0

    rnd = np.random.uniform(0.65, 1.2, size=graph.shape)
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
        # alpha=0.01,
        # colorbar=True,
        # edge_cmap='coolwarm',
        edge_cmap='turbo',
        node_size=5,
        node_color='k',
        edge_kwargs={'linewidth': 1., 'alpha': .8},
        display_mode='x',
        figure=fig
    )
    fp_out = 'result_pics/other/vendor_drawing.png'
    plt.savefig(fp_out, dpi=600)
    plt.show()
    quit()


    # plotting.plot_markers(rois, coords_clean,
    #                       colorbar=False, node_cmap='turbo')
    # plt.show()
    #
    # quit()

    rois = [0, 3, 24, 30, 31, 43, 51, 73, 76, 81, 99, 123, 142, 152, 173, 188,
            198, 202, 206, 207]
    # rois = [0, 3, 24, 30, 31,]
    coords_clean = [atlas['coords'][i] for i in rois]
    vals = [5] * len(rois)

    # atlas['maps'] = np.zeros(atlas['maps'].shape)
    data = atlas['maps'].get_fdata()
    rois = np.array(rois) + 1
    data_ = data.copy()
    data = np.zeros(data.shape)
    for roi in rois:
        if roi == 74: continue
        data[data_ == roi] = roi
    data[data == 143] = 999
    data[data == 153] = 143
    data[data == 999] = 153
    data[data < 1] = 0

    # for x in range(1, data.shape[0]-1):
    #     for y in range(1, data.shape[1]-1):
    #         for z in range(1, data.shape[2]-1):
    #             # check if any point surrounding (x, y, z) is zero
    #             if data[x, y, z] == 0:
    #                 continue
    #             # if data[x-1, y, z] == 0 or data[x+1, y, z] == 0 or \
    #             #         data[x, y-1, z] == 0 or data[x, y+1, z] == 0 or \
    #             #         data[x, y, z-1] == 0 or data[x, y, z+1] == 0:
    #             #     # continue
    #             #     continue
    #
    #             g = data[x, y, z]
    #             continue
    #             if g < .5: continue
    #             # if data[x - 1, y, z] == g or data[x + 1, y, z] == g or \
    #             #         data[x, y - 1, z] == g or data[x, y + 1, z] == g or \
    #             #         data[x, y, z - 1] == g or data[x, y, z + 1] == g:
    #             #     data[x, y, z] = 0.4
    #                 # continue
    #             if data[x - 1, y, z] != 0:
    #                 if data[x - 1, y, z] != g: data[x - 1, y, z] = 0.4
    #             if data[x + 1, y, z] != 0:
    #                 if data[x + 1, y, z] != g: data[x + 1, y, z] = 0.4
    #             if data[x, y - 1, z] != 0:
    #                 if data[x, y - 1, z] != g: data[x, y - 1, z] = 0.4
    #             if data[x, y + 1, z] == 0:
    #                 if data[x, y + 1, z] != g: data[x, y + 1, z] = 0.4
    #             if data[x, y, z - 1] == 0:
    #                 if data[x, y, z - 1] != g: data[x, y, z - 1] = 0.4
    #             if data[x, y, z + 1] == 0:
    #                 if data[x, y, z + 1] != g: data[x, y, z + 1] = 0.4


                # data[x, y, z] = 0.4

    # print(data.shape)
    # data[data > 1] = 0
    # plotting.plot_markers(rois, coords_clean,
    #                       colorbar=False, node_cmap='turbo')
    # plt.show()

    # quit()
    img = image.new_img_like(atlas['maps'], data)
    plotting.plot_glass_brain(img, cmap=plt.get_cmap('turbo'), black_bg=False,
                              threshold=0.5)
    # quit()
    # plotting.plot_roi(img, cmap=plt.get_cmap('turbo'), display_mode='y',)
    plt.savefig('result_pics/other/signif_clf_no74.png', dpi=600)
    plt.show()

if __name__ == '__main__':



    plot_quadrant_conn()









