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

    p_v_ant = np.array(p_v_ant)

    graph[np.ix_(p_d_ant, p_d_pos)] = -1
    graph[np.ix_(p_v_ant, p_v_pos)] = -1
    graph[np.ix_(p_d_pos, p_v_pos)] = 1
    graph[np.ix_(p_d_ant, p_v_ant)] = 1

    rnd = np.random.uniform(0.65, 1.2, size=graph.shape)
    graph *= rnd

    i_lower = np.tril_indices_from(graph, -1)
    graph[i_lower] = graph.T[i_lower]
    p_all = np.concatenate([p_d_ant, p_d_pos, p_v_ant, p_v_pos])

    # SchemePE significant (i think, as of June 11, 2024)
    rois = [0, 3, 24, 30, 31, 43, 51, 73, 76, 81, 99, 123, 142, 152, 173, 188,
            198, 202, 206, 207]


    data = atlas['maps'].get_fdata()
    data = prune2rois(data, rois)
    # data = do_slicing(data)

    img = image.new_img_like(atlas['maps'], data)
    plotting.plot_glass_brain(img, cmap=plt.get_cmap('turbo'), black_bg=False,
                              vmin=0.5,
                              resampling_interpolation='nearest',
                              alpha=0.7)
    # quit()
    # plotting.plot_roi(img, cmap=plt.get_cmap('turbo'), display_mode='y',)
    # plt.savefig('result_pics/other/signif_clf_no74.png', dpi=600)
    # plt.savefig('result_pics/other/Occipital_.png', dpi=600)
    # plt.savefig('result_pics/other/Occ200.png', dpi=600)

    plt.show()

def prune2rois(data, rois):
    rois = np.array(rois) + 1
    data_ = data.copy()
    data = np.zeros(data.shape)
    for roi in rois:
        if roi == 74: continue
        data[data_ == roi] = roi
    lowest_roi = np.min(rois)
    data[data >= lowest_roi] -= (lowest_roi - 1)

    return data

def do_slicing(data):

    data[data == 143] = 999
    data[data == 153] = 143
    data[data == 999] = 153
    data_ = np.zeros(data.shape)
    data_[48:52, :, :] = 1
    data[data > 1] -= 187
    data *= data_
    return data


if __name__ == '__main__':



    plot_quadrant_conn()









