import os
import pathlib

from Study1A.partition_VD_PA import get_VD_PA_partitions

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

from atlas_utils import get_atlas
import numpy as np
from nilearn import plotting, image
import matplotlib.pyplot as plt


def plot_quadrant_conn():
    atlas = get_atlas(combine_regions=False)
    p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
        get_VD_PA_partitions(age='healthy', anat=True,
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

    # good_regions = ['LOC']
    # rois = [i for i, roi in enumerate(atlas['ROI_regions']) if 'LOC' in roi]
    rois = [i for i, roi in enumerate(atlas['ROI_regions']) if 'STG' in roi]
    rois = [i for i, roi in enumerate(atlas['ROI_regions']) if 'MTG' in roi]


    data = atlas['maps'].get_fdata()
    data = prune2rois(data, rois)

    img = image.new_img_like(atlas['maps'], data)
    plotting.plot_glass_brain(img, cmap=plt.get_cmap('turbo'), black_bg=False,
                              vmin=0.5,
                              resampling_interpolation='nearest',
                              alpha=0.7)
    cur_dir = os.getcwd()
    # fp_out = f'{cur_dir}/result_pics/Fig2/Fig2E_PA_VD_anat_modules.png'
    # plt.savefig(fp_out, dpi=300)
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









