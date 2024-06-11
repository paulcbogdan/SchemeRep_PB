from atlas_utils import get_atlas
from vendor_partitioning import get_vendor_partitions
import numpy as np
from nilearn import plotting, image
import matplotlib.pyplot as plt
from information_connectivity import get_idxs


def plot_large_avg_ROIs(system):
    atlas = get_atlas(combine_regions=False)
    atlas_OG = get_atlas(combine_regions=False, lifu_labels=False)
    # for ROI, region, region_OG in zip(atlas['ROIs'],
    #                                   atlas['ROI_regions'],
    #                                   atlas_OG['ROI_regions']):
    #     print(f'{region_OG} | {region}: {ROI}')

    rois = sorted(get_idxs(system))
    data = atlas['maps'].get_fdata()
    data = prune2rois(data, rois)

    img = image.new_img_like(atlas['maps'], data)
    plotting.plot_glass_brain(img, cmap=plt.get_cmap('turbo'),
                              black_bg=False, vmin=0.5,
                              resampling_interpolation='nearest',
                              display_mode='x',
                              alpha=0.7)

    plt.savefig(f'result_pics/other/connRSA_anat_{system}.png',
                dpi=600)

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
    plot_large_avg_ROIs('ITL')
    plot_large_avg_ROIs('Occipital')
    plot_large_avg_ROIs('Parietal')
    plot_large_avg_ROIs('PFC')


