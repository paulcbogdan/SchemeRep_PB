from atlas_utils import get_atlas
import numpy as np
from nilearn import plotting, image
import matplotlib.pyplot as plt
from connRSA.information_connectivity import get_idxs


def plot_large_avg_ROIs(system, cmap='turbo', voxelwise=False): # 'turbo'


    atlas = get_atlas(combine_regions=False)
    # atlas = get_atlas(combine_regions=False, lifu_labels=False)
    # for ROI, region, region_OG in zip(atlas['ROIs'],
    #                                   atlas['ROI_regions'],
    #                                   atlas_OG['ROI_regions']):
    #     print(f'{region_OG} | {region}: {ROI}')

    rois = sorted(get_idxs(system))
    data = atlas['maps'].get_fdata()
    data = prune2rois(data, rois)

    if voxelwise:
        vmin = np.min(data[data > 0.5])
        print(f'{vmin=}')
        vmax = np.max(data)
        print(f'{vmax=}')
        sag = np.sum(data, axis=0) > 0
        data[:, :, :] = 0
        data[20, sag] = 1

        rand = np.random.randint(vmin, vmax, data.shape)
        data = np.where(data > 0.5, rand, data)

    img = image.new_img_like(atlas['maps'], data)
    fig = plt.figure(figsize=(4.2, 5))

    plotting.plot_glass_brain(img, cmap='turbo',#plt.get_cmap(cmap),
                              black_bg=False,
                              vmin=0.5,
                              resampling_interpolation='nearest',
                              display_mode='x', figure=fig,
                              alpha=.0
                              # bg_img=None
                              )
    # plotting.plot_img(img, cmap='turbo',  # plt.get_cmap(cmap),
    #                           black_bg=True,
    #                           vmin=0.5,
    #                           resampling_interpolation='nearest',
    #                           display_mode='x', figure=fig,
    #                           # bg_img=None
    #                           )
    # plotting.plot_roi(img, cmap='turbo',#plt.get_cmap(cmap),
    #                           # black_bg=False,
    #                           vmin=0.5,
    #                           resampling_interpolation='nearest',
    #                           display_mode='x', figure=fig,
    #                           bg_img=None)
    plt.show()


def prune2rois(data, rois, scramble_order=False, idx_cnt=True):
    if scramble_order:
        np.random.seed(0)
        np.random.shuffle(rois)
    rois = np.array(rois) + 1
    data_ = data.copy()
    data = np.zeros(data.shape)
    for i, roi in enumerate(rois):
        # if roi == 74: continue
        if idx_cnt:
            data[data_ == roi] = i + 1
        else:
            data[data_ == roi] = roi
    if not idx_cnt:
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
    for target in ['Occipital', 'IT', 'ITL', 'Parietal', 'PFC', 'OC_IT']:
        plot_large_avg_ROIs(target, )
