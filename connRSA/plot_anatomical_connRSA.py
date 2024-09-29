from atlas_utils import get_atlas
import numpy as np
from nilearn import plotting, image
import matplotlib.pyplot as plt
from connRSA.old_Sep29.information_connectivity import get_idxs


def plot_large_avg_ROIs(system, cmap='turbo', voxelwise=True): # 'turbo'


    atlas = get_atlas(combine_regions=False)
    atlas_OG = get_atlas(combine_regions=False, lifu_labels=False)
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
        # data[:, ~sag] = 0

        # plt.imshow(sag)
        # plt.show()
        # quit()

        # thickest_y_slice = np.argmax(np.sum(data, axis=1))
        # thickest_y_slice = np.unravel_index(thickest_y_slice,
        #                                     np.sum(data, axis=1).shape)
        # data[:thickest_y_slice[0], :, ] = 0
        # data[thickest_y_slice[0] + 1:, :, ] = 0

        rand = np.random.randint(vmin, vmax, data.shape)
        data = np.where(data > 0.5, rand, data)



    img = image.new_img_like(atlas['maps'], data)

    # test = plt.get_cmap(cmap)
    # vals_colored = cmap(np.linspace(0.05, .9, n_colored))

    fig = plt.figure(figsize=(4.2, 5))

    # vals = plt.cm.get_cmap(cmap)(np.linspace(0., 1., len(rois)))
    # idxs = np.arange(vals.shape[0])
    # np.random.shuffle(idxs)
    # print(vals)
    # vals = vals[idxs, :]
    # print(vals)
    # cmap = ListedColormap(vals)
    # print(cmap)
    # quit()

    plotting.plot_glass_brain(img,
                              cmap='turbo',#plt.get_cmap(cmap),
                              black_bg=False, vmin=0.5,
                              resampling_interpolation='nearest',
                              display_mode='x',
                              figure=fig
                              )

    # plotting.plot_roi(img, cmap=plt.get_cmap(cmap),
    #                   black_bg=False, vmin=0.5,
    #                   resampling_interpolation='nearest',
    #                   display_mode='x',
    #                   )


    # plt.savefig(f'result_pics/other/connRSA_anat_{system}.png',
    #             dpi=600)

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
    # plot_large_avg_ROIs('subcort')
    # quit()

    plot_large_avg_ROIs('Occipital')
    plot_large_avg_ROIs('IT')
    # plot_large_avg_ROIs('IT', voxelwise=False)
    # plot_large_avg_ROIs('Occipital', voxelwise=False)

    quit()
    plot_large_avg_ROIs('Parietal')
    plot_large_avg_ROIs('PFC')


