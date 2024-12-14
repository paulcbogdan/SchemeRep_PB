from Utils.atlas_funcs import get_atlas
import numpy as np
from nilearn import plotting, image
import matplotlib.pyplot as plt
from connRSA_finalizing.conn_Fig6 import get_idxs
from connRSA.jit_funcs import do_int_downsample, do_int_upsample


def plot_large_avg_ROIs(system, cmap='turbo', voxelwise=False): # 'turbo'


    atlas = get_atlas(combine_regions=False)
    # atlas = get_atlas(combine_regions=False, lifu_labels=False)
    # for ROI, region, region_OG in zip(atlas['ROIs'],
    #                                   atlas['ROI_regions'],
    #                                   atlas_OG['ROI_regions']):
    #     print(f'{region_OG} | {region}: {ROI}')

    idxs = get_idxs(system)
    print(idxs)
    # quit()
    rois = sorted(idxs)
    # rois = rois[5:6]
    # rois = rois[11:12]
    # rois = rois[1:2]


    print(f'{system} | number of ROIs: {len(rois)}')
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

        rand = np.random.randint(0, 100, data.shape)
        data = np.where(data > 0.1, rand, data)
        data[1::2, :, :] = -999_999
        data[:, 1::2, :] = -999_999
        data[:, :, 1::2] = -999_999

        # data[:, :, 0::2] = 0

    downsample = 2
    if downsample > 1:
        data = data[..., None]
        mask = data > 0.5
        data = do_int_downsample(data, downsample, mask, nan_val=-999_999,
                                edge_drop=False, fourD_mask=False)
        data[data < 0] = 0
        data = data[..., 0]
        data = do_int_upsample(data, downsample, mask)
        data[data > 0] -= np.min(data[data > 0])

    img = image.new_img_like(atlas['maps'], data)
    # plt.hist(data[data > 0].flatten())
    # plt.show()
    # quit()

    fig = plt.figure(figsize=(4.2, 5))

    plotting.plot_glass_brain(img,
                              cmap='turbo',#plt.get_cmap(cmap),
                              black_bg=False,
                              vmin=0.5,
                              resampling_interpolation='nearest',
                              display_mode='x',
                              figure=fig,
                              # alpha=.0
                              # bg_img=None
                              )

    # plotting.plot_roi(img, cmap='turbo',#plt.get_cmap(cmap),
    #                           # black_bg=False,
    #                           vmin=0.5,
    #                           resampling_interpolation='nearest',
    #                           display_mode='x', figure=fig,
    #                           bg_img=None)
    plt.show(dpi=1000)

NUM_VOXELS_ALL = []

def prune2rois(data, rois, scramble_order=False, idx_cnt=True):
    if scramble_order:
        np.random.seed(0)
        np.random.shuffle(rois)
    rois = np.array(rois) + 1
    data_ = data.copy()
    data = np.zeros(data.shape)
    global NUM_VOXELS_ALL
    num_voxels_system = []
    for i, roi in enumerate(rois):
        # if roi == 74: continue
        num_voxels = np.sum(data_ == roi)
        NUM_VOXELS_ALL.append(num_voxels)
        num_voxels_system.append(num_voxels)
        if idx_cnt:
            data[data_ == roi] = i + 1
        else:
            data[data_ == roi] = roi
    if not idx_cnt:
        lowest_roi = np.min(rois)
        data[data >= lowest_roi] -= (lowest_roi - 1)

    low_num_voxels_system = np.min(num_voxels_system)
    high_num_voxels_system = np.max(num_voxels_system)
    median_num_voxels_system = np.median(num_voxels_system)
    mean_num_voxels_system = np.mean(num_voxels_system)
    SD_num_voxels_system = np.std(num_voxels_system)
    print(f'{low_num_voxels_system=:.0f} | {high_num_voxels_system=:.0f} | {median_num_voxels_system=:.0f} | '
          f'M = {mean_num_voxels_system:.0f} [SD = {SD_num_voxels_system:.0f}]')
    # plt.hist(num_voxels_system, bins=32, range=(0, 1600))
    # plt.show()

    if len(NUM_VOXELS_ALL) > 100:
        low_num_voxels = np.min(NUM_VOXELS_ALL)
        high_num_voxels = np.max(NUM_VOXELS_ALL)
        median_num_voxels = np.median(NUM_VOXELS_ALL)
        mean_num_voxels = np.mean(NUM_VOXELS_ALL)
        SD_num_voxels = np.std(NUM_VOXELS_ALL)
        print('-' * 50)
        print(f'{low_num_voxels=:.0f} | {high_num_voxels=:.0f} | {median_num_voxels=:.0f} | '
              f'M = {mean_num_voxels:.0f} [SD = {SD_num_voxels:.0f}]')

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

# from nilearn import image
# from pathlib import Path
#
# fps = [r'C:/test.nii',
#        r'C:\test.nii',
#        Path(r'C:/test.nii'),
#        Path(r'C:\test.nii'),
#        ]
# for fp in fps:
#     image.load_img(fp)

if __name__ == '__main__':
    # from nilearn import image
    # from pathlib import Path
    #
    # fps = [r'C:/test.nii',
    #        r'C:\test.nii',
    #        Path(r'C:/test.nii'),
    #        Path(r'C:\test.nii'),
    #        ]
    # for fp in fps:
    #     image.load_img(fp)
    # quit()

    # 'ITL', 'OC_IT'
    for target in ['Occipital']:#, 'IT', 'Parietal', 'PFC']:
        plot_large_avg_ROIs(target, voxelwise=True)
