from Utils.atlas_funcs import get_atlas
import numpy as np
from nilearn import plotting, image
import matplotlib.pyplot as plt
from connRSA_finalizing.plot_Fig5_conn import get_idxs
from connRSA.jit_funcs import do_int_downsample, do_int_upsample


def plot_ROIs_SchemePE(system, cmap='turbo', voxelwise=False): # 'turbo'


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

    downsample = 1
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

    # fig = plt.figure(figsize=(4.2, 2))
    fig = plt.figure(figsize=(2., 1.9))

    plotting.plot_glass_brain(img, cmap='turbo',
                              black_bg=False, vmin=0.5,
                              resampling_interpolation='nearest',
                              display_mode='x', figure=fig,
                              # title=system
                              )
    fp_pic = fr'C:\PycharmProjects\SchemeRep\result_pics\other\{system}_ROI.png'
    plt.savefig(fp_pic, dpi=300)
    plt.show()

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

if __name__ == '__main__':
    # for target in ['ITL']:
    atlas = get_atlas(combine_regions=False)
    # print(set(atlas['ROI_regions']))
    # quit()
    ROIs = ['LOC', 'EVC', 'IPL', 'MFG', 'IFG', 'ATL']
    for ROI in ROIs:
        plot_ROIs_SchemePE(ROI, voxelwise=False)

