

from collections import defaultdict

import numpy as np
import pandas as pd
from nilearn import image
from scipy import ndimage
from nilearn import plotting
import random

def get_Schaefer_atlas():
    atlas = {}

    from nilearn.datasets import fetch_atlas_schaefer_2018
    atlas_ni = fetch_atlas_schaefer_2018(n_rois=1000, resolution_mm=2,
                                      yeo_networks=17)
    df_coords = r'Schaefer2018_1000Parcels_17Networks_order_FSLMNI152_1mm.Centroid_RAS.csv'
    df_coords = pd.read_csv(df_coords)
    atlas['coords'] = np.array(df_coords[['R', 'A', 'S']])
    print(atlas['coords'].shape)

    # fp_ref = r'fMRI_in/102/Enc_rerun/obj/' \
    #          r'ENC_sub102_run1_trial1_subset3_pairID29.nii'
    fp_ref = r'fMRI_in/102/Enc_rerun3/obj/' \
                 r'ENC_sub102_run1_trial1_subset3_pairID29.nii'
    img = image.load_img(fp_ref)
    atlas['maps'] = image.resample_to_img(atlas_ni['maps'], img,
                                          interpolation='nearest')
    ROIs = [str(ROI, encoding='utf-8') for ROI in atlas_ni['labels']]
    ROIs = [ROI.replace('PrC_', 'PrCv_').replace('Cinga_1', 'PFCmp_6').
            replace('PFCld', 'PFCl').replace('AntTemp', 'Temp')
            for ROI in ROIs]

    ROI_nums = [i + 1 for i in range(len(ROIs))]
    n_ROIs = len(ROIs)
    ROI_regions = []
    ROI_regions_laterality = []
    region_nums = defaultdict(list)
    ROIs_ = []
    for i, ROI in enumerate(ROIs):
        ROI = ROI.replace('s_LH', 's_L').replace('s_RH', 's_R')
        ROIs_.append(ROI)
        # ROI = str(ROI, encoding='utf-8')
        region = ROI.split('_')[-2]
        ROI_regions.append(region)
        LR = ROI.split('_')[1]
        region_LR = region + '_' + LR
        # print(f'{ROI} | {region_LR=}')
        region_nums[region].append(i)#int(ROI_num)-1)
        ROI_regions_laterality.append(region_LR)
    ROIs = ROIs_

    ticks = []
    tick_labels = []
    tick_lows = []
    regions = list(region_nums.keys())
    regions.sort()
    for region in regions:
        l = region_nums[region]
    # for region, l in region_nums.items():
        ticks.append(np.mean(l))
        tick_labels.append(region)
        tick_lows.append(l[0])

    # print(tick_labels)
    # quit()
    # quit()
    atlas['ROIs'] = ROIs
    atlas['ROI_nums'] = ROI_nums
    atlas['n_ROIs'] = n_ROIs
    atlas['ROI_regions'] = ROI_regions # repeats, length = # ROI
    atlas['ticks'] = ticks
    atlas['tick_labels'] = tick_labels # no repeat, length = # regions
    atlas['tick_lows'] = tick_lows
    # for tick, label in zip(ticks, tick_labels):
    #     print(f'{tick=}, {label=}')
        # TODO: sort by region
    # print(f'{tick_labels=}')

    # quit()
    atlas['shenyang'] = False
    atlas['ROI_regions_laterality'] = ROI_regions_laterality
    atlas['ROI2coord'] = dict(zip(ROIs, atlas['coords']))
    return atlas


def add_ROI_info(atlas):
    ROIs = atlas['labels']
    ROI_nums = [i + 1 for i in range(len(ROIs))]
    # ROI_nums = [int(roi.split()[0]) for roi in ROIs]
    n_ROIs = len(ROIs)

    region_nums = defaultdict(list)
    ROI_regions = []
    ROI_regions_laterality = []
    for i, ROI in enumerate(ROIs):
        # ROI_num = i + 1
        ROI_str = ROI.split(' ')[-1]
        region = ROI_str.split('_')[0]
        if '_L_' in ROI_str:
            region_LR = region + '_L'
        elif '_R_' in ROI_str:
            region_LR = region + '_R'
        else:
            region_LR = region
        # print(f'{ROI} {region} {region_LR}')
        ROI_regions.append(region)
        ROI_regions_laterality.append(region_LR)
        region_nums[region].append(i)#int(ROI_num)-1)
    ticks = []
    tick_labels = []
    tick_lows = []
    for region, l in region_nums.items():
        ticks.append(np.mean(l))
        tick_labels.append(region)
        tick_lows.append(l[0])

    atlas['ROIs'] = ROIs
    atlas['ROI_nums'] = ROI_nums
    atlas['n_ROIs'] = n_ROIs
    atlas['ROI_regions'] = ROI_regions # repeats, length = # ROI
    atlas['ticks'] = ticks
    atlas['tick_labels'] = tick_labels # no repeat, length = # regions
    atlas['tick_lows'] = tick_lows
    atlas['ROI_regions_laterality'] = ROI_regions_laterality
    atlas['ROI2coord'] = dict(zip(ROIs, atlas['coords']))
    return ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs, ROI_regions


def get_BN_atlas(combine_bilaterally=False, lifu_labels=True,
                 shenyang=True):
    if shenyang:
        fp_atlas = r'C:\PycharmProjects_C\SchemeRep\Shenyang_R\Atlas\BNA_thr25_resliced_97_115_97.nii'
        img = image.load_img(fp_atlas)
        print('Shenyang atlas')
    else:
        img = image.load_img(r'cache/BN_Atlas_246_2mm.nii.gz')
        print('Defunct atlas')

    fp_labels = r'cache/BNA_labels_Lifu.txt' if lifu_labels else r'cache/BNA_labels.txt'
    fp_labels = 'cache/BNA_labels_Lifu_ACC.txt'
    labels = pd.read_csv(fp_labels, header=None)[0].to_list()
    if combine_bilaterally:
        data = img.get_fdata()
        data = (data + 1) - (data + 1) % 2
        data /= 2
        img = image.new_img_like(img, data)
        labels = labels[::2]
        labels_ = []
        for ROI in labels:
            ROI = ROI.replace('_L', '')
            ROI_num, ROI_str = ROI.split(' ')
            ROI_num = int(ROI.split()[0])
            ROI_num = (ROI_num + 1) // 2
            ROI = f'{ROI_num} {ROI_str}'
            labels_.append(ROI)
        labels = labels_
    atlas = {'maps': img, 'labels': labels}
    atlas['coords'] = org_BNA_coords()
    add_ROI_info(atlas)
    return atlas


def get_BN_and_resample(combine_bilateral=False, new_space=True,
                        shenyang=True):
    if new_space:
        fp_ref = r'C:\PycharmProjects_C\SchemeRep/' \
                 r'fMRI_in/102/Enc_rerun3/obj/' \
                 r'ENC_sub102_run1_trial1_subset3_pairID29.nii'
    else:
        fp_ref = r'C:\PycharmProjects_C\SchemeRep/' \
                 r'fMRI_in/102/all_ENCruns_sorted/objects/' \
                 r'Day2_Run1_Trial4_UnifiedID53_StimID215_Subset2_pairID15_Con3_Resp4_IsObject1.nii'

    img = image.load_img(fp_ref)
    atlas = get_BN_atlas(combine_bilaterally=combine_bilateral,
                         shenyang=shenyang)
    atlas['maps'] = image.resample_to_img(atlas['maps'], img,
                                          interpolation='nearest')
    atlas['shenyang'] = shenyang
    # atlas['maps'].to_filename('test.nii')
    # print(atlas['maps'].shape)
    # from nilearn import plotting
    # plotting.plot_roi(atlas['maps'])
    # plotting.show()
    # quit()
    return atlas

def get_combined_BNA(combine_bilateral=False, new_space=True):
    atlas = get_BN_and_resample(combine_bilateral=combine_bilateral,
                                new_space=new_space)
    region_to_new_number = {}
    cnt = 1
    atlas_data = atlas['maps'].get_fdata()
    region_coords = defaultdict(list)
    labels = []
    for region, ROI_num, coord in zip(atlas['ROI_regions_laterality'],
                                      atlas['ROI_nums'],
                                      atlas['coords']):
        if region not in region_to_new_number:
            region_to_new_number[region] = cnt
            cnt += 1
            labels.append(region.split()[0])
        atlas_data[atlas_data == ROI_num] = region_to_new_number[region]
        region_coords[region].append(coord)
    atlas['labels'] = labels
    coords = []
    for label in labels:
        l = region_coords[label]
        l = np.mean(l, axis=0)
        coords.append(l)
        region_coords[label] = l
    atlas['coords'] = coords
    add_ROI_info(atlas)
    return atlas

def split_BNA(new_space=True, split_code='xyz'):
    atlas = get_BN_and_resample(combine_bilateral=False,
                                new_space=new_space)

    atlas_data = atlas['maps'].get_fdata()
    atlas_data_new = np.zeros_like(atlas_data)
    coords_new = []
    ROIs_new = []
    ROI_nums_new = []
    ROI_regions_new = []
    ROI_regions_LR_new = []

    # cnt = 0
    # nums = list(range(1969))
    # random.shuffle(nums)

    for ROI, region, region_LR, ROI_num, coord in zip(atlas['ROIs'],
                                           atlas['ROI_regions'],
                                           atlas['ROI_regions_laterality'],
                                           atlas['ROI_nums'],
                                           atlas['coords']):
        idxs = np.argwhere(atlas_data == ROI_num)
        region_bool = np.zeros_like(atlas_data, dtype=int)
        region_bool[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = 1
        mass_center = ndimage.center_of_mass(region_bool)

        x_options = [False, True] if 'x' in split_code else [False]
        y_options = [False, True] if 'y' in split_code else [False]
        z_options = [False, True] if 'z' in split_code else [False]

        for x_choice in x_options:
            for y_choice in y_options:
                for z_choice in z_options:
                    idxs_choice = idxs.copy()
                    # if 'x' in split_code:
                    if x_choice:
                        idxs_choice = idxs_choice[idxs_choice[:, 0] > mass_center[0]]
                    else:
                        idxs_choice = idxs_choice[idxs_choice[:, 0] < mass_center[0]]
                    # if 'y' in split_code:
                    if y_choice:
                        idxs_choice = idxs_choice[idxs_choice[:, 1] > mass_center[1]]
                    else:
                        idxs_choice = idxs_choice[idxs_choice[:, 1] < mass_center[1]]
                    # if 'z' in split_code:
                    if z_choice:
                        idxs_choice = idxs_choice[idxs_choice[:, 2] > mass_center[2]]
                    else:
                        idxs_choice = idxs_choice[idxs_choice[:, 2] < mass_center[2]]
                    adder = x_choice * 4 + y_choice * 2 + z_choice
                    x_str = 'h' if x_choice else 'l'
                    y_str = 'h' if y_choice else 'l'
                    z_str = 'h' if z_choice else 'l'
                    adder_str = f'{x_str}{y_str}{z_str}'
                    new_num = 1 + (ROI_num - 1) * 8 + adder
                    # new_num = nums[cnt]
                    # cnt += 1

                    atlas_data_new[idxs_choice[:, 0], idxs_choice[:, 1],
                                   idxs_choice[:, 2]] = new_num
                    ROI_new = f'{ROI}_{adder_str}'
                    ROIs_new.append(ROI_new)
                    ROI_regions_new.append(region)
                    ROI_regions_LR_new.append(region_LR)
                    ROI_nums_new.append(new_num)

                    if idxs_choice.shape[0] == 0:
                        coords_new.append(coord)
                    else:
                        x_low = idxs_choice[:, 0].min()
                        x_high = idxs_choice[:, 0].max()
                        y_low = idxs_choice[:, 1].min()
                        y_high = idxs_choice[:, 1].max()
                        z_low = idxs_choice[:, 2].min()
                        z_high = idxs_choice[:, 2].max()
                        x_new = (x_low + x_high) / 2
                        y_new = (y_low + y_high) / 2
                        z_new = (z_low + z_high) / 2
                        coord_new = image.coord_transform(x_new, y_new, z_new,
                                                          atlas['maps'].affine)
                        coords_new.append(coord_new)



    atlas_new = {}
    atlas_new['coords'] = coords_new
    atlas_new['ROIs'] = ROIs_new
    atlas_new['ROI_nums'] = ROI_nums_new
    atlas_new['n_ROIs'] = atlas['n_ROIs'] * 8
    atlas_new['ROI_regions'] = ROI_regions_new
    atlas_new['ticks'] = atlas['ticks']*8
    atlas_new['tick_labels'] = atlas['tick_labels']
    atlas_new['tick_lows'] = atlas['tick_lows']*8
    atlas_new['ROI_regions_laterality'] = ROI_regions_LR_new
    atlas_new['ROI2coord'] = dict(zip(ROIs_new, atlas_new['coords']))
    atlas_new['maps'] = image.new_img_like(atlas['maps'], atlas_data_new)
    return atlas_new


def get_atlas(combine_regions=False, combine_bilateral=False,
              split=False,
              new_space=True, split_code='xyz', schaefer=False,
              shenyang=True):
    if schaefer:
        atlas = get_Schaefer_atlas()
    elif split:
        assert not combine_regions, 'split and combine_regions are mutually exclusive'
        atlas = split_BNA(new_space=new_space, split_code=split_code)
    elif combine_regions:
        atlas = get_combined_BNA(combine_bilateral=combine_bilateral,
                                 new_space=new_space)
    else:
        atlas = get_BN_and_resample(combine_bilateral=combine_bilateral,
                                    new_space=new_space,
                                    shenyang=shenyang)
    return atlas

def org_BNA_coords():
    df = pd.read_csv('cache/BNA_coords_pre.csv')
    coords = []
    for l_coord, r_coord in zip(df['L_coord'], df['R_coord']):
        l_coord = [int(x.replace(' ', '')) for x in l_coord.split(',')]
        r_coord = [int(x.replace(' ', '')) for x in r_coord.split(',')]
        coords += [l_coord, r_coord]
    return coords

if __name__ == '__main__':
    # atlas = get_Schaefer_atlas()
    atlas = get_atlas()
    test = atlas['maps'].get_fdata() == 1
    print(np.sum(test))
    # print(atlas['maps'])
    # print(atlas['maps'].shape)
    # split_BNA()
    # get_atlas(combine_regions=True, combine_bilateral=False)
    # get_BN_atlas()
    # get_combined_BNA(combine_bilateral=True)