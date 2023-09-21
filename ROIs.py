from collections import defaultdict

import numpy as np
import pandas as pd
from nilearn import image


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
    atlas['ROI_regions'] = ROI_regions
    atlas['ticks'] = ticks
    atlas['tick_labels'] = tick_labels
    atlas['tick_lows'] = tick_lows
    atlas['ROI_regions_laterality'] = ROI_regions_laterality
    atlas['ROI2coord'] = dict(zip(ROIs, atlas['coords']))
    return ROIs, ROI_nums, ticks, tick_labels, tick_lows, n_ROIs, ROI_regions


def get_BN_atlas(combine_bilaterally=False, lifu_labels=True):
    img = image.load_img(r'cache/BN_Atlas_246_2mm.nii.gz')
    fp_labels = r'cache/BNA_labels_Lifu.txt' if lifu_labels else r'cache/BNA_labels.txt'
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


def get_BN_and_resample(combine_bilateral=False):
    fp_ref = r'Day2EncSingleTrialModellingLSS_sorted/102/all_ENCruns_sorted/objects/' \
             r'Day2_Run1_Trial4_UnifiedID53_StimID215_Subset2_pairID15_Con3_Resp4_IsObject1.nii'
    img = image.load_img(fp_ref)
    atlas = get_BN_atlas(combine_bilaterally=combine_bilateral)
    atlas['maps'] = image.resample_to_img(atlas['maps'], img,
                                          interpolation='nearest')
    return atlas

def get_combined_BNA(combine_bilateral=False):
    atlas = get_BN_and_resample(combine_bilateral=combine_bilateral)
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

def get_atlas(combine_regions, bilateral):
    if combine_regions:
        atlas = get_combined_BNA(combine_bilateral=bilateral)
    else:
        atlas = get_BN_and_resample(combine_bilateral=bilateral)
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
    get_atlas(combine_regions=True, bilateral=False)
    # get_BN_atlas()
    # get_combined_BNA(combine_bilateral=True)