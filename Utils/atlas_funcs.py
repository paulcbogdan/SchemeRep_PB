

from collections import defaultdict

import numpy as np
import pandas as pd
from nilearn import image
from functools import cache


def add_ROI_info(atlas):
    ROIs = atlas['labels']
    ROI_nums = [i + 1 for i in range(len(ROIs))]
    n_ROIs = len(ROIs)

    region_nums = defaultdict(list)
    ROI_regions = []
    ROI_regions_laterality = []
    for i, ROI in enumerate(ROIs):
        ROI_str = ROI.split(' ')[-1]
        region = ROI_str.split('_')[0]
        if '_L_' in ROI_str:
            region_LR = region + '_L'
        elif '_R_' in ROI_str:
            region_LR = region + '_R'
        else:
            region_LR = region
        ROI_regions.append(region)
        ROI_regions_laterality.append(region_LR)
        region_nums[region].append(i)

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


# @cache
def get_BN_atlas(combine_bilaterally=False, lifu_labels=True,
                 shenyang=True):
    if shenyang:
        fp_atlas = r'C:\PycharmProjects\SchemeRep\Shenyang_R\Atlas\BNA_thr25_resliced_97_115_97.nii'
        img = image.load_img(fp_atlas)
    else:
        img = image.load_img(r'cache/BN_Atlas_246_2mm.nii.gz')

    if lifu_labels:
        fp_labels = r'cache/BNA_labels_Lifu_ACC_ATL_fix.txt'
    else:
        fp_labels = r'cache/BNA_labels.txt'

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


def get_BN_and_resample(combine_bilateral=False, new_space=True, shenyang=True,
                        HCP=False, natview=False, lifu_labels=True):
    assert not (HCP and natview), 'HCP= and natview= are mutually exclusive'
    if new_space:
        fp_ref = (r'fMRI_in\102'
                  r'\ENC_GM20_LLS1_bpF_full\OBJ'
                  r'\ENC_sub102_run1_trial1_subset3_pairID29_object.nii')
    else:
        raise NotImplementedError
    img = image.load_img(fp_ref)
    atlas = get_BN_atlas(combine_bilaterally=combine_bilateral,
                         shenyang=shenyang, lifu_labels=lifu_labels)
    if HCP:
        affine = [[-2., 0., 0., 90.],
                  [0., 2., 0., -126.],
                  [0., 0., 2., -72.],
                  [0., 0., 0., 1.]]
        affine = np.array(affine)
        atlas['maps'] = image.resample_img(atlas['maps'], target_affine=affine,
                                           target_shape=(91, 109, 91),
                                           interpolation='nearest')
    elif natview:
        affine = [[-3., 0., 0., 90.],
                  [0., 3., 0., -126.],
                  [0., 0., 3., -72.],
                  [0., 0., 0., 1.]]
        affine = np.array(affine)
        atlas['maps'] = image.resample_img(atlas['maps'], target_affine=affine,
                                           target_shape=(61, 73, 61),
                                           interpolation='nearest')
    else:
        atlas['maps'] = image.resample_to_img(atlas['maps'], img,
                                              interpolation='nearest')
    atlas['shenyang'] = shenyang
    return atlas

def get_combined_BNA(combine_bilateral=False, new_space=True, HCP=False,
                     natview=False, lifu_labels=True):
    atlas = get_BN_and_resample(combine_bilateral=combine_bilateral,
                                new_space=new_space, HCP=HCP,
                                natview=natview, lifu_labels=lifu_labels)

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

@cache
def get_atlas(combine_regions=False, combine_bilateral=False,
              new_space=True, shenyang=True,
              HCP=False, natview=False, lifu_labels=True):
    if combine_regions:
        atlas = get_combined_BNA(combine_bilateral=combine_bilateral,
                                 new_space=new_space, HCP=HCP,
                                 natview=natview, lifu_labels=lifu_labels)
    else:
        atlas = get_BN_and_resample(combine_bilateral=combine_bilateral,
                                    new_space=new_space, shenyang=shenyang,
                                    HCP=HCP, natview=natview,
                                    lifu_labels=lifu_labels)
    return atlas

def org_BNA_coords():
    fp_coords_pre = r'cache\BNA_coords_pre.csv'
    df = pd.read_csv(fp_coords_pre)
    coords = []
    for l_coord, r_coord in zip(df['L_coord'], df['R_coord']):
        l_coord = [int(x.replace(' ', '')) for x in l_coord.split(',')]
        r_coord = [int(x.replace(' ', '')) for x in r_coord.split(',')]
        coords += [l_coord, r_coord]
    return coords


def get_BNA_ROIs(code=None):
    if code == 'BNA_region':
        ROIs = ['SFG_L', 'SFG_R', 'MFG_L', 'MFG_R', 'IFG_L', 'IFG_R', 'OrG_L', 'OrG_R', 'PrG_L', 'PrG_R', 'PCL_L', 'PCL_R', 'ATL_L', 'ATL_R', 'STG_L', 'STG_R', 'MTG_L', 'MTG_R', 'ITG_L', 'ITG_R', 'FuG_L', 'FuG_R', 'PhG_L', 'PhG_R', 'pSTS_L', 'pSTS_R', 'SPL_L', 'SPL_R', 'IPL_L', 'IPL_R', 'Pcun_L', 'Pcun_R', 'PoG_L', 'PoG_R', 'INS_L', 'INS_R', 'PCC_L', 'PCC_R', 'ACC_L', 'ACC_R', 'EVC_L', 'EVC_R', 'LOC_L', 'LOC_R', 'sOcG_L', 'sOcG_R', 'Amyg_L', 'Amyg_R', 'Hipp_L', 'Hipp_R', 'Str_L', 'Str_R', 'Tha_L', 'Tha_R']
    elif code == None or code == 'BNA':
        ROIs = ['1 SFG_L_7_1', '2 SFG_R_7_1', '3 SFG_L_7_2', '4 SFG_R_7_2', '5 SFG_L_7_3', '6 SFG_R_7_3', '7 SFG_L_7_4', '8 SFG_R_7_4', '9 SFG_L_7_5', '10 SFG_R_7_5', '11 SFG_L_7_6', '12 SFG_R_7_6', '13 SFG_L_7_7', '14 SFG_R_7_7', '15 MFG_L_7_1', '16 MFG_R_7_1', '17 MFG_L_7_2', '18 MFG_R_7_2', '19 MFG_L_7_3', '20 MFG_R_7_3', '21 MFG_L_7_4', '22 MFG_R_7_4', '23 MFG_L_7_5', '24 MFG_R_7_5', '25 MFG_L_7_6', '26 MFG_R_7_6', '27 MFG_L_7_7', '28 MFG_R_7_7', '29 IFG_L_6_1', '30 IFG_R_6_1', '31 IFG_L_6_2', '32 IFG_R_6_2', '33 IFG_L_6_3', '34 IFG_R_6_3', '35 IFG_L_6_4', '36 IFG_R_6_4', '37 IFG_L_6_5', '38 IFG_R_6_5', '39 IFG_L_6_6', '40 IFG_R_6_6', '41 OrG_L_6_1', '42 OrG_R_6_1', '43 OrG_L_6_2', '44 OrG_R_6_2', '45 OrG_L_6_3', '46 OrG_R_6_3', '47 OrG_L_6_4', '48 OrG_R_6_4', '49 OrG_L_6_5', '50 OrG_R_6_5', '51 OrG_L_6_6', '52 OrG_R_6_6', '53 PrG_L_6_1', '54 PrG_R_6_1', '55 PrG_L_6_2', '56 PrG_R_6_2', '57 PrG_L_6_3', '58 PrG_R_6_3', '59 PrG_L_6_4', '60 PrG_R_6_4', '61 PrG_L_6_5', '62 PrG_R_6_5', '63 PrG_L_6_6', '64 PrG_R_6_6', '65 PCL_L_2_1', '66 PCL_R_2_1', '67 PCL_L_2_2', '68 PCL_R_2_2', '69 ATL_L_6_1', '70 ATL_R_6_1', '71 STG_L_6_2', '72 STG_R_6_2', '73 STG_L_6_3', '74 STG_R_6_3', '75 STG_L_6_4', '76 STG_R_6_4', '77 ATL_L_6_5', '78 ATL_R_6_5', '79 STG_L_6_6', '80 STG_R_6_6', '81 MTG_L_4_1', '82 MTG_R_4_1', '83 ATL_L_4_2', '84 ATL_R_4_2', '85 MTG_L_4_3', '86 MTG_R_4_3', '87 MTG_L_4_4', '88 MTG_R_4_4', '89 ITG_L_7_1', '90 ITG_R_7_1', '91 ITG_L_7_2', '92 ITG_R_7_2', '93 ATL_L_7_3', '94 ATL_R_7_3', '95 ITG_L_7_4', '96 ITG_R_7_4', '97 ITG_L_7_5', '98 ITG_R_7_5', '99 ITG_L_7_6', '100 ITG_R_7_6', '101 ITG_L_7_7', '102 ITG_R_7_7', '103 FuG_L_3_1', '104 FuG_R_3_1', '105 FuG_L_3_2', '106 FuG_R_3_2', '107 FuG_L_3_3', '108 FuG_R_3_3', '109 PhG_L_6_1', '110 PhG_R_6_1', '111 PhG_L_6_2', '112 PhG_R_6_2', '113 PhG_L_6_3', '114 PhG_R_6_3', '115 PhG_L_6_4', '116 PhG_R_6_4', '117 PhG_L_6_5', '118 PhG_R_6_5', '119 PhG_L_6_6', '120 PhG_R_6_6', '121 pSTS_L_2_1', '122 pSTS_R_2_1', '123 pSTS_L_2_2', '124 pSTS_R_2_2', '125 SPL_L_5_1', '126 SPL_R_5_1', '127 SPL_L_5_2', '128 SPL_R_5_2', '129 SPL_L_5_3', '130 SPL_R_5_3', '131 SPL_L_5_4', '132 SPL_R_5_4', '133 SPL_L_5_5', '134 SPL_R_5_5', '135 IPL_L_6_1', '136 IPL_R_6_1', '137 IPL_L_6_2', '138 IPL_R_6_2', '139 IPL_L_6_3', '140 IPL_R_6_3', '141 IPL_L_6_4', '142 IPL_R_6_4', '143 IPL_L_6_5', '144 IPL_R_6_5', '145 IPL_L_6_6', '146 IPL_R_6_6', '147 Pcun_L_4_1', '148 Pcun_R_4_1', '149 Pcun_L_4_2', '150 Pcun_R_4_2', '151 Pcun_L_4_3', '152 Pcun_R_4_3', '153 Pcun_L_4_4', '154 Pcun_R_4_4', '155 PoG_L_4_1', '156 PoG_R_4_1', '157 PoG_L_4_2', '158 PoG_R_4_2', '159 PoG_L_4_3', '160 PoG_R_4_3', '161 PoG_L_4_4', '162 PoG_R_4_4', '163 INS_L_6_1', '164 INS_R_6_1', '165 INS_L_6_2', '166 INS_R_6_2', '167 INS_L_6_3', '168 INS_R_6_3', '169 INS_L_6_4', '170 INS_R_6_4', '171 INS_L_6_5', '172 INS_R_6_5', '173 INS_L_6_6', '174 INS_R_6_6', '175 PCC_L_7_1', '176 PCC_R_7_1', '177 ACC_L_7_2', '178 ACC_R_7_2', '179 ACC_L_7_3', '180 ACC_R_7_3', '181 PCC_L_7_4', '182 PCC_R_7_4', '183 ACC_L_7_5', '184 ACC_R_7_5', '185 PCC_L_7_6', '186 PCC_R_7_6', '187 ACC_L_7_7', '188 ACC_R_7_7', '189 EVC_L_5_1', '190 EVC_R_5_1', '191 EVC_L_5_2', '192 EVC_R_5_2', '193 EVC_L_5_3', '194 EVC_R_5_3', '195 EVC_L_5_4', '196 EVC_R_5_4', '197 EVC_L_5_5', '198 EVC_R_5_5', '199 LOC_L_4_1', '200 LOC_R_4_1', '201 LOC_L_4_2', '202 LOC_R_4_2', '203 LOC_L_4_3', '204 LOC_R_4_3', '205 LOC_L_4_4', '206 LOC_R_4_4', '207 sOcG_L_2_1', '208 sOcG_R_2_1', '209 sOcG_L_2_2', '210 sOcG_R_2_2', '211 Amyg_L_2_1', '212 Amyg_R_2_1', '213 Amyg_L_2_2', '214 Amyg_R_2_2', '215 Hipp_L_2_1', '216 Hipp_R_2_1', '217 Hipp_L_2_2', '218 Hipp_R_2_2', '219 Str_L_6_1', '220 Str_R_6_1', '221 Str_L_6_2', '222 Str_R_6_2', '223 Str_L_6_3', '224 Str_R_6_3', '225 Str_L_6_4', '226 Str_R_6_4', '227 Str_L_6_5', '228 Str_R_6_5', '229 Str_L_6_6', '230 Str_R_6_6', '231 Tha_L_8_1', '232 Tha_R_8_1', '233 Tha_L_8_2', '234 Tha_R_8_2', '235 Tha_L_8_3', '236 Tha_R_8_3', '237 Tha_L_8_4', '238 Tha_R_8_4', '239 Tha_L_8_5', '240 Tha_R_8_5', '241 Tha_L_8_6', '242 Tha_R_8_6', '243 Tha_L_8_7', '244 Tha_R_8_7', '245 Tha_L_8_8', '246 Tha_R_8_8']
    else:
        raise NotImplementedError(f'get_BNA_ROIs, {code=}')
    return ROIs

