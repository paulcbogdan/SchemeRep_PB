import os
import pathlib
from collections import defaultdict
from nilearn import plotting
import matplotlib.pyplot as plt


path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

import numpy as np
import pandas as pd
from nilearn import image

from Utils.plotting_funcs import plot_connectivity
from marinate.pkld import pkld


def get_Schaefer_atlas(HCP=False, combine_bilateral=False,
                       schaefer=True, natview=False):
    if isinstance(schaefer, tuple):
        assert len(schaefer) == 2
        assert schaefer[0]
        n_roi = schaefer[1]
    else:
        n_roi = 1000
    atlas = {}

    from nilearn.datasets import fetch_atlas_schaefer_2018
    atlas_ni = fetch_atlas_schaefer_2018(n_rois=n_roi, resolution_mm=2,
                                         yeo_networks=17)

    fp_coords = fr'Utils/schaefer_coords/Schaefer2018_{n_roi}Parcels_17Networks_order_FSLMNI152_2mm.Centroid_RAS.csv'

    df_coords = pd.read_csv(fp_coords)
    atlas['coords'] = np.array(df_coords[['R', 'A', 'S']])

    if HCP:
        affine = [[-2., 0., 0., 90.],
                  [0., 2., 0., -126.],
                  [0., 0., 2., -72.],
                  [0., 0., 0., 1.]]
        affine = np.array(affine)
        atlas['maps'] = image.resample_img(atlas_ni['maps'], target_affine=affine,
                                           target_shape=(91, 109, 91),
                                           interpolation='nearest',
                                           force_resample=True,
                                           copy_header=True)
    elif natview:
        affine = [[-3., 0., 0., 90.],
                  [0., 3., 0., -126.],
                  [0., 0., 3., -72.],
                  [0., 0., 0., 1.]]
        affine = np.array(affine)
        atlas['maps'] = image.resample_img(atlas_ni['maps'], target_affine=affine,
                                           target_shape=(61, 73, 61),
                                           interpolation='nearest',
                                           force_resample=True,
                                           copy_header=True)
    else:
        fp_ref = (r'C:\PycharmProjects\SchemeRep\fMRI_in\102'
                  r'\ENC_GM20_LLS1_bpF_full\OBJ'
                  r'\ENC_sub102_run1_trial1_subset3_pairID29_object.nii')
        img_ref = image.load_img(fp_ref)
        atlas['maps'] = image.resample_to_img(atlas_ni['maps'], img_ref,
                                              interpolation='nearest',
                                              force_resample=True,
                                              copy_header=True)

    img_atlas = image.load_img(atlas['maps'])
    data = img_atlas.get_fdata()


    labels = atlas_ni['labels']
    labels = [str(label, encoding='utf-8') for label in labels]

    labels_ = []
    for label in labels:
        label = label.replace('Cinga', 'Cing').replace('Cingm', 'Cing')  # Just 1 & 4 ROIs
        label = label.replace('PrCv', 'PrC')  # Just 3 ROIs
        labels_.append(label)
    labels = labels_

    regions = [label.split('_')[-2] for label in labels]
    regions_unique = []
    regions_old_idxs = defaultdict(list)
    for old_idx, region in enumerate(regions):
        if region not in regions_old_idxs:
            regions_unique.append(region)
        regions_old_idxs[region].append(old_idx + 1)
    old2new = {}
    new2old = {}
    new_idx = 0
    for region in regions_unique:
        for old_idx in regions_old_idxs[region]:
            old2new[old_idx] = new_idx
            new2old[new_idx] = old_idx
            new_idx += 1

    new_labels = []
    new_coords = []
    for new_idx in range(0, n_roi):
        new_labels.append(labels[new2old[new_idx] - 1])
        new_coords.append(atlas['coords'][new2old[new_idx] - 1])
    atlas['coords'] = new_coords


    data_new = np.zeros_like(data)
    for og_idx, new_idx in old2new.items():
        data_new[data == og_idx] = new_idx + 1
    data = data_new

    atlas['maps'] = image.new_img_like(img_atlas, data)

    ROIs = new_labels
    # ROIs = [ROI.replace('PrC_', 'PrCv_').replace('Cinga_1', 'PFCmp_6').
    #         replace('PFCld', 'PFCl').replace('AntTemp', 'Temp')
    #         for ROI in ROIs]
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
        region_nums[region].append(i)  # int(ROI_num)-1)
        ROI_regions_laterality.append(region_LR)
    ROIs = ROIs_

    ticks = []
    tick_labels = []
    tick_lows = []
    regions = list(region_nums.keys())
    regions.sort()
    for region in regions:
        l = region_nums[region]
        ticks.append(np.mean(l))
        tick_labels.append(region)
        tick_lows.append(l[0])

    if combine_bilateral:
        data = atlas['maps'].get_fdata()
        data = (data + 1) - (data + 1) % 2
        data /= 2
        atlas['maps'] = image.new_img_like(atlas['maps'],
                                           data)
        ROIs = ROIs[::2]
        ROIs_ = []
        for ROI in ROIs:
            ROI = ROI.replace('_L', '')
            ROI_num, ROI_str = ROI.split(' ')
            ROI_num = int(ROI.split()[0])
            ROI_num = (ROI_num + 1) // 2
            ROI = f'{ROI_num} {ROI_str}'
            ROIs_.append(ROI)
        ROIs = ROIs_


    atlas['ROIs'] = atlas['labels'] = ROIs
    atlas['ROI_nums'] = ROI_nums
    atlas['n_ROIs'] = n_ROIs
    atlas['ROI_regions'] = ROI_regions  # repeats, length = # ROI
    atlas['ticks'] = ticks
    atlas['tick_labels'] = tick_labels  # no repeat, length = # regions
    atlas['tick_lows'] = tick_lows
    # for tick, label in zip(ticks, tick_labels):
    #     print(f'{tick=}, {label=}')
    # print(f'{tick_labels=}')

    # quit()
    atlas['name'] = 'schaefer'
    atlas['shenyang'] = False
    atlas['ROI_regions_laterality'] = ROI_regions_laterality
    atlas['ROI2coord'] = dict(zip(ROIs, atlas['coords']))
    return atlas


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
    atlas['ROI_regions'] = ROI_regions  # repeats, length = # ROI
    atlas['ticks'] = ticks
    atlas['tick_labels'] = tick_labels  # no repeat, length = # regions
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
        fp_labels = r'C:\PycharmProjects\SchemeRep\cache/BNA_labels_Lifu_ACC_ATL_fix.txt'
    else:
        fp_labels = r'C:\PycharmProjects\SchemeRep\cache/BNA_labels.txt'

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
        # G:\
        fp_ref = (r'C:\PycharmProjects\SchemeRep\fMRI_in\102'
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
                                           interpolation='nearest',
                                           force_resample=True,
                                           copy_header=True)
    elif natview:
        affine = [[-3., 0., 0., 90.],
                  [0., 3., 0., -126.],
                  [0., 0., 3., -72.],
                  [0., 0., 0., 1.]]
        affine = np.array(affine)
        atlas['maps'] = image.resample_img(atlas['maps'], target_affine=affine,
                                           target_shape=(61, 73, 61),
                                           interpolation='nearest',
                                           force_resample=True,
                                           copy_header=True)
    else:
        atlas['maps'] = image.resample_to_img(atlas['maps'], img,
                                              interpolation='nearest',
                                              force_resample=True,
                                              copy_header=True)
    atlas['shenyang'] = shenyang
    return atlas


def get_combined_BNA(combine_bilateral=False, new_space=True, HCP=False,
                     natview=False, lifu_labels=True, schaefer=False):
    if schaefer:
        atlas = get_Schaefer_atlas(HCP=HCP, combine_bilateral=combine_bilateral,
                                   schaefer=schaefer, natview=natview)
    else:
        atlas = get_BN_and_resample(combine_bilateral=combine_bilateral,
                                    new_space=new_space, HCP=HCP,
                                    natview=natview, lifu_labels=lifu_labels)

    data = atlas['maps'].get_fdata().astype(int)
    # print(f'{data.shape=}')

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


@pkld(store='both', overwrite=True)
def get_atlas(combine_regions=False, combine_bilateral=False,
              new_space=True, shenyang=True,
              HCP=False, natview=False, lifu_labels=True,
              schaefer=False):
    if combine_regions:
        atlas = get_combined_BNA(combine_bilateral=combine_bilateral,
                                 new_space=new_space, HCP=HCP,
                                 natview=natview, lifu_labels=lifu_labels,
                                 schaefer=schaefer)
    elif schaefer:
        atlas = get_Schaefer_atlas(HCP=HCP, combine_bilateral=combine_bilateral,
                                   schaefer=schaefer, natview=natview)
    else:
        atlas = get_BN_and_resample(combine_bilateral=combine_bilateral,
                                    new_space=new_space, shenyang=shenyang,
                                    HCP=HCP, natview=natview,
                                    lifu_labels=lifu_labels)
    return atlas


def org_BNA_coords():
    fp_coords_pre = r'C:\PycharmProjects\SchemeRep\cache\BNA_coords_pre.csv'
    df = pd.read_csv(fp_coords_pre)
    coords = []
    for l_coord, r_coord in zip(df['L_coord'], df['R_coord']):
        l_coord = [int(x.replace(' ', '')) for x in l_coord.split(',')]
        r_coord = [int(x.replace(' ', '')) for x in r_coord.split(',')]
        coords += [l_coord, r_coord]
    return coords


def get_BNA_ROIs(code=None):
    if code == 'BNA_region':
        ROIs = ['SFG_L', 'SFG_R', 'MFG_L', 'MFG_R', 'IFG_L', 'IFG_R', 'OrG_L', 'OrG_R', 'PrG_L', 'PrG_R', 'PCL_L',
                'PCL_R', 'ATL_L', 'ATL_R', 'STG_L', 'STG_R', 'MTG_L', 'MTG_R', 'ITG_L', 'ITG_R', 'FuG_L', 'FuG_R',
                'PhG_L', 'PhG_R', 'pSTS_L', 'pSTS_R', 'SPL_L', 'SPL_R', 'IPL_L', 'IPL_R', 'Pcun_L', 'Pcun_R', 'PoG_L',
                'PoG_R', 'INS_L', 'INS_R', 'PCC_L', 'PCC_R', 'ACC_L', 'ACC_R', 'EVC_L', 'EVC_R', 'LOC_L', 'LOC_R',
                'sOcG_L', 'sOcG_R', 'Amyg_L', 'Amyg_R', 'Hipp_L', 'Hipp_R', 'Str_L', 'Str_R', 'Tha_L', 'Tha_R']
    elif code == None or code == 'BNA':
        ROIs = ['1 SFG_L_7_1', '2 SFG_R_7_1', '3 SFG_L_7_2', '4 SFG_R_7_2', '5 SFG_L_7_3', '6 SFG_R_7_3', '7 SFG_L_7_4',
                '8 SFG_R_7_4', '9 SFG_L_7_5', '10 SFG_R_7_5', '11 SFG_L_7_6', '12 SFG_R_7_6', '13 SFG_L_7_7',
                '14 SFG_R_7_7', '15 MFG_L_7_1', '16 MFG_R_7_1', '17 MFG_L_7_2', '18 MFG_R_7_2', '19 MFG_L_7_3',
                '20 MFG_R_7_3', '21 MFG_L_7_4', '22 MFG_R_7_4', '23 MFG_L_7_5', '24 MFG_R_7_5', '25 MFG_L_7_6',
                '26 MFG_R_7_6', '27 MFG_L_7_7', '28 MFG_R_7_7', '29 IFG_L_6_1', '30 IFG_R_6_1', '31 IFG_L_6_2',
                '32 IFG_R_6_2', '33 IFG_L_6_3', '34 IFG_R_6_3', '35 IFG_L_6_4', '36 IFG_R_6_4', '37 IFG_L_6_5',
                '38 IFG_R_6_5', '39 IFG_L_6_6', '40 IFG_R_6_6', '41 OrG_L_6_1', '42 OrG_R_6_1', '43 OrG_L_6_2',
                '44 OrG_R_6_2', '45 OrG_L_6_3', '46 OrG_R_6_3', '47 OrG_L_6_4', '48 OrG_R_6_4', '49 OrG_L_6_5',
                '50 OrG_R_6_5', '51 OrG_L_6_6', '52 OrG_R_6_6', '53 PrG_L_6_1', '54 PrG_R_6_1', '55 PrG_L_6_2',
                '56 PrG_R_6_2', '57 PrG_L_6_3', '58 PrG_R_6_3', '59 PrG_L_6_4', '60 PrG_R_6_4', '61 PrG_L_6_5',
                '62 PrG_R_6_5', '63 PrG_L_6_6', '64 PrG_R_6_6', '65 PCL_L_2_1', '66 PCL_R_2_1', '67 PCL_L_2_2',
                '68 PCL_R_2_2', '69 ATL_L_6_1', '70 ATL_R_6_1', '71 STG_L_6_2', '72 STG_R_6_2', '73 STG_L_6_3',
                '74 STG_R_6_3', '75 STG_L_6_4', '76 STG_R_6_4', '77 ATL_L_6_5', '78 ATL_R_6_5', '79 STG_L_6_6',
                '80 STG_R_6_6', '81 MTG_L_4_1', '82 MTG_R_4_1', '83 ATL_L_4_2', '84 ATL_R_4_2', '85 MTG_L_4_3',
                '86 MTG_R_4_3', '87 MTG_L_4_4', '88 MTG_R_4_4', '89 ITG_L_7_1', '90 ITG_R_7_1', '91 ITG_L_7_2',
                '92 ITG_R_7_2', '93 ATL_L_7_3', '94 ATL_R_7_3', '95 ITG_L_7_4', '96 ITG_R_7_4', '97 ITG_L_7_5',
                '98 ITG_R_7_5', '99 ITG_L_7_6', '100 ITG_R_7_6', '101 ITG_L_7_7', '102 ITG_R_7_7', '103 FuG_L_3_1',
                '104 FuG_R_3_1', '105 FuG_L_3_2', '106 FuG_R_3_2', '107 FuG_L_3_3', '108 FuG_R_3_3', '109 PhG_L_6_1',
                '110 PhG_R_6_1', '111 PhG_L_6_2', '112 PhG_R_6_2', '113 PhG_L_6_3', '114 PhG_R_6_3', '115 PhG_L_6_4',
                '116 PhG_R_6_4', '117 PhG_L_6_5', '118 PhG_R_6_5', '119 PhG_L_6_6', '120 PhG_R_6_6', '121 pSTS_L_2_1',
                '122 pSTS_R_2_1', '123 pSTS_L_2_2', '124 pSTS_R_2_2', '125 SPL_L_5_1', '126 SPL_R_5_1', '127 SPL_L_5_2',
                '128 SPL_R_5_2', '129 SPL_L_5_3', '130 SPL_R_5_3', '131 SPL_L_5_4', '132 SPL_R_5_4', '133 SPL_L_5_5',
                '134 SPL_R_5_5', '135 IPL_L_6_1', '136 IPL_R_6_1', '137 IPL_L_6_2', '138 IPL_R_6_2', '139 IPL_L_6_3',
                '140 IPL_R_6_3', '141 IPL_L_6_4', '142 IPL_R_6_4', '143 IPL_L_6_5', '144 IPL_R_6_5', '145 IPL_L_6_6',
                '146 IPL_R_6_6', '147 Pcun_L_4_1', '148 Pcun_R_4_1', '149 Pcun_L_4_2', '150 Pcun_R_4_2',
                '151 Pcun_L_4_3', '152 Pcun_R_4_3', '153 Pcun_L_4_4', '154 Pcun_R_4_4', '155 PoG_L_4_1',
                '156 PoG_R_4_1', '157 PoG_L_4_2', '158 PoG_R_4_2', '159 PoG_L_4_3', '160 PoG_R_4_3', '161 PoG_L_4_4',
                '162 PoG_R_4_4', '163 INS_L_6_1', '164 INS_R_6_1', '165 INS_L_6_2', '166 INS_R_6_2', '167 INS_L_6_3',
                '168 INS_R_6_3', '169 INS_L_6_4', '170 INS_R_6_4', '171 INS_L_6_5', '172 INS_R_6_5', '173 INS_L_6_6',
                '174 INS_R_6_6', '175 PCC_L_7_1', '176 PCC_R_7_1', '177 ACC_L_7_2', '178 ACC_R_7_2', '179 ACC_L_7_3',
                '180 ACC_R_7_3', '181 PCC_L_7_4', '182 PCC_R_7_4', '183 ACC_L_7_5', '184 ACC_R_7_5', '185 PCC_L_7_6',
                '186 PCC_R_7_6', '187 ACC_L_7_7', '188 ACC_R_7_7', '189 EVC_L_5_1', '190 EVC_R_5_1', '191 EVC_L_5_2',
                '192 EVC_R_5_2', '193 EVC_L_5_3', '194 EVC_R_5_3', '195 EVC_L_5_4', '196 EVC_R_5_4', '197 EVC_L_5_5',
                '198 EVC_R_5_5', '199 LOC_L_4_1', '200 LOC_R_4_1', '201 LOC_L_4_2', '202 LOC_R_4_2', '203 LOC_L_4_3',
                '204 LOC_R_4_3', '205 LOC_L_4_4', '206 LOC_R_4_4', '207 sOcG_L_2_1', '208 sOcG_R_2_1', '209 sOcG_L_2_2',
                '210 sOcG_R_2_2', '211 Amyg_L_2_1', '212 Amyg_R_2_1', '213 Amyg_L_2_2', '214 Amyg_R_2_2',
                '215 Hipp_L_2_1', '216 Hipp_R_2_1', '217 Hipp_L_2_2', '218 Hipp_R_2_2', '219 Str_L_6_1',
                '220 Str_R_6_1', '221 Str_L_6_2', '222 Str_R_6_2', '223 Str_L_6_3', '224 Str_R_6_3', '225 Str_L_6_4',
                '226 Str_R_6_4', '227 Str_L_6_5', '228 Str_R_6_5', '229 Str_L_6_6', '230 Str_R_6_6', '231 Tha_L_8_1',
                '232 Tha_R_8_1', '233 Tha_L_8_2', '234 Tha_R_8_2', '235 Tha_L_8_3', '236 Tha_R_8_3', '237 Tha_L_8_4',
                '238 Tha_R_8_4', '239 Tha_L_8_5', '240 Tha_R_8_5', '241 Tha_L_8_6', '242 Tha_R_8_6', '243 Tha_L_8_7',
                '244 Tha_R_8_7', '245 Tha_L_8_8', '246 Tha_R_8_8']
    elif code == 'schaefer':
        ROIs = ['17Networks_L_VisCent_Striate_1', '17Networks_L_VisCent_Striate_2', '17Networks_L_VisCent_Striate_3',
                '17Networks_L_VisCent_Striate_4', '17Networks_R_VisCent_Striate_1', '17Networks_R_VisCent_Striate_2',
                '17Networks_R_VisCent_Striate_3', '17Networks_R_VisCent_Striate_4', '17Networks_L_VisCent_ExStr_1',
                '17Networks_L_VisCent_ExStr_2', '17Networks_L_VisCent_ExStr_3', '17Networks_L_VisCent_ExStr_4',
                '17Networks_L_VisCent_ExStr_5', '17Networks_L_VisCent_ExStr_6', '17Networks_L_VisCent_ExStr_7',
                '17Networks_L_VisCent_ExStr_8', '17Networks_L_VisCent_ExStr_9', '17Networks_L_VisCent_ExStr_10',
                '17Networks_L_VisCent_ExStr_11', '17Networks_L_VisCent_ExStr_12', '17Networks_L_VisCent_ExStr_13',
                '17Networks_L_VisCent_ExStr_14', '17Networks_L_VisCent_ExStr_15', '17Networks_L_VisCent_ExStr_16',
                '17Networks_L_VisCent_ExStr_17', '17Networks_L_VisCent_ExStr_18', '17Networks_L_VisCent_ExStr_19',
                '17Networks_L_VisCent_ExStr_20', '17Networks_L_VisCent_ExStr_21', '17Networks_L_VisCent_ExStr_22',
                '17Networks_L_VisCent_ExStr_23', '17Networks_L_VisCent_ExStr_24', '17Networks_L_VisCent_ExStr_25',
                '17Networks_L_VisCent_ExStr_26', '17Networks_L_VisCent_ExStr_27', '17Networks_L_VisCent_ExStr_28',
                '17Networks_L_VisCent_ExStr_29', '17Networks_L_VisCent_ExStr_30', '17Networks_R_VisCent_ExStr_1',
                '17Networks_R_VisCent_ExStr_2', '17Networks_R_VisCent_ExStr_3', '17Networks_R_VisCent_ExStr_4',
                '17Networks_R_VisCent_ExStr_5', '17Networks_R_VisCent_ExStr_6', '17Networks_R_VisCent_ExStr_7',
                '17Networks_R_VisCent_ExStr_8', '17Networks_R_VisCent_ExStr_9', '17Networks_R_VisCent_ExStr_10',
                '17Networks_R_VisCent_ExStr_11', '17Networks_R_VisCent_ExStr_12', '17Networks_R_VisCent_ExStr_13',
                '17Networks_R_VisCent_ExStr_14', '17Networks_R_VisCent_ExStr_15', '17Networks_R_VisCent_ExStr_16',
                '17Networks_R_VisCent_ExStr_17', '17Networks_R_VisCent_ExStr_18', '17Networks_R_VisCent_ExStr_19',
                '17Networks_R_VisCent_ExStr_20', '17Networks_R_VisCent_ExStr_21', '17Networks_R_VisCent_ExStr_22',
                '17Networks_R_VisCent_ExStr_23', '17Networks_R_VisCent_ExStr_24', '17Networks_R_VisCent_ExStr_25',
                '17Networks_R_VisCent_ExStr_26', '17Networks_R_VisCent_ExStr_27', '17Networks_R_VisCent_ExStr_28',
                '17Networks_R_VisCent_ExStr_29', '17Networks_R_VisCent_ExStr_30', '17Networks_R_VisCent_ExStr_31',
                '17Networks_R_VisCent_ExStr_32', '17Networks_R_VisCent_ExStr_33', '17Networks_R_VisCent_ExStr_34',
                '17Networks_R_VisCent_ExStr_35', '17Networks_L_VisPeri_StriCal_1', '17Networks_L_VisPeri_StriCal_2',
                '17Networks_L_VisPeri_StriCal_3', '17Networks_L_VisPeri_StriCal_4', '17Networks_L_VisPeri_StriCal_5',
                '17Networks_L_VisPeri_StriCal_6', '17Networks_L_VisPeri_StriCal_7', '17Networks_L_VisPeri_StriCal_8',
                '17Networks_L_VisPeri_StriCal_9', '17Networks_L_VisPeri_StriCal_10', '17Networks_R_VisPeri_StriCal_1',
                '17Networks_R_VisPeri_StriCal_2', '17Networks_R_VisPeri_StriCal_3', '17Networks_R_VisPeri_StriCal_4',
                '17Networks_R_VisPeri_StriCal_5', '17Networks_R_VisPeri_StriCal_6', '17Networks_L_VisPeri_ExStrInf_1',
                '17Networks_L_VisPeri_ExStrInf_2', '17Networks_L_VisPeri_ExStrInf_3', '17Networks_L_VisPeri_ExStrInf_4',
                '17Networks_L_VisPeri_ExStrInf_5', '17Networks_L_VisPeri_ExStrInf_6', '17Networks_L_VisPeri_ExStrInf_7',
                '17Networks_L_VisPeri_ExStrInf_8', '17Networks_L_VisPeri_ExStrInf_9',
                '17Networks_L_VisPeri_ExStrInf_10', '17Networks_R_VisPeri_ExStrInf_1',
                '17Networks_R_VisPeri_ExStrInf_2', '17Networks_R_VisPeri_ExStrInf_3', '17Networks_R_VisPeri_ExStrInf_4',
                '17Networks_R_VisPeri_ExStrInf_5', '17Networks_R_VisPeri_ExStrInf_6', '17Networks_R_VisPeri_ExStrInf_7',
                '17Networks_R_VisPeri_ExStrInf_8', '17Networks_R_VisPeri_ExStrInf_9',
                '17Networks_R_VisPeri_ExStrInf_10', '17Networks_R_VisPeri_ExStrInf_11',
                '17Networks_R_VisPeri_ExStrInf_12', '17Networks_L_VisPeri_ExStrSup_1',
                '17Networks_L_VisPeri_ExStrSup_2', '17Networks_L_VisPeri_ExStrSup_3', '17Networks_L_VisPeri_ExStrSup_4',
                '17Networks_L_VisPeri_ExStrSup_5', '17Networks_L_VisPeri_ExStrSup_6', '17Networks_L_VisPeri_ExStrSup_7',
                '17Networks_L_VisPeri_ExStrSup_8', '17Networks_L_VisPeri_ExStrSup_9',
                '17Networks_L_VisPeri_ExStrSup_10', '17Networks_L_VisPeri_ExStrSup_11',
                '17Networks_L_VisPeri_ExStrSup_12', '17Networks_R_VisPeri_ExStrSup_1',
                '17Networks_R_VisPeri_ExStrSup_2', '17Networks_R_VisPeri_ExStrSup_3', '17Networks_R_VisPeri_ExStrSup_4',
                '17Networks_R_VisPeri_ExStrSup_5', '17Networks_R_VisPeri_ExStrSup_6', '17Networks_R_VisPeri_ExStrSup_7',
                '17Networks_R_VisPeri_ExStrSup_8', '17Networks_R_VisPeri_ExStrSup_9',
                '17Networks_R_VisPeri_ExStrSup_10', '17Networks_R_VisPeri_ExStrSup_11', '17Networks_L_SomMotA_1',
                '17Networks_L_SomMotA_2', '17Networks_L_SomMotA_3', '17Networks_L_SomMotA_4', '17Networks_L_SomMotA_5',
                '17Networks_L_SomMotA_6', '17Networks_L_SomMotA_7', '17Networks_L_SomMotA_8', '17Networks_L_SomMotA_9',
                '17Networks_L_SomMotA_10', '17Networks_L_SomMotA_11', '17Networks_L_SomMotA_12',
                '17Networks_L_SomMotA_13', '17Networks_L_SomMotA_14', '17Networks_L_SomMotA_15',
                '17Networks_L_SomMotA_16', '17Networks_L_SomMotA_17', '17Networks_L_SomMotA_18',
                '17Networks_L_SomMotA_19', '17Networks_L_SomMotA_20', '17Networks_L_SomMotA_21',
                '17Networks_L_SomMotA_22', '17Networks_L_SomMotA_23', '17Networks_L_SomMotA_24',
                '17Networks_L_SomMotA_25', '17Networks_L_SomMotA_26', '17Networks_L_SomMotA_27',
                '17Networks_L_SomMotA_28', '17Networks_L_SomMotA_29', '17Networks_L_SomMotA_30',
                '17Networks_L_SomMotA_31', '17Networks_L_SomMotA_32', '17Networks_L_SomMotA_33',
                '17Networks_L_SomMotA_34', '17Networks_L_SomMotA_35', '17Networks_L_SomMotA_36',
                '17Networks_L_SomMotA_37', '17Networks_L_SomMotA_38', '17Networks_L_SomMotA_39',
                '17Networks_L_SomMotA_40', '17Networks_L_SomMotA_41', '17Networks_L_SomMotA_42',
                '17Networks_R_SomMotA_1', '17Networks_R_SomMotA_2', '17Networks_R_SomMotA_3', '17Networks_R_SomMotA_4',
                '17Networks_R_SomMotA_5', '17Networks_R_SomMotA_6', '17Networks_R_SomMotA_7', '17Networks_R_SomMotA_8',
                '17Networks_R_SomMotA_9', '17Networks_R_SomMotA_10', '17Networks_R_SomMotA_11',
                '17Networks_R_SomMotA_12', '17Networks_R_SomMotA_13', '17Networks_R_SomMotA_14',
                '17Networks_R_SomMotA_15', '17Networks_R_SomMotA_16', '17Networks_R_SomMotA_17',
                '17Networks_R_SomMotA_18', '17Networks_R_SomMotA_19', '17Networks_R_SomMotA_20',
                '17Networks_R_SomMotA_21', '17Networks_R_SomMotA_22', '17Networks_R_SomMotA_23',
                '17Networks_R_SomMotA_24', '17Networks_R_SomMotA_25', '17Networks_R_SomMotA_26',
                '17Networks_R_SomMotA_27', '17Networks_R_SomMotA_28', '17Networks_R_SomMotA_29',
                '17Networks_R_SomMotA_30', '17Networks_R_SomMotA_31', '17Networks_R_SomMotA_32',
                '17Networks_R_SomMotA_33', '17Networks_R_SomMotA_34', '17Networks_R_SomMotA_35',
                '17Networks_R_SomMotA_36', '17Networks_R_SomMotA_37', '17Networks_R_SomMotA_38',
                '17Networks_R_SomMotA_39', '17Networks_R_SomMotA_40', '17Networks_R_SomMotA_41',
                '17Networks_R_SomMotA_42', '17Networks_R_SomMotA_43', '17Networks_R_SomMotA_44',
                '17Networks_R_SomMotA_45', '17Networks_R_SomMotA_46', '17Networks_R_SomMotA_47',
                '17Networks_R_SomMotA_48', '17Networks_L_SomMotB_Cent_1', '17Networks_L_SomMotB_Cent_2',
                '17Networks_L_SomMotB_Cent_3', '17Networks_L_SomMotB_Cent_4', '17Networks_L_SomMotB_Cent_5',
                '17Networks_L_SomMotB_Cent_6', '17Networks_L_SomMotB_Cent_7', '17Networks_L_SomMotB_Cent_8',
                '17Networks_L_SomMotB_Cent_9', '17Networks_L_SomMotB_Cent_10', '17Networks_L_SomMotB_Cent_11',
                '17Networks_R_SomMotB_Cent_1', '17Networks_R_SomMotB_Cent_2', '17Networks_R_SomMotB_Cent_3',
                '17Networks_R_SomMotB_Cent_4', '17Networks_R_SomMotB_Cent_5', '17Networks_R_SomMotB_Cent_6',
                '17Networks_R_SomMotB_Cent_7', '17Networks_R_SomMotB_Cent_8', '17Networks_R_SomMotB_Cent_9',
                '17Networks_R_SomMotB_Cent_10', '17Networks_R_SomMotB_Cent_11', '17Networks_R_SomMotB_Cent_12',
                '17Networks_L_SomMotB_S2_1', '17Networks_L_SomMotB_S2_2', '17Networks_L_SomMotB_S2_3',
                '17Networks_L_SomMotB_S2_4', '17Networks_L_SomMotB_S2_5', '17Networks_L_SomMotB_S2_6',
                '17Networks_L_SomMotB_S2_7', '17Networks_L_SomMotB_S2_8', '17Networks_L_SomMotB_S2_9',
                '17Networks_L_SomMotB_S2_10', '17Networks_L_SomMotB_S2_11', '17Networks_L_SomMotB_S2_12',
                '17Networks_L_SomMotB_S2_13', '17Networks_R_SomMotB_S2_1', '17Networks_R_SomMotB_S2_2',
                '17Networks_R_SomMotB_S2_3', '17Networks_R_SomMotB_S2_4', '17Networks_R_SomMotB_S2_5',
                '17Networks_R_SomMotB_S2_6', '17Networks_R_SomMotB_S2_7', '17Networks_R_SomMotB_S2_8',
                '17Networks_R_SomMotB_S2_9', '17Networks_R_SomMotB_S2_10', '17Networks_R_SomMotB_S2_11',
                '17Networks_R_SomMotB_S2_12', '17Networks_R_SomMotB_S2_13', '17Networks_R_SomMotB_S2_14',
                '17Networks_R_SomMotB_S2_15', '17Networks_L_SomMotB_Ins_1', '17Networks_L_SalVentAttnA_Ins_1',
                '17Networks_L_SalVentAttnA_Ins_2', '17Networks_L_SalVentAttnA_Ins_3', '17Networks_L_SalVentAttnA_Ins_4',
                '17Networks_L_SalVentAttnA_Ins_5', '17Networks_L_SalVentAttnA_Ins_6', '17Networks_L_SalVentAttnA_Ins_7',
                '17Networks_L_SalVentAttnA_Ins_8', '17Networks_L_SalVentAttnA_Ins_9', '17Networks_L_SalVentAttnB_Ins_1',
                '17Networks_L_SalVentAttnB_Ins_2', '17Networks_L_SalVentAttnB_Ins_3', '17Networks_L_SalVentAttnB_Ins_4',
                '17Networks_L_SalVentAttnB_Ins_5', '17Networks_L_SalVentAttnB_Ins_6', '17Networks_L_SalVentAttnB_Ins_7',
                '17Networks_L_SalVentAttnB_Ins_8', '17Networks_R_SomMotB_Ins_1', '17Networks_R_SalVentAttnA_Ins_1',
                '17Networks_R_SalVentAttnA_Ins_2', '17Networks_R_SalVentAttnA_Ins_3', '17Networks_R_SalVentAttnA_Ins_4',
                '17Networks_R_SalVentAttnA_Ins_5', '17Networks_R_SalVentAttnA_Ins_6', '17Networks_R_SalVentAttnA_Ins_7',
                '17Networks_R_SalVentAttnA_Ins_8', '17Networks_R_SalVentAttnA_Ins_9', '17Networks_R_SalVentAttnB_Ins_1',
                '17Networks_R_SalVentAttnB_Ins_2', '17Networks_R_SalVentAttnB_Ins_3', '17Networks_R_SalVentAttnB_Ins_4',
                '17Networks_R_SalVentAttnB_Ins_5', '17Networks_R_SalVentAttnB_Ins_6', '17Networks_R_SalVentAttnB_Ins_7',
                '17Networks_R_SalVentAttnB_Ins_8', '17Networks_R_SalVentAttnB_Ins_9', '17Networks_L_SomMotB_Aud_1',
                '17Networks_L_SomMotB_Aud_2', '17Networks_L_SomMotB_Aud_3', '17Networks_L_SomMotB_Aud_4',
                '17Networks_L_SomMotB_Aud_5', '17Networks_L_SomMotB_Aud_6', '17Networks_L_SomMotB_Aud_7',
                '17Networks_L_SomMotB_Aud_8', '17Networks_L_SomMotB_Aud_9', '17Networks_L_SomMotB_Aud_10',
                '17Networks_L_SomMotB_Aud_11', '17Networks_L_SomMotB_Aud_12', '17Networks_L_SomMotB_Aud_13',
                '17Networks_L_SomMotB_Aud_14', '17Networks_R_SomMotB_Aud_1', '17Networks_R_SomMotB_Aud_2',
                '17Networks_R_SomMotB_Aud_3', '17Networks_R_SomMotB_Aud_4', '17Networks_R_SomMotB_Aud_5',
                '17Networks_R_SomMotB_Aud_6', '17Networks_R_SomMotB_Aud_7', '17Networks_R_SomMotB_Aud_8',
                '17Networks_R_SomMotB_Aud_9', '17Networks_R_SomMotB_Aud_10', '17Networks_R_SomMotB_Aud_11',
                '17Networks_R_SomMotB_Aud_12', '17Networks_R_SomMotB_Aud_13', '17Networks_R_SomMotB_Aud_14',
                '17Networks_L_DorsAttnA_TempOcc_1', '17Networks_L_DorsAttnA_TempOcc_2',
                '17Networks_L_DorsAttnA_TempOcc_3', '17Networks_L_DorsAttnA_TempOcc_4',
                '17Networks_L_DorsAttnA_TempOcc_5', '17Networks_L_DorsAttnA_TempOcc_6',
                '17Networks_L_DorsAttnA_TempOcc_7', '17Networks_L_DorsAttnA_TempOcc_8',
                '17Networks_L_DorsAttnA_TempOcc_9', '17Networks_L_DorsAttnA_TempOcc_10',
                '17Networks_L_DorsAttnB_TempOcc_1', '17Networks_L_DorsAttnB_TempOcc_2',
                '17Networks_R_DorsAttnA_TempOcc_1', '17Networks_R_DorsAttnA_TempOcc_2',
                '17Networks_R_DorsAttnA_TempOcc_3', '17Networks_R_DorsAttnA_TempOcc_4',
                '17Networks_R_DorsAttnA_TempOcc_5', '17Networks_R_DorsAttnA_TempOcc_6',
                '17Networks_R_DorsAttnA_TempOcc_7', '17Networks_R_DorsAttnA_TempOcc_8',
                '17Networks_R_DorsAttnB_TempOcc_1', '17Networks_L_DorsAttnA_ParOcc_1',
                '17Networks_L_DorsAttnA_ParOcc_2', '17Networks_L_DorsAttnA_ParOcc_3', '17Networks_L_DorsAttnA_ParOcc_4',
                '17Networks_L_DorsAttnA_ParOcc_5', '17Networks_L_DorsAttnA_ParOcc_6', '17Networks_L_DorsAttnA_ParOcc_7',
                '17Networks_R_DorsAttnA_ParOcc_1', '17Networks_R_DorsAttnA_ParOcc_2', '17Networks_R_DorsAttnA_ParOcc_3',
                '17Networks_R_DorsAttnA_ParOcc_4', '17Networks_R_DorsAttnA_ParOcc_5', '17Networks_R_DorsAttnA_ParOcc_6',
                '17Networks_R_DorsAttnA_ParOcc_7', '17Networks_R_DorsAttnA_ParOcc_8', '17Networks_R_DorsAttnA_ParOcc_9',
                '17Networks_L_DorsAttnA_SPL_1', '17Networks_L_DorsAttnA_SPL_2', '17Networks_L_DorsAttnA_SPL_3',
                '17Networks_L_DorsAttnA_SPL_4', '17Networks_L_DorsAttnA_SPL_5', '17Networks_L_DorsAttnA_SPL_6',
                '17Networks_L_DorsAttnA_SPL_7', '17Networks_L_DorsAttnA_SPL_8', '17Networks_L_DorsAttnA_SPL_9',
                '17Networks_L_DorsAttnA_SPL_10', '17Networks_L_DorsAttnA_SPL_11', '17Networks_L_DorsAttnA_SPL_12',
                '17Networks_L_DorsAttnA_SPL_13', '17Networks_L_DorsAttnA_SPL_14', '17Networks_L_DorsAttnA_SPL_15',
                '17Networks_L_DorsAttnA_SPL_16', '17Networks_L_DorsAttnA_SPL_17', '17Networks_R_DorsAttnA_SPL_1',
                '17Networks_R_DorsAttnA_SPL_2', '17Networks_R_DorsAttnA_SPL_3', '17Networks_R_DorsAttnA_SPL_4',
                '17Networks_R_DorsAttnA_SPL_5', '17Networks_R_DorsAttnA_SPL_6', '17Networks_R_DorsAttnA_SPL_7',
                '17Networks_R_DorsAttnA_SPL_8', '17Networks_R_DorsAttnA_SPL_9', '17Networks_R_DorsAttnA_SPL_10',
                '17Networks_R_DorsAttnA_SPL_11', '17Networks_R_DorsAttnA_SPL_12', '17Networks_R_DorsAttnA_SPL_13',
                '17Networks_R_DorsAttnA_SPL_14', '17Networks_R_DorsAttnA_SPL_15', '17Networks_R_DorsAttnA_SPL_16',
                '17Networks_R_DorsAttnA_SPL_17', '17Networks_L_DorsAttnB_PostC_1', '17Networks_L_DorsAttnB_PostC_2',
                '17Networks_L_DorsAttnB_PostC_3', '17Networks_L_DorsAttnB_PostC_4', '17Networks_L_DorsAttnB_PostC_5',
                '17Networks_L_DorsAttnB_PostC_6', '17Networks_L_DorsAttnB_PostC_7', '17Networks_L_DorsAttnB_PostC_8',
                '17Networks_L_DorsAttnB_PostC_9', '17Networks_L_DorsAttnB_PostC_10', '17Networks_L_DorsAttnB_PostC_11',
                '17Networks_L_DorsAttnB_PostC_12', '17Networks_L_DorsAttnB_PostC_13', '17Networks_L_DorsAttnB_PostC_14',
                '17Networks_L_DorsAttnB_PostC_15', '17Networks_L_DorsAttnB_PostC_16', '17Networks_L_DorsAttnB_PostC_17',
                '17Networks_L_DorsAttnB_PostC_18', '17Networks_R_DorsAttnB_PostC_1', '17Networks_R_DorsAttnB_PostC_2',
                '17Networks_R_DorsAttnB_PostC_3', '17Networks_R_DorsAttnB_PostC_4', '17Networks_R_DorsAttnB_PostC_5',
                '17Networks_R_DorsAttnB_PostC_6', '17Networks_R_DorsAttnB_PostC_7', '17Networks_R_DorsAttnB_PostC_8',
                '17Networks_R_DorsAttnB_PostC_9', '17Networks_R_DorsAttnB_PostC_10', '17Networks_R_DorsAttnB_PostC_11',
                '17Networks_R_DorsAttnB_PostC_12', '17Networks_R_DorsAttnB_PostC_13', '17Networks_R_DorsAttnB_PostC_14',
                '17Networks_R_DorsAttnB_PostC_15', '17Networks_R_DorsAttnB_PostC_16', '17Networks_R_DorsAttnB_PostC_17',
                '17Networks_R_DorsAttnB_PostC_18', '17Networks_R_DorsAttnB_PostC_19', '17Networks_R_DorsAttnB_PostC_20',
                '17Networks_L_DorsAttnB_FEF_1', '17Networks_L_DorsAttnB_FEF_2', '17Networks_L_DorsAttnB_FEF_3',
                '17Networks_L_DorsAttnB_FEF_4', '17Networks_L_DorsAttnB_FEF_5', '17Networks_L_DorsAttnB_FEF_6',
                '17Networks_L_DorsAttnB_FEF_7', '17Networks_R_DorsAttnB_FEF_1', '17Networks_R_DorsAttnB_FEF_2',
                '17Networks_R_DorsAttnB_FEF_3', '17Networks_R_DorsAttnB_FEF_4', '17Networks_R_DorsAttnB_FEF_5',
                '17Networks_R_DorsAttnB_FEF_6', '17Networks_L_DorsAttnB_PrCv_1', '17Networks_L_DorsAttnB_PrCv_2',
                '17Networks_R_DorsAttnB_PrCv_1', '17Networks_L_SalVentAttnA_ParOper_1',
                '17Networks_L_SalVentAttnA_ParOper_2', '17Networks_L_SalVentAttnA_ParOper_3',
                '17Networks_L_SalVentAttnA_ParOper_4', '17Networks_L_SalVentAttnA_ParOper_5',
                '17Networks_L_SalVentAttnA_ParOper_6', '17Networks_L_SalVentAttnA_ParOper_7',
                '17Networks_R_SalVentAttnA_ParOper_1', '17Networks_R_SalVentAttnA_ParOper_2',
                '17Networks_R_SalVentAttnA_ParOper_3', '17Networks_R_SalVentAttnA_ParOper_4',
                '17Networks_R_SalVentAttnA_ParOper_5', '17Networks_R_SalVentAttnA_ParOper_6',
                '17Networks_R_SalVentAttnA_ParOper_7', '17Networks_R_SalVentAttnA_ParOper_8',
                '17Networks_R_SalVentAttnA_ParOper_9', '17Networks_R_SalVentAttnA_ParOper_10',
                '17Networks_L_SalVentAttnA_FrOper_1', '17Networks_L_SalVentAttnA_FrOper_2',
                '17Networks_L_SalVentAttnA_FrOper_3', '17Networks_L_SalVentAttnA_FrOper_4',
                '17Networks_L_SalVentAttnA_FrOper_5', '17Networks_L_SalVentAttnA_FrOper_6',
                '17Networks_L_SalVentAttnA_FrOper_7', '17Networks_L_SalVentAttnA_FrOper_8',
                '17Networks_L_SalVentAttnA_FrOper_9', '17Networks_R_SalVentAttnA_FrOper_1',
                '17Networks_R_SalVentAttnA_FrOper_2', '17Networks_R_SalVentAttnA_FrOper_3',
                '17Networks_R_SalVentAttnA_FrOper_4', '17Networks_R_SalVentAttnA_FrOper_5',
                '17Networks_R_SalVentAttnA_FrOper_6', '17Networks_R_SalVentAttnA_FrOper_7',
                '17Networks_R_SalVentAttnA_FrOper_8', '17Networks_L_SalVentAttnA_ParMed_1',
                '17Networks_L_SalVentAttnA_ParMed_2', '17Networks_L_SalVentAttnA_ParMed_3',
                '17Networks_L_SalVentAttnA_ParMed_4', '17Networks_L_SalVentAttnA_ParMed_5',
                '17Networks_L_SalVentAttnA_ParMed_6', '17Networks_L_SalVentAttnA_ParMed_7',
                '17Networks_L_SalVentAttnA_ParMed_8', '17Networks_R_SalVentAttnA_ParMed_1',
                '17Networks_R_SalVentAttnA_ParMed_2', '17Networks_R_SalVentAttnA_ParMed_3',
                '17Networks_R_SalVentAttnA_ParMed_4', '17Networks_R_SalVentAttnA_ParMed_5',
                '17Networks_R_SalVentAttnA_ParMed_6', '17Networks_R_SalVentAttnA_ParMed_7',
                '17Networks_R_SalVentAttnA_ParMed_8', '17Networks_R_SalVentAttnA_ParMed_9',
                '17Networks_R_SalVentAttnA_ParMed_10', '17Networks_R_SalVentAttnA_ParMed_11',
                '17Networks_L_SalVentAttnA_FrMed_1', '17Networks_L_SalVentAttnA_FrMed_2',
                '17Networks_L_SalVentAttnA_FrMed_3', '17Networks_L_SalVentAttnA_FrMed_4',
                '17Networks_L_SalVentAttnA_FrMed_5', '17Networks_L_SalVentAttnA_FrMed_6',
                '17Networks_L_SalVentAttnA_FrMed_7', '17Networks_R_SalVentAttnA_FrMed_1',
                '17Networks_R_SalVentAttnA_FrMed_2', '17Networks_R_SalVentAttnA_FrMed_3',
                '17Networks_R_SalVentAttnA_FrMed_4', '17Networks_R_SalVentAttnA_FrMed_5',
                '17Networks_R_SalVentAttnA_FrMed_6', '17Networks_R_SalVentAttnA_FrMed_7',
                '17Networks_R_SalVentAttnA_FrMed_8', '17Networks_R_SalVentAttnA_FrMed_9',
                '17Networks_R_SalVentAttnA_FrMed_10', '17Networks_L_SalVentAttnB_IPL_1',
                '17Networks_L_SalVentAttnB_IPL_2', '17Networks_L_ContB_IPL_1', '17Networks_L_ContB_IPL_2',
                '17Networks_L_ContB_IPL_3', '17Networks_L_ContB_IPL_4', '17Networks_L_ContB_IPL_5',
                '17Networks_L_ContB_IPL_6', '17Networks_L_ContB_IPL_7', '17Networks_L_ContB_IPL_8',
                '17Networks_L_DefaultA_IPL_1', '17Networks_L_DefaultA_IPL_2', '17Networks_L_DefaultA_IPL_3',
                '17Networks_L_DefaultA_IPL_4', '17Networks_L_DefaultA_IPL_5', '17Networks_L_DefaultA_IPL_6',
                '17Networks_L_DefaultA_IPL_7', '17Networks_L_DefaultB_IPL_1', '17Networks_L_DefaultB_IPL_2',
                '17Networks_L_DefaultB_IPL_3', '17Networks_L_DefaultB_IPL_4', '17Networks_L_DefaultB_IPL_5',
                '17Networks_L_DefaultB_IPL_6', '17Networks_L_DefaultB_IPL_7', '17Networks_L_DefaultB_IPL_8',
                '17Networks_L_DefaultB_IPL_9', '17Networks_L_DefaultC_IPL_1', '17Networks_L_DefaultC_IPL_2',
                '17Networks_L_DefaultC_IPL_3', '17Networks_R_SalVentAttnB_IPL_1', '17Networks_R_SalVentAttnB_IPL_2',
                '17Networks_R_SalVentAttnB_IPL_3', '17Networks_R_SalVentAttnB_IPL_4', '17Networks_R_ContB_IPL_1',
                '17Networks_R_ContB_IPL_2', '17Networks_R_ContB_IPL_3', '17Networks_R_ContB_IPL_4',
                '17Networks_R_ContB_IPL_5', '17Networks_R_ContB_IPL_6', '17Networks_R_ContB_IPL_7',
                '17Networks_R_ContB_IPL_8', '17Networks_R_ContB_IPL_9', '17Networks_R_ContB_IPL_10',
                '17Networks_R_ContB_IPL_11', '17Networks_R_ContB_IPL_12', '17Networks_R_DefaultA_IPL_1',
                '17Networks_R_DefaultA_IPL_2', '17Networks_R_DefaultA_IPL_3', '17Networks_R_DefaultA_IPL_4',
                '17Networks_R_DefaultA_IPL_5', '17Networks_R_DefaultA_IPL_6', '17Networks_R_DefaultA_IPL_7',
                '17Networks_R_DefaultA_IPL_8', '17Networks_R_DefaultC_IPL_1', '17Networks_R_DefaultC_IPL_2',
                '17Networks_L_SalVentAttnB_PFCd_1', '17Networks_L_ContA_PFCd_1', '17Networks_L_ContA_PFCd_2',
                '17Networks_L_ContB_PFCd_1', '17Networks_L_ContB_PFCd_2', '17Networks_L_ContB_PFCd_3',
                '17Networks_L_DefaultA_PFCd_1', '17Networks_L_DefaultA_PFCd_2', '17Networks_L_DefaultA_PFCd_3',
                '17Networks_L_DefaultA_PFCd_4', '17Networks_L_DefaultB_PFCd_1', '17Networks_L_DefaultB_PFCd_2',
                '17Networks_L_DefaultB_PFCd_3', '17Networks_L_DefaultB_PFCd_4', '17Networks_L_DefaultB_PFCd_5',
                '17Networks_L_DefaultB_PFCd_6', '17Networks_L_DefaultB_PFCd_7', '17Networks_L_DefaultB_PFCd_8',
                '17Networks_L_DefaultB_PFCd_9', '17Networks_L_DefaultB_PFCd_10', '17Networks_L_DefaultB_PFCd_11',
                '17Networks_L_DefaultB_PFCd_12', '17Networks_R_ContA_PFCd_1', '17Networks_R_DefaultA_PFCd_1',
                '17Networks_R_DefaultA_PFCd_2', '17Networks_R_DefaultA_PFCd_3', '17Networks_R_DefaultA_PFCd_4',
                '17Networks_R_DefaultA_PFCd_5', '17Networks_R_DefaultB_PFCd_1', '17Networks_R_DefaultB_PFCd_2',
                '17Networks_R_DefaultB_PFCd_3', '17Networks_R_DefaultB_PFCd_4', '17Networks_R_DefaultB_PFCd_5',
                '17Networks_R_DefaultB_PFCd_6', '17Networks_R_DefaultB_PFCd_7', '17Networks_R_DefaultB_PFCd_8',
                '17Networks_L_SalVentAttnB_PFCl_1', '17Networks_L_SalVentAttnB_PFCl_2',
                '17Networks_L_SalVentAttnB_PFCl_3', '17Networks_L_SalVentAttnB_PFCl_4',
                '17Networks_L_SalVentAttnB_PFCl_5', '17Networks_L_SalVentAttnB_PFCl_6', '17Networks_L_ContA_PFCl_1',
                '17Networks_L_ContA_PFCl_2', '17Networks_L_ContA_PFCl_3', '17Networks_L_ContA_PFCl_4',
                '17Networks_L_ContA_PFCl_5', '17Networks_L_ContA_PFCl_6', '17Networks_L_ContA_PFCl_7',
                '17Networks_L_ContA_PFCl_8', '17Networks_L_ContA_PFCl_9', '17Networks_L_ContB_PFCl_1',
                '17Networks_L_ContB_PFCl_2', '17Networks_L_ContB_PFCl_3', '17Networks_L_DefaultB_PFCl_1',
                '17Networks_L_DefaultB_PFCl_2', '17Networks_R_SalVentAttnB_PFCl_1', '17Networks_R_SalVentAttnB_PFCl_2',
                '17Networks_R_SalVentAttnB_PFCl_3', '17Networks_R_SalVentAttnB_PFCl_4',
                '17Networks_R_SalVentAttnB_PFCl_5', '17Networks_R_SalVentAttnB_PFCl_6', '17Networks_R_ContA_PFCl_1',
                '17Networks_R_ContA_PFCl_2', '17Networks_R_ContA_PFCl_3', '17Networks_R_ContA_PFCl_4',
                '17Networks_R_ContA_PFCl_5', '17Networks_R_ContA_PFCl_6', '17Networks_R_ContA_PFCl_7',
                '17Networks_R_ContA_PFCl_8', '17Networks_R_ContA_PFCl_9', '17Networks_R_ContA_PFCl_10',
                '17Networks_R_ContA_PFCl_11', '17Networks_L_SalVentAttnB_OFC_1', '17Networks_L_LimbicB_OFC_1',
                '17Networks_L_LimbicB_OFC_2', '17Networks_L_LimbicB_OFC_3', '17Networks_L_LimbicB_OFC_4',
                '17Networks_L_LimbicB_OFC_5', '17Networks_L_LimbicB_OFC_6', '17Networks_L_LimbicB_OFC_7',
                '17Networks_L_LimbicB_OFC_8', '17Networks_L_LimbicB_OFC_9', '17Networks_L_LimbicB_OFC_10',
                '17Networks_L_LimbicB_OFC_11', '17Networks_L_LimbicB_OFC_12', '17Networks_L_LimbicB_OFC_13',
                '17Networks_L_LimbicB_OFC_14', '17Networks_L_LimbicB_OFC_15', '17Networks_L_LimbicB_OFC_16',
                '17Networks_R_LimbicB_OFC_1', '17Networks_R_LimbicB_OFC_2', '17Networks_R_LimbicB_OFC_3',
                '17Networks_R_LimbicB_OFC_4', '17Networks_R_LimbicB_OFC_5', '17Networks_R_LimbicB_OFC_6',
                '17Networks_R_LimbicB_OFC_7', '17Networks_R_LimbicB_OFC_8', '17Networks_R_LimbicB_OFC_9',
                '17Networks_R_LimbicB_OFC_10', '17Networks_R_LimbicB_OFC_11', '17Networks_R_LimbicB_OFC_12',
                '17Networks_R_LimbicB_OFC_13', '17Networks_R_LimbicB_OFC_14', '17Networks_R_LimbicB_OFC_15',
                '17Networks_R_LimbicB_OFC_16', '17Networks_L_SalVentAttnB_PFCmp_1', '17Networks_L_SalVentAttnB_PFCmp_2',
                '17Networks_L_SalVentAttnB_PFCmp_3', '17Networks_L_SalVentAttnB_PFCmp_4', '17Networks_L_ContB_PFCmp_1',
                '17Networks_L_ContB_PFCmp_2', '17Networks_R_SalVentAttnB_PFCmp_1', '17Networks_R_SalVentAttnB_PFCmp_2',
                '17Networks_R_SalVentAttnB_PFCmp_3', '17Networks_R_SalVentAttnB_PFCmp_4',
                '17Networks_R_SalVentAttnB_PFCmp_5', '17Networks_R_ContB_PFCmp_1', '17Networks_R_ContB_PFCmp_2',
                '17Networks_R_ContB_PFCmp_3', '17Networks_L_LimbicA_TempPole_1', '17Networks_L_LimbicA_TempPole_2',
                '17Networks_L_LimbicA_TempPole_3', '17Networks_L_LimbicA_TempPole_4', '17Networks_L_LimbicA_TempPole_5',
                '17Networks_L_LimbicA_TempPole_6', '17Networks_L_LimbicA_TempPole_7', '17Networks_L_LimbicA_TempPole_8',
                '17Networks_L_LimbicA_TempPole_9', '17Networks_L_LimbicA_TempPole_10',
                '17Networks_L_LimbicA_TempPole_11', '17Networks_L_LimbicA_TempPole_12',
                '17Networks_L_LimbicA_TempPole_13', '17Networks_L_LimbicA_TempPole_14',
                '17Networks_R_LimbicA_TempPole_1', '17Networks_R_LimbicA_TempPole_2', '17Networks_R_LimbicA_TempPole_3',
                '17Networks_R_LimbicA_TempPole_4', '17Networks_R_LimbicA_TempPole_5', '17Networks_R_LimbicA_TempPole_6',
                '17Networks_R_LimbicA_TempPole_7', '17Networks_R_LimbicA_TempPole_8', '17Networks_R_LimbicA_TempPole_9',
                '17Networks_R_LimbicA_TempPole_10', '17Networks_R_LimbicA_TempPole_11',
                '17Networks_R_LimbicA_TempPole_12', '17Networks_R_LimbicA_TempPole_13',
                '17Networks_R_LimbicA_TempPole_14', '17Networks_R_LimbicA_TempPole_15', '17Networks_L_ContA_Temp_1',
                '17Networks_L_ContA_Temp_2', '17Networks_L_ContA_Temp_3', '17Networks_L_ContB_Temp_1',
                '17Networks_L_ContB_Temp_2', '17Networks_L_ContB_Temp_3', '17Networks_L_ContB_Temp_4',
                '17Networks_L_DefaultB_Temp_1', '17Networks_L_DefaultB_Temp_2', '17Networks_L_DefaultB_Temp_3',
                '17Networks_L_DefaultB_Temp_4', '17Networks_L_DefaultB_Temp_5', '17Networks_L_DefaultB_Temp_6',
                '17Networks_L_DefaultB_Temp_7', '17Networks_L_DefaultB_Temp_8', '17Networks_L_DefaultB_Temp_9',
                '17Networks_L_DefaultB_Temp_10', '17Networks_L_DefaultB_Temp_11', '17Networks_L_DefaultB_Temp_12',
                '17Networks_L_DefaultB_Temp_13', '17Networks_R_ContA_Temp_1', '17Networks_R_ContA_Temp_2',
                '17Networks_R_ContB_Temp_1', '17Networks_R_ContB_Temp_2', '17Networks_R_ContB_Temp_3',
                '17Networks_R_ContB_Temp_4', '17Networks_R_ContB_Temp_5', '17Networks_R_DefaultA_Temp_1',
                '17Networks_R_DefaultA_Temp_2', '17Networks_R_DefaultB_Temp_1', '17Networks_R_DefaultB_Temp_2',
                '17Networks_R_DefaultB_Temp_3', '17Networks_L_ContA_IPS_1', '17Networks_L_ContA_IPS_2',
                '17Networks_L_ContA_IPS_3', '17Networks_L_ContA_IPS_4', '17Networks_L_ContA_IPS_5',
                '17Networks_L_ContA_IPS_6', '17Networks_L_ContA_IPS_7', '17Networks_L_ContA_IPS_8',
                '17Networks_L_ContA_IPS_9', '17Networks_L_ContA_IPS_10', '17Networks_R_ContA_IPS_1',
                '17Networks_R_ContA_IPS_2', '17Networks_R_ContA_IPS_3', '17Networks_R_ContA_IPS_4',
                '17Networks_R_ContA_IPS_5', '17Networks_R_ContA_IPS_6', '17Networks_R_ContA_IPS_7',
                '17Networks_R_ContA_IPS_8', '17Networks_R_ContA_IPS_9', '17Networks_R_ContA_IPS_10',
                '17Networks_R_ContA_IPS_11', '17Networks_R_ContA_IPS_12', '17Networks_L_ContA_PFClv_1',
                '17Networks_L_ContA_PFClv_2', '17Networks_L_ContA_PFClv_3', '17Networks_L_ContA_PFClv_4',
                '17Networks_L_ContA_PFClv_5', '17Networks_L_ContB_PFClv_1', '17Networks_L_ContB_PFClv_2',
                '17Networks_L_ContB_PFClv_3', '17Networks_L_ContB_PFClv_4', '17Networks_L_ContB_PFClv_5',
                '17Networks_R_SalVentAttnB_PFClv_1', '17Networks_R_SalVentAttnB_PFClv_2', '17Networks_R_ContB_PFClv_1',
                '17Networks_R_ContB_PFClv_2', '17Networks_R_ContB_PFClv_3', '17Networks_R_ContB_PFClv_4',
                '17Networks_R_ContB_PFClv_5', '17Networks_R_ContB_PFClv_6', '17Networks_R_ContB_PFClv_7',
                '17Networks_R_ContB_PFClv_8', '17Networks_L_ContA_Cingm_1', '17Networks_L_ContA_Cingm_2',
                '17Networks_R_ContA_Cingm_1', '17Networks_R_ContA_Cingm_2', '17Networks_L_ContC_pCun_1',
                '17Networks_L_ContC_pCun_2', '17Networks_L_ContC_pCun_3', '17Networks_L_ContC_pCun_4',
                '17Networks_L_ContC_pCun_5', '17Networks_L_ContC_pCun_6', '17Networks_L_ContC_pCun_7',
                '17Networks_L_ContC_pCun_8', '17Networks_L_ContC_pCun_9', '17Networks_L_ContC_pCun_10',
                '17Networks_L_ContC_pCun_11', '17Networks_L_ContC_pCun_12', '17Networks_L_ContC_pCun_13',
                '17Networks_R_ContC_pCun_1', '17Networks_R_ContC_pCun_2', '17Networks_R_ContC_pCun_3',
                '17Networks_R_ContC_pCun_4', '17Networks_R_ContC_pCun_5', '17Networks_R_ContC_pCun_6',
                '17Networks_R_ContC_pCun_7', '17Networks_R_ContC_pCun_8', '17Networks_R_ContC_pCun_9',
                '17Networks_R_ContC_pCun_10', '17Networks_R_ContC_pCun_11', '17Networks_L_ContC_Cingp_1',
                '17Networks_L_ContC_Cingp_2', '17Networks_L_ContC_Cingp_3', '17Networks_L_ContC_Cingp_4',
                '17Networks_L_ContC_Cingp_5', '17Networks_L_ContC_Cingp_6', '17Networks_R_ContC_Cingp_1',
                '17Networks_R_ContC_Cingp_2', '17Networks_R_ContC_Cingp_3', '17Networks_R_ContC_Cingp_4',
                '17Networks_L_DefaultA_pCunPCC_1', '17Networks_L_DefaultA_pCunPCC_2', '17Networks_L_DefaultA_pCunPCC_3',
                '17Networks_L_DefaultA_pCunPCC_4', '17Networks_L_DefaultA_pCunPCC_5', '17Networks_L_DefaultA_pCunPCC_6',
                '17Networks_L_DefaultA_pCunPCC_7', '17Networks_L_DefaultA_pCunPCC_8', '17Networks_L_DefaultA_pCunPCC_9',
                '17Networks_L_DefaultA_pCunPCC_10', '17Networks_L_DefaultA_pCunPCC_11',
                '17Networks_L_DefaultA_pCunPCC_12', '17Networks_L_DefaultA_pCunPCC_13',
                '17Networks_L_DefaultA_pCunPCC_14', '17Networks_L_DefaultA_pCunPCC_15',
                '17Networks_L_DefaultA_pCunPCC_16', '17Networks_L_DefaultA_pCunPCC_17',
                '17Networks_L_DefaultA_pCunPCC_18', '17Networks_L_DefaultA_pCunPCC_19',
                '17Networks_L_DefaultA_pCunPCC_20', '17Networks_L_DefaultA_pCunPCC_21',
                '17Networks_L_DefaultA_pCunPCC_22', '17Networks_R_DefaultA_pCunPCC_1',
                '17Networks_R_DefaultA_pCunPCC_2', '17Networks_R_DefaultA_pCunPCC_3', '17Networks_R_DefaultA_pCunPCC_4',
                '17Networks_R_DefaultA_pCunPCC_5', '17Networks_R_DefaultA_pCunPCC_6', '17Networks_R_DefaultA_pCunPCC_7',
                '17Networks_R_DefaultA_pCunPCC_8', '17Networks_R_DefaultA_pCunPCC_9',
                '17Networks_R_DefaultA_pCunPCC_10', '17Networks_R_DefaultA_pCunPCC_11',
                '17Networks_R_DefaultA_pCunPCC_12', '17Networks_L_DefaultA_PFCm_1', '17Networks_L_DefaultA_PFCm_2',
                '17Networks_L_DefaultA_PFCm_3', '17Networks_L_DefaultA_PFCm_4', '17Networks_L_DefaultA_PFCm_5',
                '17Networks_L_DefaultA_PFCm_6', '17Networks_L_DefaultA_PFCm_7', '17Networks_L_DefaultA_PFCm_8',
                '17Networks_L_DefaultA_PFCm_9', '17Networks_L_DefaultA_PFCm_10', '17Networks_L_DefaultA_PFCm_11',
                '17Networks_L_DefaultA_PFCm_12', '17Networks_R_DefaultA_PFCm_1', '17Networks_R_DefaultA_PFCm_2',
                '17Networks_R_DefaultA_PFCm_3', '17Networks_R_DefaultA_PFCm_4', '17Networks_R_DefaultA_PFCm_5',
                '17Networks_R_DefaultA_PFCm_6', '17Networks_R_DefaultA_PFCm_7', '17Networks_R_DefaultA_PFCm_8',
                '17Networks_R_DefaultA_PFCm_9', '17Networks_R_DefaultA_PFCm_10', '17Networks_L_DefaultB_PFCv_1',
                '17Networks_L_DefaultB_PFCv_2', '17Networks_L_DefaultB_PFCv_3', '17Networks_L_DefaultB_PFCv_4',
                '17Networks_L_DefaultB_PFCv_5', '17Networks_L_DefaultB_PFCv_6', '17Networks_L_DefaultB_PFCv_7',
                '17Networks_L_DefaultB_PFCv_8', '17Networks_L_DefaultB_PFCv_9', '17Networks_R_DefaultB_PFCv_1',
                '17Networks_R_DefaultB_PFCv_2', '17Networks_R_DefaultB_PFCv_3', '17Networks_R_DefaultB_PFCv_4',
                '17Networks_R_DefaultB_PFCv_5', '17Networks_R_DefaultB_PFCv_6', '17Networks_L_DefaultC_Rsp_1',
                '17Networks_L_DefaultC_Rsp_2', '17Networks_L_DefaultC_Rsp_3', '17Networks_L_DefaultC_Rsp_4',
                '17Networks_L_DefaultC_Rsp_5', '17Networks_L_DefaultC_Rsp_6', '17Networks_L_DefaultC_Rsp_7',
                '17Networks_L_DefaultC_Rsp_8', '17Networks_R_DefaultC_Rsp_1', '17Networks_R_DefaultC_Rsp_2',
                '17Networks_R_DefaultC_Rsp_3', '17Networks_R_DefaultC_Rsp_4', '17Networks_L_DefaultC_PHC_1',
                '17Networks_L_DefaultC_PHC_2', '17Networks_L_DefaultC_PHC_3', '17Networks_L_DefaultC_PHC_4',
                '17Networks_L_DefaultC_PHC_5', '17Networks_L_DefaultC_PHC_6', '17Networks_R_DefaultC_PHC_1',
                '17Networks_R_DefaultC_PHC_2', '17Networks_R_DefaultC_PHC_3', '17Networks_R_DefaultC_PHC_4',
                '17Networks_L_TempPar_1', '17Networks_L_TempPar_2', '17Networks_L_TempPar_3', '17Networks_L_TempPar_4',
                '17Networks_L_TempPar_5', '17Networks_L_TempPar_6', '17Networks_L_TempPar_7', '17Networks_L_TempPar_8',
                '17Networks_L_TempPar_9', '17Networks_L_TempPar_10', '17Networks_L_TempPar_11',
                '17Networks_L_TempPar_12', '17Networks_L_TempPar_13', '17Networks_L_TempPar_14',
                '17Networks_L_TempPar_15', '17Networks_L_TempPar_16', '17Networks_R_TempPar_1',
                '17Networks_R_TempPar_2', '17Networks_R_TempPar_3', '17Networks_R_TempPar_4', '17Networks_R_TempPar_5',
                '17Networks_R_TempPar_6', '17Networks_R_TempPar_7', '17Networks_R_TempPar_8', '17Networks_R_TempPar_9',
                '17Networks_R_TempPar_10', '17Networks_R_TempPar_11', '17Networks_R_TempPar_12',
                '17Networks_R_TempPar_13', '17Networks_R_TempPar_14', '17Networks_R_TempPar_15',
                '17Networks_R_TempPar_16', '17Networks_R_TempPar_17', '17Networks_R_TempPar_18',
                '17Networks_R_TempPar_19', '17Networks_R_TempPar_20', '17Networks_R_TempPar_21',
                '17Networks_R_TempPar_22', '17Networks_R_SalVentAttnA_PrCv_1', '17Networks_R_SalVentAttnA_PrCv_2',
                '17Networks_R_SalVentAttnB_PFCmp_6', '17Networks_R_ContB_PFCl_1', '17Networks_R_ContB_PFCl_2',
                '17Networks_R_ContB_PFCl_3', '17Networks_R_ContB_PFCl_4', '17Networks_R_ContB_PFCl_5',
                '17Networks_R_ContB_PFCl_6', '17Networks_R_ContB_PFCl_7', '17Networks_R_ContB_PFCl_8',
                '17Networks_R_ContB_PFCl_9', '17Networks_R_ContB_PFCl_10', '17Networks_R_ContB_PFCl_11',
                '17Networks_R_DefaultB_Temp_1', '17Networks_R_DefaultB_Temp_2', '17Networks_R_DefaultB_Temp_3',
                '17Networks_R_DefaultB_Temp_4']
    else:
        raise NotImplementedError(f'get_BNA_ROIs, {code=}')
    return ROIs


if __name__ == '__main__':
    plot_ROI()
    quit()
    from nilearn.datasets import fetch_atlas_schaefer_2018

    atlas_ni = fetch_atlas_schaefer_2018(n_rois=400, resolution_mm=2,
                                         yeo_networks=17)
    data = image.load_img(atlas_ni['maps']).get_fdata()
    # print(np.max(data))
    # quit()
    labels = atlas_ni['labels']
    for i, label in enumerate(labels):
        print(f'{i + 1} | {label}')

    # atlas_me = get_atlas(schaefer=(True, 400), combine_regions=False)
    # data = image.load_img(atlas_me['maps']).get_fdata()
    # print(np.max(data))
    # quit()
    # labels = atlas_me['labels']
    ROI_choose_idx = 126
    ROI_choose = labels[ROI_choose_idx - 1]
    ROI = data == ROI_choose_idx
    print(np.argwhere(ROI))


    img = image.new_img_like(atlas_ni['maps'], ROI)
    plotting.plot_roi(img, title=f'ROI: {ROI_choose} ({ROI_choose_idx})')
    plt.show()
    quit()

    atlas_BN = get_atlas(combine_regions=False)
    labels = atlas_BN['labels']
    for i, label in enumerate(labels):
        print(f'{i + 1} | {label}')
    data_BN = image.load_img(atlas_BN['maps']).get_fdata()
    ROI_choose_idx = 138
    ROI_IPL = data_BN == ROI_choose_idx
    ROI_choose = labels[ROI_choose_idx - 1]

    img_IPL = image.new_img_like(atlas_BN['maps'], ROI_IPL)
    plotting.plot_roi(img_IPL, title=f'ROI: {ROI_choose} ({ROI_choose_idx})')
    plt.show()
    # plt.imshow(ROI[:, 50, :], cmap='viridis')
    # plt.show()
    quit()


    atlas = get_atlas(schaefer=(True, 400), combine_regions=True)
    # print(atlas['ROIs'])
    t = np.random.normal(0, 1, size=(len(atlas['ROIs']),
                                     len(atlas['ROIs'])))
    print(f'{len(atlas["ROIs"])=}')

    for i in range(len(atlas['ticks'])):
        tick = atlas['ticks'][i]
        label = atlas['tick_labels'][i]
        low = atlas['tick_lows'][i]
        print(f'{i} | {tick=:<3}, {low=:<4}, {label=}')

    plot_connectivity(t, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'], title='test', tile=.01,
                      no_avg=True, cbar_label='t-value',
                      vmin=-2, vmax=2)

    # print(atlas['ROI_regions'])ff
