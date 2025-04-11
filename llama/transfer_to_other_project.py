import shutil
from functools import cache
from pathlib import Path

import numpy as np
from tqdm import tqdm

from Utils.atlas_funcs import get_atlas
from organize_bhv import get_trial_info


def get_DistRep_sns():
    sns = ['102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '117',
           '118', '119', '120', '123', '124', '126', '127', '128', '129', '130', '131', '132', '134', '135', '136',
           '137', '138', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '214',
           '216', '217', '218', '219', '221', '222', '224', '225', '227', '230', '232', '233', '234', '235', '239']
    sns = [int(sn) for sn in sns]
    return sns


def move_fp_RSM(sn, ROI, fp_fMRI='obj7_fMRI'):
    dir_in = fr'C:\PycharmProjects\SchemeRep\cache\conn_RSA\ars\RSA\{fp_fMRI}_corr_spear_within_nan_False'

    # if small_M:
    fn = f'{sn}_{ROI}_BOLD_cmb.npy'
    fp = Path(dir_in) / fn
    if not fp.exists():
        fn = f'{sn}_{ROI}_BOLD.npy'
        fp = Path(dir_in) / fn
        if not fp.exists():
            print(f'Missing: {sn}, {ROI}, {fp_fMRI}')
            # quit()
            return
    # print('Good: ', sn, ROI, fp_fMRI)

    dir_out = r'P:\UnpackLlama_P\fMRI_RSMs'
    fn_out = fr'{sn}\{fp_fMRI}_{ROI}.npy'
    fp_out = Path(dir_out) / fn_out
    fp_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(fp, fp_out)


def rename_obj(obj):
    if obj == 'oversize tire':
        print('A')
        return 'oversized tire'
    else:
        return obj


def rename_scn(scn):
    if scn == 'monster truck':
        print('B')
        return 'monster truck area'
    elif scn == 'inside of a car':
        print('C')
        return 'car'
    elif scn == 'office space':
        print('D')
        return 'office'
    elif scn == 'arch':
        print('E')
        return 'desert arch'
    else:
        return scn


@cache
def get_sorted_sn_df(sn, fp):
    df_sn = get_trial_info(sn, easy_override=False, verbose=-1)
    sess = fp.split('_')[0].replace('7', '')
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)
    print(list(df_sn.columns))
    # quit()
    keep_cols = ['sn', 'inc', 'vis_hit', 'con_hit',
                 'obj', 'scene', 'inc_rt', 'per_inc',
                 'obj_trial', 'con_trial', 'bl_trial',
                 'vis_trial', 'con_resp', 'vis_type']

    df_sn = df_sn[keep_cols]
    df_sn['obj'] = df_sn['obj'].apply(rename_obj)
    df_sn['scene'] = df_sn['scene'].apply(rename_scn)
    print(df_sn['vis_type'])

    return df_sn


@cache
def get_act_all(fp_fMRI, combine_regions=False, mci=False):
    from Study1A.load_Study1A_funcs import load_FC
    from Utils.pickle_wrap_funcs import pickle_wrap

    kwargs = {'fp': fp_fMRI,
              'key': 'inc',
              'atlas_name': 'BNA',
              'key_vals': (1, 2, 3),
              'get_df_sn': True,
              'combine_regions': combine_regions,
              'combine_bilateral': combine_regions,
              }
    if mci:
        kwargs['strict_sns'] = 'MCI'
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
        pickle_wrap(load_FC, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache',
                    RAM_cache=True)
    sns = [int(df_sn['sn'].unique()) for df_sn in df_sns]

    sn2act = {}
    assert len(sns) == len(sn_inc_activity)
    for i, sn in enumerate(sns):
        sn2act[sn] = sn_inc_activity[i]

    return sn2act

def get_ROI_idx(ROI, combine_regions=True):
    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=combine_regions)
    ROIs = atlas['ROIs']
    # print(ROIs)
    idx = ROIs.index(ROI)
    return idx


def move_act(sn, fp_fMRI, ROI, combine_regions=True):

    sn2act = get_act_all(fp_fMRI, combine_regions=combine_regions,
                         mci=str(sn)[0] == '3')
    # print(list(sn2act))
    # print(f'{sn=}')
    # quit()
    # quit()
    try:
        act = sn2act[sn]
    except KeyError:
        print(f'Bad: {sn}/{fp_fMRI}/{ROI}')
        return
    act = np.nanmean(act, axis=0)

    # comb_str = '_cmb' if combine_regions else ''
    dir_out = r'P:\UnpackLlama_P\fMRI_RSMs'

    idx = get_ROI_idx(ROI, combine_regions=combine_regions)
    act = act[idx]
    print(f'{sn}/{fp_fMRI}: {act.shape=}')

    fn_out = fr'{sn}\{fp_fMRI}_act_{ROI}.npy'
    fp_out = Path(dir_out) / fn_out
    act = act.astype(np.float32)
    Path(fp_out).parent.mkdir(parents=True, exist_ok=True)
    np.save(fp_out, act)
    assert act.shape == (114, )

def send_region_bilateral_ROI(sn, fp_fMRI, region):
    atlas = get_atlas(combine_regions=False)
    # print(list(atlas))
    # print(atlas['ROI_regions_laterality'])
    # quit()

    nsms = []
    for i, ROI in enumerate(atlas['ROIs']):
        if region == atlas['ROI_regions_laterality'][i]:
            dir_in = fr'C:\PycharmProjects\SchemeRep\cache\conn_RSA\ars\RSA\{fp_fMRI}_corr_spear_within_nan_False'
            fn = f'{sn}_{ROI}_BOLD.npy'
            fp = Path(dir_in) / fn
            if fp.exists():
                nsm = np.load(fp)
                nsms.append(nsm)
    assert len(nsms) > 0, f'No ROIs found for {sn}/{fp_fMRI}: {region}'
    nsm = np.nanmean(nsms, axis=0)

    dir_out = r'P:\UnpackLlama_P\fMRI_RSMs'
    fn_out = fr'{sn}\{fp_fMRI}_{region}.npy'
    fp_out = Path(dir_out) / fn_out
    np.save(fp_out, nsm)





if __name__ == '__main__':
    # send_region_bilateral_ROI(102, 'obj7_fMRI', 'SFG_L')

    # atlas = get_atlas(combine_regions=False)
    # ROIs = atlas['ROIs']

    fps_fMRI = ['obj7_fMRI', 'bl7_fMRI']#, 'vis7_fMRI', 'con7_fMRI']
    # fps_fMRI = ['scn7_fMRI']
    sns = get_DistRep_sns()

    sns = [x for x in range(301, 317)]
    sns = [sn for sn in sns if sn not in [307, 314]]

    # ROIs = ['Occipital', 'ITL', 'Parietal', 'PFC']
    # ROIs = ['IT']

    REGIONS = get_atlas(combine_regions=True,
                        combine_bilateral=True)['ROIs']
    # ROIs = get_atlas()['ROIs']
    # ROIs = ['mOFC', 'lOFC']
    # REGIONS = ['cortical'] + REGIONS
    ROIs = REGIONS


    for sn in tqdm(sns, desc=f'Copying over sns'):
        if sn == '215': continue
        for fp_fMRI in fps_fMRI:
            # print(f'{sn=}')
            # print(f'{fp_fMRI=}')
            # quit()
            # for region in ROIs:
            #     send_region_bilateral_ROI(sn, fp_fMRI, region)
            # continue
            # if 'obj' not in fp_fMRI:
            #     continue

            for ROI in ROIs:
                move_act(sn, fp_fMRI, ROI, combine_regions=True)
            # continue
            # quit()

            df_sn = get_sorted_sn_df(sn, fp_fMRI)
            fp_df_sn_fp = fr'C:\PycharmProjects\UnpackLlama\fMRI_RSMs\{sn}\{sn}_{fp_fMRI}_df_sn.csv'
            df_sn.to_csv(fp_df_sn_fp, index=False)
            # continue
            for ROI in ROIs:
                move_fp_RSM(sn, ROI, fp_fMRI)
