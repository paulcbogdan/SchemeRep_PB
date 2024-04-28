from collections import defaultdict
from glob import glob
from pathlib import Path

from tqdm import tqdm

from organize_bhv import get_trial_info
import warnings
import pandas as pd


def get_bad_sns_fp(fp):
    if 'obj_' in fp: f'Stop using an old fp: {fp=}'
    if fp == 'all':
        bad_sns = {'116', '125', '215', '133', '213', '231'}
        # raise ValueError('Stop using all for get sns!')
    # if fp == 'all':
    #     bad_sns = {'116', '125', '135', '212'}
    elif 'obj' in fp or 'scn' in fp:
        bad_sns = set()
    elif 'bl' in fp:
        bad_sns = {'212'}
    elif 'vis' in fp:
        bad_sns = {'116', '125', '215', '133', '213',} #   '138'
    elif 'con' in fp:
        # TODO: add 231 back to CON after shenyang preprocesses it
        bad_sns = {'116', '125',  '133', '215', '231'}
    elif 'cmb' in fp:
        bad_sns = set()
    elif fp == 'loose':
        bad_sns = set()
    else:
        raise ValueError(f'Unknown fp: {fp}')
    # bad_sns.add('133') # Missing memory (CON & VIS) behavioral data
    # bad_sns.add('138') # Bad retrieval session
    # bad_sns.add('213') # Bad retrieval session
    # bad_sns.add('215') # Missing memory (CON & VIS) behavioral & fMRI data
    # bad_sns.add('224') # Was not able to finish the last run of encoding
    # bad_sns.add('231') # No conceptual retrieval or pdata
    return bad_sns

def get_sns_l(fps_fMRI):
    age2sns0 = get_sns(fps_fMRI[0])
    for fp1 in fps_fMRI[1:]:
        age2sns1 = get_sns(fp1)
        for age, l in age2sns1.items():
            age2sns0[age] = set(age2sns0[age]) & set(l)
    for age, l in age2sns0.items():
        age2sns0[age] = sorted(list(set(l)))
    return age2sns0

def get_sns(fp_fMRI='loose', sh=False):
    age2sn = defaultdict(list)
    bad_sns = get_bad_sns_fp(fp_fMRI)
    sh_sns = get_shenyang_subjects()

    # bad_sns = get_bad_sns(ret=ret)
    for age in range(1, 4):
        bhv_root = fr'behavFiles/ENC/S{age}*_run1.mat'
        fps = glob(bhv_root)
        for fp in fps:
            sn = Path(fp).stem

            sn_no_S = sn[1:]
            sn = sn_no_S.replace('_run1', '')
            # sn = fn.replace(r'behavFiles/ENC/S', '').replace('_run1.mat', '')
            if sn in bad_sns:
                continue
            if sh and sn not in sh_sns:
                continue
            age2sn[age].append(sn)
    age2sn['healthy'] = age2sn[1] + age2sn[2]
    return age2sn


def test_fp(fp='obj3_fMRI', ages=(1, 2)):
    age2sn = get_sns('loose')
    # print(age2sn[1])
    # print(age2sn[2])
    # quit()
    # print(f'{age2sn=}')
    # quit()
    for age in ages:
        for sn in tqdm(age2sn[age], desc=f'Testing org_bhv for {fp}, {age=}'):
            print(f'Running ({fp}): {sn=}')
            try:
                df_sn = get_trial_info(sn, easy_override=True)
            except Exception as e:
                print(f'ERROR ({sn}): {e}')
                continue
            num_nans = df_sn[fp].isna().sum()
            has_con_hits = df_sn['con_hit'].notna().sum() > 0
            has_vis_hits = df_sn['vis_hit'].notna().sum() > 0
            if num_nans > 0:
                warnings.warn(f'{sn} has {num_nans} nans in {fp}')
            if not has_con_hits:
                warnings.warn(f'{sn} has no conceptual memory data in {fp}')
            if not has_vis_hits:
                warnings.warn(f'{sn} has no visual memory data in {fp}')


def get_shenyang_subjects():
    sns_str = '102 103 105 106 107 108 110 111 112 114 116 117 120 123 124 ' \
              '125 126 127 128 130 132 134 135 136 137 ' \
              '201 202 203 205 206 207 208 210 211 212 214 216 217 218 219 ' \
              '221 222 225 230 233 234'
    return set(sns_str.split())

# if __name__ == '__main__':
#     pd.set_option('display.max_rows', 115)
#     # df_sn = get_trial_info('231', easy_override=True)
#     # print(df_sn['con7_fMRI'])
#     # quit()
#
#     # df_sn = get_trial_info('138', easy_override=True)
#     # print(df_sn['vis7_fMRI'])
#     # quit()
#     # df_sn = get_trial_info('213', easy_override=True)
#     # print(df_sn['vis7_fMRI'])
#     # quit()
#
#     df_sn = get_trial_info('212', easy_override=True)
#     print(df_sn['bl7_fMRI'])
#     # print(df_sn['vis_hit'].value_counts(dropna=False))
#     quit()

if __name__ == '__main__':
    # test = get_sns('loos')
    # print(test)


    # test_fp('bl7_fMRI')
    # test_fp('obj7_fMRI')
    # test_fp('vis7_fMRI')

    test_fp('bl8_fMRI')
    test_fp('vis8_fMRI')
    test_fp('con8_fMRI')


    # df_sn = get_trial_info('105', easy_override=True)
    # pd.set_option('display.max_rows', 115)
    # print(df_sn[['obj', 'con_resp']])
    quit()
    for FP in ['bl3_fMRI', 'obj3_fMRI', 'scn3_fMRI', 'con3_fMRI', 'vis3_fMRI']:
        test_fp(FP)
