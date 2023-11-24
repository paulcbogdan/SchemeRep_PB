from collections import defaultdict

from atlas_utils import get_atlas
from organize_bhv import get_trial_info
from single_trial_conn import run_settings, report_results, prep_networks
from utils import pickle_wrap
import pandas as pd
import numpy as np
from tqdm import tqdm

def get_idx_from_key(l, substring):
    idxs = []
    keys = []
    not_idxs = []
    not_keys = []
    for i, key in enumerate(l):
        if substring in key:
            idxs.append(i)
            keys.append(key)
        else:
            not_idxs.append(i)
            not_keys.append(key)
    return idxs, keys

def get_BNA_ROIs():
    ROIs = ['1 SFG_L_7_1', '2 SFG_R_7_1', '3 SFG_L_7_2', '4 SFG_R_7_2', '5 SFG_L_7_3', '6 SFG_R_7_3', '7 SFG_L_7_4', '8 SFG_R_7_4', '9 SFG_L_7_5', '10 SFG_R_7_5', '11 SFG_L_7_6', '12 SFG_R_7_6', '13 SFG_L_7_7', '14 SFG_R_7_7', '15 MFG_L_7_1', '16 MFG_R_7_1', '17 MFG_L_7_2', '18 MFG_R_7_2', '19 MFG_L_7_3', '20 MFG_R_7_3', '21 MFG_L_7_4', '22 MFG_R_7_4', '23 MFG_L_7_5', '24 MFG_R_7_5', '25 MFG_L_7_6', '26 MFG_R_7_6', '27 MFG_L_7_7', '28 MFG_R_7_7', '29 IFG_L_6_1', '30 IFG_R_6_1', '31 IFG_L_6_2', '32 IFG_R_6_2', '33 IFG_L_6_3', '34 IFG_R_6_3', '35 IFG_L_6_4', '36 IFG_R_6_4', '37 IFG_L_6_5', '38 IFG_R_6_5', '39 IFG_L_6_6', '40 IFG_R_6_6', '41 OrG_L_6_1', '42 OrG_R_6_1', '43 OrG_L_6_2', '44 OrG_R_6_2', '45 OrG_L_6_3', '46 OrG_R_6_3', '47 OrG_L_6_4', '48 OrG_R_6_4', '49 OrG_L_6_5', '50 OrG_R_6_5', '51 OrG_L_6_6', '52 OrG_R_6_6', '53 PrG_L_6_1', '54 PrG_R_6_1', '55 PrG_L_6_2', '56 PrG_R_6_2', '57 PrG_L_6_3', '58 PrG_R_6_3', '59 PrG_L_6_4', '60 PrG_R_6_4', '61 PrG_L_6_5', '62 PrG_R_6_5', '63 PrG_L_6_6', '64 PrG_R_6_6', '65 PCL_L_2_1', '66 PCL_R_2_1', '67 PCL_L_2_2', '68 PCL_R_2_2', '69 ATL_L_6_1', '70 ATL_R_6_1', '71 STG_L_6_2', '72 STG_R_6_2', '73 STG_L_6_3', '74 STG_R_6_3', '75 STG_L_6_4', '76 STG_R_6_4', '77 ATL_L_6_5', '78 ATL_R_6_5', '79 ATL_L_6_6', '80 ATL_R_6_6', '81 MTG_L_4_1', '82 MTG_R_4_1', '83 ATL_L_4_2', '84 ATL_R_4_2', '85 MTG_L_4_3', '86 MTG_R_4_3', '87 MTG_L_4_4', '88 MTG_R_4_4', '89 ITG_L_7_1', '90 ITG_R_7_1', '91 ITG_L_7_2', '92 ITG_R_7_2', '93 ATL_L_7_3', '94 ATL_R_7_3', '95 ITG_L_7_4', '96 ITG_R_7_4', '97 ITG_L_7_5', '98 ITG_R_7_5', '99 ITG_L_7_6', '100 ITG_R_7_6', '101 ITG_L_7_7', '102 ITG_R_7_7', '103 FuG_L_3_1', '104 FuG_R_3_1', '105 FuG_L_3_2', '106 FuG_R_3_2', '107 FuG_L_3_3', '108 FuG_R_3_3', '109 PhG_L_6_1', '110 PhG_R_6_1', '111 PhG_L_6_2', '112 PhG_R_6_2', '113 PhG_L_6_3', '114 PhG_R_6_3', '115 PhG_L_6_4', '116 PhG_R_6_4', '117 PhG_L_6_5', '118 PhG_R_6_5', '119 PhG_L_6_6', '120 PhG_R_6_6', '121 pSTS_L_2_1', '122 pSTS_R_2_1', '123 pSTS_L_2_2', '124 pSTS_R_2_2', '125 SPL_L_5_1', '126 SPL_R_5_1', '127 SPL_L_5_2', '128 SPL_R_5_2', '129 SPL_L_5_3', '130 SPL_R_5_3', '131 SPL_L_5_4', '132 SPL_R_5_4', '133 SPL_L_5_5', '134 SPL_R_5_5', '135 IPL_L_6_1', '136 IPL_R_6_1', '137 IPL_L_6_2', '138 IPL_R_6_2', '139 IPL_L_6_3', '140 IPL_R_6_3', '141 IPL_L_6_4', '142 IPL_R_6_4', '143 IPL_L_6_5', '144 IPL_R_6_5', '145 IPL_L_6_6', '146 IPL_R_6_6', '147 Pcun_L_4_1', '148 Pcun_R_4_1', '149 Pcun_L_4_2', '150 Pcun_R_4_2', '151 Pcun_L_4_3', '152 Pcun_R_4_3', '153 Pcun_L_4_4', '154 Pcun_R_4_4', '155 PoG_L_4_1', '156 PoG_R_4_1', '157 PoG_L_4_2', '158 PoG_R_4_2', '159 PoG_L_4_3', '160 PoG_R_4_3', '161 PoG_L_4_4', '162 PoG_R_4_4', '163 INS_L_6_1', '164 INS_R_6_1', '165 INS_L_6_2', '166 INS_R_6_2', '167 INS_L_6_3', '168 INS_R_6_3', '169 INS_L_6_4', '170 INS_R_6_4', '171 INS_L_6_5', '172 INS_R_6_5', '173 INS_L_6_6', '174 INS_R_6_6', '175 CG_L_7_1', '176 CG_R_7_1', '177 CG_L_7_2', '178 CG_R_7_2', '179 CG_L_7_3', '180 CG_R_7_3', '181 CG_L_7_4', '182 CG_R_7_4', '183 CG_L_7_5', '184 CG_R_7_5', '185 CG_L_7_6', '186 CG_R_7_6', '187 CG_L_7_7', '188 CG_R_7_7', '189 EVC_L_5_1', '190 EVC_R_5_1', '191 EVC_L_5_2', '192 EVC_R_5_2', '193 EVC_L_5_3', '194 EVC_R_5_3', '195 EVC_L_5_4', '196 EVC_R_5_4', '197 EVC_L_5_5', '198 EVC_R_5_5', '199 LOC_L_4_1', '200 LOC_R_4_1', '201 LOC_L_4_2', '202 LOC_R_4_2', '203 LOC_L_4_3', '204 LOC_R_4_3', '205 LOC_L_4_4', '206 LOC_R_4_4', '207 sOcG_L_2_1', '208 sOcG_R_2_1', '209 sOcG_L_2_2', '210 sOcG_R_2_2', '211 Amyg_L_2_1', '212 Amyg_R_2_1', '213 Amyg_L_2_2', '214 Amyg_R_2_2', '215 Hipp_L_2_1', '216 Hipp_R_2_1', '217 Hipp_L_2_2', '218 Hipp_R_2_2', '219 Str_L_6_1', '220 Str_R_6_1', '221 Str_L_6_2', '222 Str_R_6_2', '223 Str_L_6_3', '224 Str_R_6_3', '225 Str_L_6_4', '226 Str_R_6_4', '227 Str_L_6_5', '228 Str_R_6_5', '229 Str_L_6_6', '230 Str_R_6_6', '231 Tha_L_8_1', '232 Tha_R_8_1', '233 Tha_L_8_2', '234 Tha_R_8_2', '235 Tha_L_8_3', '236 Tha_R_8_3', '237 Tha_L_8_4', '238 Tha_R_8_4', '239 Tha_L_8_5', '240 Tha_R_8_5', '241 Tha_L_8_6', '242 Tha_R_8_6', '243 Tha_L_8_7', '244 Tha_R_8_7', '245 Tha_L_8_8', '246 Tha_R_8_8']
    return ROIs

def load_for_lmer(RSA, semantic, split, four_tasks, do_networks,
                  trial_similarity):
    dir_results = r'cache/conn_RSA'

    settings = {'RSA': RSA, 'semantic': semantic, 'do_networks': do_networks,
                'conn': 'euc', 'trial_similarity': trial_similarity,  # change trial_similarity=spear
                'second_order': 'spear', 'four_tasks': four_tasks,
                'combine_regions': False, 'split': split,
                'RDM_method': 'clever_std'}

    results_conn = pickle_wrap(None, run_settings, kwargs=settings, verbose=1,
                               cache_dir=dir_results, easy_override=False)
    print('-' * 100)
    report_results(results_conn)
    print('-' * 100)

    settings = {'RSA': RSA, 'semantic': semantic, 'do_networks': False,
                'conn': 'BOLD', 'trial_similarity': 'corr',
                'second_order': 'spear', 'four_tasks': four_tasks,
                'combine_regions': False, 'split': False,
                'RDM_method': 'clever_std'}
    results_bold = pickle_wrap(None, run_settings, kwargs=settings, verbose=1,
                               cache_dir=dir_results, easy_override=False)
    return results_bold, results_conn

def organize_df(results_conn, results_bold, do_networks, target_ROI):
    scores_conn = results_conn['scores_by_ROI']
    scores_conn = np.array(scores_conn)
    scores_bold = results_bold['scores_by_ROI']
    scores_bold = np.array(scores_bold)

    network2ROI, BOLD_keys_short = prep_network2ROI(do_networks, target_ROI)

    ROI_to_bold_keys = {}
    df_as_d = defaultdict(list)
    print('Organizing df for IRAF analysis')
    for sn_i, sn in tqdm(enumerate(results_conn['sns']), desc='Adding subjects...'):
        df_sn = get_trial_info(sn, easy_override=True)

        df_sn.sort_values(by='bl_trial', inplace=True)
        for ROI_j, ROI_name in enumerate(results_conn['keys']):
            if ROI_name != target_ROI:
                continue
            for fp_idx in range(scores_conn.shape[1]):
                scores_conn_fp = scores_conn[sn_i, fp_idx, ROI_j, :]
                n = scores_conn_fp.shape[0]
                df_as_d['fp_idx'].extend([str(fp_idx)] * n)
                df_as_d['ROI'].extend([ROI_name] * n)
                df_as_d['conn_score'].extend(scores_conn_fp)
                if do_networks and ROI_name in network2ROI:
                    idxs = []
                    ROI_bold_keys_short = []
                    for ROI_name2 in network2ROI[ROI_name]:
                        # if ROI_name2 not in ['LOC', 'EVC', 'sOcG']:
                        #     continue
                        idxs2, ROI_bold_keys_short2 = \
                            get_idx_from_key(BOLD_keys_short, ROI_name2)
                        idxs.extend(idxs2)
                        ROI_bold_keys_short.extend(ROI_bold_keys_short2)
                else:
                    idxs, ROI_bold_keys_short = \
                        get_idx_from_key(BOLD_keys_short, ROI_name)

                ROI_to_bold_keys[ROI_name] = ROI_bold_keys_short
                for idx, key_short in zip(idxs, ROI_bold_keys_short):
                    scores_bold_fp = scores_bold[sn_i, fp_idx, idx, :]
                    df_as_d[f'{key_short}'].extend(scores_bold_fp)

                for col in df_sn.columns:
                    # print(f'{col=} | {df_sn[col].values}')
                    df_as_d[col].extend(df_sn[col].values)
        # break

    # for key, l in df_as_d.items():
    #     print(f'{key}: {len(l)}')

    df = pd.DataFrame(df_as_d)

    print(len(df))
    def mean_or_first(l):
        try:
            M = l.mean()
        except TypeError:
            M = l.iloc[0]
        return M

    # df = df.groupby(['sn', 'obj', 'ROI']).agg(mean_or_first).reset_index()
    # print(len(df))
    # print(list(df.columns))
    # quit()
    return df, ROI_to_bold_keys

def prep_network2ROI(do_networks, target_ROI):
    network2ROI = prep_networks(setting=do_networks)
    for key, l in network2ROI.items():
        if isinstance(l, tuple):
            network2ROI[key] = l[0] + l[1]
    assert target_ROI in network2ROI.keys(), f'{target_ROI} not in settings choice'

    BOLD_keys = get_BNA_ROIs()
    key2short = {key: key.split(' ')[1] for key in BOLD_keys}
    BOLD_keys_short = [key2short[key] for key in BOLD_keys]

    return network2ROI, BOLD_keys_short


def lmer_stats(df, ROI_to_bold_keys, target_ROI):
    df_M = df.groupby('sn').mean()
    M_score = df_M['conn_score'].mean()
    SD_score = df_M['conn_score'].std()
    SE_score = SD_score / np.sqrt(df_M.shape[0])
    t_score = M_score / SE_score

    print(f'Single FP: {M_score=:.3f}, {SD_score=:.3f}, {SE_score=:.3f}, {t_score=:.3f}')
    # quit()

    # print(f'{list(ROI_to_bold_keys)=}')
    target_bold_keys = ROI_to_bold_keys[target_ROI]
    print(f'{target_bold_keys=}')
    print(f'{len(target_bold_keys)=}')
    print()
    cols = ['ROI', 'sn', 'conn_score', 'hit_hit', 'inc', 'vis_hit', 'con_hit',
            'inc_str', 'per_inc_str', 'fp_idx'] + \
           target_bold_keys
    df = df[cols]
    df.dropna(inplace=True)
    df['vis_hit'] = df['vis_hit'].astype(int)
    df['con_hit'] = df['con_hit'].astype(int)

    from pymer4.models import Lmer

    print('-' * 120)
    formula = 'conn_score ~ 1 + ' + ' + '.join(target_bold_keys) + '+  (1|sn) + (1|fp_idx)'
    print(f'{formula=}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())

    print('-' * 120)
    formula = 'conn_score ~ 1 + hit_hit + (1|sn) + (1|fp_idx)'
    print(f'{formula=}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())

    print('-' * 120)
    formula = 'conn_score ~ 1 + inc_str + (1|sn) + (1|fp_idx)'
    print(f'{formula=}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())

    print('-' * 120)
    formula = 'conn_score ~ 1 + (1|sn) + (1|fp_idx)'
    print(f'{formula=}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())

    print('-' * 120)
    formula = 'conn_score ~ 1 + (1|sn)'
    print(f'{formula=}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())

    print('-' * 120)
    formula = 'conn_score ~ 1 + (1|fp_idx)'
    print(f'{formula=}')
    model = Lmer(formula, data=df)
    model.fit(REML=True, verbose=True, summary=True)
    print(model.summary())

def run_lmer():
    RSA = True
    semantic = -1
    split = False
    four_tasks = True
    do_networks = 1
    target_ROI = 'Ventral'
    trial_similarity = 'corr'

    results_bold, results_conn = load_for_lmer(RSA, semantic, split,
                                               four_tasks, do_networks,
                                               trial_similarity)
    df, ROI_to_bold_keys = organize_df(results_conn, results_bold, do_networks,
                                       target_ROI)

    lmer_stats(df, ROI_to_bold_keys, target_ROI)


            

if __name__ == '__main__':
    # atlas = get_atlas(False, False)
    # print(atlas['tick_labels'])
    # quit()
    run_lmer()

