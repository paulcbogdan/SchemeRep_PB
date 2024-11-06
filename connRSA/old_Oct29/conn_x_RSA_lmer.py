import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas.errors import PerformanceWarning
from tqdm import tqdm

from atlas_utils import get_atlas, get_BNA_ROIs
from connRSA.DistRep_ROI_RSA import ROI2NETWORK
from connRSA.conn_regress import prep_ROI_avg, do_regr_RSA_sn
from connRSA.single_trial_conn import prep_fps
from networks.old.networks import prep_networks

from org_sns import get_sns
from organize_bhv import get_trial_info, sort_df_sn
from utils import pickle_wrap, get_formula_cols, stdize

import os
# os.chdir(r'/')

# disable settingswithcopyerror
pd.options.mode.chained_assignment = None


# suppress
import warnings
warnings.filterwarnings("ignore", category=PerformanceWarning)

def run_IRAF_connRSA_ROI(kwargs, RSA, ISPC, easy_override, plot,
                         voxel_small_M, voxel_small_all, voxel_large,
                         avg_large):
    sns = get_sns('all')['healthy']
    fps = prep_fps(kwargs['four_tasks'])
    df_sn_l = []
    for sn in sns:
        if sn in ['131', '138', '224', '230', '234', '239']: continue
        kwargs['sn'] = sn
        for fp in fps:
            names = ['small_M', 'voxel_large', 'avg_large']
            foci = [voxel_small_M, voxel_large, avg_large]
            df_sn = get_trial_info(sn, verbose=-1)
            df_sn, _ = sort_df_sn(df_sn, fp)
            sess = (fp.split('_')[0].replace('2', '').replace('3', '').
                    replace('4', '').replace('7', '').replace('8', ''))
            df_sn['sess'] = sess
            for foc, name in zip(foci, names):
                kwargs['fp'] = fp
                kwargs['ROI_focus'] = foc
                kwargs['ROIs_ctrl'] = []
                try:
                    IRAFs = pickle_wrap(do_regr_RSA_sn, None, kwargs=kwargs,
                                        easy_override=easy_override, verbose=-1,
                                        )
                except FileNotFoundError:
                    print(f'FileNotFound: {sn}, {name}, {fp=}')
                    continue
                df_sn[f'IRAF_{name}'] = IRAFs
            df_sn_l.append(df_sn)
    df = pd.concat(df_sn_l)
    return df

def run_connRSA_lmer(df):
    from pymer4.models import Lmer

    df = df[df['sess'] != 'bl']
    df = df[df['sess'] != 'obj']

    formula = f'IRAF_avg_large ~ 1 + hit_hit + (1 | sn)'
    cols = get_formula_cols(df, formula)

    model = Lmer(formula, data=df[cols])
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())


def run_conn_x_RSA(df, target_ROI):
    pd.set_option('display.max_columns', None)
    # stop pd wrap

    print(list(df.columns))

    df['IRAF_dif'] = df['IRAF_avg_large'] - df['IRAF_small_M']

    cols = ['IRAF_avg_large', 'IRAF_small_M', 'IRAF_dif',
            'M_', 'M_o', 'M_c', 'M_pfc',
            'FC_v', 'FC_o', 'FC_pfc', 'FC_c']
    cols = ['IRAF_avg_large', 'IRAF_small_M', 'IRAF_dif',
            f'M_{target_ROI}', f'SD_{target_ROI}', f'FC_{target_ROI}']
    df_sn = df.groupby('sn')[cols].mean()
    corr = df_sn.corr()
    print(corr)

    from pymer4.models import Lmer

    df = df[df['sess'] != 'con']
    # df = df[df['sess'] != 'obj']

    # IRAF_small_M + FC_v +
    # df['FC_v'] = df['FC_v'] - df['FC_c']

    formula = f'IRAF_avg_large ~ 1 + SD_{target_ROI} + (1 | sn)'
    # plt.scatter(df['IRAF_small_M'], df['IRAF_avg_large'])
    # plt.show()
    cols = get_formula_cols(df, formula)
    df.dropna(subset=cols, inplace=True)
    for col in cols:
        try:
            df[col] = stats.zscore(df[col])
            df = df[df[col].abs() < 3]
        except TypeError:
            pass

    model = Lmer(formula, data=df[cols])
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())
    # quit()

def run_IRAF_connRSA():
    ISPC = False
    RSA = True
    semantic = True
    conn = 'prod'
    trial_similarity = 'corr'  # euc
    second_order = 'spear'
    RDM_method = 'within_nan'
    four_tasks = '7'
    stdize_by_run = True if trial_similarity == 'euc' else False
    # stdize_by_run = False

    target_ROIs = ['Occipital', 'Ventral', 'Dorsal', 'PFC',  # 'cingulate',
                   'subcort']
    # target_ROIs = ['FuG', 'ITG', 'PhG', 'MTG', 'ATL']
    # atlas = get_atlas()
    # target_ROIs = list(set(atlas['tick_labels']))
    # print(target_ROIs)
    # quit()

    df_all = None

    M_IRAFs = []
    M_FCs = []

    for target_ROI in tqdm(target_ROIs, desc='ROIs IRAF x conn'):

        kwargs = {'semantic': semantic, 'fp': None, 'fp0': None,
                  'fp1': None, 'trial_similarity': trial_similarity,
                  'second_order': second_order, 'RDM_method': RDM_method,
                  'stdize_by_run': stdize_by_run, 'regress_row': True,
                  'four_tasks': four_tasks,
                  }

        outer_kwargs = {'kwargs': kwargs, 'RSA': RSA, 'ISPC': ISPC,
                        'easy_override': False,
                        'plot': len(target_ROIs) < 10}

        regions = set(prep_networks(
            network_setting=ROI2NETWORK[target_ROI])[target_ROI])
        ROIs_match = [ROI for region in regions
                      for ROI in get_BNA_ROIs() if region in ROI]
        target_name = fr'{target_ROI}_M'
        ROI_lvl_control = [f'{ROI}_BOLD' for ROI in ROIs_match]
        prep_ROI_avg(target_name, ROI_lvl_control, RSA=RSA, ISPC=ISPC,
                     ERS_alt=False, **kwargs)

        outer_kwargs['voxel_small_M'] = target_name
        outer_kwargs['voxel_small_all'] = ROI_lvl_control
        outer_kwargs['voxel_large'] = f'{target_ROI}_BOLD_cmb'
        outer_kwargs['avg_large'] = f'{target_ROI}_BOLD'

        df = pickle_wrap(run_IRAF_connRSA_ROI, None, kwargs=outer_kwargs,
                         verbose=-1, easy_override=False)
        if not 'sess' in df.columns: # ????
            df = pickle_wrap(run_IRAF_connRSA_ROI, None, kwargs=outer_kwargs,
                             verbose=-1, easy_override=True)

        df[f'IRAFs_{target_ROI}_al'] = df[f'IRAF_avg_large']

        sns = df['sn'].unique()

        conn_kwargs = {'drop_con': False, 'four_tasks': four_tasks,
                       'sns': sns, 'do_base': False, 'target_ROI': target_ROI}
        df_conn = pickle_wrap(get_df_single_trial_conn, None,
                              kwargs=conn_kwargs,
                              verbose=-1, easy_override=True)

        print(f'{len(df_conn)=}')
        print(f'{len(df)=}')

        df = pd.merge(df, df_conn, on=['sn', 'sess', 'obj'])
        if df_all is None:
            df_all = df
        else:
            df_all[f'IRAFs_{target_ROI}_al'] = df[f'IRAF_avg_large']
            df_all[f'IRAFs_{target_ROI}_sm'] = df[f'IRAF_small_M']
            df_all[f'IRAFs_{target_ROI}_dif'] = (df[f'IRAF_avg_large'] -
                                                 df[f'IRAF_small_M'])
            df_all[f'FC_{target_ROI}'] = df[f'FC_{target_ROI}']
            df_all[f'M_{target_ROI}'] = df[f'M_{target_ROI}']
            M_FC = df_all[f'FC_{target_ROI}'].mean()
            key_IRAF = f'IRAFs_{target_ROI}_al'
            key_IRAF = f'IRAFs_{target_ROI}_dif'
            M_IRAF = df_all[key_IRAF].mean() * 1000
            print(f'{target_ROI}, {M_FC=:.3f}, {M_IRAF=:.3f}')
            M_FCs.append(M_FC)
            M_IRAFs.append(M_IRAF)
        run_conn_x_RSA(df, target_ROI)

    print(f'{M_FCs=}')
    print(f'{M_IRAFs=}')
    r, p = stats.pearsonr(M_FCs, M_IRAFs)
    n_ROIs = len(target_ROIs)
    print(f'FC x IRAF (n = {n_ROIs}): {r=:.3f}, {p=:.3f}')

def get_idxs(ROI):
    regions = set(prep_networks(
        network_setting=ROI2NETWORK[ROI])[ROI])
    ROIs_match = [i for region in regions
                  for i, ROI in enumerate(get_BNA_ROIs()) if region in ROI]
    return ROIs_match


def get_df_single_trial_conn(sns, drop_con=False, four_tasks='7',
                             target_ROI=None, do_base=True):
    fps = prep_fps(four_tasks)
    if drop_con:
        fps = [fp for fp in fps if 'con' not in fp]

    if do_base:
        ventral_idxs = get_idxs('Ventral')
        occ_idxs = get_idxs('Occipital')
        cort_idxs = get_idxs('cortical')
        PFC_idxs = get_idxs('PFC')
    target_idxs = get_idxs(target_ROI)

    df_sn_l = []
    for fp in fps:
        kwargs = {'fp': fp,
                  'split': False,
                  'key': 'inc',
                  'key_vals': (1, 2, 3),
                  'strict_sns': True,
                  'get_df_sn': True
                  }
        sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
            pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                        easy_override=False, verbose=-1, cache_dir='cache',
                        RAM_cache=True)
        sn_inc_activity = np.nanmean(sn_inc_activity, axis=1)
        sn_inc_activity = stdize(sn_inc_activity, axis=-1, nans=True)
        sn_act_target = np.nanmean(sn_inc_activity[:, target_idxs, :], axis=1)
        sn_sd_target = np.nanstd(sn_inc_activity[:, target_idxs, :], axis=1)


        sn_single_trial_conn = (sn_inc_activity[:, :, None] *
                                sn_inc_activity[:, None, :])
        sn_single_trial_conn[:, *np.diag_indices(246), :] = np.nan

        sn_fc_target = sn_single_trial_conn[:, *np.ix_(target_idxs,
                                                       target_idxs), :]
        sn_sd_fc = np.nanstd(sn_fc_target, axis=(1, 2))
        sn_fc_target = np.nanmean(sn_fc_target, axis=(1, 2))

        # if do_base:
        #     sn_act_ventral = np.nanmean(sn_inc_activity[:, ventral_idxs, :], axis=1)
        #     sn_act_occ = np.nanmean(sn_inc_activity[:, occ_idxs, :], axis=1)
        #     sn_act_cort = np.nanmean(sn_inc_activity[:, cort_idxs, :], axis=1)
        #     sn_act_PFC = np.nanmean(sn_inc_activity[:, PFC_idxs, :], axis=1)
        #
        #     sn_st_ventral = sn_single_trial_conn[:, *np.ix_(ventral_idxs,
        #                                                     ventral_idxs), :]
        #     sn_st_ventral = np.nanmean(sn_st_ventral, axis=(1, 2))
        #
        #     sn_st_occipital = sn_single_trial_conn[:, *np.ix_(occ_idxs,
        #                                                       occ_idxs), :]
        #     sn_st_occipital = np.nanmean(sn_st_occipital, axis=(1, 2))
        #
        #     sn_st_cort = sn_single_trial_conn[:, *np.ix_(cort_idxs,
        #                                                  cort_idxs), :]
        #     sn_st_cort = np.nanmean(sn_st_cort, axis=(1, 2))
        #
        #     sn_st_PFC = sn_single_trial_conn[:, *np.ix_(PFC_idxs,
        #                                                  PFC_idxs), :]
        #     sn_st_PFC = np.nanmean(sn_st_PFC, axis=(1, 2))

        sess = (fp.split('_')[0].replace('2', '').replace('3', '').
                replace('4', '').replace('7', '').replace('8', ''))

        for i, df_sn in enumerate(df_sns):
            if df_sn['sn'].iloc[0] not in sns: continue
            # if do_base:
            #     df_sn['FC_v'] = sn_st_ventral[i]
            #     df_sn['FC_o'] = sn_st_occipital[i]
            #     df_sn['FC_c'] = sn_st_cort[i]
            #     df_sn['FC_pfc'] = sn_st_PFC[i]
            #
            #     df_sn['M_v'] = sn_act_ventral[i]
            #     df_sn['M_o'] = sn_act_occ[i]
            #     df_sn['M_c'] = sn_act_cort[i]
            #     df_sn['M_pfc'] = sn_act_PFC[i]
            df_sn[f'FC_{target_ROI}'] = sn_fc_target[i]
            df_sn[f'M_{target_ROI}'] = sn_act_target[i]
            df_sn[f'SD_{target_ROI}'] = sn_sd_target[i]
            df_sn[f'SD_FC_{target_ROI}'] = sn_sd_fc[i]

            cols = ['sn', 'obj', f'FC_{target_ROI}', f'M_{target_ROI}',
                    f'SD_{target_ROI}', f'SD_FC_{target_ROI}']
            if do_base:
                cols += ['FC_v', 'FC_o', 'FC_c', 'FC_pfc',
                         'M_v', 'M_o', 'M_c', 'M_pfc']
            df_sn_pruned = df_sn[cols]
            df_sn_pruned['sess'] = sess
            df_sn_l.append(df_sn_pruned)
    df_sn_all = pd.concat(df_sn_l)
    return df_sn_all




if __name__ == '__main__':
    # get_single_trial_conn()
    run_IRAF_connRSA()

