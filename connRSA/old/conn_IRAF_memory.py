import pandas as pd

from connRSA.conn_analyze_IRAFs import ROI2NETWORK
from connRSA.conn_regress import prep_ROI_avg, do_regr_RSA_sn
from connRSA.conn_utils import get_BNA_ROIs
from connRSA.single_trial_conn import prep_fps
from old.networks import prep_networks
from org_sns import get_sns
from organize_bhv import get_trial_info, sort_df_sn
from utils import pickle_wrap, get_formula_cols

import os
os.chdir(r'/')



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
            foci = ['small_M', 'voxel_large', 'avg_large']
            df_sn = get_trial_info(sn, verbose=-1)
            df_sn, _ = sort_df_sn(df_sn, fp)
            sess = (fp.split('_')[0].replace('2', '').replace('3', '').
                    replace('4', '').replace('7', '').replace('8', ''))
            df_sn['sess'] = sess
            for foc in foci:
                kwargs['fp'] = fp
                kwargs['ROI_focus'] = voxel_large
                kwargs['ROIs_ctrl'] = []
                try:
                    IRAFs = pickle_wrap(do_regr_RSA_sn, None, kwargs=kwargs,
                                        easy_override=easy_override, verbose=-1,
                                        )
                except FileNotFoundError:
                    print(f'FileNotFound: {sn}, {fp=}')
                    continue
                df_sn[f'IRAF_{foc}'] = IRAFs
            df_sn_l.append(df_sn)
    df = pd.concat(df_sn_l)
    return df

def run_connRSA_lmer(df):
    from pymer4.models import Lmer
    # print(df['sess'].unique())
    # quit()

    df = df[df['sess'] != 'bl']
    df = df[df['sess'] != 'obj']

    formula = f'IRAF_avg_large ~ 1 + hit_hit + (1 | sn)'
    cols = get_formula_cols(df, formula)

    model = Lmer(formula, data=df[cols])
    model.fit(REML=True, verbose=False, summary=False)
    print(model.summary())


def run_IRAF_connRSA():
    ISPC = False
    RSA = True
    semantic = False
    conn = 'prod'
    trial_similarity = 'corr'  # euc
    second_order = 'spear'
    RDM_method = 'within_nan'
    four_tasks = '7'
    stdize_by_run = True if trial_similarity == 'euc' else False
    # stdize_by_run = False

    target_ROIs = ['Occipital', 'Ventral', 'Dorsal', 'PFC',  # 'cingulate',
                   'subcort']
    # target_ROIs = ['Ventral']

    for target_ROI in target_ROIs:

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
                         verbose=0, easy_override=True)

        # df = run_IRAF_connRSA_ROI(**outer_kwargs)
        run_connRSA_lmer(df)
        # quit()

if __name__ == '__main__':
    run_IRAF_connRSA()

