from collections import defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from tqdm import tqdm

from Utils.atlas_funcs import get_atlas
from connRSA.conn_regress import do_regr_RSA_sn
from connRSA_finalizing.plot_Fig5_conn import get_IC_mat, get_cross_IC_mat, get_cross_ERS_mat, get_cross_IRAF_mat
from connRSA.old_Oct29.conn_x_RSA_lmer import get_idxs
from connRSA.single_trial_conn import prep_fps
from Study1A.load_Study1A_funcs import load_FC
# from old.network_funcs import load_FC_for_Lifu
from org_sns import get_sns
from Utils.pickle_wrap_funcs import pickle_wrap
import matplotlib

import os
os.chdir(r'C:\PycharmProjects\SchemeRep')

def prep_var_ERS(trialwise=True):
    # semantic = False
    # regress_FC = False

    dt_max = datetime(2024, 6, 8, 0, 0, 0, 0)

    ERS_nan_block = False

    cross = False

    drop_con = False
    trial_similarity = 'corr' # euc
    stdize_by_run = False
    atlas = get_atlas()
    ROIs = atlas['ROIs']


    sns = ['102', '103', '104', '105', '106', '107', '108', '109', '110',
           '111', '112', '113', '114', '115', '117',
           '118', '119', '120', '123', '124', '126', '127', '128', '129',
           '130', '131', '132', '134', '135', '136',
           '137', '138', '201', '202', '203', '204', '205', '206', '207',
           '208', '209', '210', '211', '212', '214',
           '216', '217', '218', '219', '221', '222', '224', '225', '227',
           '230', '232', '233', '234', '235', '239']

    # sns = sns[:40]

    four_tasks = '7'
    fps = prep_fps(four_tasks)

    kwargs = {'trial_similarity': trial_similarity,
              'stdize_by_run': stdize_by_run,
              'ROIs': ROIs,
              }

    corrs = []
    # TODO: maybe regress out the activation normal FC matrix?

    if drop_con:
        fps = [fp for fp in fps if 'con' not in fp]
    ERS_scores_all = []

    dfs_l = []
    for i, sn in enumerate(sns):#, desc=f'Looping IC: {cross=}'):

        kwargs['fps'] = fps
        kwargs['cross'] = cross
        kwargs['nan_block'] = ERS_nan_block
        kwargs['get_var'] = 'IT'
        kwargs['sn'] = sn
        print(f'Doing: {sn}')
        try:
            fp2ROI2iNPS = pickle_wrap(get_cross_ERS_mat, kwargs=kwargs, verbose=-1,
                            easy_override=True, dt_max=dt_max)

        except AttributeError:
            fp2ROI2iNPS = pickle_wrap(get_cross_ERS_mat, kwargs=kwargs, verbose=-1,
                            easy_override=True, dt_max=dt_max)
        v = np.nanstd(fp2ROI2iNPS, axis=1)
        print(f'{v=:}')
        # v = np.sqrt(v)

        fp_pair2cnt = {}

        cnt = 0
        for fp0 in fps:
            for fp1 in fps:
                if fp0 >= fp1:
                    continue
                fp_pair2cnt[(fp0, fp1)] = cnt
                fp_pair2cnt[(fp1, fp0)] = cnt
                cnt += 1

        corrs_all_ = []
        for fp0 in fps:
            idxs = [fp_pair2cnt[(fp0, fp1)] for fp1 in fps if fp1 != fp0]
            corrs_all_.append(v[idxs])

        corrs_all = np.nanmean(np.array(corrs_all_), axis=1)


        df_as_l = defaultdict(list)
        for i in range(len(fps)):
            df_as_l['sn'].extend([sn]*len(corrs_all[i]))
            df_as_l['fp'].extend([fps[i]]*len(corrs_all[i]))
            df_as_l['conn'].extend(corrs_all[i])
            df_as_l['obj'].extend(range(len(corrs_all[i])))
        df_sn = pd.DataFrame(df_as_l)
        dfs_l.append(df_sn)
    df = pd.concat(dfs_l)
    if trialwise:
        return df

    df_M = df.groupby(['sn', 'fp'])[['conn']].mean().reset_index()

    return df



def prep_conn_corrs(key, ERS=False, semantic=True, trialwise=True):
    # semantic = False
    # regress_FC = False

    dt_max = datetime(2024, 6, 8, 0, 0, 0, 0)


    # ERS = False
    ERS_nan_block = False

    cross = False
    IRAF = False

    drop_con = False
    same_RSM_corr = False
    trial_similarity = 'corr' # euc
    stdize_by_run = False
    second_order = 'spear'
    atlas = get_atlas()
    ROIs = atlas['ROIs']
    # print(f'{ROIs=}')
    # quit()
    if cross and ERS:
        assert not drop_con

    sns = ['102', '103', '104', '105', '106', '107', '108', '109', '110',
           '111', '112', '113', '114', '115', '117',
           '118', '119', '120', '123', '124', '126', '127', '128', '129',
           '130', '131', '132', '134', '135', '136',
           '137', '138', '201', '202', '203', '204', '205', '206', '207',
           '208', '209', '210', '211', '212', '214',
           '216', '217', '218', '219', '221', '222', '224', '225', '227',
           '230', '232', '233', '234', '235', '239']

    four_tasks = '7'
    fps = prep_fps(four_tasks)

    kwargs = {'trial_similarity': trial_similarity,
              'stdize_by_run': stdize_by_run,
              'ROIs': ROIs,
              }

    corrs = []
    # TODO: maybe regress out the activation normal FC matrix?

    if drop_con:
        fps = [fp for fp in fps if 'con' not in fp]
    ERS_scores_all = []

    for i, sn in enumerate(tqdm(sns, desc='prepping conn')):#, desc=f'Looping IC: {cross=}'):
        # print(f'Onto: {sn}')
        kwargs['sn'] = sn
        if IRAF:
            kwargs['fps'] = fps
            kwargs['second_order'] = 'spear'
            kwargs['RDM_method'] = 'within_nan'
            kwargs['semantic'] = semantic
            sn_corrs = pickle_wrap(get_cross_IRAF_mat,
                                   kwargs=kwargs, verbose=-1,
                                   easy_override=False,
                                   dt_max=dt_max)

        elif ERS:
            kwargs['fps'] = fps
            kwargs['cross'] = cross
            kwargs['nan_block'] = ERS_nan_block

            sn_corrs, ERS_scores = pickle_wrap(get_cross_ERS_mat,
                                               kwargs=kwargs, verbose=-1,
                                               easy_override=False,
                                               dt_max=dt_max)
            ERS_scores = np.array(ERS_scores)

            ERS_scores_all.append(ERS_scores)
            # print(ERS_scores)
        elif cross:
            kwargs['second_order'] = second_order
            kwargs['within_nan'] = True
            kwargs['fps'] = fps
            kwargs['same_RSM_corr'] = same_RSM_corr
            sn_corrs = pickle_wrap(get_cross_IC_mat, kwargs=kwargs, verbose=-1,
                                   easy_override=False, dt_max=dt_max)
        else:
            kwargs['second_order'] = second_order
            kwargs['within_nan'] = True
            sn_corrs = []
            for fp in fps:
                kwargs['fp'] = fp
                corr = pickle_wrap(get_IC_mat, kwargs=kwargs, verbose=-1,
                                   easy_override=False, dt_max=dt_max)
                sn_corrs.append(corr)
            sn_corrs = np.array(sn_corrs)

        # corr = np.nanmean(sn_corrs, axis=0)
        corrs.append(sn_corrs)

    corrs_all = np.array(corrs)
    df_all = []

    if ERS:
        fp_pair2cnt = {}

        cnt = 0
        for fp0 in fps:
            for fp1 in fps:
                if fp0 >= fp1:
                    continue
                print(f'{fp0}, {fp1}: {cnt=}')
                fp_pair2cnt[(fp0, fp1)] = cnt
                fp_pair2cnt[(fp1, fp0)] = cnt
                cnt += 1

        corrs_all_ = []
        for fp0 in fps:
            idxs = [fp_pair2cnt[(fp0, fp1)] for fp1 in fps if fp1 != fp0]
            corrs_all_.append(corrs_all[:, idxs])
        corrs_all = (np.nanmean(np.array(corrs_all_), axis=2).
                     transpose((1, 0, 2, 3)))


    for i in range(corrs_all.shape[1]):
        corrs = corrs_all[:, i]

        networks = ['Occipital', 'ITL', 'Parietal', 'PFC',
                    'OC_T', 'Ventral']
        network2name = {'Occipital': 'Occipital', 'ITL': 'Temporal',
                        'Parietal': 'Parietal', 'PFC': 'PFC',
                        'IT': 'Temporal', 'OC': 'Occipital',
                        'OC_T': 'OC_T', 'Ventral': 'Ventral'}

        # networks = [key]
        # network2name = {key: key}
        names = [network2name[net] for net in networks]
        net2idxs = {}
        df_as_l = defaultdict(list)

        colors = ['dodgerblue', 'darkorange', 'crimson', 'limegreen']
        # plt.gcf().add_axes([0.1,0.1, 0.35,0.8])

        for net in networks:
            idxs = get_idxs(net)
            net2idxs[net] = idxs
            net_corr = corrs[:, idxs][:, :, idxs]
            net_sn_vals = np.nanmean(net_corr, axis=(1, 2))

            # net_M = np.nanmean(net_sn_vals)
            # net_SD = np.nanstd(net_sn_vals, ddof=1)
            # net_SE = net_SD / np.sqrt(np.sum(~np.isnan(net_sn_vals)))
            df_as_l['net'].extend([network2name[net]]*len(sns))
            df_as_l['sn'].extend(sns)
            df_as_l['conn'].extend(net_sn_vals)

        df = pd.DataFrame(df_as_l)

        if key == 'OC_IT':
            df = df[df['net'].isin(['Occipital', 'Temporal'])]
            df = df.groupby('sn')[['conn']].mean().reset_index()
        else:
            # df_oc = df[df['net'] == 'Occipital']
            # df_it = df[df['net'] == 'Temporal']
            # df_dif = df_oc['conn'].values - df_it['conn'].values
            # df = pd.DataFrame({'sn': df_it['sn'].values, 'conn': df_dif})

            df = df[df['net'].isin([[key], network2name[key]])]

        df['fp'] = fps[i]
        df_all.append(df)
    df = pd.concat(df_all)

    # print(df)
    return df

def get_RSA_betas(key, ctrl_within=True, get_local=True, semantic=True,
                  ERS=False, big_voxelwise=False):
    if key == 'OC_IT' and False:
        df_OC = get_RSA_betas('Occipital', ctrl_within=ctrl_within)
        df_ITL = get_RSA_betas('ITL', ctrl_within=ctrl_within)
        df = pd.concat([df_OC, df_ITL])
        df = df.groupby(['sn', 'fp']).mean().reset_index()
        return df


    kwargs = {'semantic': semantic,
              'trial_similarity': 'corr',
              'second_order': 'spear',
              'RDM_method': 'within_nan',
              'stdize_by_run': False,
              'regress_row': False,
              'ROI_focus': f'{key}_M' if get_local else
              (f'{key}_BOLD_cmb' if big_voxelwise else f'{key}_BOLD'),
              'ROIs_ctrl': [f'{key}_M'] if ctrl_within else [],
              }

    fps = prep_fps('7')
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    bad_tups = {('132', 'obj7_fMRI'), ('138', 'vis7_fMRI'),
                ('224', 'obj7_fMRI'), ('234', 'obj7_fMRI')}

    betas1_all = np.full((len(sns), len(fps)), np.nan)
    betas2_all = np.full((len(sns), len(fps)), np.nan)
    betas_dif_all = np.full((len(sns), len(fps)), np.nan)
    for i, sn in enumerate(sns):
        for j, fp in enumerate(fps):
            kwargs['return_dif'] = True
            kwargs['sn'] = sn
            kwargs['fp'] = fp
            if (sn, fp) in bad_tups:
                continue

            beta1, beta2, dif = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                               verbose=-1, easy_override=True)
            if np.isnan(beta1):
                continue

            betas1_all[i, j] = beta1
            betas2_all[i, j] = beta2
            betas_dif_all[i, j] = dif
    # print(betas1_all.shape)
    # quit()
    df_all = []
    for i in range(len(fps)):
        df = pd.DataFrame({'sn': sns, 'beta1': betas1_all[:, i],
                           'fp': fps[i], 'beta2': betas2_all[:, i],})
        df_all.append(df)
    df = pd.concat(df_all)
    return df

def get_dist_IRAFs(key='IT', ctrl_within=False, semantic=True,
                   get_local=False, out_key=None):

    kwargs = {'semantic': semantic, 'fp': None, 'fp0': None,
              'fp1': None, 'trial_similarity': 'corr',
              'second_order': 'spear',
              'RDM_method': 'within_nan',
              'stdize_by_run': False,
              'regress_row': False, 'four_tasks': '7',
              # 'ROI_focus': f'{key}_M' if get_local else f'{key}_BOLD',
              # 'ROIs_ctrl': [f'{key}_M'] if ctrl_within else [],
              }

    if get_local:
        kwargs['ROI_focus'] = f'{key}_M'
        if ctrl_within:
            kwargs['ROIs_ctrl'] = [f'{key}_BOLD']
        else:
            kwargs['ROIs_ctrl'] = []
    else:
        kwargs['ROI_focus'] = f'{key}_BOLD'
        if ctrl_within:
            kwargs['ROIs_ctrl'] = [f'{key}_M']
        else:
            kwargs['ROIs_ctrl'] = []


    fps = prep_fps(kwargs['four_tasks'])
    sns = get_sns('all')['healthy']
    bad_sns = ['116', '125', '133', '213', '215', '231']
    sns = [sn for sn in sns if sn not in bad_sns]
    bad_tups = {('132', 'obj7_fMRI'), ('138', 'vis7_fMRI'),
                ('224', 'obj7_fMRI'), ('234', 'obj7_fMRI')}

    betas1_all = np.full((len(sns), len(fps), 114), np.nan)
    fp_idxs = np.full((len(sns), len(fps), 114), np.nan)
    sn_idxs = np.full((len(sns), len(fps), 114), np.nan)
    obj_idxs = np.full((len(sns), len(fps), 114), np.nan)
    for i, sn in enumerate(sns):
        for j, fp in enumerate(fps):
            kwargs['cv'] = False
            kwargs['return_dif'] = True
            kwargs['sn'] = sn
            kwargs['fp'] = fp
            fp_idxs[i, j, :] = j
            sn_idxs[i, j, :] = i
            obj_idxs[i, j, :] = np.arange(114)
            if (sn, fp) in bad_tups:
                continue

            kwargs['regress_row'] = True
            beta1 = pickle_wrap(do_regr_RSA_sn, kwargs=kwargs,
                                verbose=-1, easy_override=True)
            has_nan = np.isnan(beta1).any()
            if has_nan:
                print(beta1)
                print('end')
                quit()
            # if np.isnan(beta1):
            #     continue

            betas1_all[i, j, :] = beta1


    betas1 = betas1_all.reshape(-1)
    fp_idxs = fp_idxs.reshape(-1)
    sn_idxs = sn_idxs.reshape(-1)
    obj_idxs = obj_idxs.reshape(-1)
    df = pd.DataFrame({'sn': sn_idxs, 'fp': fp_idxs,
                       out_key if out_key else 'beta1': betas1,
                       'obj': obj_idxs})
    df['fp'] = df['fp'].apply(lambda x: fps[int(x)])
    df['sn'] = df['sn'].apply(lambda x: sns[int(x)])
    return df


def get_plain_corr():
    four_tasks = '7'
    fps = prep_fps(four_tasks)

    corrs = []
    sns = None
    prev_sns = None
    # fps = ['obj7_fMRI']
    for fp in fps:
        kwargs = {'fp': fp,
                  'split': False,
                  'key': 'inc',
                  'key_vals': (1, 2, 3),
                  'strict_sns': True,
                  'get_df_sn': True
                  }
        sn_inc_conn, sn_conn, age2idxs, sn_inc_activity, df_sns = \
            pickle_wrap(load_FC, None, kwargs=kwargs,
                        easy_override=False, verbose=1, cache_dir='cache',
                        RAM_cache=True)
        sns = [df_sn['sn'].iloc[0] for df_sn in df_sns]
        if prev_sns is None:
            prev_sns = sns
        else:
            assert tuple(prev_sns) == tuple(sns)
        corrs.append(sn_conn)
    return np.array(corrs).transpose((1, 0, 2, 3))

# TODO: FINALIZE IT or ITL
def corr_RSA_conn(main_key='ITL', semantic=False):
    df_conn = prep_conn_corrs(main_key, ERS=False, semantic=semantic)

    df_rsa = get_RSA_betas(main_key, ctrl_within=False, get_local=False,
                           semantic=True, big_voxelwise=True)

    df_rsa_per = get_RSA_betas(main_key, ctrl_within=False, get_local=False,
                           semantic=False, big_voxelwise=True)
    df_rsa['beta1_per'] = df_rsa_per['beta1']

    df_rsa_local = get_RSA_betas(main_key, ctrl_within=False, get_local=True,
                                 semantic=True)

    df_rsa_local_per = get_RSA_betas(main_key, ctrl_within=False, get_local=True,
                                 semantic=False)

    df_rsa['local'] = df_rsa_local['beta1']
    df_rsa['local_per'] = df_rsa_local_per['beta1']
    df = pd.merge(df_conn, df_rsa, on=['sn', 'fp'])

    df = df.dropna(subset=['local', 'beta1'])

    df.sort_values('conn', inplace=True)

    # df = df[df['conn'] < .5] # one outlier

    import statsmodels.formula.api as smf

    formula = 'beta1 ~ conn'
    model = smf.ols(formula, data=df)
    results = model.fit()
    print(results.summary())
    # formula = 'conn ~ local'
    # model = smf.ols(formula, data=df)
    # results = model.fit()
    # print(results.summary())

    # df = df[df['conn'].abs() < 3]
    # df = df[df['beta1'].abs() < 3]
    plt.rcParams.update({'font.sans-serif': 'Arial',
                         'font.size': 14,
                         'figure.figsize': (5, 4),
                         'mathtext.default': 'regular' })

    r, p = stats.spearmanr(df['conn'], df['beta1'])
    # plt.title(f'{main_key=}, {r=:.3f}, {p=:.4f}')
    plt.scatter(df['conn'], df['beta1'],
                facecolors=('dodgerblue', 0.5),
                edgecolors=(0, 0, 0, 0.5),
                )
    plt.xlabel('ITL $NSM_{ROI}$-$NSM_{ROI}$ correlation')
    plt.ylabel('ITL semantic RSA effect')

    low = np.nanquantile(df['conn'], 0.00)
    high = np.nanquantile(df['conn'], 1.0)

    df_pred = pd.DataFrame({'conn': np.linspace(low, high, 1000),})
    pred = results.predict(exog=df_pred)
    plt.plot(df_pred['conn'], pred, color='k',
             linestyle='--', alpha=.8)

    matplotlib.colors.colorConverter.to_rgba('mediumseagreen', alpha=.5)
    plt.gca().spines[['right', 'top']].set_visible(False)
    plt.tight_layout()
    semantic_str = '_semantic' if semantic else '_perceptual'
    fp_fig = fr'result_pics/connRSA/RSM_RSM_x_RSA_{main_key}{semantic_str}.png'
    plt.savefig(fp_fig, dpi=600)
    plt.show()
    df.to_csv(r'C:\PycharmProjects\SchemeRep\df_conn_x_RSA.csv', index=False)
    print('SAVED .CSV')
    quit()
    # quit()
    # print(df['conn'])


    print(f'{len(df)=}')

    # from pymer4.models import Lmer
    # formula = 'conn ~ beta1 + (1 |sn) + (1|fp)' #
    # model = Lmer(formula, data=df) # local +
    # model.fit(summarize=False)
    # print(model.summary())


    # df = df.groupby('sn')[['conn', 'beta1']].mean()


    r, p = stats.spearmanr(df['conn'], df['beta1'])#, nan_policy='omit')
    print(f'distr: {r=:.3f}, {p=:.4f}')
    # r, p = stats.spearmanr(df['conn'], df['local'])#, nan_policy='omit')
    # print(f'local: {r=:.3f}, {p=:.4f}')

def trialwise_corr():
    # df_ERS = prep_var_ERS(trialwise=True)
    df = get_dist_IRAFs(key='IT', ctrl_within=True)
    df = pd.merge(df, df_ERS, on=['sn', 'fp', 'obj'])
    r, p = stats.spearmanr(df['conn'], df['beta1'], nan_policy='omit')
    plt.scatter(df['conn'], df['beta1'], alpha=0.1)
    plt.show()
    df.dropna(inplace=True)
    print(f'Spearman: {r=:.3f}, {p=:.4f}')

    from pymer4.models import Lmer
    formula = 'conn ~ beta1 + (1 + beta1 |sn)'  #
    model = Lmer(formula, data=df)  # local +
    model.fit(summarize=False)
    print(model.summary())


def RSA_x_RSA():
    df_OC = get_dist_IRAFs(key='Occipital', ctrl_within=False,
                           get_local=False,
                           semantic=False, out_key='OC_IRAF')
    df_IT = get_dist_IRAFs(key='IT', ctrl_within=False,
                           get_local=True,
                           semantic=True, out_key='IT_IRAF')

    # df_OC = get_dist_IRAFs(key='Occipital', ctrl_within=False,
    #                        get_local=True,
    #                        semantic=True, out_key='OC_IRAF')
    # df_IT = get_dist_IRAFs(key='IT', ctrl_within=False,
    #                        get_local=False,
    #                        semantic=False, out_key='IT_IRAF')

    # df_IT_local = get_dist_IRAFs(key='IT', ctrl_within=False,
    #                              get_local=True,
    #                              semantic=False, out_key='IT_IRAF_l')
    df = pd.merge(df_OC, df_IT, on=['sn', 'fp', 'obj'])

    df_M = df.groupby(['sn', 'fp'])[['OC_IRAF', 'IT_IRAF']].mean().reset_index()
    r, p = stats.spearmanr(df_M['OC_IRAF'], df_M['IT_IRAF'], nan_policy='omit')
    print(f'Group: {r=:.3f}, {p=:.4f}')

    df.dropna(inplace=True)
    r, p = stats.spearmanr(df['OC_IRAF'], df['IT_IRAF'], nan_policy='omit')
    print(f'Spearman: {r=:.3f}, {p=:.4f}')
    # plt.scatter(df['OC_IRAF'], df['IT_IRAF'], alpha=0.1)
    # plt.show()

    from pymer4.models import Lmer
    formula = 'IT_IRAF ~ OC_IRAF + (1 + OC_IRAF | sn)'
    model = Lmer(formula, data=df)  # local +
    model.fit(summarize=False)
    print(model.summary())


if __name__ == '__main__':
    # RSA_x_RSA()
    # trialwise_corr()
    # prep_var_ERS()
    # get_dist_IRAFs(key='IT')
    corr_RSA_conn()
    # DF = prep_conn_corrs(ERS=True)
    # get_RSA_betas()

    # corr_RSA_conn(main_key='Occipital', semantic=True)







