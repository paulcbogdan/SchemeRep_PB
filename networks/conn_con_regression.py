import os

from connsearch import print_list_stats

os.chdir(r'H:\PycharmProjects_H\SchemeRep')
import pickle
import pandas as pd
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from org_sns import get_sns
from old_Apr6.ttest_mat import get_stats_graphs


from scipy import stats

from atlas_utils import get_atlas
from old.network_funcs import load_FC_for_Lifu
from old.plot_gen import plot_connectivity, my_plot_surf
from utils import pickle_wrap, stdize
import numpy as np
import statsmodels.formula.api as smf

import warnings
warnings.filterwarnings("ignore",
                        message='Series.__getitem__ treating keys as positions')

FIRST_LOAD = True

N_REPEATS = 128
def evaluate_acc_ps(accs, kernel='rbf', subj_std=False, by_run=False,
                    strict=False):
    if subj_std:
        num = 11
        subj_std_str = '_subjstd'
    else:
        num = 24
        subj_std_str = ''
    by_run_str = '' if by_run else '_notbyrun'
    strict_str = '_strict' if strict else ''
    fp_perm = (fr'cache/accs_all_perm_{num}_{kernel}_noneu'
               fr'{subj_std_str}{by_run_str}{strict_str}.pkl')
    # print(fp_perm)
    with open(fp_perm, 'rb') as f:
        accs_all_perm = pickle.load(f)
    global FIRST_LOAD
    if FIRST_LOAD:
        print(f'{fp_perm=}')
        print_list_stats(accs_all_perm)
        FIRST_LOAD = False
    if len(accs) > 10:
        print_list_stats(accs)
    accs_all_perm = np.array(accs_all_perm)
    accs_all_perm = np.sort(accs_all_perm)
    accs_all_perm = accs_all_perm[~np.isnan(accs_all_perm)]
    ps = []
    for acc in accs:
        if np.isnan(acc):
            continue
        p = np.mean(accs_all_perm > acc) #+ 0.5 / len(accs_all_perm)
        if p == 0:
            print('Minimal!')
            p += 0.5 / len(accs_all_perm)
        # print(f'{acc=:.2f}, {p=:.4f}')
        ps.append(p)
    return ps

def combine_bl(sn_inc_conn):
    n_sn = sn_inc_conn.shape[0]
    n_cond = sn_inc_conn.shape[1]
    nroi = sn_inc_conn.shape[2]
    sn_inc_conn_mid = np.full((n_sn, n_cond, nroi // 2, nroi), np.nan)
    sn_inc_conn_new = np.full((n_sn, n_cond, nroi // 2, nroi // 2), np.nan)
    for i in range(nroi // 2):
        i_low = i * 2
        i_high = i_low + 2
        sn_inc_conn_mid[:, :, i, :] = (
            sn_inc_conn[:, :, i_low:i_high, :].mean(axis=2))
        # print(sn_inc_conn_mid[:, :, i, :])
        # quit()
    for i in range(nroi // 2):
        i_low = i * 2
        i_high = i_low + 2
        sn_inc_conn_new[:, :, :, i] = (
            sn_inc_conn_mid[:, :, :, i_low:i_high].mean(axis=3))
    return sn_inc_conn_new

def get_con_reg_zs(fp='obj7_fMRI', combine_regions=False, age='healthy',
                   comb_bl=True, semi_combine=True, only_cortical=True,
                   perm=False, by_run=False, kernel='linear', subj_std=True,
                   strict_perm=True):
    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc',
              'key_vals': (1, 2, 3),
              'combine_regions': combine_regions,
              'combine_bilateral': comb_bl if combine_regions else False
              }

    print(f'{by_run=}')
    print(f'{subj_std=}')
    print(f'{semi_combine=}')
    print(f'{only_cortical=}')

    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs,
                    easy_override=False, verbose=1, cache_dir='cache')
    # print(age2idxs[1])
    # print(age2idxs[2])
    # print(sn_inc_conn.shape)
    # quit()
    if by_run:
        sn_inc_act_ = np.concatenate([sn_inc_activity[..., :38],
                                      sn_inc_activity[..., 38:76],
                                      sn_inc_activity[..., 76:]], axis=1)
        sn_inc_act_ = stdize(sn_inc_act_, axis=-1, nans=True)
        sn_inc_conn_ = sn_inc_act_[..., None, :] * sn_inc_act_[..., None, :, :]
        sn_inc_conn = np.nanmean(sn_inc_conn_, axis=-1)
    if only_cortical:
        atlas = get_atlas(combine_regions=combine_regions)
        bad_rois = {'Amyg', 'Hipp', 'Str', 'Tha'}
        bad_j = [j for j, roi in enumerate(atlas['ROI_regions'])
                 if roi in bad_rois]
        sn_inc_conn[..., bad_j, :] = np.nan
        sn_inc_conn[..., :, bad_j] = np.nan
        print('Pruned subcortical')

    n_sn = sn_inc_conn.shape[0]
    sns = [str(i) for i in range(n_sn) for _ in range(sn_inc_conn.shape[1])]
    incs = list(range(1, len(kwargs['key_vals']) + 1)) * n_sn
    if by_run: incs *= 3
    new_nrows = sn_inc_conn.shape[0]*sn_inc_conn.shape[1]
    nrois = sn_inc_conn.shape[-1]
    sn_inc_conn = np.reshape(sn_inc_conn, (new_nrows, nrois, nrois))

    cols = []
    cols_full = []
    vals = []
    vals_full = []
    np.set_printoptions(precision=4)
    for i in range(nrois):
        for j in range(i):
            cols.append(f'r{i}_{j}')
            z = sn_inc_conn[:, i, j]
            vals.append(z)
            cols_full.append(f'r{j}_{i}')
            vals_full.append(z)
    vals_full = vals + vals_full
    cols_full = cols + cols_full
    vals = np.array(vals).T
    vals_full = np.array(vals_full).T
    df = pd.DataFrame(vals, columns=cols)
    df['sn'] = sns
    df['inc'] = incs
    df_full = pd.DataFrame(vals_full, columns=cols_full)
    df_full['sn'] = sns
    df_full['inc'] = incs
    df_full['fp'] = fp
    # print(df_full)
    # quit()
    if perm:
        accs_max_perm = []
        accs_all_perm = []
        print('PERMY!!!')
        no_neu = True
        if no_neu:
            df_full = df_full[df_full['inc'] != 2]

        for nsim in range(1000):
            df_full_ = df_full.copy()

            for i, (sn, df_sn) in enumerate(df_full_.groupby('sn')):
                # print(df_full_.loc[df_sn.index, 'inc'])

                if strict_perm and by_run:
                    if i % 2 == 0: # flip 1 of each
                        idx1 = df_full_.loc[df_sn.index, 'inc'] == 1
                        idx1 = idx1[idx1].sample(1).index
                        idx3 = df_full_.loc[df_sn.index, 'inc'] == 3
                        idx3 = idx3[idx3].sample(1).index
                        df_full_.loc[idx1, 'inc'] = 3
                        df_full_.loc[idx3, 'inc'] = 1
                        # print('test')
                        # print(idx1)
                    else: # flip 2 of each
                        idx1 = df_full_.loc[df_sn.index, 'inc'] == 1
                        idx1 = idx1[idx1].sample(2).index
                        idx3 = df_full_.loc[df_sn.index, 'inc'] == 3
                        idx3 = idx3[idx3].sample(2).index
                        # print(idx1)
                        # print(idx3)
                        df_full_.loc[idx1, 'inc'] = 3
                        df_full_.loc[idx3, 'inc'] = 1
                elif strict_perm:
                    # raise NotImplementedError
                    if i % 2:
                        df_full_.loc[df_sn.index, 'inc'] = [1, 3]
                    else:
                        df_full_.loc[df_sn.index, 'inc'] = [3, 1]
                else:
                    incs_shuffled = np.random.permutation(df_sn['inc'])
                    df_full_.loc[df_sn.index, 'inc'] = incs_shuffled
            # print(df_full_['inc'].value_counts())
            # quit()
            accs = conn_clf(df_full_, combine_regions=combine_regions,
                            nrois=nrois, semi_combine=semi_combine,
                            kernel=kernel, subj_std=subj_std,
                            comb_bl=comb_bl, by_run=by_run, strict=strict_perm)
            max_acc = np.max(accs)
            print_list_stats(accs)
            accs_all_perm.extend(accs)
            accs_max_perm.append(max_acc)
            print_list_stats(accs_max_perm)
            print(f'{accs_all_perm=}')
            print_list_stats(accs_all_perm)
            semi_str = '_semi' if semi_combine else ''
            no_neu_str = '_noneu' if no_neu else ''
            subj_std_str = '_subjstd' if subj_std else ''
            semi_combine_str = '_semi' if semi_combine else ''
            by_run_str = '' if by_run else '_notbyrun'
            strict_str = '_strict' if strict_perm else ''
            with open(f'cache/accs_all_perm_{nsim}_'
                      f'{kernel}{semi_str}{no_neu_str}{subj_std_str}'
                      f'{semi_combine_str}{by_run_str}{strict_str}.pkl',
                      'wb') as f:
                pickle.dump(accs_all_perm, f)
    else:
        conn_clf(df_full, combine_regions=combine_regions, nrois=nrois,
                 semi_combine=semi_combine, kernel=kernel, subj_std=subj_std,
                 comb_bl=comb_bl, by_run=by_run, strict=strict_perm)

    t_mat = np.full((nrois, nrois), np.nan)
    p_l = []
    idx2roi = []

    formula_gen = '{ROI} ~ inc + sn'
    nans_mat = np.full((nrois, nrois), np.nan)
    for i in tqdm(range(nrois), desc='Looping lm'):
        # if i < 2: continue
        for j in range(i):
            # if (i, j) != (20, 19): continue
            idx2roi.append((i, j))
            n_nans = np.sum(pd.isna(df[f'r{i}_{j}']))
            nans_mat[i, j] = n_nans

            df_vars = df[['sn', 'inc', f'r{i}_{j}']].dropna()
            if len(df_vars) < 1:
                t_mat[i, j] = t_mat[j, i] = np.nan
                p_l.append(np.nan)
                continue

            formula = formula_gen.format(ROI=f'r{i}_{j}')
            model = smf.ols(formula=formula, data=df_vars)
            res = model.fit()

            p = res.pvalues.loc['inc']
            t = -res.tvalues.loc['inc']

            t_mat[i, j] = t_mat[j, i] = t
            # p_mat[i, j] = p_mat[j, i] = p
            p_l.append(p)

    p_mat = stats.t.cdf(t_mat, df=len(df)-1)
    z_mat = stats.norm.ppf(p_mat)
    # z_mat[t_mat > 0] *= -1

    p_l = np.array(p_l)
    return z_mat, t_mat, p_l, idx2roi

def get_paired_ttest_zs(fp='obj7_fMRI', combine_regions=False,
                        age='healthy'):
    kwargs = {'fp': fp,
              'split': False,
              'key': 'inc',
              'key_vals': (1, 3),
              'combine_regions': combine_regions
              }
    sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
        pickle_wrap(load_FC_for_Lifu, None, kwargs=kwargs, easy_override=False, verbose=1, cache_dir='cache')
    age2idxs['healthy'] = age2idxs[1] + age2idxs[2]
    idxs = age2idxs[age]
    M_graph, SD_graph, SE_graph, N_graph, t_graph, p_graph, z_graph = \
        get_stats_graphs(sn_inc_conn[idxs, 0, :, :],
                         sn_inc_conn[idxs, 1, :, :])

    trils = np.tril_indices_from(p_graph, k=-1)
    p_l = p_graph[trils]
    return z_graph, t_graph, p_l, trils

def perm_conn_clf():
    pass

def do_ROI_clf(df, grps, cols, kernel='linear', groupkfold=True):
    from hotelling.stats import hotelling_t2

    if kernel == 'hotel':

        # print(df.columns)
        # print(f'{cols=}')
        df_ = df#.dropna(subset=cols, axis=0)
        # print(f'{df_.shape=}')
        # quit()
        # cols = cols[:2]
        X0 = np.array(df_[df_['inc'] == 0][cols])
        nan_cols = np.isnan(X0).sum(axis=0)
        # print(nan_cols)
        # quit()
        # print(nan_cols.shape)
        # quit()
        X1 = np.array(df_[df_['inc'] == 1][cols])
        nan_cols1 = np.isnan(X1).sum(axis=0)
        nan_cols = nan_cols + nan_cols1
        X0 = X0[:, nan_cols == 0]
        X1 = X1[:, nan_cols == 0]

        X0 = X0 - X1

        stat, F, p_value, _ = hotelling_t2(X0)

        return None, None, p_value


    else:
        from sklearn.svm import SVC
        from sklearn.model_selection import (StratifiedGroupKFold, cross_val_score)
        if groupkfold:
            Y = df['inc']
            X = df[cols]
            ex_grps = df['sn']
            accs = []
            for seed in range(N_REPEATS):
                df = df.sample(frac=1)
                Y = df['inc']
                X = df[cols]

                groups = df['sn']

                grps_unq = np.sort(np.unique(groups))
                grps_unq_ = np.sort(np.unique(groups))
                np.random.shuffle(grps_unq_)
                grp_mapper = {}
                for i, grp in enumerate(grps_unq):
                    grp_mapper[grp] = grps_unq_[i]
                groups = np.array([grp_mapper[grp] for grp in groups])

                cv = StratifiedGroupKFold(n_splits=2, )
                acc = cross_val_score(SVC(kernel=kernel), X, Y, groups=groups,
                                      cv=cv, scoring='accuracy')
                accs.extend(acc)
            return None, None, accs

        else:
            y_trues = []
            y_preds = []
            acc = []

            for grp in grps:
                clf = SVC(kernel=kernel)

                idx_match = df['sn'] == grp
                X_test = df.loc[idx_match, cols]
                y_test = df.loc[idx_match, 'inc']
                y_trues.extend(y_test)

                idx_train = ~idx_match
                X_train = df.loc[idx_train, cols]
                y_train = df.loc[idx_train, 'inc']
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_preds.extend(y_pred)
                acc.append(np.mean(y_pred == y_test))

            return y_trues, y_preds, acc

def conn_clf(df, combine_regions=False, nrois=54, kernel='linear',
             thresh=2.32, semi_combine=False, groupkfold=True,
             subj_std=False, comb_bl=False, by_run=False, strict=False):
    # sns = get_sns()
    # sns = sns[1] + sns[2]
    # print(f'{sns=}')
    # print(f'{kernel=}')
    # print(df.shape)
    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=False)
    regions = []
    for region in atlas['ROI_regions' if comb_bl else 'ROI_regions_laterality']:
        if region not in regions:
            regions.append(region)
    scores = []

    i2cols = []
    M_accs = []
    if semi_combine:
        labels = regions
        # print(f'{labels=}')
        for region in regions:
            cols = []
            for i, roi0 in enumerate(atlas['ROIs']):
                if region not in roi0:
                    # print(f'{region} not in {roi0}')
                    continue
                for j, roi1 in enumerate(atlas['ROIs']):
                    if region in roi1:
                        continue
                    cols.append(f'r{i}_{j}')
            i2cols.append(cols)
            # print(f'{region} | {len(cols)=}')
    else:
        labels = atlas['ROIs']
        for i in range(nrois):
            # cols = [f'r{roi_i}_{j}' for j in range(nrois) if j != roi_i]
            cols = [f'r{i}_{j}' for j in range(nrois) if j != i]
            i2cols.append(cols)

    for i, label in enumerate(labels):
        df_ = df.copy()
        # cv = LeaveOneGroupOut()
        cols = i2cols[i]
        grps = df_['sn'].unique()

        df_ = df_[df_['inc'] != 2] # MOVED AT 4/1 11:43 am
        if kernel in ['rbf', 'linear']:
            if subj_std:
                for grp in grps:
                    z = stats.zscore(df_.loc[df_['sn'] == grp, cols],
                                     axis=0, nan_policy='omit',
                                     )
                    # print(z.shape)
                    # print(df_.loc[df_['sn'] == grp, cols].shape )
                    # print(len(z))
                    df_.loc[df_['sn'] == grp, cols] = z

                    # df_.loc[df_['sn'] == grp, cols] -= (
                    #     np.nanmean(df_.loc[df_['sn'] == grp, cols], axis=0))
        df_['inc'] = (df_['inc'] - 1) / 2

        nan_cols = df_[cols].isna().sum()
        cols = [col for i, col in enumerate(cols) if nan_cols[i] < 10]
        # print(cols)
        df_ = df_[cols + ['sn', 'inc']]
        df_['num_nans'] = df_[cols].isna().sum(axis=1)
        df_ = df_[df_['num_nans'] < 1] # TODO: toggle
        df_.fillna(0, inplace=True) # some subjects have 1 or 2 edges missing
        df_[cols] = stats.zscore(df_[cols], axis=0)
        # print(df_[cols])
        # print(df_.shape)
        # quit()

        try:
            y_trues, y_preds, acc = do_ROI_clf(df_, grps, cols, kernel=kernel,
                                               groupkfold=groupkfold)
        except ValueError as e:
            print(f'error: do_ROI_clf {e=}')
            # scores.append(np.nan)
            continue

        M_acc = np.mean(acc)
        M_accs.append(M_acc)
        if groupkfold:
            try:
                p_acc = evaluate_acc_ps([M_acc], kernel=kernel,
                                        subj_std=subj_std, by_run=by_run,
                                        strict=strict)[0]
            except FileNotFoundError:
                scores.append(M_acc)
                print(f'{label}: {M_acc=:.2%}')
                continue

            if np.isnan(p_acc):
                print('HOW IS THIS NAN??')
                continue
            print(f'{label}: {M_acc=:.2%}, {p_acc=:.4f}')
            scores.append(M_acc)
            continue

        if kernel == 'hotel':
            p = acc
            print(f'{label}: {p=:.4f}')
            scores.append(p)
            continue

        r, p = stats.pearsonr(y_trues, y_preds)
        y_trues = np.array(y_trues).astype(int)
        y_preds = np.array(y_preds).astype(int)
        hits = np.sum(y_trues == y_preds) // 2
        chances = len(y_trues) // 2

        p_binom = 1 - stats.binom.cdf(hits, chances, .5)
        corr = .05 / p_binom
        z = -stats.norm.ppf(p_binom)
        #  {r=:.3f}, {p=:.4f} |
        print(f'{label}: {M_acc=:.2%}, {p_binom=:.4f} '
              f'({z=:.2f}, {corr=:.1f}) '
              f'| {r=:.3f}, {p=:.4f}')
        # ps.append(p_binom)
        scores.append(p) # TODO: toggle p_binom probably is better
    if semi_combine or groupkfold:
        print(f'{scores=}')
        try:
            ps = evaluate_acc_ps(scores, kernel=kernel, subj_std=subj_std,
                                 by_run=by_run, strict=strict)
            sigs, p_corr, alpha_sidak, alpha_bon = \
                multipletests(ps, alpha=.05, method='fdr_bh')
            num_sig = np.sum(sigs)
            print(f'{num_sig=}')
            print(f'{ps=}')
            np.sort(p_corr)
            print(f'{p_corr=}')
            ps_sig = [p for p, sig in zip(ps, sigs) if sig]
            print(f'{ps_sig=}')
            if len(ps_sig):
                print(f'Weakest sig: {max(ps_sig)}')
            else:
                print('No sig at all')
        except FileNotFoundError:
            pass
        return M_accs
    zs = -stats.norm.ppf(scores)
    zs[zs < 0] = np.nan
    print(f'{zs=}')
    atlas = get_atlas(combine_regions=combine_regions)

    title_str = f'Shows effect'
    num_rois = len(zs)
    fp_str = '_obj8' if 'obj8' in df['fp'].iloc[0] else ''
    print(f'{fp_str=}')
    fp_out = f'result_pics/FC/ROIwise_clf_{num_rois}_{kernel}_thr{thresh}.png'
    my_plot_surf(zs, atlas, title_str, fp_out=fp_out,
                 neg='', pos='Effect', thresh=thresh)



def plot_reg_zs(combine_regions=False, combine_bl=False, lm=True,
                fp='obj7_fMRI', semi_combine=False, only_cortical=False):
    if lm:
        z_mat, t_mat, p_l, idx2roi = \
            pickle_wrap(get_con_reg_zs, None,
                        kwargs={'combine_regions': combine_regions,
                                'age': 'healthy', 'comb_bl': combine_bl,
                                'fp': fp, 'semi_combine': semi_combine,
                                'only_cortical': only_cortical},
                        easy_override=True)
    else:
        z_mat, t_mat, p_l, idx2roi = \
            pickle_wrap(get_paired_ttest_zs, None,
                        kwargs={'combine_regions': combine_regions,
                                'age': 'healthy'}, easy_override=False)

    p_l = p_l[~np.isnan(p_l)] / 2
    n_tests = np.sum(~np.isnan(p_l))
    # print(f'{n_tests=}')
    p_fwe = .05 / n_tests
    z_fwe = stats.norm.ppf(p_fwe)
    z_l = stats.norm.ppf(p_l)
    corr = 1 / (n_tests * p_l)
    for z, corr in zip(np.sort(z_l)[:10], np.sort(corr)[::-1]):
        print(f'{z=:.2f}, corr={corr:.2f}')

    sigs, p_corr, alpha_sidak, alpha_bon = \
        multipletests(p_l, alpha=.05, method='fdr_bh')
    if sum(sigs) == 0:
        cutoff_str = f'(NS, FWE_z = {z_fwe:.2f})'
    else:
        cutoff_z = round(np.min(np.abs(z_l[sigs])), 2)
        cutoff_str = f'(FDR, z > {cutoff_z})'

    print(f'{cutoff_str=}')

    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=combine_bl)
    # atlas['ticks'] = atlas['ticks'][::2]
    # atlas['tick_labels'] = atlas['tick_labels'][::2]
    # atlas['tick_lows'] = atlas['tick_lows'][::2]
    combine_str = '_combined' if combine_regions else ''
    fp_out = fr'result_pics/FC_matrix/congruency_regression{combine_str}.png'
    plot_connectivity(z_mat, atlas['ticks'], atlas['tick_labels'],
                      atlas['tick_lows'],
                      title=f'Connectivity ~ congruency '
                            f'{cutoff_str}', fp=fp_out, no_avg=True,
                      cbar_label='z-score')

if __name__ == '__main__':
    # t2z(-2.85)
    plot_reg_zs()
    # get_con_reg_zs()