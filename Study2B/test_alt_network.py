from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from Study1A.plot_Fig2CD_partitions import get_VD_PA_partitions
from Study1B.analyze_plot_Fig3 import (get_wl_contrast_conn, get_combo, make_conn,
                                       get_PE_x_Conn_effect)
from Study2B.analyze_Study2B import get_sns_roi_ar_std
from Utils.atlas_funcs import get_atlas
from Utils.pickle_wrap_funcs import pickle_wrap
from marinate.pkld import pkld
import pandas as pd

def get_alt_network_ef(combine_regions=False, bilateral=False, corr_z=False,
                       sub_ROI_expected=False, schaefer=False):
    if isinstance(schaefer, tuple):
        combine_regions = (combine_regions, ('schaefer', schaefer[1]))
    elif schaefer:
        combine_regions = (combine_regions, 'schaefer')
    kw = {'combine_regions': combine_regions, 'bilateral': False,
          'only': 'combo', 'num_sns': 1000, 'learning_rate': 0.3,
          'drop_first': False, 'reset_trial0': True, }

    if kw['only'] == 'wl':
        conn_highs, conn_lows, sns = (
            get_wl_contrast_conn(kw, easy_override=False))
    elif kw['only'] == 'combo':
        # can either be run while averaging a loss matrix & win matrix ('combo')
        #   or just making a single one covering both PE ('both')
        # the manuscript uses 'combo'
        conn_highs, conn_lows, sns = (
            get_combo(kw, easy_override=False))
    else:
        conn_highs, conn_lows, sns = (
            pickle_wrap(make_conn, kwargs=kw, easy_override=False))
    if schaefer:
        combine_regions = combine_regions[0]

    #
    # itr, dd, vv, dv_ant, dv_pos, M_overall = get_PE_x_Conn_effect(
    #     conn_highs, combine_regions=combine_regions, combine_bilateral=bilateral,
    #     schaefer=schaefer)
    # df = pd.DataFrame({'high_PA': dd + vv, 'high_VD': dv_ant + dv_pos,
    #                    'high_dd': dd, 'high_vv': vv, 'high_dv_ant': dv_ant,
    #                    'high_dv_pos': dv_pos, })
    #
    # itr, dd, vv, dv_ant, dv_pos, M_overall = get_PE_x_Conn_effect(
    #     conn_lows, combine_regions=combine_regions, combine_bilateral=bilateral,
    #     schaefer=schaefer)
    # df['low_PA'] = dd + vv
    # df['low_VD'] = dv_ant + dv_pos
    # df['low_dd'] = dd
    # df['low_vv'] = vv
    # df['low_dv_ant'] = dv_ant
    # df['low_dv_pos'] = dv_pos
    #
    # for ef in ['PA', 'VD', 'dd', 'vv', 'dv_ant', 'dv_pos']:
    #     df[f'{ef}_diff'] = df[f'high_{ef}'] - df[f'low_{ef}']
    #     t, p = stats.ttest_rel(df[f'high_{ef}'], df[f'low_{ef}'])
    #     N = np.sum(~np.isnan(df[f'high_{ef}']))
    #     if ef in ['PA', 'dd', 'vv']:
    #         extra = ' (expected negative)'
    #     elif ef in ['VD', 'dv_ant', 'dv_pos']:
    #         extra = ' (expected positive)'
    #     else:
    #         raise ValueError
    #     print(f'{ef}: t[{N - 1}] = {t:.2f}, {p=:.4f} {extra}')

    # print(conn_highs.shape)
    networks = ['VD', 'PA',
                'FPCN', 'VAN', 'DMN', 'DAN', 'Limbic', 'SM', 'Visual']


    # p_dorsal, p_ventral, p_d_ant, p_d_pos, p_v_ant, p_v_pos, matrix_mask = \
    #     get_VD_PA_partitions(age='healthy', anat=True,
    #                          do_PA=True, thr=.9, anat_ver=3,
    #                          combine_regions=combine_regions,
    #                          schaefer=schaefer)

    # print(conn_highs.shape)

    for network in networks:
        idxs = get_yeo_idxs(network, combine_regions=combine_regions)
        # idxs0 = idxs[0::2]
        # idxs1 = idxs[1::2]
        # idxs0 = p_v_ant
        # idxs1 = p_d_ant
        # network_highs = np.nanmean(conn_highs[:, *np.ix_(idxs0, idxs1)], axis=(1, 2))
        # network_lows = np.nanmean(conn_lows[:, *np.ix_(idxs0, idxs1)], axis=(1, 2))
        # print(idxs)
        # quit()
        network_highs = np.nanmean(conn_highs[:, *np.ix_(idxs, idxs)], axis=(1, 2))
        network_lows = np.nanmean(conn_lows[:, *np.ix_(idxs, idxs)], axis=(1, 2))
        t, p = stats.ttest_rel(network_highs, network_lows)
        print(f'{network} | {t=:.3f}, {p=:.3f}')
        # quit()


def test_fluc_mag_networks(lr='LR', combine_regions=True,
                           num_sns=1000):
    kw = {'lr': lr, 'combine_regions': combine_regions,
          'bilateral': False, 'reg_global': False,
          'no_compcor': False, 'rs': True}
    fp = r'Study1B/final_HCP_subjects.txt'
    with open(fp, 'r') as f:
        s = f.read()
    s = s.replace('\n', '').replace(' ', '')
    sns = s.split(',')
    sns = sns[:num_sns]
    if combine_regions:
        sns = tuple(sns)
        sn_roi_act = get_sns_roi_ar_std(tuple(sns), **kw)
        sn_roi_act = sn_roi_act[:, :, :120]
    else:
        # sns = tuple(sns[::5])
        sn_roi_act = get_sns_roi_ar_std(tuple(sns), **kw)
        sn_roi_act = sn_roi_act[:, :, :120]

    conn_trials = sn_roi_act[..., None, :] * \
                  sn_roi_act[..., None, :, :]
    # print(conn_trials.shape)
    conn_trials = stats.zscore(conn_trials, axis=(1, 2))
    conn_trials = stats.zscore(conn_trials, axis=-1)

    # 'VD', 'PA',
    networks = ['vv', 'dd', 'vd_ant', 'vd_pos',
                'FPCN', 'VAN', 'DMN', 'DAN',
                'Limbic', 'SM', 'Visual']

    VD_PA_networks = ['vv', 'dd', 'vd_ant', 'vd_pos']

    mat = np.full((len(networks), len(networks)), np.nan)
    for i, network0 in enumerate(networks):
        for j, network1 in enumerate(networks):
            if i >= j: continue
            both_in = network0 in VD_PA_networks and network1 in VD_PA_networks
            both_out = network0 not in VD_PA_networks and network1 not in VD_PA_networks
            if not (both_in or both_out):
                continue
            if network0 in ['vv', 'dd'] and network1 in ['vv', 'dd']:
                continue
            if network0 in ['vd_ant', 'vd_pos'] and network1 in ['vd_ant', 'vd_pos']:
                continue

            # if network0 == 'VD':
            #     network0a = 'vd_ant'

            # if network0 in VD_PA_networks and network1 in VD_PA_networks:
            # idxs0 = get_yeo_idxs(network0, combine_regions=combine_regions)
            # idxs1 = get_yeo_idxs(network1, combine_regions=combine_regions)
            # # print(f'{network0}: {idxs0=}')
            # if isinstance(idxs0, tuple):
            #     tvc0 = np.nanmean(conn_trials[:, *np.ix_(idxs0[0], idxs0[1]), :], axis=(1, 2))
            # else:
            #     tvc0 = np.nanmean(conn_trials[:, *np.ix_(idxs0, idxs0), :], axis=(1, 2))
            # if isinstance(idxs1, tuple):
            #     print(f'tup: {network1}')
            #     tvc1 = np.nanmean(conn_trials[:, *np.ix_(idxs1[0], idxs1[1]), :], axis=(1, 2))
            # else:
            #     tvc1 = np.nanmean(conn_trials[:, *np.ix_(idxs1, idxs1), :], axis=(1, 2))
            tvc0 = get_tvc(network0, conn_trials, combine_regions)
            # tvc0 = stats.zscore(tvc0, nan_policy='omit')
            tvc1 = get_tvc(network1, conn_trials, combine_regions)
            # tvc1 = stats.zscore(tvc1, nan_policy='omit')

            mag_sns = np.nanmean(np.abs(tvc0 - tvc1), axis=1)
            M_mag = np.nanmean(mag_sns)
            SE_mag = stats.sem(mag_sns, nan_policy='omit')
            print(f'{network0} - {network1} | {M_mag=:.4f} ({SE_mag:.4f})')
            mat[i, j] = M_mag
            mat[j, i] = M_mag

            # network_highs = np.nanmean(conn_trials[:, *np.ix_(idxs0, idxs1)], axis=(1, 2))
            # print(f'{network0} - {network1} | {np.nanmean(network_highs)=}')
    vmin = np.nanquantile(mat, .1)
    vmax = np.nanquantile(mat, .9)
    plot_heatmap(mat, networks, attn=False,
                 title='Fluctuation Magnitude', rotation=0,
                 vmin=vmin, vmax=vmax)


def get_tvc(network, conn_trials, combine_regions):
    if network == 'VD':
        tvc_ant = get_tvc('vd_ant', conn_trials, combine_regions)
        return tvc_ant
        # tvc_pos = get_tvc('vd_pos', conn_trials, combine_regions)
        # return (tvc_ant + tvc_pos) / 2
    elif network == 'PA':
        tvc_dd = get_tvc('dd', conn_trials, combine_regions)
        return tvc_dd
        # tvc_vv = get_tvc('vv', conn_trials, combine_regions)
        # return (tvc_dd + tvc_vv) / 2
    else:
        idxs = get_yeo_idxs(network, combine_regions=combine_regions)
        if isinstance(idxs, tuple):
            tvc = np.nanmean(conn_trials[:, *np.ix_(idxs[0], idxs[1]), :], axis=(1, 2))
        else:
            tvc = np.nanmean(conn_trials[:, *np.ix_(idxs, idxs), :], axis=(1, 2))
        return tvc



def plot_heatmap(corr_t, region_labels, labels_horizontal=None,
                 region=None, attn=None,
                 vmin=-5, vmax=5, cmap='coolwarm', title=None,
                 rotation=50):
    fig, ax = plt.subplots()
    im = ax.imshow(corr_t, cmap=cmap, interpolation='nearest',
                   vmin=vmin, vmax=vmax)
    if title is not None:
        plt.title(title, pad=10)

    # Add text annotations
    for i in range(len(region_labels)):
        for j in range(len(region_labels)):
            if np.isnan(corr_t[i, j]):
                continue
            if cmap == 'coolwarm':
                color = 'k'
                color = 'w' if corr_t[i, j] > vmax else 'k'
            else:
                color = 'k' if corr_t[i, j] > .1 else 'w'
            text = ax.text(j, i, f"{corr_t[i, j]:.2f}",
                           ha="center", va="center", color=color,
                           )

    # Set the ticks and labels
    if region is not None:
        attn_str = 'attn' if attn else 'item'
        plt.title(f'{region}: {attn_str}')
    ax.set_xticks(np.arange(len(region_labels)),
                  )
    ax.set_yticks(np.arange(len(region_labels)))
    if labels_horizontal is None:
        ax.set_xticklabels(region_labels,
                           rotation=rotation)
    else:
        ax.set_xticklabels(labels_horizontal,
                           rotation=rotation)
    ax.set_yticklabels(region_labels)

    fig.tight_layout()
    plt.show()


@pkld(store='both')
def get_bna2yeo(combine_regions=False):
    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=False)

    from nichord.coord_labeler import get_idx_to_label
    coords = atlas['coords']
    idx_to_label = get_idx_to_label(coords, atlas='yeo')
    # print(list(idx_to_label.values()))
    return idx_to_label


@cache
def get_yeo_idxs(network, combine_regions=False, no_quad=True):
    if network == 'vd_ant':
        p_dorsal, p_ventral, p_d_ant_, p_d_pos_, p_v_ant_, p_v_pos_, matrix_mask = \
            get_VD_PA_partitions(age='healthy', do_PA=True, anat=True,
                                 anat_ver=3, combine_regions=combine_regions)
        if no_quad: # otherwise, you have literally the same edges
            return list(p_v_ant_), list(p_d_ant_)
        else:
            return list(p_v_ant_) + list(p_d_ant_)
    elif network == 'vd_pos':
        p_dorsal, p_ventral, p_d_ant_, p_d_pos_, p_v_ant_, p_v_pos_, matrix_mask = \
            get_VD_PA_partitions(age='healthy', do_PA=True, anat=True,
                                 anat_ver=3, combine_regions=combine_regions)
        if no_quad:
            return list(p_v_pos_), list(p_d_pos_)
        else:
            return list(p_v_pos_) + list(p_d_pos_)
    elif network == 'vv':
        p_dorsal, p_ventral, p_d_ant_, p_d_pos_, p_v_ant_, p_v_pos_, matrix_mask = \
            get_VD_PA_partitions(age='healthy', do_PA=True, anat=True,
                                 anat_ver=3, combine_regions=combine_regions)
        if no_quad:
            return list(p_v_pos_), list(p_v_ant_)
        else:
            return list(p_v_pos_) + list(p_v_ant_)
        # return list(p_v_pos_), list(p_v_ant_)
    elif network == 'dd':
        p_dorsal, p_ventral, p_d_ant_, p_d_pos_, p_v_ant_, p_v_pos_, matrix_mask = \
            get_VD_PA_partitions(age='healthy', do_PA=True, anat=True,
                                 anat_ver=3, combine_regions=combine_regions)
        if no_quad:
            return list(p_d_pos_), list(p_d_ant_)
        else:
            return list(p_d_pos_) + list(p_d_ant_)
        # return list(p_d_pos_), list(p_d_ant_)
    bna2yeo = get_bna2yeo(combine_regions=combine_regions)
    idxs = np.array([i for i, label in bna2yeo.items() if label == network])
    idxs_odd = [idx for idx in idxs if idx % 2 == 1]
    idxs_even = [idx for idx in idxs if idx % 2 == 0]
    # return idxs_odd, idxs_even
    return list(idxs)


if __name__ == '__main__':
    # networks = ['FPCN', 'VAN', 'DMN', 'DAN', 'Limbic', 'SM', 'Visual']
    # for network in networks:
    #     idxs = get_yeo_idxs(network, combine_regions=True)
    #     print(f'{network=}, {len(idxs)=}')

    test_fluc_mag_networks()
    # get_bna2yeo()
    # quit()
    # get_alt_network_ef()
