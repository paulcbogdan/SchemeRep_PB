import pandas as pd

from Utils.atlas_funcs import get_atlas
from data_driven_fluc import load_rs_BOLD
from Study2A.rs_connectivity_funcs import partial_corr_df
import numpy as np
from scipy import stats


def calc_anat_corr(shuffle=False):
    conn_trials, _ = load_rs_BOLD()
    atlas = get_atlas(combine_regions=False)
    if shuffle:
        regions_all = list(set(atlas['ROI_regions']))
        regions = np.random.choice(regions_all, 4)
        while len(regions) > len(set(regions)):
            regions = np.random.choice(regions_all, 4)
    else:
        regions = ['MFG', 'IPL', 'ATL', 'LOC', ]
    quads = []
    for region in regions:
        quad0 = [i for i, r in enumerate(atlas['ROI_regions']) if r == region]
        quads.append(quad0)

    non_used_nodes = (set(range(246)) -
                      set(quads[0]) - set(quads[1]) -
                      set(quads[2]) - set(quads[3]))
    non_used_nodes = list(non_used_nodes)

    VD0 = conn_trials[:, *np.ix_(quads[0], quads[1])]
    VD0 = np.nanmean(VD0, axis=(1, 2))
    VD1 = conn_trials[:, *np.ix_(quads[2], quads[3])]
    VD1 = np.nanmean(VD1, axis=(1, 2))
    PA0 = conn_trials[:, *np.ix_(quads[0], quads[2])]
    PA0 = np.nanmean(PA0, axis=(1, 2))
    PA1 = conn_trials[:, *np.ix_(quads[1], quads[3])]
    PA1 = np.nanmean(PA1, axis=(1, 2))

    glob = np.nanmean(conn_trials, axis=(1, 2))

    nde0 = conn_trials[:, *np.ix_(quads[0], non_used_nodes), :]
    nde0 = np.nanmean(nde0, axis=(1, 2))
    nde1 = conn_trials[:, *np.ix_(quads[1], non_used_nodes), :]
    nde1 = np.nanmean(nde1, axis=(1, 2))
    nde2 = conn_trials[:, *np.ix_(quads[2], non_used_nodes), :]
    nde2 = np.nanmean(nde2, axis=(1, 2))
    nde3 = conn_trials[:, *np.ix_(quads[3], non_used_nodes), :]
    nde3 = np.nanmean(nde3, axis=(1, 2))

    df = pd.DataFrame({'VD0': VD0.flatten(), 'VD1': VD1.flatten(),
                       'PA0': PA0.flatten(), 'PA1': PA1.flatten(),
                       'global': glob.flatten(),
                       # 'NDE': nde.flatten(),
                       'NDE0': nde0.flatten(), 'NDE1': nde1.flatten(),
                       'NDE2': nde2.flatten(), 'NDE3': nde3.flatten()})

    df['dd_vv'] = df['VD0'] + df['VD1']
    df['dv_dv'] = df['PA0'] + df['PA1']

    ar = partial_corr_df(df, ['dd_vv', 'dv_dv'],
                         ['NDE0', 'NDE1', 'NDE2', 'NDE3', ],
                         verbose=0)
    r = ar[0, 1]
    print(f'{r=}')
    return r

def calc_anat_perm(res=-.154):
    print('-' * 10)
    n = 1000
    rs = []
    for i in range(n):
        rs.append(calc_anat_corr(shuffle=True))
        cutoff = sorted(rs)[int((i + 1) * .05)]
        prop_lower = len([r for r in rs if r < res]) / (i + 1)
        if i > 1:
            M = np.mean(rs)
            SD = np.std(rs)
            res_z = (res - M) / SD
            res_p = stats.norm.sf(-res_z)
            print(f'\t{cutoff=:.3f}, {M=:.3f}, {prop_lower=:.1%} '
                  f'[{SD:.3f}, N = {len(rs)}] | '
                  f'{res:.3f}, {res_z=:.2f} ({res_p=:.3f})')
    rs = np.array(rs)
    print(f'{rs=}')
    print(f'{np.mean(rs)=}')
    print(f'{np.std(rs)=}')

if __name__ == '__main__':
    # calc_anat_corr()
    calc_anat_perm()

