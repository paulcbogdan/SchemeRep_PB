import numpy as np
import scipy.stats as stats
import pandas as pd
import pingouin as pg


if __name__ == '__main__':

    # x = np.random.normal(0, 1, (5, 100000))

    num_ROIS = 5
    # cov = np.full((num_ROIS, num_ROIS), 0.1)
    # cov[np.diag_indices_from(cov)] = 1

    # cov = [[1.0, 0.569792179220749, 0.0863418131649791, 0.11049860890265359],
    #        [0.569792179220749, 1.0, 0.09626888456090456, 0.2340526294594836],
    #        [0.0863418131649791, 0.09626888456090456, 1.0, 0.09142026050694677],
    #        [0.11049860890265359, 0.2340526294594836, 0.09142026050694677, 1.0]]

    cov = [[1.0, 0.569792179220749, 0.0863418131649791, 0.11049860890265359, 0.4744860197168674],
           [0.569792179220749, 1.0, 0.09626888456090456, 0.2340526294594836, 0.574046017913156],
           [0.0863418131649791, 0.09626888456090456, 1.0, 0.09142026050694677, 0.4176132290023326],
           [0.11049860890265359, 0.2340526294594836, 0.09142026050694677, 1.0, 0.36198624591242934],
           [0.4744860197168674, 0.574046017913156, 0.4176132290023326, 0.36198624591242934, 1.0]]

    x = np.random.multivariate_normal(np.zeros(num_ROIS), cov, 100_000).T
    # x = np.random.normal(0, 1, (num_ROIS, 100000))

    # x -= x.mean(axis=0) * .1
    # new_cov = np.cov(x)
    # print(new_cov)
    # quit()

    # x = stats.zscore(x, axis=1)
    dd = x[0] * x[1]
    vv = x[2] * x[3]
    dv_ant = x[0] * x[2]
    dv_pos = x[1] * x[3]
    cov_regions = x[4:, :]
    # cov_regions = cov_regions + np.random.normal(0, 0.1, (100, 100_000))

    pd_no = np.nanmean(x[0] * cov_regions, axis=0)
    ad_no = np.nanmean(x[1] * cov_regions, axis=0)
    av_no = np.nanmean(x[2] * cov_regions, axis=0)
    pv_no = np.nanmean(x[3] * cov_regions, axis=0)
    no_no = np.nanmean(cov_regions * cov_regions, axis=0)

    ar = np.stack([dd, vv, dv_ant, dv_pos, pd_no, ad_no, av_no, pv_no, no_no],
                  axis=0)
    df = pd.DataFrame(ar.T, columns=['dd', 'vv', 'dv_ant', 'dv_pos',
                                     'pd_no', 'ad_no', 'av_no', 'pv_no',
                                     'no_no'])

    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']

    # out = pg.partial_corr(data=df, x='dd_vv', y='dv_dv',
    #                       covar=['pd_no', 'ad_no', 'av_no', 'pv_no',
    #                              'no_no'])

    out = pg.partial_corr(data=df, x='dd_vv', y='dv_dv',
                          covar=['pd_no', 'ad_no', 'av_no', 'pv_no',
                                 # 'no_no'
                                 ])

    # dd_vv = dd + vv
    # dv_dv = dv_ant + dv_pos

    # r, p = stats.pearsonr(df['dd_vv'], df['dv_dv'])
    # print(f'{r=:.3f}, {p=:.3f}')
    print(out)

    r, p = stats.pearsonr(df['dd_vv'], df['dv_dv'])
    print(f'dd_vv x dv_dv: {r=:.3f}, {p=:.3f}')

    r, p = stats.pearsonr(df['dd'], df['vv'])
    print(f'dd x vv: {r=:.3f}, {p=:.3f}')


