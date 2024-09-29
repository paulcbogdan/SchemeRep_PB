from collections import defaultdict

import numpy as np
from scipy import stats as stats
from tqdm import tqdm

from connRSA.single_trial_conn import prep_fps
from connRSA.conn_utils import get_ROI_vecs_wrap
from connRSA.old_Sep29.conn_RSA import get_trial_x_trial_RSM
from atlas_utils import get_atlas
from organize_bhv import get_trial_info
from org_sns import get_sns


def conn_autocorrelation(combine_regions=False, split=False, four_tasks=False,
                         conn='euc'):

    # this suggests that to measure the similarity between trials 0 and 1, you
    #   should standardize the two by the mean and std calculated solely with
    #   trials 3-38.

    # Autocorrelation is very small, like .08
    atlas = get_atlas(combine_regions=combine_regions,
                      combine_bilateral=False,
                      split=split, split_code='xyz')
    fps = prep_fps(four_tasks)

    fp = fps[0]
    sess = fp.split('_')[0].replace('2', '')
    age2sn = get_sns(ret=True)
    sns = age2sn[1]
    corr_by_sn = []
    corr_by_ROI = defaultdict(list)
    for i, sn in tqdm(enumerate(sns), desc='looping subjects'):
        df_sn = get_trial_info(sn)
        df_sn.sort_values(by=f'{sess}_trial', inplace=True)
        # for idx, row in df_sn.iterrows():
        #     print(row[fp])
        # print(df_sn[fp])
        # quit()
        ROI2vecs = get_ROI_vecs_wrap(sn, atlas, fp, df_sn, fp1=None,
                                     networks=False)
        sn_corr = []
        for i, (ROI, vecs) in enumerate(ROI2vecs.items()):
            if i % 10 != 0: continue
            vecs = ROI2vecs[ROI]

            keeps = ~np.isnan(vecs).any(axis=0)
            vecs = vecs[:, keeps]
            if vecs.shape[-1] < 10: continue
            # print(f'{vecs.shape=}')
            # quit()
            # print(vecs.shape)
            # quit()
            # vecs = stdize(vecs, axis=1, nans=True)
            # print(f'{vecs.shape=}')
            # vecs = stdize(vecs, axis=1, nans=True)
            # vecs = get_conn_vecs(vecs, conn=conn)
            # ROI_corr = []

            # print(vecs.shape)
            # quit()
            # vecs = np.vstack([vecs[:10], vecs[38:48], vecs[76:86]])
            trial_per_run = vecs.shape[0] // 3

            # print(vecs.shape)
            # quit()

            # for run in range(3):
            #     low = run * trial_per_run
            #     high = (run + 1) * trial_per_run
            #     # vecs_run = vecs[low:high, :]
            #     # vecs_run = stdize(vecs_run, axis=0, nans=True)
            #     vecs[low:high, :] = stdize(vecs[low:high, :], axis=0, nans=True)

                # for trial in range(trial_per_run):
                #     if trial == trial_per_run - 1: continue
                #     data0 = vecs_run[trial, :]
                #     data1 = vecs_run[trial + 1, :]
                #     r_order = stats.pearsonr(data0, data1)[0]
                #
                #     alts = []
                #     for trial_alt in range(trial_per_run):
                #         if trial_alt == trial: continue
                #         data1 = vecs_run[trial_alt, :]
                #         r = stats.pearsonr(data0, data1)[0]
                #         alts.append(r)
                #
                #     r_dif = r_order - np.mean(alts)
                #     ROI_corr.append(r_order)
                    # sn_corr.append(r_dif)

            # RDM_fMRI = np.corrcoef(vecs)
            RDM_fMRI = get_trial_x_trial_RSM(vecs)


            # print(RDM_fMRI.shape)
            # print(vecs.shape)
            # quit()

            RDM_stim = np.zeros_like(RDM_fMRI)
            for run0 in range(3):
                for run1 in range(3):
                    if run0 != run1: continue
                    low0 = run0 * trial_per_run
                    high0 = (run0 + 1) * trial_per_run
                    low1 = run1 * trial_per_run
                    high1 = (run1 + 1) * trial_per_run
                    for i in range(trial_per_run):
                        for j in range(trial_per_run):
                            RDM_stim[low0 + i, low1 + j] = 38 - abs(i - j)


            # plt.imshow(RDM_stim)
            # plt.show()
            trils = np.tril_indices_from(RDM_fMRI, k=-1)
            RDM_fMRI_flat = RDM_fMRI[trils]
            RDM_stim_flat = RDM_stim[trils]
            r, p = stats.spearmanr(RDM_fMRI_flat, RDM_stim_flat)
            corr_by_ROI[ROI].append(r)
            # plt.imshow(RDM_fMRI)
            # plt.show()
            # quit()
            # plt.imshow(RDM_stim)
            # plt.show()
            # z = RDM_x_RDM_by_run(RDM_fMRI, RDM_stim, corr='spear')
            # corr_by_ROI[ROI].append(z)
            # quit()
            # print(f'{r=}')
            # quit()



            # quit()

            # n_edges = vecs.shape[1]
            # for j in range(n_edges):
            #     rs = []
            #     for run in range(3):
            #         low = run * trial_per_run
            #         high = (run + 1) * trial_per_run
            #         time_series = vecs[low:high, j]
            #         # print(f'{vecs.shape=}, {run=},{time_series.shape=}')
            #         time_series0 = time_series[:-1]
            #         time_series1 = time_series[1:]
            #         # plt.scatter(range(len(time_series0)), time_series0)
            #         # plt.show()
            #         r = stats.pearsonr(time_series0, time_series1)[0]
            #         rs.append(r)
            #     sn_corr.append(np.mean(rs))
            #     ROI_corr.append(np.mean(rs))
            # corr_by_ROI[ROI].append(np.mean(ROI_corr))

        for ROI, l in corr_by_ROI.items():
            print(f'{ROI}, {np.mean(l)=:.3f} ({np.std(l)=:.3f})')


        sn_corr = np.array(sn_corr)
        # plt.title(f'{sn=}')
        # plt.hist(sn_corr)
        # corr_sn_region_val.append(corr_by_sn)
        # print(f'{corr_by_sn.shape=}')
        # plt.imshow(corr_by_sn)
        # plt.colorbar()
        # plt.show()
        continue
        M = np.mean(autocorrelation_all_sn)
        SD = np.std(autocorrelation_all_sn)
        print(f'N = {i + 1} ({len(autocorrelation_all_sn)}): '
              f'{M=:.5f} ({SD=:.5f})')
        #     print(f'{ROI}, {autocorrelation=:.3f}')
        # quit()


if __name__ == '__main__':
    conn_autocorrelation()