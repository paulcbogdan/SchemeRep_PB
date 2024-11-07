from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt

from old_Apr6.corr_RSA_x_vendor import get_plain_df_sn

import matplotlib.ticker as mtick
import scipy.stats as stats


# os.chdir(r'/')

def plot_group(df, group, title):
    df = df[df['age'] == group]
    d = defaultdict(list)
    for sn in df['sn'].unique():
        df_sn = df[df['sn'] == sn]
        for vis_type, df_vt in df_sn.groupby('vis_type'):
            n_trials = len(df_vt)
            for vis_resp in ['old', 'similar', 'new']:
                df_vr = df_vt[df_vt['vis_resp'] == vis_resp]
            # for vis_resp, df_vr in df_vt.groupby('vis_resp'):
                n_resp = len(df_vr)
                # n = len(df_vr)
                p = n_resp / n_trials
                # print(f'{vis_type=} {vis_resp=} {p=:.4f}')
                # ls.append(p)
                d[(vis_type, vis_resp)].append(p)
                # print(f'{vis_type=} {vis_resp=} {n=}')
                # print(f'{n / n_sn=:.4f}')
    plt.rcParams.update({'font.size': 14})
    plt.title(f'{title}')
    for i, resp in enumerate(['old', 'similar', 'new']):
        l_resp_sim = [d[('old', resp)],
                      d[('similar', resp)],
                      d[('new', resp)]]
        l_resp_sim = np.array(l_resp_sim)
        l_resp_SE = np.nanstd(l_resp_sim, axis=1) / np.sqrt(l_resp_sim.shape[1])
        l_resp_sim = np.nanmean(l_resp_sim, axis=1)

        # print(l_resp_sim.shape)
        # quit()
        plt.bar(np.arange(3) + i * .2 - .2,
                l_resp_sim, .2, label=f'Response: {resp}',
                yerr=l_resp_SE, capsize=5)
    plt.xticks(np.arange(3),
               ['Condition:\nold', 'Condition:\nsimilar',
                'Condition:\nnew'])
    plt.ylim(0, 1)
    plt.ylabel('Percentage of trials')
    plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
    plt.legend(frameon=False, loc='upper left')
    plt.tight_layout()
    plt.show()

    index = (np.array(d[('similar', 'similar')]) -
             np.array(d[('new', 'similar')]))
    print(f'{title} (N = {len(index)}): '
          f'{np.mean(index)=:.4f} {np.std(index)=:.4f}')
    return index

if __name__ == '__main__':
    # age2sns = pickle_wrap(get_sns, None, kwargs={'fp_fMRI': 'loose'},
    #                       easy_override=True)

    # df = get_trial_info('103', easy_override=True, ret=True,
    #                     incl_lures=True,
    #                     verbose=0)
    # print(len(df))
    # sns = age2sns['healthy']
    # for sn in sns:
    #     fp_bhv = fr'behavFiles/RET_vis/S{sn}_run{run}_RV.mat'
    #     mat_enc = io.loadmat(fp_bhv)

    df, _ = get_plain_df_sn(bad_sns=None, incl_lures=True)
    n_sn = len(df['sn'].unique())
    n_row = len(df)
    print(n_sn)
    print(n_row)
    print(n_row / n_sn)

    # df_similar = df[df['vis_type'] == 'similar']
    # df_sim_sim = df_similar[df_similar['vis_resp'] == 'similar']
    # p_sim_sim = len(df_sim_sim) / len(df_similar)
    # print(f'{p_sim_sim=:.4f}')
    #
    # df_new = df[df['vis_type'] == 'new']
    # df_new_sim = df_new[df_new['vis_resp'] == 'similar']
    # p_new_sim = len(df_new_sim) / len(df_new)
    # print(f'{p_new_sim=:.4f}')
    #
    # age = 'YA'

    idx_YA = plot_group(df, 1, 'Younger adults')
    idx_OA = plot_group(df, 2, 'Older adults')
    t, p = stats.ttest_ind(idx_YA, idx_OA)
    print(f'{t=:.4f} {p=:.4f}')





