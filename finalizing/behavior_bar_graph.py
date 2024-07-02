import os

os.chdir(r'H:\PycharmProjects_H\SchemeRep')

import pandas as pd
import matplotlib.ticker as mtick

from old_Apr6.corr_RSA_x_vendor import get_plain_df_sn
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

np.float = float
np.bool = bool
np.int = int

def per_inc_bar(key='per_inc'):
    df, _ = get_plain_df_sn()
    # print(df['vis_hit'])
    # quit()
    # df = df.groupby(['sn', 'inc'])[[key]].mean()
    df.reset_index(inplace=True)
    # df['per_inc'] = df[key].astype(float)
    df.dropna(subset=['inc', 'per_inc'], inplace=True)
    # df['per_inc'] = df['per_inc'].astype(int).astype(str)
    df['inc'] = df['inc'].astype(str)
    # df.dropna(inplace=True)

    pd.set_option('display.max_rows', None)
    # df_ = df.sort_values(by='per_inc')


    # set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'
    # set fontsize to 16
    plt.rcParams.update({'font.size': 20})

    jitter_mag = 0.1
    inc2color = {1: 'r', 2: 'purple', 3: 'dodgerblue'}

    # df['rt_rank'] = df['inc_rt'].rank()
    # max_rank = df['rt_rank'].max()
    # df['rt_rank'] /= max_rank
    # key = 'rt_rank'
    key = 'per_inc'
    df_grp = df.groupby(['sn', 'inc'])[key].mean()
    # df_inc = df_grp.groupby('inc').mean()
    # df_dif = df_inc.diff().dropna()
    print(df_grp)
    key2 = '2'
    df_dif = df_grp.loc[:, '1'] - df_grp.loc[:, key2]
    M1 = df_grp.loc[:, '1'].mean()
    SD1 = df_grp.loc[:, '1'].std()
    M3 = df_grp.loc[:, key2].mean()
    SD3 = df_grp.loc[:, key2].std()
    print(f'{M1=:.3f}, {SD1=:.3f}, {M3=:.3f}, {SD3=:.3f}')
    t, p = stats.ttest_rel(df_grp.loc[:, '1'], df_grp.loc[:, key2])
    n = len(df_grp.loc[:, '1'])
    d = t / np.sqrt(n)
    print(f'{t=:.3f} (N = {n}) {p=:.3f}, {d=:.3f}')

    # df_inc = df.groupby(['inc'])[key].mean()



    # M1 = df_inc.loc['1']

    quit()

    for sn, df_sn in df.groupby('sn'):
        x = []
        for inc in range(1, 4):
            inc_score = df_sn[df_sn['inc'] == str(inc)][key].values[0]

            jitter = np.random.uniform(-jitter_mag, jitter_mag)
            plt.scatter(inc + jitter, inc_score,
                        alpha=0.65, c=inc2color[inc], s=30, linewidth=0)
            plt.scatter(inc + jitter, inc_score,
                        alpha=0.65, c='w', s=1, linewidth=0)
            x.append(inc + jitter)
        plt.plot(x, df_sn['per_inc'], color='k', alpha=0.25,
                 linestyle=(5, (10, 3)),
                 linewidth=0.35, zorder=-1)

    plt.xlabel('')
    plt.xticks([1, 2, 3], ['High PE', 'Medium PE', 'Low PE'])

    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.gca().tick_params(axis='x', which='major', pad=10, length=8)

    if key == 'per_inc':
        plt.ylim(1, 4)
        plt.yticks([1, 2, 3, 4])
        plt.ylabel('Perceived likelihood', labelpad=8)
        fp_out = r'result_pics/other/behavior_bar_graph.png'

    elif key == 'con_hit':
        plt.ylim(0, 1)
        plt.yticks([0, .25, .5, .75, 1.])
        plt.ylabel('Retrieval accuracy', labelpad=8)
        plt.gca().yaxis.set_major_formatter(mtick.StrMethodFormatter('{x:.0%}'))
        # plt.tight_layout(h_pad=0.)
        fp_out = r'result_pics/other/con_hit_bar_graph.png'
    else:
        raise ValueError

    plt.gcf().subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.15)
    plt.savefig(fp_out, dpi=600)
    plt.show()


def con_memory_bar():
    pass

if __name__ == '__main__':
    # per_inc_bar()
    per_inc_bar('inc_rt')

    # per_inc_bar('con_hit')



