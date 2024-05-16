import os

os.chdir(r'E:\PycharmProjects_E\SchemeRep')

import pandas as pd

from old_Apr6.corr_RSA_x_vendor import get_plain_df_sn
import matplotlib.pyplot as plt
import numpy as np

np.float = float
np.bool = bool
np.int = int

if __name__ == '__main__':
    df, _ = get_plain_df_sn()
    df = df.groupby(['sn', 'inc'])[['per_inc']].mean()
    df.reset_index(inplace=True)
    df['per_inc'] = df['per_inc'].astype(float)
    df['inc'] = df['inc'].astype(str)
    df.dropna(inplace=True)
    print(df)
    # df = df[df['inc'] == '3']
    # df['inc'] = df['inc'].map({'1': 'Incongruent', '2': 'Neutral',
    #                            '3': 'Congruent'})
    pd.set_option('display.max_rows', None)
    df_ = df.sort_values(by='per_inc')
    print(df_)

    # g = sns.catplot(x='inc', y='per_inc',
    #                     # hue='orange',
    #                     data=df,
    #                 jitter=0.1,
    #
    #                     )

    # set font to Arial
    plt.rcParams['font.sans-serif'] = 'Arial'
    # set fontsize to 16
    plt.rcParams.update({'font.size': 20})

    jitter_mag = 0.1
    inc2color = {1: 'r', 2: 'g', 3: 'dodgerblue'}
    for sn, df_sn in df.groupby('sn'):
        x = []
        for inc in range(1, 4):
            inc_score = df_sn[df_sn['inc'] == str(inc)]['per_inc'].values[0]
            jitter = np.random.uniform(-jitter_mag, jitter_mag)
            plt.scatter(inc + jitter, inc_score,
                        alpha=0.65, c=inc2color[inc], s=30, linewidth=0)
            plt.scatter(inc + jitter, inc_score,
                        alpha=0.65, c='w', s=1, linewidth=0)
            x.append(inc + jitter)
        plt.plot(x, df_sn['per_inc'], color='k', alpha=0.25,
                 linestyle=(5, (10, 3)),
                 linewidth=0.35, zorder=-1)

    # for sn, df_sn in df.groupby('sn'):
    #     # if sn != '227': continue
    #     plt.plot([1, 2, 3], df_sn['per_inc'], color='k', alpha=0.2,
    #              linestyle='--', linewidth=1)

    plt.xlabel('')
    # plt.xticks([1, 2, 3], ['Incongruent', 'Neutral', 'Congruent'])
    plt.xticks([1, 2, 3], ['High PE', 'Medium PE', 'Low PE'])

    plt.ylabel('Likelihood response', labelpad=8)
    # plt.gca().set_ylabel('Likelihood response', labelpad=5)
    plt.yticks([1, 2, 3, 4])
    plt.gca().spines[['top', 'right']].set_visible(False)
    plt.gca().tick_params(axis='x', which='major', pad=10, length=8)

    plt.ylim(1, 4)
    fp_out = r'result_pics/other/behavior_bar_graph.png'
    plt.savefig(fp_out, dpi=600)
    plt.show()



