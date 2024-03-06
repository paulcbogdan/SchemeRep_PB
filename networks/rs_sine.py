import pdb

import pandas as pd

from dFC_control_FC import get_all_task_vendor
from factor_analysis import identify_extremely_low_variance_sn
from org_sns import get_sns
from organize_bhv import get_trial_info
import os
os.chdir(r'E:\PycharmProjects_E\SchemeRep')


def do_rs_sine(fp='rs'):
    # df = load_FA(fp)
    df = get_all_task_vendor(anat=True, zscore=True)



    # df = df[df['task'] != 'RS']
    # df = df[df['task'] == 'RS']
    bad_sns = identify_extremely_low_variance_sn(df)
    bad_sns.add('138')
    bad_sns.add('217') # almost always near zero for Horz
    bad_sns.add('110') # almost always nere zero for Vendor
    df = df[~df['sn'].isin(bad_sns)]
    df.dropna(subset=['da', 'dp', 'va', 'vp'], inplace=True)

    df['vert'] = df['da'] + df['dp'] - df['va'] - df['vp']
    df['horz'] = df['da'] - df['dp'] + df['va'] - df['vp']

    df['dd_vv'] = df['dd'] + df['vv']
    df['dv_dv'] = df['dv_ant'] + df['dv_pos']
    df['vendor'] = df['dd_vv'] - df['dv_dv']


def test():
    sns = get_sns()
    dfs_l = []
    for sn in sns[1] + sns[2]:
        df_sn = get_trial_info(sn, easy_override=False)
        dfs_l.append(df_sn)
    df = pd.concat(dfs_l)
    #
    # print(df[['vis_hit', 'con_hit']])
    # quit()
    # print(list(df.columns))
    # quit()

    df['vis_hit_strict'] = df['vis_hit'] == 1
    df['con_hit_strict'] = df['con_hit'] == 1
    # print(df['vis_hit'].value_counts())
    # print(df['vis_hit_strict'].value_counts())
    # quit()

    df_grp = (df.groupby('sn')[['vis_hit', 'con_hit',
                                'vis_hit_strict', 'con_hit_strict']].
              mean())



    pd.set_option('display.max_rows', None)
    df_grp.to_csv('SchemeRep_hit_rate.csv')
    print(df_grp)


    # print(df['vis_hit'])
    quit()

if __name__ == '__main__':
    # do_rs_sine()
    test()