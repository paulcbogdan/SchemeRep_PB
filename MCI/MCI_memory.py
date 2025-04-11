import os

from organize_bhv import get_trial_info
from org_sns import get_sns
import pandas as pd
import numpy as np

os.chdir(r"C:\PycharmProjects\SchemeRep")


if __name__ == "__main__":
    sns = get_sns("all")["healthy"]
    sns_YA = [sn for sn in sns if sn[0] == "1"]
    # print(len(sns_YA))
    sns_OA = [sn for sn in sns if sn[0] == "2"]
    # print(len(sns_OA))
    sns_MCI = [str(x) for x in range(301, 317)]
    # 307 has no encoding behavioral data, neither does 314
    sns_MCI = [sn for sn in sns if sn not in ['307', "314"]]
    key = 'vis_hit'

    dfs_YA = [get_trial_info(sn, ret=True, easy_override=False) for sn in sns_YA]
    YA_l = []
    for df in dfs_YA:
        x = np.nanmean(df[key])
        YA_l.append(x)

    dfs_OA = [get_trial_info(sn, ret=True, easy_override=False) for sn in sns_OA]
    OA_l = []
    for df in dfs_OA:
        x = np.nanmean(df[key])
        OA_l.append(x)


    dfs_MCI = [get_trial_info(sn, ret=True, easy_override=False) for sn in sns_MCI]
    MCI_l = []
    for df in dfs_MCI:
        x = np.nanmean(df[key])
        MCI_l.append(x)

    M_YA = np.nanmean(YA_l)
    SE_YA = np.nanstd(YA_l) / np.sqrt(len(YA_l))
    M_OA = np.nanmean(OA_l)
    SE_OA = np.nanstd(OA_l) / np.sqrt(len(OA_l))
    M_MCI = np.nanmean(MCI_l)
    SE_MCI = np.nanstd(MCI_l) / np.sqrt(len(MCI_l))

    print(f'{M_YA=:.2f} ({SE_YA:.2f})')
    print(f'{M_OA=:.2f} ({SE_OA:.2f})')
    print(f'{M_MCI=:.2f} ({SE_MCI:.2f})')

    # sns_YA = get_sns("all")["YA"]
    # sns_OA = get_sns("all")["OA"]
    # print(len(sns))
