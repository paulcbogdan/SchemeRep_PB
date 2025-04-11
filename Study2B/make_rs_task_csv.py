import pickle
import pandas as pd
import numpy as np

import os
import pathlib

from Study2B.plot_Fig5B_histogram import get_np_t_task

path = pathlib.Path(__file__).parent.parent.resolve()
os.chdir(path)

if __name__ == '__main__':
    WL = True
    SCHAEFER = False
    COMBINE_REGIONS = False
    JUST_LR = None # 'L'
    np_rs, np_t = get_np_t_task(wl=WL, schaefer=SCHAEFER,
                                combine_regions=COMBINE_REGIONS,
                                just_lr=JUST_LR)

    n_roi = np_rs.shape[0]
    n_sn = np_rs.shape[1]

    idxs = np.arange(n_roi)
    np.random.shuffle(idxs)
    np_rs = np_rs[idxs]
    np_t = np_t[idxs]

    df_as_l = []
    for i in range(n_roi):
        # if np_t.shape[0] > 1000:
        #     if i % 10 != 0:
        #         continue
        for j in range(n_sn):
            d = {'roi': i, 'sn': j,
                 'rs': np_rs[i, j],
                 'task': np_t[i, j]}
            df_as_l.append(d)
    df = pd.DataFrame(df_as_l)

    n_df = len(df)
    lr_str = f'_{JUST_LR}' if JUST_LR else ''

    if WL:
        if SCHAEFER:
            df.to_csv(fr'Study2B/rs_task_schaefer_wl_df_{n_df}.csv', index=False)
        else:
            df.to_csv(fr'Study2B/rs_task_wl_df_{n_df}.csv', index=False)
    else:
        if SCHAEFER:
            if COMBINE_REGIONS:
                fp = fr'Study2B/rs_task_schaefer_comb_df_{n_df}.csv'
                df.to_csv(fp, index=False)
                print(f'Out: {fp=}')
            else:
                df.to_csv(fr'Study2B/rs_task_schaefer_df{lr_str}_{n_df}.csv', index=False)
        else:
            df.to_csv(fr'Study2B/rs_task_df_{n_df}.csv', index=False)
