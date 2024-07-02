from collections import defaultdict
from datetime import datetime
from time import time

from nilearn import image

from org_sns import get_sns
from utils import pickle_wrap
from scipy import io
from glob import glob
import pandas as pd
import numpy as np
import os
from pathlib import Path


if __name__ == '__main__':
    dir_in = r'H:\PycharmProjects_H\SchemeRep\nuisance_regressors'

    age2sn = get_sns()
    sns = age2sn[1] + age2sn[2]
    for sn in sns:
        for run in range(1, 4):

            fp_mat = fr'{dir_in}\sub-{sn}_ses-2_task-ENC_run-0{run}_desc-confounds_timeseries_use_univ.mat'
            if not os.path.isfile(fp_mat):
                print(f'missing: {sn}, {run}')
                continue
            mat_enc = io.loadmat(fp_mat)
            # print(mat_enc['names'][0])
            names = [name[0] for name in mat_enc['names'][0]]
            # print(mat_enc['R'])
            df = pd.DataFrame(mat_enc['R'], columns=names)
            # print(df.columns)
    quit()
    # quit()

    sess = 2
    run = 1
    fp_tsv = rf'cache\confounds\102_{sess}_{run}_confounds.tsv'
    df_confounds = pd.read_csv(fp_tsv, delimiter='\t')

    nuisance_cols = ["global_signal", "white_matter", "csf",
                     "dvars", "framewise_displacement", "rmsd",
                     "trans_x", "trans_y", "trans_z", "rot_x",
                     "rot_y", "rot_z"]
    df_confounds = df_confounds[nuisance_cols]
    print(df_confounds)
