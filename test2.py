from scipy import io
from glob import glob
import pandas as pd

def organize_subj_df(sn):
    bhv_root = r'SchemRep_tasks/PTBtasks/results'
    fmri_root = fr'Day2EncSingleTrialModellingLSS_sorted/{sn}/all_ENCruns_sorted/objects'
    df_sn_as_l = []
    for run in range(1, 4):
        fp_bhv = fr'{bhv_root}/S{sn}_run{run}.mat'
        mat = io.loadmat(fp_bhv)
        for trial in range(1, 39):
            glob_fMRI = fr'{fmri_root}/Day2_Run{run}_Trial{trial}_*.nii'
            fp_fMRI = glob(glob_fMRI)[0]
            # print(mat['pdata'][0][0][9])
            # quit()
            obj = mat['pdata'][0][0][7][0][trial - 1][0]
            scene = mat['pdata'][0][0][8][0][trial - 1][0]
            CIN = mat['pdata'][0][0][9][0][trial - 1][0][0]
            d = {'trial': trial,
                 'run': run,
                 'fp_fMRI': fp_fMRI,
                 'obj': obj,
                 'scene': scene,
                 'CIN': CIN}
            df_sn_as_l.append(d)
    df_sn = pd.DataFrame(df_sn_as_l)
    return df_sn


if __name__ == '__main__':
    organize_subj_df('138')
