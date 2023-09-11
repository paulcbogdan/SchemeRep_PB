from scipy import io
from glob import glob
import pandas as pd

# TODO: measure where congruent is more correlated object x scene

NAME_RENAMER = {'inside of a car': 'car',
               'surfing board': 'surfboard',
               'tropical volcano': 'volcano',
               'coffee shop': 'cafe',
               'religious statue': 'statuette',
               'oversize tire': 'tire',
               'restroom stall': 'bathroom',
               'front porch': 'porch',
               'ice stadium': 'hockey rink',
               'laundry hamper': 'laundry basket',
               'dumb bell': 'dumbbell',
               # 'rock-climbing shoe': 'climbing shoe',
               'rock-climbing shoe': 'shoe', # maybe not the best
               'book bag': 'backpack',
               'sea gull': 'seagull',
               'hair salon': 'salon',
               'movie theater': 'theater',
               'snowy mountains': 'mountain',
               'dining chair': 'chair',
               'dining table': 'table',
               'police baton': 'baton',
               'grocery store': 'supermarket',
               'apartment complex': 'apartment',
               'concert hall': 'orchestra',
               'display cabinet': 'cabinet',
               'potted plant': 'plant',
               'construction helmet': 'hard hat',
               'game token': 'token',
               'Eiffel Tower': 'landmark',
               'binder clip': 'clip',
               'office space': 'office',
               'haircomb': 'comb',
               'soccerball': 'soccer ball',
               'McDonald\'s': 'fast food',
               'ATM': 'automatic teller machine',
               'college quad': 'college campus',
               'picnic blanket': 'blanket',
               }

def organize_subj_df(sn):
    renamer = NAME_RENAMER

    bhv_root = r'SchemRep_tasks/PTBtasks/results'
    bhv_root = fr'behavFiles/ENC'
    fmri_root = fr'Day2EncSingleTrialModellingLSS_sorted/{sn}/all_ENCruns_sorted/objects'
    df_sn_as_l = []
    for run in range(1, 4):
        fp_bhv = fr'{bhv_root}/S{sn}_run{run}.mat'
        mat = io.loadmat(fp_bhv)
        for trial in range(1, 39):
            glob_fMRI = fr'{fmri_root}/Day2_Run{run}_Trial{trial}_*.nii'
            #print(glob_fMRI)
            fp_fMRI = glob(glob_fMRI)[0]
            obj = mat['pdata'][0][0][7][0][trial - 1][0]
            scene = mat['pdata'][0][0][8][0][trial - 1][0]
            CIN = mat['pdata'][0][0][9][0][trial - 1][0][0]
            if obj in renamer:
                # print('In:', obj)
                obj = renamer[obj]
            if scene in renamer:
                # print('In:', scene)
                scene = renamer[scene]
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
