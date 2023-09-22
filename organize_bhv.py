from collections import defaultdict

from scipy import io
from glob import glob
from pprint import pprint
import pandas as pd

# TODO: measure where congruent is more correlated object x scene

NAME_RENAMER_OLD = {'inside of a car': 'car',
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

NAME_RENAMER = {'inside of a car': 'car',
               'surfing board': 'surfboard', # object
               'tropical volcano': 'volcano',
               'coffee shop': 'cafe',
               #'religious statue': 'statuette',
               #'oversize tire': 'tire',
               'restroom stall': 'bathroom',
               'front porch': 'porch',
               'ice stadium': 'hockey rink',
               #'laundry hamper': 'laundry basket',
               'dumb bell': 'dumbbell',
               # 'rock-climbing shoe': 'climbing shoe',
               'rock-climbing shoe': 'rock climbing shoe', # maybe not the best
               # 'book bag': 'backpack',
               'sea gull': 'seagull', # object
               'hair salon': 'salon',
               #'movie theater': 'theater',
               'snowy mountains': 'mountain',
               #'dining chair': 'chair',
               #'dining table': 'table',
               #'police baton': 'police baton',
               'grocery store': 'supermarket',
               'apartment complex': 'apartment',
               'concert hall': 'orchestra',
               'display cabinet': 'cabinet',
               #'potted plant': 'plant',
               #'construction helmet': 'hard hat',
               #'game token': 'game token',
               'Eiffel Tower': 'paris landmark',
               #'binder clip': 'binder clip',
               'office space': 'office',
               'haircomb': 'hair comb', # object
               'soccerball': 'soccer ball', # object
               'McDonald\'s': 'fast food',
               #'ATM': 'ATM', # automatic teller machine
               'college quad': 'college campus',
               #'picnic blanket': 'blanket',
               }

# NAME_RENAMER_NEW = {'inside of a car': 'car',
#                'surfing board': 'surfboard',
#                'tropical volcano': 'volcano',
#                'coffee shop': 'cafe',
#                'religious statue': 'statuette',
#                'oversize tire': 'tire',
#                'restroom stall': 'bathroom',
#                'front porch': 'porch',
#                'ice stadium': 'hockey rink',
#                'laundry hamper': 'laundry basket',
#                'dumb bell': 'dumbbell',
#                # 'rock-climbing shoe': 'climbing shoe',
#                'rock-climbing shoe': 'climbing shoe', # maybe not the best
#                'book bag': 'backpack',
#                'sea gull': 'seagull',
#                'hair salon': 'salon',
#                'movie theater': 'theater',
#                'snowy mountains': 'mountain',
#                'dining chair': 'chair',
#                'dining table': 'table',
#                'police baton': 'police baton',
#                'grocery store': 'supermarket',
#                'apartment complex': 'apartment',
#                'concert hall': 'orchestra',
#                'display cabinet': 'cabinet',
#                'potted plant': 'plant',
#                'construction helmet': 'hard hat',
#                'game token': 'game token',
#                'Eiffel Tower': 'paris landmark',
#                'binder clip': 'binder clip',
#                'office space': 'office',
#                'haircomb': 'hair comb',
#                'soccerball': 'soccer ball',
#                'McDonald\'s': 'fast food',
#                'ATM': 'ATM', # automatic teller machine
#                'college quad': 'college campus',
#                'picnic blanket': 'blanket',
#                }


def get_trial_info(sn):
    renamer = NAME_RENAMER

    obj_root = fr'Day2EncSingleTrialModellingLSS_sorted/{sn}/all_ENCruns_sorted/objects'
    scn_root = fr'Day2EncSingleTrialModellingLSS_sorted/{sn}/all_ENCruns_sorted/scenes'
    df_sn_as_l = []
    for run in range(1, 4):
        fp_enc = fr'behavFiles/ENC/S{sn}_run{run}.mat'
        mat_enc = io.loadmat(fp_enc)
        for trial in range(1, 39):
            glob_obj = fr'{obj_root}/Day2_Run{run}_Trial{trial}_*.nii'
            fp_obj = glob(glob_obj)[0]
            glob_scn = fr'{scn_root}/Day2_Run{run}_Trial{trial}_*.nii'
            fp_scn = glob(glob_scn)[0]

            obj = mat_enc['pdata'][0][0][7][0][trial - 1][0]
            scene = mat_enc['pdata'][0][0][8][0][trial - 1][0]
            CIN = mat_enc['pdata'][0][0][9][0][trial - 1][0][0]
            resp = mat_enc['pdata'][0][0][11][0][trial - 1][0][0]

            if obj in renamer:
                obj_rename = renamer[obj]
            else:
                obj_rename = obj
            if scene in renamer:
                scene_rename = renamer[scene]
            else:
                scene_rename = scene

            d = {'trial': trial,
                 'run': run,
                 'fp_fMRI': fp_obj,
                 'obj_fMRI': fp_obj,
                 'scn_fMRI': fp_scn,
                 'obj': obj,
                 'scene': scene,
                 'obj_rename': obj_rename,
                 'scene_rename': scene_rename,
                 'CIN': CIN,
                 'perceived_con': resp # higher (up to 4) = seen as congruent
                 }
            df_sn_as_l.append(d)
    df_sn = pd.DataFrame(df_sn_as_l)
    df_sn = process_conc_retrieval(df_sn, sn)
    return df_sn

def process_conc_retrieval(df_sn, sn):
    obj2resp = {}
    obj2otype = {} # unused
    for run in range(1, 4):
        fp_enc = fr'behavFiles/RET_con/S{sn}_run{run}_RC.mat'
        try:
            mat_enc = io.loadmat(fp_enc)
        except FileNotFoundError:
            break
        objs = []
        for stim in mat_enc['pdata'][0][0][6][0]:
            objs.append(str(stim[0]))
        otypes = [] # 0 = old, 1 = new
        for otype in mat_enc['pdata'][0][0][8][0]:
            otypes.append(int(otype[0]))
        resps = []
        for resp in mat_enc['pdata'][0][0][9][0]:
            try:
                resps.append(int(resp[0]))
            except ValueError: # no response
                resps.append(None)
        obj2resp_run = dict(zip(objs, resps))
        obj2resp.update(obj2resp_run)
        obj2otype_run = dict(zip(objs, otypes))
        obj2otype.update(obj2otype_run)
    else:
        df_sn['ON'] = df_sn['obj'].map(obj2resp)
        df_sn['hit_bool'] = df_sn['ON'] >= 3 # Old is 3 or 4
    return df_sn


def get_all_sns():
    age2sn = defaultdict(list)
    bad_sns = {'126', '131',
               '201', '224', '231', '232', '233', '234', '235'}
    for age in range(1, 4):
        bhv_root = fr'behavFiles/ENC/S{age}*_run1.mat'
        fns = glob(bhv_root)
        for fn in fns:
            sn = fn.replace('behavFiles/ENC\\S', '').replace('_run1.mat', '')
            # sn = fn.replace(r'behavFiles/ENC/S', '').replace('_run1.mat', '')
            if sn in bad_sns:
                continue
            age2sn[age].append(sn)
    age2sn['healthy'] = age2sn[1] + age2sn[2]
    return age2sn

if __name__ == '__main__':
    get_trial_info('138')