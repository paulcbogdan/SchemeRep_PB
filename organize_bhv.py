from collections import defaultdict
from datetime import datetime
from time import time

# from nilearn import image

from utils import pickle_wrap
from scipy import io
from glob import glob
import pandas as pd
import numpy as np
import os
from pathlib import Path
os.chdir(r'C:\PycharmProjects\SchemeRep')

# TODO: measure where congruent is more correlated object x scene

STIM_CATEGORY = {'ambulance1.jpg': 'dead_large',
                 'apple1.jpg': 'living_plant',
                 'atm_exemplar1.jpg': 'dead_medium',
                 'banana_exemplar1.jpg': 'living_plant',
                 'beer1.jpg': 'dead_small', #??
                 'bench_exemplar1.jpg': 'dead_medium',
                 'briefcase_exemplar1.jpg': 'dead_small',
                 'binderclip1.jpg': 'dead_small',
                 'bookbag1.jpg': 'dead_small',
                 'cabin1.jpg': 'dead_large',
                 'cactus1.jpg': 'living_plant',
                 'camel_exemplar1.jpg': 'living_animal',
                 'cashregister1.jpg': 'dead_medium',
                 'car_exemplar1.jpg': 'dead_large',
                 'chalice1.jpg': 'dead_small',
                 'chandelier1.jpg': 'dead_medium',
                 'cheeseburger_exemplar1.jpg': 'dead_small',
                 'clown_exemplar1.jpg': 'living_animal',
                 'cockroach_exemplar1.jpg': 'living_animal',
                 'coffeemachine1.jpg': 'dead_small',
                 'construction_helmet_exemplar1.jpg': 'dead_small',
                 'cookie1.jpg': 'dead_small',
                 'cow_exemplar1.jpg': 'living_animal',
                 'crib_exemplar1.jpg': 'dead_medium',
                 'cross_exemplar1.jpg': 'dead_small',
                 'cuttingboard1.jpg': 'dead_small',
                 'dining_chair_exemplar1.jpg': 'dead_medium',
                 'dining_table_exemplar1.jpg': 'dead_medium',
                 'displaycabinet1.jpg': 'dead_medium',
                 'dogtoy1.jpg': 'dead_small',
                 'donut_exemplar1.jpg': 'dead_small',
                 'doorknob_exemplar1.jpg': 'dead_small',
                 'dragonfly_exemplar1.jpg': 'living_animal',
                 'dumbbell1.jpg': 'dead_small',
                 'ferriswheel1.jpg': 'dead_large',
                 'firetruck_exemplar1.jpg': 'dead_large',
                 'flower1.jpg': 'living_plant',
                 'flower_pot_exemplar1.jpg': 'dead_small',
                 'football_exemplar1.jpg': 'dead_small',
                 'frame_exemplar1.jpg': 'dead_small',
                 'game_token_exemplar1.jpg': 'dead_small',
                 'gas_can_exemplar1.jpg': 'dead_small',
                 'gavel_exemplar1.jpg': 'dead_small',
                 'gift_bag_exemplar1.jpg': 'dead_small',
                 'goggles_exemplar1.jpg': 'dead_small',
                 'golfclub1.jpg': 'dead_medium',
                 'grill1.jpg': 'dead_medium',
                 'haircomb_exemplar1.jpg': 'dead_small',
                 'hairdryer_exemplar1.jpg': 'dead_small',
                 'helmet1.jpg': 'dead_small',
                 'horse_exemplar1.jpg': 'living_animal',
                 'iceskate1.jpg': 'dead_small',
                 'jeep1.jpg': 'dead_medium',
                 'kayak_exemplar1.jpg': 'dead_medium',
                 'ladder1.jpg': 'dead_medium',
                 'laundry_hamper_exemplar1.jpg': 'dead_medium',
                 'lawmower1.jpg': 'dead_medium',
                 'lei_exemplar1.jpg': 'dead_small',
                 'log_exemplar1.jpg':'dead_medium',
                 'loudspeaker1.jpg': 'dead_medium',
                 'mailbox_exemplar1.jpg': 'dead_medium',
                 'microscope_exemplar1.jpg': 'dead_small',
                 'notebook1.jpg': 'dead_small',
                 'office_chair_exemplar1.jpg': 'dead_medium',
                 'ostrich1.jpg': 'living_animal',
                 'oversizetire1.jpg': 'dead_medium',
                 'palm_tree_exemplar1.jpg': 'living_plant',
                 'piano_exemplar1.jpg': 'dead_large',
                 'picnicblanket1.jpg': 'dead_medium',
                 'pillow_exemplar1.jpg': 'dead_medium',
                 'pine_exemplar1.jpg': 'living_plant',
                 'plant_exemplar1.jpg': 'living_plant',
                 'podium1.jpg': 'dead_medium',
                 'poker_table_exemplar1.jpg': 'dead_medium',
                 'polar_bear_exemplar1.jpg': 'living_animal',
                 'policebaton1.jpg': 'dead_medium',
                 'policecar1.jpg': 'dead_large',
                 'popcorn_machine_exemplar1.jpg': 'dead_medium',
                 'printer_exemplar1.jpg': 'dead_medium',
                 'raft_exemplar1.jpg': 'dead_large',
                 'railroad_exemplar1.jpg': 'dead_large',
                 'reed1.jpg': 'living_plant',
                 'religious_statue_exemplar1.jpg': 'dead_small',
                 'rock-climbing_shoe_exemplar1.jpg': 'dead_small',
                 'rollerskates_exemplar1.jpg': 'dead_small',
                 'rose11.jpg': 'living_plant',
                 'ruler1.jpg': 'dead_small',
                 'sailboat_exemplar1.jpg': 'dead_large',
                 'schoolbus1.jpg': 'dead_large',
                 'scorpion_exemplar1.jpg': 'living_animal',
                 'seagull_exemplar1.jpg': 'living_animal',
                 'seal1.jpg': 'living_animal',
                 'seashell_exemplar1.jpg': 'living_animal', # ?
                 'seatbelt_exemplar1.jpg': 'dead_small',
                 'sewing_machine_exemplar1.jpg': 'dead_medium',
                 'shopping_cart_exemplar1.jpg': 'dead_medium',
                 'ski_exemplar1.jpg': 'dead_medium',
                 'slide_exemplar1.jpg': 'dead_large',
                 'soccerball_exemplar1.jpg': 'dead_small',
                 'spotlight1.jpg': 'dead_medium',
                 'sprayer1.jpg': 'dead_medium',
                 'steering-wheel_exemplar1.jpg': 'dead_small',
                 'surfboard_exemplar1.jpg': 'dead_medium',
                 'swimsuit_exemplar1.jpg': 'dead_small',
                 'television_exemplar1.jpg': 'dead_medium',
                 'tennis_ball_exemplar1.jpg': 'dead_small',
                 'tile1.jpg': 'dead_small',
                 'toilet_exemplar1.jpg': 'dead_medium',
                 'trafficsign1.jpg': 'dead_medium',
                 'truck1.jpg': 'dead_large',
                 'violin_exemplar1.jpg': 'dead_small',
                 'wheat1.jpg': 'living_plant',
                 'whistle_exemplar1.jpg': 'dead_small',
                 'wineglass1.jpg': 'dead_small'}

SCENE_CATEGORY =  {'Airport1.jpg': 'outdoor_developed',
 'amphitheater1.jpg': 'outdoor_developed',
 'amusementpark1.jpg': 'outdoor_developed',
 'apartment1.jpg': 'outdoor_developed',
 'aquarium1.jpg': 'indoor',
 'arcade1.jpg': 'indoor',
 'arch1.jpg': 'outdoor_nature',
 'attic1.jpg': 'indoor',
 'bakery1.jpg': 'indoor',
 'balcony1.jpg': 'outdoor_developed',
 'bank1.jpg': 'outdoor_developed',
 'bar4.jpg': 'indoor',
 'barn_2.jpg': 'outdoor_developed',
 'bathroom2.jpg': 'indoor',
 'beach_2.jpg': 'outdoor_nature',
 'bedroom1.jpg': 'indoor',
 'bikerack.jpg': 'outdoor_developed',
 'bridge1.jpg': 'outdoor_developed',
 'buffet1.jpg': 'indoor',
 'bus2.jpg': 'indoor',
 'busstop1.jpg': 'outdoor_developed',
 'campsite1.jpg': 'outdoor_nature',
 'canal3.jpg': 'outdoor_developed',
 'canyon_1.jpg': 'outdoor_nature',
 'carinside1.jpg': 'indoor',
 'casino1.jpg': 'indoor',
 'castle1.jpg': 'outdoor_developed',
 'cemetery3.jpg': 'outdoor_developed',
 'church_1.jpg': 'outdoor_developed',
 'circusinside1.jpg': 'indoor',
 'classroom1.jpg': 'indoor',
 'climbingwall1.jpg': 'indoor', # ish
 'coast1.jpg': 'outdoor_nature',
 'coffeeshop2.jpg': 'indoor',
 'collegequad1.jpg': 'outdoor_developed',
 'conferenceroom1.jpg': 'indoor',
 'constructionsite1.jpg': 'outdoor_developed',
 'countryroad2.jpg': 'nature',
 'courtroom_3.jpg': 'indoor',
 'deli1.jpg': 'indoor',
 'desert1.jpg': 'outdoor_nature',
 'door1.jpg': 'outdoor_developed',
 'driveway1.jpg': 'outdoor_developed',
 'dump1.jpg': 'outdoor_developed',
 'eiffeltower2.jpg': 'outdoor_developed',
 'firestation1.jpg': 'outdoor_developed',
 'flowershop1.jpg': 'outdoor_developed',
 'footballfield2.jpg': 'outdoor_developed',
 'garage1.jpg': 'indoor',
 'garden1.jpg': 'outdoor_nature',
 'gasstation4.jpg': 'outdoor_developed',
 'golfcourse1.jpg': 'outdoor_nature', # ish
 'grassland1.jpg': 'outdoor_nature',
 'greenhouse1.jpg': 'indoor',
 'grocerystore1.jpg': 'indoor',
 'gym1.jpg': 'indoor',
 'hairsalon3.jpg': 'indoor',
 'homeoffice1.jpg': 'indoor',
 'hospital1.jpg': 'indoor',
 'hotellobby1.jpg': 'indoor',
 'house1.jpg': 'outdoor_developed',
 'iceberg1.jpg': 'outdoor_nature',
 'icestadium1.jpg': 'indoor',
 'islands1.jpg': 'outdoor_nature',
 'kitchen1.jpg': 'indoor',
 'lab3.jpg': 'indoor',
 'laundryroom1.jpg': 'indoor',
 'lecturehall1.jpg': 'indoor',
 'library1.jpg': 'indoor',
 'livingroom1.jpg': 'indoor',
 'mall2.jpg': 'indoor',
 'market1.jpg': 'outdoor_developed',
 'mcdonalds.jpg': 'outdoor_developed',
 'monstertruck1.jpg': 'indoor',
 'mountainssnow1.jpg': 'outdoor_nature',
 'movietheater3.jpg': 'indoor',
 'museum4.jpg': 'indoor',
 'musicstudio10.jpg': 'indoor',
 'nursery1.jpg': 'indoor',
 'office1.jpg': 'indoor',
 'orchard1.jpg': 'outdoor_nature',
 'orchestra2.jpg': 'indoor',
 'park1.jpg': 'outdoor_nature',
 'petstore1.jpg': 'outdoor_developed',
 'pier1.jpg': 'outdoor_nature',
 'playground1.jpg': 'outdoor_nature',
 'policestation1.jpg': 'outdoor_developed',
 'pond1.jpg': 'outdoor_nature',
 'postoffice1.jpg': 'outdoor_developed',
 'prison1.jpg': 'indoor',
 'pyramid1.jpg': 'outdoor_nature',
 'racetrack1.jpg': 'outdoor_nature',
 'restaurant1.jpg': 'indoor',
 'restroom-stall1.jpg': 'indoor',
 'rollerrink1.jpg': 'indoor',
 'sauna1.jpg': 'indoor',
 'seaport1.jpg': 'outdoor_developed',
 'sewingroom1.jpg': 'indoor',
 'shopfront1.jpg': 'indoor',
 'soccerfield1.jpg': 'outdoor_nature',
 'stage1.jpg': 'indoor',
 'swamp1.jpg': 'outdoor_nature',
 'swimmingpool1.jpg': 'indoor',
 'temple1.jpg': 'outdoor_developed',
 'tenniscourt1.jpg': 'outdoor_developed',
 'trainstation1.jpg': 'indoor',
 'treehouse1.jpg': 'outdoor_nature',
 'tropicalvolcano1.jpg': 'outdoor_nature',
 'volleyballcourt1.jpg': 'outdoor_developed',
 'waterfall1.jpg': 'outdoor_nature',
 'waves_1.jpg': 'outdoor_nature',
 'winery1.jpg': 'indoor',
 'woods1.jpg': 'outdoor_nature',
 'zoo1.jpg': 'outdoor_developed'}

NAME_RENAMER = {'inside of a car': 'car',
               'surfing board': 'surfboard', # object
               'tropical volcano': 'volcano',
               'coffee shop': 'cafe',
               'restroom stall': 'bathroom',
               'front porch': 'porch',
               'ice stadium': 'hockey rink',
               'dumb bell': 'dumbbell',
               'rock-climbing shoe': 'rockclimbing shoe', # maybe not the best
               'sea gull': 'seagull', # object
               'hair salon': 'salon',
               'snowy mountains': 'mountain',
               'grocery store': 'supermarket',
               'apartment complex': 'apartment',
               'concert hall': 'orchestra',
               'display cabinet': 'cabinet',
               'Eiffel Tower': 'paris landmark',
               'office space': 'office',
               'haircomb': 'comb', # object
               'soccerball': 'soccer ball', # object
               'McDonald\'s': 'fast food',
               'college quad': 'college campus',
               }

def sort_df_sn(df_sn, fp):
    sess = (fp.split('_')[0].replace('2', '').replace('3', '').replace('4', '').
            replace('7', '').replace('8', ''))
    df_sn.sort_values(by=f'{sess}_trial', inplace=True)
    return df_sn, sess

def get_trial_info(sn, easy_override=False, ret=True, incl_lures=False,
                   verbose=-1, only_one=False):
    # ret may not be needed. added in 11/25/2025 but it wasnt needed
    ret_str = '_NoRet' if not ret else ''
    # fp = fr'H:\PycharmProjects_H\SchemeRep\cache/trial_info/{sn}{ret_str}.pkl'


    # dt_max = datetime(2024, 2, 17, 1, 0, 0, 0)

    if only_one:
        if only_one == 'con':
            kwargs = {'df_sn': None, 'sn': sn}
            df_sn = pickle_wrap(include_conceptual, None, kwargs=kwargs,
                                verbose=verbose, easy_override=easy_override)
            df_sn['sn'] = sn
        elif only_one == 'vis':
            kwargs = {'df_sn': None, 'sn': sn}
            df_sn = pickle_wrap(include_vis, None, kwargs=kwargs,
                                verbose=verbose, easy_override=easy_override)
            df_sn['sn'] = sn
            # print(df_sn)
            # quit()
    else:
        kwargs = {'sn': sn, 'ret': ret, 'incl_lures': incl_lures}
        df_sn = pickle_wrap(get_trial_info_, None, kwargs=kwargs,
                            verbose=verbose, easy_override=easy_override)


    # df_sn = pickle_wrap(lambda: get_trial_info_(sn, ret), fp,
    #                     easy_override=easy_override, verbose=verbose,
    #                     dt_max=dt_max)
    return df_sn

def get_trial_info_(sn, ret=True, incl_lures=False):
    renamer = NAME_RENAMER

    # obj_root = fr'fMRI_in/{sn}/all_ENCruns_sorted/objects'
    # scn_root = fr'fMRI_in/{sn}/all_ENCruns_sorted/scenes'
    # obj_root3 = fr'fMRI_in/{sn}/ENC_rerun3/OBJ'
    # scn_root3 = fr'fMRI_in/{sn}/ENC_rerun3/SCN'
    # cmb_root3 = fr'fMRI_in/{sn}/ENC_rerun3/CMB'
    # obj_root2 = fr'fMRI_in/{sn}/Enc_rerun/obj'
    # LSS1b_root4 = fr'fMRI_in/{sn}/ENC_LSS1b/OBJ'
    # LSS1_root5 = fr'fMRI_in/{sn}/ENC_LSS1/OBJ'
    # LSS2_root6 = fr'fMRI_in/{sn}/ENC_LSS2/OBJ'
    new_GM_root = fr'fMRI_in/{sn}/ENC_GM20_LLS1_bpF_full/OBJ'
    no_GRS_enc_root = fr'fMRI_in/{sn}/Enc_NoGSR_8/OBJ'

    df_sn_as_l = []
    inc_run_cnt = defaultdict(lambda: 0)
    for run in range(1, 4):
        fp_bhv = fr'behavFiles/ENC/S{sn}_run{run}.mat'
        mat_enc = io.loadmat(fp_bhv)


        for i in range(38):
            trial = i + 1


            glob_obj7 = fr'{new_GM_root}/ENC_sub{sn}_run{run}_trial{trial}_*_object.nii'
            glob_obj7 = glob(glob_obj7)
            glob_scn7 = fr'{new_GM_root}/ENC_sub{sn}_run{run}_trial{trial}_*_scene.nii'
            glob_scn7 = glob(glob_scn7)
            try:
                fp_obj7 = glob_obj7[0]
                fp_scn7 = glob_scn7[0]
            except IndexError:
                if sn == '224' and run == 3:
                    fp_obj7 = None
                    fp_scn7 = None
                elif sn == '234' and run == 1:
                    fp_obj7 = None
                    fp_scn7 = None
                else:
                    raise IndexError(f'{sn}, {run}, {trial}')

            glob_obj8 = fr'{no_GRS_enc_root}/ENC_sub{sn}_run{run}_trial{trial}_*_object.nii'
            glob_obj8 = glob(glob_obj8)
            glob_scn8 = fr'{no_GRS_enc_root}/ENC_sub{sn}_run{run}_trial{trial}_*_scene.nii'
            glob_scn8 = glob(glob_scn8)
            try:
                fp_obj8 = glob_obj8[0]
                fp_scn8 = glob_scn8[0]
            except IndexError:
                if sn in ['116', '125', '138', '213', '230', '234',
                          '239']:
                    fp_obj8 = None
                    fp_scn8 = None
                elif sn == '224' and run == 3:
                    fp_obj8 = None
                    fp_scn8 = None
                elif sn == '234' and run == 1:
                    fp_obj8 = None
                    fp_scn8 = None
                else:
                    raise IndexError(f'fp8 IndexError: {sn}, {run}, {trial}')

            obj = mat_enc['pdata'][0][0][7][0][i][0]
            scene = mat_enc['pdata'][0][0][8][0][i][0]
            inc = mat_enc['pdata'][0][0][9][0][i][0][0]
            resp = mat_enc['pdata'][0][0][11][0][i][0][0]
            rt = mat_enc['pdata'][0][0][12][0][i][0][0]

            if obj in renamer:
                obj_rename = renamer[obj]
            else:
                obj_rename = obj
            if scene in renamer:
                scene_rename = renamer[scene]
            else:
                scene_rename = scene

            inc2str = {1: 'incongruent', 2: 'neutral', 3: 'congruent'}
            inc_str = inc2str[inc]
            per2str = {1: 'incongruent', 2:' neutral',
                       3: 'neutral', 4: 'congruent'}

            trial_full = trial + 38 * (run - 1)
            per_inc14 = resp if resp in [1, 4] else np.nan
            per_inc_str = np.nan if pd.isna(resp) else per2str[resp]
            per_inc14_str = np.nan if pd.isna(per_inc14) else per2str[per_inc14]
            per_inc_bool = 1 if resp < 2.5 else 3

            if pd.isna(resp):
                inc_match14 = inc_match = inc_match14_strict = np.nan
            elif inc == 1:
                inc_match14 = resp <= 2
                inc_match = inc_match14_strict = resp == 1
            elif inc == 3:
                inc_match14 = resp >= 3
                inc_match = inc_match14_strict = resp == 4
            else:
                inc_match14 = inc_match14_strict = np.nan
                inc_match = resp in [2, 3]

            inc_run = run * 10 + int(inc)
            inc_run_cnt[inc_run] += 1

            d = {'sn': sn,
                 'enc_trial': trial_full,
                 'obj_trial': trial_full,
                 'scn_trial': trial_full,
                 'cmb_trial': trial_full,
                 'enc_run': run,
                 'obj_run': run,
                 'scn_run': run,

                 'obj7_fMRI': fp_obj7,
                 'obj8_fMRI': fp_obj8,

                 'scn7_fMRI': fp_scn7,
                 'scn8_fMRI': fp_scn8,

                 'obj': obj,
                 'scene': scene,
                 'obj_rename': obj_rename,
                 'scene_rename': scene_rename,
                 'inc': inc,
                 'inc_run': inc_run,
                 'inc_run_cnt': inc_run_cnt[inc_run] - 1,
                 'inc_str': inc_str,
                 'inc_rt': rt,
                 'per_inc_bool': per_inc_bool,
                 'per_inc': resp, # higher (up to 4) = seen as congruent
                 'per_inc_str': per_inc_str,
                 'per_inc14': per_inc14,
                 'per_inc14_str': per_inc14_str,
                 'inc_match': inc_match,
                 'inc_match14': inc_match14,
                 'inc_match14_strict': inc_match14_strict,
                 }
            df_sn_as_l.append(d)
    df_sn = pd.DataFrame(df_sn_as_l)
    if sn == '132': # run 3 is corrupted?
        df_sn.loc[76:, 'obj7_fMRI'] = np.nan
        df_sn.loc[76:, 'scn7_fMRI'] = np.nan

    matches1234 = df_sn['inc_match'].astype(np.float64).sum() # Gives wrong number for 102 if i dont astype??
    matches14 = df_sn['inc_match14'].astype(np.float64).sum()
    matches14_strict = df_sn['inc_match14_strict'].astype(np.float64).sum()
    # print(f'{matches1234=}, {matches14=}, {matches14_strict=}')

    df_sn = include_BL(df_sn, sn)

    if ret:
        df_sn = include_conceptual(df_sn, sn)
        df_sn = include_vis(df_sn, sn, incl_lures=incl_lures)
    try:
        df_sn['hit_hit'] = df_sn['con_hit'] & df_sn['vis_hit']
        def f(row):
            if pd.isna(row['con_hit']) or pd.isna(row['vis_hit']):
                return np.nan
            else:
                return row['con_hit'] & row['vis_hit']
        df_sn['hit_hit_nan'] = df_sn.apply(f, axis=1)
    except KeyError as e:
        pass

    df_sn['hit_hit'] = df_sn['con_hit'] & df_sn['vis_hit']
    def f(row):
        if pd.isna(row['con_hit']) or pd.isna(row['vis_hit']):
            return np.nan
        else:
            return row['con_hit'] & row['vis_hit']
    df_sn['hit_hit_nan'] = df_sn.apply(f, axis=1)


    df_sn['inc_hit_hit'] = df_sn.apply(
        lambda row: np.nan if pd.isna(row['hit_hit']) else
        f'{row["inc"]}{int(row["hit_hit"])}', axis=1)
    df_sn['inc_vis_hit'] = df_sn.apply(
        lambda row: np.nan if pd.isna(row['vis_hit']) else
        f'{row["inc"]}{int(row["vis_hit"])}', axis=1)
    df_sn['inc_con_hit'] = df_sn.apply(
        lambda row: np.nan if pd.isna(row['con_hit']) else
        f'{row["inc"]}{int(row["con_hit"])}', axis=1)
    # df_sn = prep_dif(df_sn, sn)
    df_sn['true'] = True

    df_sn['i_nc'] = df_sn['inc'].apply(lambda x: 'i' if x == 1 else 'nc')
    df_sn['in_c'] = df_sn['inc'].apply(lambda x: 'c' if x == 3 else 'in')

    add_fns(df_sn, sn)
    df_sn['sn'] = sn

    for key in ['obj', 'scn', 'con', 'vis']:
        num_nans = df_sn[f'{key}_trial'].isna().sum()
        assert num_nans == 0, f'Bad _trial {num_nans=}, {key=}, {sn=}'

    return df_sn

def add_fns(df_sn, sn):
    trial_info = r'behavFiles/trial_info_all.csv'
    df_info = pd.read_csv(trial_info)
    df_info = df_info[df_info['Subject'] == int(sn)]
    if len(df_info) == 114: # missing retrieval data
        pass
    else:
        match_con = df_info['ObjectTypeLabel_RCON'] == 'Old'
        match_vis = df_info['ObjectTypeLabel_RVIS'].isin(['Old', 'Similar'])
        df_info = df_info[match_con | match_vis]


    obj_l = df_info['Object'].values
    fns = df_info['ObjectFile'].values
    fns_scn = df_info['SceneFile'].values
    obj2fns = dict(zip(obj_l, fns))

    obj2fns_scn = dict(zip(obj_l, fns_scn))
    df_sn['obj_fn'] = df_sn['obj'].map(obj2fns)
    df_sn['scn_fn'] = df_sn['obj'].map(obj2fns_scn)
    df_sn['obj_cat'] = df_sn['obj_fn'].map(STIM_CATEGORY)
    df_sn['scn_cat'] = df_sn['scn_fn'].map(SCENE_CATEGORY)
    try:
        df_sn['living'] = df_sn['obj_cat'].apply(lambda x: 'living' in x)
    except TypeError as e:
        print(f'Failied to do living/non-living: {sn}: {e=}')


def add_onset_time(df_sn, sn):
    trial_info = r'behavFiles/trial_info_all.csv'
    df_info = pd.read_csv(trial_info)
    df_info = df_info[df_info['ObjectTypeLabel_RCON'] == 'Old']
    df_info_sn = df_info[df_info['Subject'] == int(sn)]
    obj_l = df_info_sn['Object'].values
    for name, stim, key in [('BL', 'Obj', 'bl'),
                            ('ENC', 'Obj', 'obj'),
                            ('ENC', 'Scene', 'scn'),
                            ('RCON', 'Obj', 'con'),
                            ('RVIS', 'Obj', 'vis')]:
        onset_col = f'Onset{stim}_{name}'
        onsets = df_info_sn[onset_col].values
        obj2onsets = dict(zip(obj_l, onsets))
        df_sn[f'{key}_onset'] = df_sn['obj'].map(obj2onsets)
        df_sn[f'{key}_onset_TR'] = 4 + df_sn[f'{key}_onset'] // 2# + 1
        try:
            df_sn[f'{key}_onset_TR'] = df_sn[f'{key}_onset_TR'].astype(int)
        except pd.errors.IntCastingNaNError:
            assert sn == '133' or sn == '138'
            # TODO: investigate 212...


def include_conceptual(df_sn, sn):
    conc_root = fr'fMRI_in/{sn}'
    obj2resp = {}
    obj2run = {}
    obj2old_new = {}
    obj2rt = {}
    obj2fp = {}
    obj2fp2 = {}
    obj2fp3 = {}
    obj2fp7 = {}
    obj2fp8 = {}
    obj2trial = {}
    obj_l = []
    for run in range(1, 4):
        # if sn == '231': # TODO: use their data if I can
        #     continue
        fp_bhv = fr'behavFiles/RET_con/S{sn}_run{run}_RC.mat'
        try:
            mat_enc = io.loadmat(fp_bhv)
        except FileNotFoundError:
            print(f'Missing CON behavioral data: {sn}, {run=}')
            continue

        for i in range(48):
            # Warning 231 may be bad.
            # 212 BL / 231 RCON -- the order of objects in pdata
            # ;(psychtoolbox output) doesn't match the stimtbl (which
            # supposedly determines the object presentation
            # order beforehand). I've gone ahead and make single-trial model
            # based on the psychtoolbox output, but maybe proceed with caution
            # with these two ppl
            if sn == '231' and run == 1: continue
            trial = i + 1
            obj = mat_enc['pdata'][0][0][6][0][i][0]
            obj_l.append(obj)
            obj2run[obj] = run
            obj2trial[obj] = trial + (run - 1) * 48
            old_new = mat_enc['pdata'][0][0][8][0][i][0][0]
            assert old_new in [0, 1], f'Old new not 0 or 1: {old_new=}'
            obj2old_new[obj] ='new' if old_new else 'old'
            resp = mat_enc['pdata'][0][0][9][0][i][0]
            obj2resp[obj] = None if pd.isna(resp) else int(resp)
            obj2rt[obj] = mat_enc['pdata'][0][0][10][0][i][0]

            glob_conc7 = fr'{conc_root}/CON_rerun7/CONC/RCON_sub{sn}_run{run}_trial{trial}_*.nii'
            glob_conc7 = glob(glob_conc7)
            if len(glob_conc7):
                assert len(glob_conc7) == 1
                obj2fp7[obj] = glob_conc7[0]
            else:
                if len(glob_conc7) < 1:
                    obj2fp7[obj] = None

            glob_conc8 = fr'{conc_root}/Con_NoGSR_8/CONC/RCON_sub{sn}_run{run}_trial{trial}_*.nii'
            glob_conc8 = glob(glob_conc8)
            if len(glob_conc8):
                assert len(glob_conc8) == 1
                obj2fp8[obj] = glob_conc8[0]
            else:
                if len(glob_conc8) < 1:
                    obj2fp8[obj] = None
    else:
        if df_sn is None:
            df_sn = pd.DataFrame({'obj': obj_l})
            df_sn['old_new'] = df_sn['obj'].map(obj2old_new)
        # else:
        # print(len(obj2resp))
        # quit()
        df_sn['con_resp'] = df_sn['obj'].map(obj2resp)
        df_sn['con_hit'] = df_sn['con_resp'].apply(
            lambda x: np.nan if pd.isna(x) else x >= 3)
        df_sn['con7_fMRI'] = df_sn['obj'].map(obj2fp7)
        df_sn['con8_fMRI'] = df_sn['obj'].map(obj2fp8)

        obj2trial, obj2run = fill_obj2trial_obj2run(obj2trial, obj2run)

        df_sn['con_run'] = df_sn['obj'].map(obj2run)
        df_sn['con_trial'] = df_sn['obj'].map(obj2trial)
    return df_sn

def fill_obj2trial_obj2run(obj2trial, obj2run):

    contained_runs = set(obj2run.values())
    missing_runs = set(range(1, 4)) - contained_runs
    assert len(missing_runs) <= 1, f'{missing_runs=}'
    obj2run_ = defaultdict(lambda: list(missing_runs)[0])
    obj2run_.update(obj2run)
    obj2run = obj2run_
    obj2trial_ = defaultdict(lambda: list(missing_runs)[0] * 48) #cover 38 or 48
    obj2trial_.update(obj2trial)
    obj2trial = obj2trial_
    return obj2trial, obj2run

def include_vis(df_sn, sn, incl_lures=False):
    # TODO: investigate why 138 is missing run3 visual retrieval
    # Figure out the trial breakdown
    vis_root = fr'fMRI_in/{sn}'
    obj2resp = {}
    obj2type = {} # unused
    obj2fp = {}
    obj2fp2 = {}
    obj2fp3 = {}
    obj2fp7 = {}
    obj2fp8 = {}
    obj2rt = {}
    obj2run = {}
    obj2trial = {}
    obj_l = []
    for run in range(1, 4):
        fp_bhv = fr'behavFiles/RET_vis/S{sn}_run{run}_RV.mat'
        try:
            mat_enc = io.loadmat(fp_bhv)
        except FileNotFoundError:
            print(f'Missing VIS behavioral data: {sn}, {run=}')
            continue
        if sn == '213' and run == 2: # not recorded
            continue
        for i in range(42):
            trial = i + 1
            obj = mat_enc['pdata'][0][0][6][0][i][0]
            obj_l.append(obj)
            obj2run[obj] = run
            obj2trial[obj] = trial + 42 * (run - 1)
            resp = mat_enc['pdata'][0][0][8][0][i][0]
            resp = 'old' if resp == 3 else \
                   'similar' if resp == 2 else \
                   'new' if resp == 1 else np.nan
            obj2resp[obj] = resp
            obj2rt[obj] = mat_enc['pdata'][0][0][9][0][i][0]
            old_similar_new = mat_enc['pdata'][0][0][11][0][i][0][0]
            # TODO: For visual retrieval, confirm that 0 = old, 1 = similar, 2 = new
            old_similar_new = 'old' if old_similar_new == 0 else \
                'similar' if old_similar_new == 1 else 'new'
            obj2type[obj] = old_similar_new

            if sn == '138' and run == 3: # has original preprocessing but no rerun betas?
                continue

            glob_vis7 = fr'{vis_root}/VIS_rerun7/VIS/RVIS_sub{sn}_run{run + 3}_trial{trial}_*.nii'
            glob_vis7 = glob(glob_vis7)
            if len(glob_vis7):
                obj2fp7[obj] = glob_vis7[0]
            else:
                obj2fp7[obj] = None

            glob_vis8 = fr'{vis_root}/Vis_NoGSR_8/VIS/RVIS_sub{sn}_run{run + 3}_trial{trial}_*.nii'
            glob_vis8 = glob(glob_vis8)
            if len(glob_vis8):
                obj2fp8[obj] = glob_vis8[0]
            else:
                obj2fp8[obj] = None
    else:
        if df_sn is None:
            df_sn = pd.DataFrame({'obj': obj_l})

        df_sn['vis_resp'] = df_sn['obj'].map(obj2resp)
        df_sn['vis_type'] = df_sn['obj'].map(obj2type)
        f = lambda row: np.nan if pd.isna(row['vis_resp']) else \
            row['vis_resp'] == row['vis_type']
        df_sn['vis_hit'] = df_sn.apply(f, axis=1)
        df_sn['vis_fMRI'] = df_sn['obj'].map(obj2fp)
        df_sn['vis2_fMRI'] = df_sn['obj'].map(obj2fp2)
        df_sn['vis3_fMRI'] = df_sn['obj'].map(obj2fp3)
        df_sn['vis7_fMRI'] = df_sn['obj'].map(obj2fp7)
        df_sn['vis8_fMRI'] = df_sn['obj'].map(obj2fp8)
        obj2trial, obj2run = fill_obj2trial_obj2run(obj2trial, obj2run)
        df_sn['vis_run'] = df_sn['obj'].map(obj2run)
        df_sn['vis_trial'] = df_sn['obj'].map(obj2trial)

        if incl_lures:
            l_lures = []
            for obj in obj2resp:
                if obj2type[obj] == 'new':
                    d = {'obj': obj,
                         'vis_resp': obj2resp[obj],
                         'vis8_fMRI': obj2fp8[obj],
                         'vis_type': obj2type[obj],
                         'vis_hit': np.nan if pd.isna(obj2resp[obj]) else
                                    obj2resp[obj] == obj2type[obj],
                         'vis_run': obj2run[obj],
                         'vis_trial': obj2trial[obj],
                         }
                    l_lures.append(d)
            df_lures = pd.DataFrame(l_lures)
            df_sn = pd.concat([df_sn, df_lures], ignore_index=True)

    return df_sn

def do_BL_move(bl_root, run, trial):
    glob_BL_pre = fr'{bl_root}/Day1_Run{run}_Trial{trial}_*.nii'
    glob_BL_pre = glob(glob_BL_pre)
    if len(glob_BL_pre) > 1:
        print(f'Bad more than one pre: {glob_BL_pre=}')
        quit()
    elif len(glob_BL_pre) == 1:
        import shutil
        fp_BL_pre = Path(glob_BL_pre[0])
        dir_BL_post = fp_BL_pre.parent.joinpath('all_BLruns_sorted')
        dir_BL_post.mkdir(exist_ok=True)
        fp_BL_post = dir_BL_post.joinpath(fp_BL_pre.name)
        shutil.move(fp_BL_pre, fp_BL_post)

def include_BL(df_sn, sn):
    bl_root = fr'fMRI_in/{sn}'
    obj2resp = {}
    obj2run = {}
    obj2fp = {}
    obj2fp2 = {}
    obj2fp3 = {}
    obj2fp7 = {}
    obj2fp8 = {}
    obj2trial = {}
    for run in range(1, 4):
        fp_bhv = fr'behavFiles/bl/S{sn}_run{run}.mat'
        try:
            mat_enc = io.loadmat(fp_bhv)
        except FileNotFoundError:
            print(f'Missing BL behavioral data: {sn}, {run=}')
            continue

        for i in range(38):
            trial = i + 1
            obj = mat_enc['pdata'][0][0][6][0][i][0]

            obj2run[obj] = run
            obj2trial[obj] = trial + 38 * (run - 1)
            do_BL_move(bl_root, run, trial)

            # print(f'BL: {obj}')

            glob_BL7 = fr'{bl_root}/BL_rerun7/BL/BL_sub{sn}_run{run}_trial{trial}_*.nii'
            glob_BL7 = glob(glob_BL7)
            if len(glob_BL7):
                obj2fp7[obj] = glob_BL7[0]
            else:
                obj2fp7[obj] = None

            glob_BL8 = fr'{bl_root}/Bl_NoGSR_8/BL/BL_sub{sn}_run{run}_trial{trial}_*.nii'
            glob_BL8 = glob(glob_BL8)
            if len(glob_BL8):
                obj2fp8[obj] = glob_BL8[0]
            else:
                obj2fp8[obj] = None

    else:
        df_sn['bl_resp'] = df_sn['obj'].map(obj2resp)
        df_sn['bl_fMRI'] = df_sn['obj'].map(obj2fp)
        df_sn['bl2_fMRI'] = df_sn['obj'].map(obj2fp2)
        df_sn['bl3_fMRI'] = df_sn['obj'].map(obj2fp3)
        df_sn['bl7_fMRI'] = df_sn['obj'].map(obj2fp7)
        df_sn['bl8_fMRI'] = df_sn['obj'].map(obj2fp8)
        df_sn['bl_run'] = df_sn['obj'].map(obj2run)
        df_sn['bl_trial'] = df_sn['obj'].map(obj2trial)
    df_sn['bl_resp'] = df_sn['bl_resp'].apply(
        lambda x: x[0] if isinstance(x, np.ndarray) else x)

    return df_sn


def prep_dif(df, sn):
    t = time()
    dif_root = Path(fr'fMRI_in\{sn}')
    key_pairs = [('bl', 'obj'), ('bl', 'vis'), ('obj', 'vis')]
    for key0, key1 in key_pairs:
        fp_key0 = f'{key0}_fMRI'
        fp_key1 = f'{key1}_fMRI'
        dif_fps = []
        for trial, run, fp0, fp1 in zip(df['enc_trial'], df['run'],
                                        df[fp_key0], df[fp_key1]):
            if pd.isna(fp0) or pd.isna(fp1):
                dif_fps.append(None)
                continue
            dir_out = dif_root.joinpath(fr'dif_{key0}_{key1}')
            dir_out.mkdir(exist_ok=True)
            # fp_out = dir_out.joinpath(f'EncTrial{trial}.nii')
            fp_out = dir_out.joinpath(f'Trial{trial}_Run{run}.nii')
            dif_fps.append(str(fp_out))
            if fp_out.is_file():
                continue

            fp_out = str(fp_out) # nilearn errors with pathlib Paths
            img0 = image.load_img(fp0)
            img1 = image.load_img(fp1)
            dif_img = image.math_img('img0 - img1', img0=img0, img1=img1)
            dif_img.to_filename(str(fp_out))
        df[f'dif_{key0}-{key1}'] = dif_fps
    print(f'Prepped difs for {sn} in {time() - t:.2f}s')
    return df

if __name__ == '__main__':
    # df_sn = include_conceptual(None, '239')
    # print(df_sn)
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    df_sn = get_trial_info_('102')
    df_sn.sort_values(by='obj_trial', inplace=True)
    print(df_sn[['obj_trial', 'bl_trial', 'obj']])

    # df_sn = get_trial_info_('132')
    # df_sn.sort_values(by='obj_trial', inplace=True)
    # print(df_sn[['obj7_fMRI', 'obj_trial']])
