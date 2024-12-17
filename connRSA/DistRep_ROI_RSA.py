import os
from datetime import datetime

from connRSA.single_trial_conn import run_settings

os.chdir(r'C:\PycharmProjects\SchemeRep')

# from single_trial_conn import run_settings
from connRSA.old_Oct29.conn_report import report_results
from Utils.pickle_wrap_funcs import pickle_wrap

ROI2NETWORK = {'Occipital': 1, 'Ventral': 1, 'Dorsal': 1, 'else_cortical': 2,
               'PFC': 16, 'PFC_ACC': 14, #'FP': 14,
               'perceptual': 17,
               'full_frontal': 11, 'full_frontal_CG': 11, 'MTL': 9, 'MTL2': 20,
               'subcort': 21 , 'INScc': 22, 'cingulate': 22, 'cortical': 23,
               'IT': 24, 'ITL': 25, 'Parietal': 26, 'OC_IT': 27, 'OC_T': 27,
               'cortex': 28, 'FP': 29, 'DMN': 29, 'FPT': 29,
               'Temporal': 30,
               'PL': 31, 'LPFC': 31,
               'tha_str': 32,
               'PFC_no_OFC': 33}
ROI2network_anat = {'SFG': 19, 'MFG': 19, 'IFG': 19, 'OrG': 19, 'PrG': 19,
                    'PCL': 19, 'ATL': 19, 'STG': 19, 'MTG': 19, 'ITG': 19,
                    'FuG': 19, 'PhG': 19, 'pSTS': 19, 'SPL': 19, 'IPL': 19,
                    'Pcun': 19, 'PoG': 19, 'INS': 19, 'PCC': 19, 'ACC': 19,
                    'EVC': 19, 'LOC': 19, 'sOcG': 19, 'Amyg': 19, 'Hipp': 19,
                    'Str': 19, 'Tha': 19}
ROI2NETWORK.update(ROI2network_anat)


def DistRep_ROI_RSA(RSA=True, semantic=False, do_networks=False,
                    conn='euc', trial_similarity='euc',
                    second_order='spear', four_tasks=False,
                    combine_regions=False, split=False, RDM_method='by_run',
                    age=1, plotting=None, atlas='BNA', stdize_by_run=False):
    settings = locals().copy()
    if settings['stdize_by_run'] == False:
        del settings['stdize_by_run']

    dir_results = r'cache/conn_RSA'

    if not RSA:
        settings['RDM_method'] = None
        settings['second_order'] = 'spear'

    dt_max = datetime(2024, 6, 8, 0, 0, 0, 0)
    # dt_max = datetime(2024, 6, 8, 0, 0, 0, 0)

    # if not settings['combine_regions'] and False:
    #
    #     results_conn = pickle_wrap(run_settings, None, kwargs=settings,
    #                                easy_override=False, verbose=1,
    #                                cache_dir=dir_results)
    #     report_results(results_conn, do_lmer=False)

    if ('RDM_method' in settings and (settings['RDM_method'] is not None) and
            'complex_mean' in settings['RDM_method']):
        # settings['RDM_method'] = 'clever_std'
        settings['RDM_method'] = 'within_nan'


    settings['conn'] = 'BOLD'
    # print(f'BOLD ' * 10)
    # print(RSA)
    results_bold_comb = pickle_wrap(run_settings, None,
                                   kwargs=settings, easy_override=False,
                                   verbose=1, cache_dir=dir_results,
                                   dt_max=dt_max)
    report_results(results_bold_comb, do_lmer=False)
    # quit()

    if settings['do_networks'] != 19: # this is covered by combine one below
        print(f'COMBINE BIG ' * 10)
        settings['combine_regions'] = True
        results_bold_cmb_big = pickle_wrap(run_settings, None,
                                           kwargs=settings, easy_override=False,
                                           verbose=1, cache_dir=dir_results,
                                           dt_max=dt_max)
        report_results(results_bold_cmb_big, do_lmer=False)

    if settings['do_networks'] != 19: # this is covered by combine one below
        settings['conn'] = 'BOLD_ctrl'
        print(f'COMBINE BIG BOLD ctrl' * 10)
        settings['combine_regions'] = True
        results_bold_cmb_big = pickle_wrap(run_settings, None,
                                           kwargs=settings, easy_override=False,
                                           verbose=1, cache_dir=dir_results,
                                           dt_max=dt_max)
        report_results(results_bold_cmb_big, do_lmer=False)
    return

    settings['conn'] = 'BOLD'
    # dt_max = datetime(2024, 6, 10, 0, 0, 0, 0)
    print('TOAST')
    settings['do_networks'] = False
    settings['combine_regions'] = False
    results_bold_sep = pickle_wrap(run_settings, None,
                                   kwargs=settings, easy_override=False,
                                   verbose=1, cache_dir=dir_results,
                                   dt_max=dt_max)
    report_results(results_bold_sep, do_lmer=False)
    # print('TEST')
    # quit()
    # quit()
    #
    # print(settings)
    # quit()


    settings['do_networks'] = False
    settings['combine_regions'] = True
    print(f'COMBINE ' * 10)

    results_bold_sep_big = pickle_wrap(run_settings, None,
                                       kwargs=settings, easy_override=False,
                                       verbose=1, cache_dir=dir_results)
    report_results(results_bold_sep_big, do_lmer=False)


    print(f'BOLD SUB MEAN ' * 10)
    settings['conn'] = 'BOLD_ctrl'
    settings['combine_regions'] = True

    results_bold_sep_big_sub_mean = pickle_wrap(run_settings, None,
                                       kwargs=settings, easy_override=True,
                                       verbose=1, cache_dir=dir_results)
    report_results(results_bold_sep_big, do_lmer=False)

    # return (results_conn, results_bold_comb, results_bold_sep_big,
    #         results_bold_sep)


def run_DistRep_ROI_RSA(semantic=True, target_ROI='ITL', RSA=False):
    # 'SFG' and most single regions represent all combined_regions
    # Don't modify the below ones
    # RSA = True
    conn = 'prod'
    trial_similarity = 'corr'
    # trial_similarity = 'euc'
    second_order = 'spear'
    four_tasks = '7'
    combine_regions = False
    split = False
    RDM_method = 'within_nan'
    age = 'healthy'
    # stdize_by_run = False # True if trialc_similarity == 'euc' else False
    stdize_by_run = True if trial_similarity == 'euc' else False

    do_networks = ROI2NETWORK[target_ROI]

    kwargs = {'RSA': RSA, 'semantic': semantic, 'do_networks': do_networks,
              'conn': conn, 'trial_similarity': trial_similarity,
              'second_order': second_order, 'four_tasks': four_tasks,
              'combine_regions': combine_regions, 'split': split,
              'RDM_method': RDM_method, 'age': age,
              'stdize_by_run': stdize_by_run
              }

    DistRep_ROI_RSA(**kwargs)


if __name__ == '__main__':
    targets = ['Occipital', 'IT', 'ITL', 'Parietal', 'PFC', 'OC_IT',
               'SFG']
    # targets = ['OC', 'IT', 'OC_IT']
    targets = ['OC_T', 'Occipital', 'IT', 'OC_IT']
    targets = ['Ventral']
    targets = ['Occipital', 'IT', 'ITL']
    # targets = ['tha_str']
    targets = ['PFC_no_OFC']
    targets = ['cortical']
    for sem_per in [True, False]:
        for target in targets:#[::-1]:
            run_DistRep_ROI_RSA(semantic=sem_per, target_ROI=target,
                                RSA=True)

