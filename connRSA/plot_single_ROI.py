from atlas_utils import get_atlas
from connRSA.conn_report import report_results
from old.plot_gen import my_plot_surf
from utils import pickle_wrap
from single_trial_conn import run_settings

import os
os.chdir(r'E:\PycharmProjects_E\SchemeRep')

def plot_single_ROI():

    RSA = True
    semantic = False
    conn = 'BOLD'
    trial_similarity = 'corr'
    second_order = 'spear'
    four_tasks = '8'
    split = False
    RDM_method = 'within_nan'
    age = 'healthy'
    stdize_by_run = True if trial_similarity == 'euc' else False

    kwargs = {'RSA': RSA, 'semantic': semantic, 'do_networks': False,
              'conn': conn, 'trial_similarity': trial_similarity,
              'second_order': second_order, 'four_tasks': four_tasks,
              'combine_regions': False, 'split': split,
              'RDM_method': RDM_method, 'age': age,
              'stdize_by_run': stdize_by_run
              }
    if not RSA:
        kwargs['RDM_method'] = None

    # kwargs2 = {'RSA': False, 'semantic': False, 'do_networks': False,
    #           'conn': 'BOLD', 'trial_similarity': 'corr',
    #            'second_order': 'spear', 'four_tasks': '7',
    #            'combine_regions': False, 'split': False,
    #            'RDM_method': None, 'age': 'healthy',
    #            'plotting': None, 'atlas': 'BNA', 'stdize_by_run': False}
    #

    #
    # for key in kwargs:
    #     val0 = kwargs[key]
    #     val1 = kwargs2[key]
    #     assert val0 == val1, f'{key}: {val0} vs {val1}'


    dir_results = r'cache/conn_RSA'
    results_bold_sep = pickle_wrap(run_settings, None,
                                   kwargs=kwargs, easy_override=False,
                                   verbose=1, cache_dir=dir_results)
    scores, scores_by_fp = report_results(results_bold_sep, do_lmer=False)

    atlas = get_atlas()
    my_plot_surf(scores, atlas, 'big', vmax=4, thresh=2)


if __name__ == '__main__':
    plot_single_ROI()

