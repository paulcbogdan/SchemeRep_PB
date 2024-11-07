from Utils.atlas_funcs import get_atlas, get_BNA_ROIs
from connRSA.conn_regress import run_all_sn, get_title
from old.plot_gen import my_plot_surf
from Utils.pickle_wrap_funcs import pickle_wrap


def plot_basic():
    easy_override = False
    ISPC = False
    RSA = True
    semantic = False
    ERS_alt = False
    conn = 'prod'
    trial_similarity = 'corr' # euc
    second_order = 'spear'
    RDM_method = 'within_nan'
    four_tasks = '7'
    regress_row = False
    stdize_by_run = True if trial_similarity == 'euc' else False

    ROI_foci = [f'{ROI}_BOLD' for ROI in get_BNA_ROIs()]

    ts_ROI, ts_BOLD, ts_conn = [], [], []
    kwargs_ = None
    kwargs = {'semantic': semantic, 'fp': None, 'fp0': None,
              'fp1': None, 'trial_similarity': trial_similarity,
              'second_order': second_order,
              'RDM_method': RDM_method,
              'stdize_by_run': stdize_by_run,
              'regress_row': regress_row, 'four_tasks': four_tasks,
              }
    for ROI_focus in ROI_foci:
        kwargs_ = kwargs.copy()
        kwargs_['ROI_focus'] = ROI_focus
        kwargs_['ROIs_ctrl'] = []
        kw_outer = {'RSA': RSA, 'ISPC': ISPC, 'ERS_alt': ERS_alt,
                    'kwargs': kwargs_}
        t_ROIs_all, r_sqs_ROIs_all = pickle_wrap(run_all_sn, kwargs=kw_outer,
                                                 easy_override=easy_override)
        # t_ROIs_all, r_sqs_ROIs_all = run_all_sn(kwargs_, RSA, ISPC, ERS_alt,
        #                                         easy_override=easy_override,
        #                                         )
        ts_ROI.append(t_ROIs_all)

    title, fn, fontsize = get_title(RSA, ISPC, kwargs_, 28)
    print(f'{ts_ROI=}')
    vmax = 5
    thresh = 2
    title_ROI = title.split(':')[0]# + ': ROIs'
    atlas = get_atlas()
    my_plot_surf(ts_ROI, atlas, title_ROI, vmax=vmax, thresh=thresh)




