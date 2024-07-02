from atlas_utils import get_atlas
from connRSA.conn_analyze_IRAFs import ROI2NETWORK
from connRSA.conn_regress import plot_stacked_bars, prep_ROI_avg, run_all_sn
from connRSA.conn_utils import get_BNA_ROIs
from old.networks import prep_networks
from old.plot_gen import my_plot_surf
from utils import pickle_wrap


def plot_cortex_RSA():
    easy_override = False
    ctrl_strict = False

    ISPC = False
    RSA = True
    semantic = True
    ERS_alt = False
    trial_similarity = 'corr' # euc
    second_order = 'spear'
    RDM_method = 'within_nan'
    four_tasks = '7'
    stdize_by_run = True if trial_similarity == 'euc' else False

    # target_ROIs = ['Occipital', 'Ventral', 'Dorsal', 'PFC', 'subcort']
    # target_ROIs = ['Dorsal', 'pSTS', 'Parietal']
    # target_ROIs = ['Parietal']
    # , 'IT', 'Ventral']

    # target_ROIs = ['Occipital', 'ITL', 'Parietal', 'PFC', 'subcort']
    target_ROIs = get_BNA_ROIs()
    target_ROIs = [f'{ROI}_BOLD' for ROI in target_ROIs]

    ts_ROI, ts_BOLD, ts_conn = [], [], []
    for regress_row in [False]:
        for target_ROI in target_ROIs:
            kwargs = {'semantic': semantic, 'fp': None, 'fp0': None,
                      'fp1': None, 'trial_similarity': trial_similarity,
                      'second_order': second_order,
                      'RDM_method': RDM_method,
                      'stdize_by_run': stdize_by_run,
                      'regress_row': regress_row, 'four_tasks': four_tasks,
                      }

            kwargs_ = kwargs.copy()
            kwargs_['ROI_focus'] = target_ROI
            kwargs_['ROIs_ctrl'] = []
            t_ROIs_all, r_sqs_ROIs_all = run_all_sn(kwargs_, RSA, ISPC, ERS_alt,
                                                    easy_override=False,
                                                    )
            # print(t_ROIs_all)
            ts_ROI.append(t_ROIs_all)

    if len(target_ROIs) >= 10:
        if len(target_ROIs) > 40:
            cbl = False
        else:
            cbl = True
        atlas = get_atlas(combine_regions=False, combine_bilateral=cbl)

        vmax = 4
        thresh = 1.65

        title_ROI = ''
        my_plot_surf(ts_ROI, atlas, title_ROI, vmax=vmax, thresh=thresh)


if __name__ == '__main__':
    plot_cortex_RSA()



