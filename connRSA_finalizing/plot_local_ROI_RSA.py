import utils
from Utils.atlas_funcs import get_atlas, get_BNA_ROIs
from connRSA.conn_regress import run_all_sn, run_all_sn_
from old.plot_gen import my_plot_surf
import numpy as np
from nilearn import image, plotting
import matplotlib.pyplot as plt

def plot_cortex_RSA(semantic=True, RSA=True, control_network=False):
    trial_similarity = 'corr'
    second_order = 'spear'
    RDM_method = 'within_nan'
    four_tasks = '7'
    stdize_by_run = False

    target_ROIs = get_BNA_ROIs()
    target_ROIs = [f'{ROI}_BOLD' for ROI in target_ROIs]

    region2network = {'SFG': 'PFC', 'MFG': 'PFC', 'IFG': 'PFC', 'OrG': 'PFC',
                      'SPL': 'Parietal', 'IPL': 'Parietal', 'Pcun': 'Parietal',
                      'EVC': 'Occipital', 'LOC': 'Occipital', 'sOcG': 'Occipital',
                      'ITG': 'ITL', 'FuG': 'ITL', 'PhG': 'ITL', 'ATL': 'ITL'}

    ts_ROI, ts_BOLD, ts_conn = [], [], []
    for regress_row in [False]:
        for target_ROI in target_ROIs:
            region = target_ROI.split(' ')[1].split('_')[0]

            kwargs = {'semantic': semantic, 'fp': None, 'fp0': None,
                      'fp1': None, 'trial_similarity': trial_similarity,
                      'second_order': second_order,
                      'RDM_method': RDM_method,
                      'stdize_by_run': stdize_by_run,
                      'regress_row': regress_row, 'four_tasks': four_tasks,
                      }

            kwargs_ = kwargs.copy()
            kwargs_['ROI_focus'] = target_ROI
            if control_network and region in region2network:
                network = region2network[region]
                kwargs_['ROIs_ctrl'] = [f'{network}_BOLD', ]
            else:
                kwargs_['ROIs_ctrl'] = []
            # print(target_ROI)
            # quit()

            # t_ROIs_all, r_sqs_ROIs_all = run_all_sn(kwargs_, RSA,
            #                                         easy_override=False,
            #                                         )

            t_ROIs_all, r_sqs_ROIs_all = (
                utils.pickle_wrap(run_all_sn_, kwargs={'kwargs': kwargs_,
                                                       'RSA': RSA,
                                                       'ISPC': False,
                                                       'ERS_alt': False},))

            # print(t_ROIs_all)
            # quit()
            ts_ROI.append(t_ROIs_all)
            print(f'{target_ROI=}, t = {t_ROIs_all:.3f}')

    if len(target_ROIs) >= 10:
        if len(target_ROIs) > 40:
            cbl = False
        else:
            cbl = True
        atlas = get_atlas(combine_regions=False, combine_bilateral=cbl)

        vmax = 6
        thresh = 2

        title_ROI = 'Semantic' if semantic else 'Perceptual'
        title_ROI = title_ROI if RSA else 'NPS'
        fp_out = fr'result_pics\ROI_RSA\cortex_RSA_{title_ROI}.png'
        print(ts_ROI)
        my_plot_surf(ts_ROI, atlas, title_ROI, vmax=vmax, thresh=thresh,
                     fp_out=fp_out, only_positive=True,
                     cmap='RdPu' if semantic else 'BuPu')


if __name__ == '__main__':
    plot_cortex_RSA()



