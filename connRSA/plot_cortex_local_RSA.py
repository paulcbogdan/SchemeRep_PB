from atlas_utils import get_atlas
from connRSA.conn_analyze_IRAFs import ROI2NETWORK
from connRSA.conn_regress import plot_stacked_bars, prep_ROI_avg, run_all_sn
from connRSA.conn_utils import get_BNA_ROIs
from old.networks import prep_networks
from old.plot_gen import my_plot_surf
from utils import pickle_wrap


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

            t_ROIs_all, r_sqs_ROIs_all = run_all_sn(kwargs_, RSA,
                                                    easy_override=False,
                                                    )
            ts_ROI.append(t_ROIs_all)

    if len(target_ROIs) >= 10:
        if len(target_ROIs) > 40:
            cbl = False
        else:
            cbl = True
        atlas = get_atlas(combine_regions=False, combine_bilateral=cbl)

        vmax = 5
        thresh = 2.5

        title_ROI = 'Semantic' if semantic else 'Perceptual'
        fp_out = fr'result_pics\ROI_RSA\cortex_RSA_{title_ROI}.png'
        my_plot_surf(ts_ROI, atlas, title_ROI, vmax=vmax, thresh=thresh,
                     fp_out=fp_out, only_positive=False)


if __name__ == '__main__':
    plot_cortex_RSA()



