from pathlib import Path

import pandas as pd

from atlas_utils import get_atlas
from ttest_mat import get_stats_graphs
from old.networks import load_FC_for_Lifu
from utils import pickle_wrap, run_two_sample_on_2D
import numpy as np

def out_csv(mat, atlas, fp, regionwise=False):
    df_mat = pd.DataFrame(mat, index=atlas['ROIs'], columns=atlas['ROIs'])
    print(f'{len(atlas["ROIs"])=}')
    df_mat.to_csv(fp)



if __name__ == '__main__':
    fp2name = {'obj4_fMRI': 'object_betas',
               'scn4_fMRI': 'scene_betas'}
    combine_regions = [True, False]
    for cr in combine_regions:
        for fp in ['obj4_fMRI']: # , 'scn4_fMRI'
            kwargs = {'fp': fp,
                      'split': False,
                      'key': 'inc',
                      'key_vals': (1, 3),
                      }
            if cr:
                kwargs['combine_regions'] = True
            sn_inc_conn, sn_conn, age2idxs, sn_inc_activity = \
                pickle_wrap(None, load_FC_for_Lifu, kwargs=kwargs, verbose=1,
                            easy_override=False, cache_dir='../cache')

            M_YA_graph = np.nanmean(sn_inc_conn[age2idxs[1], :, :, :], axis=(0, 1))
            M_OA_graph = np.nanmean(sn_inc_conn[age2idxs[2], :, :, :], axis=(0, 1))

            M1_graph, SD1_graph, SE1_graph, N1_graph, t1_graph, p1_graph, z1_graph = \
                get_stats_graphs(sn_inc_conn[age2idxs[1], 0, :, :],
                                 sn_inc_conn[age2idxs[1], 1, :, :])
            M2_graph, SD2_graph, SE2_graph, N2_graph, t2_graph, p2_graph, z2_graph = \
                get_stats_graphs(sn_inc_conn[age2idxs[2], 0, :, :],
                                 sn_inc_conn[age2idxs[2], 1, :, :])
            atlas = get_atlas(schaefer=False, combine_regions=cr)
            beta_dir_name = fp2name[fp]
            if cr:
                out_dir = f'simon_csvs_out/regions_{beta_dir_name}'
            else:
                out_dir = f'simon_csvs_out/ROIs_{beta_dir_name}'

            t1_graph = -t1_graph
            t2_graph = -t2_graph

            Path(out_dir).mkdir(exist_ok=True, parents=True)
            fp_M1 = f'{out_dir}/YA_mean_both_cond.csv'
            out_csv(M_YA_graph, atlas, fp_M1)
            fp_t1 = f'{out_dir}/YA_t-test_con-inc.csv'
            out_csv(t1_graph, atlas, fp_t1)
            fp_M2 = f'{out_dir}/OA_mean_both_cond.csv'
            out_csv(M_OA_graph, atlas, fp_M2)
            fp_t2 = f'{out_dir}/OA_t-test_con-inc.csv'
            out_csv(t2_graph, atlas, fp_t2)

            M12_graph, SD12_graph, SE12_graph, N12_graph, t12_graph, p12_graph, z12_graph = \
                get_stats_graphs(sn_inc_conn[:, 0, :, :],
                                 sn_inc_conn[:, 1, :, :])
            t12_graph = -t12_graph

            fp_t12 = f'{out_dir}/YA_and_OA_t-test_con-inc.csv'
            out_csv(t12_graph, atlas, fp_t12)

            mYA_graph = np.nanmean(sn_inc_conn[age2idxs[1], :, :, :], axis=1)
            mOA_graph = np.nanmean(sn_inc_conn[age2idxs[2], :, :, :], axis=1)
            age_t_graph = run_two_sample_on_2D(mYA_graph, mOA_graph)
            fp_age = f'{out_dir}/age_effect_t-test_both_cond.csv'
            out_csv(t2_graph, atlas, fp_age)

            age_itr = run_two_sample_on_2D(-M1_graph, -M2_graph)
            fp_itr = f'{out_dir}/age_x_congruency_t-vals.csv'
            out_csv(t2_graph, atlas, fp_itr)