import pickle
import numpy as np

from analyze_ROIs import prune_bad_sns
from atlas_utils import get_atlas
from modularity_testing import get_modules, get_partition_matrix, plot_nichord
from single_trial_conn import corr_matrix_last_two_dim
from utils import get_RSA_fn, stdize
import matplotlib.pyplot as plt

def load_data(fp_fMRI_col = 'obj_fMRI', age=1, semantic=False, inc=None,
              bilateral=False, combine_regions=False, vec_prod=False,
              org_by_region=False):
    fn = get_RSA_fn(inc=inc, age=age, semantic=semantic, DNN_layer=2,
                    fp_fMRI_col=fp_fMRI_col, bilateral=bilateral,
                    combine_regions=combine_regions, vec_prod=vec_prod,
                    org_by_region=org_by_region)
    fp = fr'cache/RSA/{fn}.pkl'
    with open(fp, 'rb') as file:
        d = pickle.load(file)
    return d

def get_partitions(d_YA, d_OA, atlas, dim='inc', vals=(1, 2, 3)):
    M_corr = []
    age2corrs = {}
    for age, d in zip(['YA', 'OA'], [d_YA, d_OA]):
        l_M_corr_inc = []
        corrs_age = []
        # for inc in range(1, 4):
        for inc in vals:
            inc_ar = d['bhv'][dim]
            # inc_ar = d['bhv']['inc']
            inc_mask = inc_ar == inc
            act_ar = np.array(list(d['activity'].values()))
            act_ar_inc = []
            for sn in range(act_ar.shape[1]):
                act_sn_ar = act_ar[:, sn, inc_mask[sn, :]]
                act_ar_inc.append(act_sn_ar)
            try:
                act_ar_inc = np.array(act_ar_inc)
                corrs_inc, _ = corr_matrix_last_two_dim(act_ar_inc, nans=True)
            except ValueError:
                # unequal size for each sn
                corrs_inc = []
                for acr_sn_ar in act_ar_inc:
                    corrs_sn, _ = corr_matrix_last_two_dim(acr_sn_ar, nans=True)
                    corrs_inc.append(corrs_sn)
            M_corr_inc = np.nanmean(corrs_inc, axis=0)
            l_M_corr_inc.append(M_corr_inc)
            corrs_age.append(corrs_inc)
        M_corr_age = np.mean(l_M_corr_inc, axis=0)
        M_corr.append(M_corr_age)
        corrs_age = np.array(corrs_age)
        good_sns = []
        for sn in range(corrs_age.shape[1]):
            all_not_nans = True
            for cond in range(corrs_age.shape[0]):
                has_not_nans = np.any(~np.isnan(corrs_age[cond, sn, :, :]))
                all_not_nans = all_not_nans and has_not_nans
            if all_not_nans:
                good_sns.append(sn)
        corrs_age = corrs_age[:, good_sns, :, :]
        age2corrs[age] = corrs_age

    M_corr = np.mean(M_corr, axis=0)
    # plt.imshow(M_corr)
    # plt.colorbar()
    # plt.show()
    # quit()
    partitions = get_modules(M_corr)

    for i, p in enumerate(partitions):
        if len(p) < 5:
            continue
        continue
        # print(f'Partition {i}: {p}')
        # print(atlas['coords'])
        M_corr_part = get_partition_matrix(M_corr, p, w_zeros=True)
        print(f'{M_corr_part.shape=}')
        fn = f'newer_p{i}_M_corr.png'
        title = f'Partition {i}'
        plot_nichord(M_corr_part, atlas['coords'], fn, title)

    return partitions, age2corrs

def calculate_within_between(age2corrs, p):
    for age in ['YA', 'OA']:
        print(f'\t{age}')
        p_mat = get_partition_matrix(age2corrs[age], p)
        p_M_within_connectivity = np.nanmean(p_mat, axis=(-2, -1))
        between_mask = np.full(age2corrs[age].shape[-2:], False)
        between_mask[p, :] = True
        between_mask[:, p] = True
        between_mask[np.ix_(p, p)] = False
        p_between_edges = age2corrs[age][:, :, between_mask]
        p_M_between_connectivity = np.nanmean(p_between_edges, axis=-1)
        # print(p_M_within_connectivity)
        # print(p_M_between_connectivity)
        integration = p_M_between_connectivity / p_M_within_connectivity
        integration = p_M_between_connectivity
        M_by_inc = np.mean(integration, axis=1)
        SD_by_inc = np.std(integration, axis=1)
        SE_by_inc = SD_by_inc / np.sqrt(integration.shape[1])
        for i in range(M_by_inc.shape[0]):
            print(f'Inc {i+1}: {M_by_inc[i]:.2f} +/- {SE_by_inc[i]:.2f}')


def do():
    atlas = get_atlas(combine_regions=False, bilateral=False)
    d_YA = load_data(age=1)
    d_YA = prune_bad_sns(d_YA, drop_ret=False)
    d_OA = load_data(age=2)
    d_OA = prune_bad_sns(d_OA, drop_ret=False)

    partitions, age2corrs = get_partitions(d_YA, d_OA, atlas)
    # partitions, age2corrs = get_partitions(d_YA, d_OA, atlas, dim='con_hit',
    #                                        vals=(False, True))

    for i in range(5):
        print(f'Partition: {i}')
        p_MTL = partitions[i]
        calculate_within_between(age2corrs, p_MTL)


    for p in partitions:
        if len(p) < 10:
            continue
        p_corrs = get_partition_matrix(age2corrs, p)




    # print(part)
    # print(M_corr_age)
    # plt.imshow(M_corr_age)
    # plt.show()

def FC_RSA():
    pass



if __name__ == '__main__':
    FC_RSA()
    # do()

