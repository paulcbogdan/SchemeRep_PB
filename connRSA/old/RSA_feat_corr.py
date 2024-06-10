import numpy as np
from scipy import stats

from atlas_utils import get_atlas
from connRSA.old.RSA_feat_var import analyze_var_ROI
from connRSA.single_trial_conn import prep_fps


def scratch():
    cov = [[1, 0.99, 0.1],
           [0.99, 1, 0.1],
           [0.1, 0.1, 1]]
    x = np.random.multivariate_normal([0, 0, 0], cov, size=1000)
    regressors = x[:, :2]
    # print(regressors.mean(axis=1))
    regressors -= regressors.mean(axis=1)[..., None]
    regressors = stats.zscore(regressors, axis=0)
    print(regressors)
    y = x[:, 2]

    ones = np.ones((regressors.shape[0], 1))
    regressors = np.concatenate((ones, regressors), axis=1)

    XTX_inv = np.linalg.inv(np.dot(regressors.T, regressors))
    XTX_invX = np.dot(XTX_inv, regressors.T)
    betas = np.dot(XTX_invX, y.T)
    print(betas)

def analyze_ROI_pair(ROI0, ROI1, sns,
                     four_tasks, trial_similarity, stdize_by_run,
                     semantic, second_order, regress_global=True,
                     shuffle_seed=None):
    fps = prep_fps(four_tasks)

    ar = []
    for fp in fps:
        fp_l = []
        for sn in sns:
            vals0 = analyze_var_ROI(sn, ROI0, fp, trial_similarity,
                                   stdize_by_run, second_order, semantic,
                                   regress_global=regress_global,
                                    shuffle_seed=shuffle_seed)
            vals1 = analyze_var_ROI(sn, ROI1, fp, trial_similarity,
                                      stdize_by_run, second_order, semantic,
                                      regress_global=regress_global,
                                      shuffle_seed=shuffle_seed)
            r, p  = stats.spearmanr(vals0, vals1)
            fp_l.append(r)
        ar.append(fp_l)
    return np.mean(ar)

def analyze_all_ROI():
    ROIs = get_atlas()['ROIs']

    semantic = True
    trial_similarity = 'corr'  # euc
    stdize_by_run = False
    second_order = 'spear'
    four_tasks = '8'

    sns = ['102', '103', '104', '105', '106', '107', '108', '109', '110',
           '111', '112', '113', '114', '115', '117', '118', '119', '120',
           '123', '124', '126', '127', '128', '129', '130', '134', '135',
           '136', '137', '201', '202', '203', '204', '205', '206', '207',
           '208', '209', '210', '211', '212', '214', '216', '217', '218',
           '219', '221', '222', '225', '227', '232', '233', '235']
    # sns = sns[:3]
    # sns = sns[:20]

    for i, ROI0 in enumerate(ROIs):
        for j, ROI1 in enumerate(ROIs):
            # if i < 190 or j < 190: continue
            if 'FuG' not in ROI0: continue
            if 'FuG' not in ROI1: continue
            if i >= j - 1: continue
            score = analyze_ROI_pair(ROI0, ROI1, sns, four_tasks,
                                     trial_similarity, stdize_by_run, semantic,
                                     second_order, regress_global=False)
            print(f'{score=:.3f} {ROI0=}, {ROI1=}')
            l_perm = []
            for k in range(100):
                score = analyze_ROI_pair(ROI0, ROI1, sns, four_tasks,
                                         trial_similarity,
                                         stdize_by_run, semantic, second_order,
                                         regress_global=False,
                                         shuffle_seed=k)
                l_perm.append(score)
                print(f'Perm score: {score:.3f}')


if __name__ == '__main__':
    analyze_all_ROI()

