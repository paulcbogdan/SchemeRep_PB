from sim_RSA import create_feature_vector
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def make_trial_features(n_trials, n_features_per_trial, n_features):
    ar = []
    for i in range(n_trials):
        onehot_vec = create_feature_vector(n_features, n_features_per_trial)
        ar.append(onehot_vec)
    ar = np.array(ar)
    return ar


def make_feature_voxels(n_voxels, n_features):
    out = np.random.normal(0, 1, (n_voxels, n_features))
    return out


def make_sparse(mat, n_high=10, axis=1):
    mat = mat.copy()
    # mat = mat.T
    for j in range(mat.shape[1]):
        col = mat[:, j]
        sorted_col = np.sort(col)[::-1]
        cutoff = sorted_col[n_high]
        col[col < cutoff] = 0
        col[col > 0] = 1
        mat[:, j] = col
    # mat = mat.T
    return mat


def test_one_dud_region(n_trials, n_features_per_trial, n_features, n_voxels, n_high):
    trial_features = make_trial_features(n_trials, n_features_per_trial, n_features)
    feature_voxels_L = make_feature_voxels(n_voxels, n_features)
    feature_voxels_L_sp = make_sparse(feature_voxels_L, n_high=n_high, axis=1)

    feature_voxels_R = make_feature_voxels(n_voxels, n_features)
    feature_voxels_R_sp = make_sparse(feature_voxels_R, n_high=n_high, axis=1)

    trial_features_noise = make_trial_features(n_trials, n_features_per_trial, n_features)
    rsm_model = np.corrcoef(trial_features)

    vecs_L = np.dot(trial_features, feature_voxels_L_sp.T)
    print(f"{vecs_L.shape=}")

    rsm_L = np.corrcoef(stats.rankdata(vecs_L, axis=1))
    trils = np.tril_indices_from(rsm_L, k=-1)
    rsm_L_ = rsm_L[trils]
    rsm_model_ = rsm_model[trils]
    r, p = stats.spearmanr(rsm_model_, rsm_L_)
    print(f"Left: {r=:.3f}")

    vecs_R = np.dot(trial_features_noise, feature_voxels_R_sp.T)
    rsm_R = np.corrcoef(stats.rankdata(vecs_R, axis=1))
    print(f"{rsm_R.shape=}")
    rsm_R_ = rsm_R[trils]
    r, p = stats.spearmanr(rsm_model_, rsm_R_)
    print(f"Right: {r=:.3f}")

    # vecs_concat = np.dot(trial_features, feature_voxels_concat.T)
    vecs_concat = np.concatenate([vecs_L, vecs_R], axis=1)
    rsm_concat = np.corrcoef(vecs_concat)
    # print(f"{rsm_concat.shape=}")
    rsm_concat_ = rsm_concat[trils]
    r, p = stats.spearmanr(rsm_model_, rsm_concat_)
    print(f"Distributed (concat): {r=:.4f}")

    # vecs_LR = np.dot(trial_features, feature_voxels_LR_sp.T)
    # rsm_LR = np.corrcoef(vecs_LR)
    rsm_avg = np.mean([rsm_L, rsm_R], axis=0)
    rsm_avg_ = rsm_avg[trils]
    r, p = stats.spearmanr(rsm_model_, rsm_avg_)
    print(f"Local (avging): {r=:.4f}")
    print("------------------")


def test_different_trials_each_region(n_trials, n_features_per_trial, n_features, n_voxels, n_high):
    trial_features = make_trial_features(n_trials, n_features_per_trial, n_features)
    feature_voxels_all = make_feature_voxels(n_voxels * 2, n_features)
    feature_voxels_all_sp = make_sparse(feature_voxels_all, n_high=n_high, axis=1)
    vecs_all = np.dot(trial_features, feature_voxels_all_sp.T)

    for i in range(n_trials):
        if i < n_trials / 2:
            vecs_all[i, :n_voxels] = np.random.permutation(vecs_all[i, :n_voxels])
        else:
            vecs_all[i, n_voxels:] = np.random.permutation(vecs_all[i, n_voxels:])

    rsm_model = np.corrcoef(trial_features)
    trils = np.tril_indices_from(rsm_model, k=-1)
    rsm_model_ = rsm_model[trils]

    rsm_all = np.corrcoef(vecs_all)
    rsm_all_ = rsm_all[trils]
    r, p = stats.spearmanr(rsm_model_, rsm_all_)
    print(f"Distributed (concat): {r=:.4f}")

    vecs_L = vecs_all[:, :n_voxels]
    vecs_R = vecs_all[:, n_voxels:]

    rsm_L = np.corrcoef(vecs_L)
    rsm_L_ = rsm_L[trils]
    # print(f"{rsm_L.shape=}")
    # print(f"{rsm_model.shape=}")
    r, p = stats.spearmanr(rsm_model_, rsm_L_)
    print(f"Left: {r=:.4f}")
    rsm_R = np.corrcoef(vecs_R)
    rsm_R_ = rsm_R[trils]
    r, p = stats.spearmanr(rsm_model_, rsm_R_)
    print(f"Right: {r=:.4f}")

    rsm_avg = np.mean([rsm_L, rsm_R], axis=0)
    rsm_avg_ = rsm_avg[trils]
    r, p = stats.spearmanr(rsm_model_, rsm_avg_)
    print(f"Local (avging): {r=:.4f}")


def test_differt_features_each_region(n_trials, n_features_per_trial, n_features, n_voxels, n_high):
    trial_features = make_trial_features(n_trials, n_features_per_trial, n_features)
    feature_voxels_L = make_feature_voxels(n_voxels, n_features)
    feature_voxels_R = make_feature_voxels(n_voxels, n_features)

    feature_voxels_LR = np.concatenate([feature_voxels_L, feature_voxels_R], axis=0)
    feature_voxels_LR_sp = make_sparse(feature_voxels_LR, n_high=n_high * 2, axis=1)

    feature_voxels_LR_sp[:n_voxels, :n_features_per_trial] = 0
    feature_voxels_LR_sp[n_voxels:, n_features_per_trial:] = 0

    feature_voxels_LR_sp_L = feature_voxels_LR_sp[:n_voxels, :]
    feature_voxels_LR_sp_R = feature_voxels_LR_sp[n_voxels:, :]

    vecs_LR = np.dot(trial_features, feature_voxels_LR_sp.T)
    rsm_LR = np.corrcoef(vecs_LR)

    trils = np.tril_indices_from(rsm_LR, k=-1)
    rsm_model = np.corrcoef(trial_features)
    rsm_model_ = rsm_model[trils]

    rsm_LR_ = rsm_LR[trils]
    r, p = stats.spearmanr(rsm_model_, rsm_LR_)
    print(f"Distributed: {r=:.4f}")

    vecs_dist_L = np.dot(trial_features, feature_voxels_LR_sp_L.T)
    rsm_dist_L = np.corrcoef(vecs_dist_L)
    rsm_dist_L_ = rsm_dist_L[trils]
    r, p = stats.spearmanr(rsm_model_, rsm_dist_L_)
    print(f"Distributed L: {r=:.4f}")

    vecs_dist_R = np.dot(trial_features, feature_voxels_LR_sp_R.T)
    rsm_dist_R = np.corrcoef(vecs_dist_R)
    rsm_dist_R_ = rsm_dist_R[trils]
    r, p = stats.spearmanr(rsm_model_, rsm_dist_R_)
    print(f"Distributed R: {r=:.4f}")

    rsm_dist_avg = np.mean([rsm_dist_L, rsm_dist_R], axis=0)
    rsm_dist_avg_ = rsm_dist_avg[trils]
    r, p = stats.spearmanr(rsm_model_, rsm_dist_avg_)
    print(f"Distributed avg: {r=:.4f}")


def test_different_categories(n_trials, n_features_per_trial, n_features, n_voxels, n_high):

    trial_features_0 = make_trial_features(n_trials, n_features_per_trial, n_features)
    feature_voxels_L_0 = make_feature_voxels(n_voxels, n_features)
    feature_voxels_R_0 = make_feature_voxels(n_voxels, n_features)
    # feature_voxels_L_0 = np.random.permutation(feature_voxels_L_0)

    vecs_L_0 = np.dot(trial_features_0, feature_voxels_L_0.T)
    print(f"{vecs_L_0.shape=}")
    vecs_R_0 = np.dot(trial_features_0, feature_voxels_R_0.T)

    for col in range(n_features):
        vecs_L_0[:, col] = stats.zscore(vecs_L_0[:, col])
        vecs_R_0[:, col] = stats.zscore(vecs_R_0[:, col])

    trial_split = n_trials // 2
    vecs_L_0[:trial_split, :] = np.random.normal(
        0, 1, (trial_split, n_voxels)
    )  # np.random.permutation(vecs_L_0[:trial_split, :])
    vecs_R_0[trial_split:, :] = np.random.normal(
        0, 1, (n_trials - trial_split, n_voxels)
    )  # np.random.permutation(vecs_R_0[trial_split:, :])

    # for col in range(n_features):
    #     vecs_L_0[:, col] = stats.zscore(vecs_L_0[:, col])
    #     vecs_R_0[:, col] = stats.zscore(vecs_R_0[:, col])

    vecs_concat = np.concatenate([vecs_L_0, vecs_R_0], axis=1)

    rsm_model_0 = np.corrcoef(trial_features_0)
    rsm_L_0 = np.corrcoef(vecs_L_0)
    rsm_R_0 = np.corrcoef(vecs_R_0)
    rsm_concat = np.corrcoef(vecs_concat)
    rsm_avg = np.mean([rsm_L_0, rsm_R_0], axis=0)

    trils = np.tril_indices_from(rsm_model_0, k=-1)
    rsm_model_0_ = rsm_model_0[trils]
    rsm_L_0_ = rsm_L_0[trils]
    rsm_R_0_ = rsm_R_0[trils]
    rsm_concat_ = rsm_concat[trils]
    rsm_avg_ = rsm_avg[trils]



    r, p = stats.spearmanr(rsm_model_0_, rsm_concat_)
    print(f"Distributed (concat): {r=:.4f}")

    r, p = stats.spearmanr(rsm_model_0_, rsm_avg_)
    print(f"Local (avging): {r=:.4f}")

    r, p = stats.spearmanr(rsm_L_0_, rsm_R_0_)
    print(f"Left x Right: {r=:.4f}")

    r, p = stats.spearmanr(rsm_model_0_, rsm_L_0_)
    print(f"Left: {r=:.4f}")

    r, p = stats.spearmanr(rsm_model_0_, rsm_R_0_)
    print(f"Right: {r=:.4f}")

    # r, p = stats.spearmanr(rsm_model_0_, rsm_L_0_)
    # r, p = stats.spearmanr(rsm_model_0_, rsm_L_0_)
    # print(f"Left: {r=:.4f}")

    # rsm_R_0 = np.corrcoef(vecs_R_0)
    # rsm_R_0_ = rsm_R_0[trils]
    # r, p = stats.spearmanr(rsm_model_0_, rsm_R_0_)
    # print(f"Right: {r=:.4f}")

    # print(f"{vecs_L_0.shape=}")
    # quit()

    # trial_features_1 = make_trial_features(n_trials, n_features_per_trial, n_features)
    # feature_voxels_L_1 = make_feature_voxels(n_voxels, n_features)
    # feature_voxels_R_1 = make_feature_voxels(n_voxels, n_features)
    # feature_voxels_R_1 = np.random.permutation(feature_voxels_R_1)

    # vecs_L_1 = np.dot(trial_features_1, feature_voxels_L_1.T)
    # vecs_R_1 = np.dot(trial_features_1, feature_voxels_R_1.T)

    # vecs_concat = np.concatenate([vecs_L_0, vecs_R_0, vecs_L_1, vecs_R_1], axis=1)
    # rsm_concat = np.corrcoef(vecs_concat)



if __name__ == "__main__":
    print("--*" * 10 + "--")
    n_features = 20
    n_voxels = 200
    n_trials = 10_000
    n_features_per_trial = 5
    n_high = 3
    null_R = True

    # test_one_dud_region(n_trials, n_features_per_trial, n_features, n_voxels, n_high)
    n_high = 10
    # test_differt_features_each_region(
    #     n_trials, n_features_per_trial, n_features, n_voxels, n_high
    # )
    test_different_categories(n_trials, n_features_per_trial, n_features, n_voxels, n_high)
    # test_different_trials_each_region(
    #     n_trials, n_features_per_trial, n_features, n_voxels, n_high
    # )

# print(vecs_L.shape)
