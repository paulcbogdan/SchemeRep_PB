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


if __name__ == "__main__":
    print("--*" * 10 + "--")
    n_features = 20
    n_voxels = 1000
    n_trials = 1_000
    n_features_per_trial = 2
    n_high = 5
    null_R = True
    trial_features = make_trial_features(n_trials, n_features_per_trial, n_features)
    feature_voxels_L = make_feature_voxels(n_voxels, n_features)
    feature_voxels_L_sp = make_sparse(feature_voxels_L, n_high=n_high, axis=1)
    # feature_voxels_L_sp = feature_voxels_L

    feature_voxels_R = make_feature_voxels(n_voxels, n_features)
    # feature_voxels_R = np.random.normal(0, 1, feature_voxels_R.shape)
    feature_voxels_R_sp = make_sparse(feature_voxels_R, n_high=n_high, axis=1)
    # feature_voxels_R_sp = feature_voxels_R

    # feature_voxels_concat = np.concatenate(
    #     [feature_voxels_L_sp, feature_voxels_R_sp], axis=0
    # )

    rsm_model = np.corrcoef(trial_features)

    vecs_L = np.dot(trial_features, feature_voxels_L_sp.T)
    rsm_L = np.corrcoef(vecs_L)
    trils = np.tril_indices_from(rsm_L, k=-1)
    rsm_L_ = rsm_L[trils]
    rsm_model_ = rsm_model[trils]
    r, p = stats.spearmanr(rsm_model_, rsm_L_)
    print(f"Left: {r=:.3f}")

    vecs_R = np.dot(trial_features, feature_voxels_R_sp.T)
    if null_R:
        np.random.shuffle(vecs_R)
    rsm_R = np.corrcoef(vecs_R)
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

    n_high = 5

    feature_voxels_LR = np.concatenate([feature_voxels_L, feature_voxels_R], axis=0)
    # print(f"{feature_voxels_LR.shape=}")
    # quit()
    feature_voxels_LR_sp = make_sparse(feature_voxels_LR, n_high=n_high * 2, axis=1)
    # print(f"{feature_voxels_LR_sp.shape=}")
    # print(f"{n_high=}")
    # quit()
    # print(f"{n_voxels=}")
    # print(f"{n_high=}")
    # quit()
    # print(f"{feature_voxels_LR_sp.shape=}")
    # print(f"{n_features_per_trial=}")
    # quit()

    feature_voxels_LR_sp[:n_voxels, :n_features_per_trial] = 0
    feature_voxels_LR_sp[n_voxels:, n_features_per_trial:] = 0
    # plt.imshow(feature_voxels_LR_sp)
    # plt.show()
    # quit()
    # print(f"{feature_voxels_LR=}")
    # print(f"{n_voxels=}")
    # print(f"{n_features_per_trial=}")
    # quit()
    # quit()

    feature_voxels_LR_sp_L = feature_voxels_LR_sp[:n_voxels, :]
    feature_voxels_LR_sp_R = feature_voxels_LR_sp[n_voxels:, :]

    vecs_LR = np.dot(trial_features, feature_voxels_LR_sp.T)
    rsm_LR = np.corrcoef(vecs_LR)
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


# print(vecs_L.shape)
