import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

import numpy as np


def create_feature_vector(n_voxels, n_voxels_per_feature):
    """
    Creates a vector of zeros with specific elements set to 1.

    Args:
        n_voxels: The total size of the vector
        n_voxels_per_feature: Number of elements to set to 1

    Returns:
        A numpy array of size n_voxels with n_voxels_per_feature elements set to 1
    """
    if n_voxels_per_feature > n_voxels:
        raise ValueError("n_voxels_per_feature cannot exceed n_voxels")

    # Create a vector of zeros
    vec = np.zeros(n_voxels)

    # Randomly select indices to set to 1
    indices = np.random.choice(n_voxels, n_voxels_per_feature, replace=False)

    # Set selected indices to 1
    vec[indices] = 1

    return vec


def generate_random_covariance_matrix_unit_sd(
    size: int, epsilon: float = 1e-9
) -> np.ndarray:
    """
    Generates a randomized covariance matrix of a given size, where each variable
    has a unit standard deviation (i.e., variances on the diagonal are 1).
    This results in a correlation matrix.

    The matrix is guaranteed to be symmetric and positive semi-definite.

    Args:
      size: The dimension of the square covariance matrix (e.g., 114 for a 114x114 matrix).
      epsilon: A small positive value to add to the diagonal of the initial
               matrix to ensure strict positive definiteness before normalization.
               This helps in avoiding division by zero or numerical instability if
               the initial matrix is not strictly positive definite.

    Returns:
      A numpy.ndarray representing the random correlation matrix (covariance matrix
      with unit standard deviations).
    """
    if not isinstance(size, int) or size <= 0:
        raise ValueError("Size must be a positive integer.")
    if not isinstance(epsilon, float) or epsilon < 0:
        raise ValueError("Epsilon must be a non-negative float.")

    # 1. Generate a random matrix A.
    #    The elements are drawn from a standard normal distribution.
    A = np.random.randn(size, size)

    # 2. Compute an initial symmetric positive semi-definite matrix S = A * A.T.
    S = np.dot(A, A.T)

    # 3. Ensure strict positive definiteness by adding a small multiple of the identity matrix.
    #    This helps prevent issues if S has diagonal entries that are zero or very close to zero,
    #    which could happen if A is not full rank (though unlikely with randn for typical sizes).
    S_reg = S + np.eye(size) * epsilon

    # 4. Calculate the inverse of standard deviations.
    #    std_devs_inv = 1.0 / sqrt(diag(S_reg))
    #    Create a diagonal matrix from these inverse standard deviations.
    diag_S_reg = np.diag(S_reg)
    if np.any(diag_S_reg <= 0):
        # This case should be extremely rare due to epsilon addition if A was not all zeros in a row/col
        raise ValueError(
            "Initial matrix has non-positive diagonal elements even after regularization. Try a larger epsilon."
        )

    inv_std_devs = 1.0 / np.sqrt(diag_S_reg)
    D_inv_sqrt = np.diag(inv_std_devs)

    # 5. Compute the correlation matrix R = D_inv_sqrt * S_reg * D_inv_sqrt.
    #    This scales S_reg such that the diagonal elements of R become 1.
    correlation_matrix = np.dot(np.dot(D_inv_sqrt, S_reg), D_inv_sqrt)

    # Ensure the matrix is numerically symmetric after operations (it should be)
    correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2.0

    # Clamp diagonal elements to 1.0 to correct any tiny floating point inaccuracies
    np.fill_diagonal(correlation_matrix, 1.0)

    return correlation_matrix


def sample_from_rsm(rsm, n_samples=100, error=0):
    vecs = np.random.multivariate_normal(np.zeros(rsm.shape[0]), rsm, size=n_samples).T
    vecs = vecs + np.random.normal(0, error, size=vecs.shape)
    return vecs


def do_analysis():
    rsm_model = generate_random_covariance_matrix_unit_sd(114)
    vecs_L = sample_from_rsm(rsm_model)
    # vecs_R = np.copy(vecs_L)
    # np.random.shuffle(vecs_R)
    vecs_R = sample_from_rsm(rsm_model)

    vecs_L_rand = np.copy(vecs_L)
    np.random.shuffle(vecs_L_rand)
    vecs_R_rand = np.copy(vecs_R)
    np.random.shuffle(vecs_R_rand)
    vecs_L = np.concatenate([vecs_L, vecs_L_rand], axis=1)
    vecs_R = np.concatenate([vecs_R, vecs_R_rand], axis=1)

    vecs_concat = np.concatenate([vecs_L, vecs_R], axis=1)

    rsm_L = np.cov(vecs_L)
    rsm_R = np.cov(vecs_R)
    rsm_concat = np.cov(vecs_concat)
    rsm_avg = (rsm_L + rsm_R) / 2

    trils = np.tril_indices_from(rsm_model, k=-1)
    rsm_model_ = rsm_model[trils]
    rsm_L_ = rsm_L[trils]
    rsm_R_ = rsm_R[trils]
    rsm_avg_ = rsm_avg[trils]
    rsm_concat_ = rsm_concat[trils]
    r, p = stats.spearmanr(rsm_model_, rsm_L_)
    print(f"Left corr: {r=:.3f}")
    r, p = stats.spearmanr(rsm_model_, rsm_R_)
    print(f"Right corr: {r=:.3f}")
    r, p = stats.spearmanr(rsm_model_, rsm_avg_)
    print(f"Avg corr: {r=:.3f}")
    r, p = stats.spearmanr(rsm_model_, rsm_concat_)
    print(f"Concat corr: {r=:.3f}")

    # print(f"{np.linalg.norm(rsm_L_ - rsm_L_est_)=}")


def do_analysis_feature_vecs(n_trials=114, n_features_per_trial=10):
    feature2onehot_l = make_feature2onehot(
        n_features=20, n_active=10, n_voxels=50
    )
    feature2onehot_r = make_feature2onehot(
        n_features=20, n_active=10, n_voxels=50
    )
    feature2onehot_lr = make_feature2onehot(
        n_features=20, n_active=20, n_voxels=50
    )

    trial2features = []
    for i in range(n_trials):
        trial2features.append(
            np.random.choice(
                list(feature2onehot_l.keys()), n_features_per_trial, replace=False
            )
        )

    vecs_model = make_feature_vecs(
        feature2onehot_l,
        trial2features,
        n_trials=100,
        n_voxels=50,
        error=0,
    )
    rsm_model = np.corrcoef(vecs_model)

    vecs_l = make_feature_vecs(
        feature2onehot_l,
        trial2features,
        n_trials=100,
        n_voxels=50,
        error=1,
    )
    rsm_l = np.corrcoef(vecs_l)
    trils = np.tril_indices_from(rsm_model, k=-1)
    rsm_model_ = rsm_model[trils]
    rsm_l_ = rsm_l[trils]
    r, p = stats.spearmanr(rsm_model_, rsm_l_)
    print(f"Left corr: {r=:.3f}")


def make_feature2onehot(n_features, n_active, n_voxels):
    feature2onehot_l = {}
    for i in range(n_features):
        onehot_vec = create_feature_vector(n_voxels, n_active)
        feature2onehot_l[i] = onehot_vec
    return feature2onehot_l


def make_feature_vecs(
    feature2onehot_l,
    trial2features,
    n_trials=100,
    n_voxels=50,
    error=1,
    n_features_per_trial=10,
):
    vecs_l = []
    for i in range(n_trials):
        vec = np.random.randn(n_voxels) * error
        features_included = trial2features[i]
        for feature in features_included:
            vec += feature2onehot_l[feature]
        vecs_l.append(vec)
    vecs_l = np.array(vecs_l)
    return vecs_l


if __name__ == "__main__":
    # do_analysis()
    do_analysis_feature_vecs()
