
import pandas as pd

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from time import time

import numpy as np
from numba import jit
import numpy.linalg as la


@jit(nopython=True)
def compute_t_value(y, X_itr):
    """
    Helper function to compute t-value for a single regression
    y ~ 1 + X where X is a single column
    """
    # print(X.shape)
    # quit()

    n = len(y)
    X_itr = np.column_stack((np.ones(n), X_itr))

    # Compute beta: (X'X)^-1 X'y
    XtX = X_itr.T @ X_itr
    Xty = X_itr.T @ y
    beta = la.solve(XtX, Xty)

    # Compute residuals
    residuals = y - X_itr @ beta

    # Compute standard error
    mse = np.sum(residuals ** 2) / (n - X_itr.shape[1])
    var_beta = mse * la.inv(XtX)
    se = np.sqrt(np.diag(var_beta))

    # Return t-value for the slope coefficient
    return beta[1] / se[1]


@jit(nopython=True)
def pairwise_interaction_t_values(y, X):
    """
    Compute t-values for all pairwise interaction regressions
    y ~ 1 + X1 * X2 for all pairs of columns in X

    Parameters:
    -----------
    y : array-like of shape (n_samples,)
        Target variable
    X : array-like of shape (n_samples, n_features)
        Feature matrix

    Returns:
    --------
    t_values : array-like of shape (n_features, n_features)
        Matrix where entry (i,j) contains the t-value for the
        interaction term in regression y ~ 1 + X_i * X_j
    """
    n_features = X.shape[1]
    t_values = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i, n_features):
            # Compute interaction term
            interaction = X[:, i] * X[:, j]


            # Compute t-value
            t_values[i, j] = compute_t_value(y, interaction)
            # t_values[i, j] = compute_t_value(y, interaction)
            t_values[j, i] = t_values[i, j]  # Matrix is symmetric

    return t_values


@jit(nopython=True)
def compute_t_value_(y, X1, X2):
    """
    Helper function to compute t-value for regression
    y ~ 1 + X1 + X2 + X1*X2
    Returns t-value for the interaction term
    """
    n = len(y)
    interaction = X1 * X2

    # Create design matrix with intercept, main effects, and interaction
    X = np.column_stack((np.ones(n), X1, X2, interaction))
    # print(X)

    # Compute beta: (X'X)^-1 X'y
    XtX = X.T @ X
    Xty = X.T @ y
    beta = la.solve(XtX, Xty)

    # Compute residuals
    residuals = y - X @ beta

    # Compute standard error
    mse = np.sum(residuals ** 2) / (n - X.shape[1])
    var_beta = mse * la.inv(XtX)
    se = np.sqrt(np.diag(var_beta))

    # Return t-value for the interaction coefficient (last coefficient)
    return beta[3] / se[3]


@jit(nopython=True)
def pairwise_interaction_t_values_(y, X):
    """
    Compute t-values for all pairwise interaction regressions
    y ~ 1 + X1 + X2 + X1*X2 for all pairs of columns in X

    Parameters:
    -----------
    y : array-like of shape (n_samples,)
        Target variable
    X : array-like of shape (n_samples, n_features)
        Feature matrix

    Returns:
    --------
    t_values : array-like of shape (n_features, n_features)
        Matrix where entry (i,j) contains the t-value for the
        interaction term in regression y ~ 1 + X_i + X_j + X_i * X_j
    """
    n_features = X.shape[1]
    t_values = np.zeros((n_features, n_features))

    for i in range(n_features):
        for j in range(i, n_features):
            if i == j: continue
            # Compute t-value with both main effects and interaction
            t_values[i, j] = compute_t_value_(y, X[:, i], X[:, j])
            t_values[j, i] = t_values[i, j]  # Matrix is symmetric

    return t_values


if __name__ == '__main__':
    # Y = np.random.randint(0, 2, (100))
    Y = np.random.normal(size=(1000))
    Y = np.array(Y, dtype=np.float64)
    Y = stats.zscore(Y, axis=0, nan_policy='omit')
    X = np.random.normal(size=(1000, 300))
    # print(X)
    # quit()
    X = stats.zscore(X, axis=0, nan_policy='omit')
    t_st = time()
    out = pairwise_interaction_t_values_(Y, X)
    # print(out[0, 1])
    #
    # df_test = pd.DataFrame({'Y': Y, 'X1': X[:, 0], 'X2': X[:, 1]})
    # # do regression with smf
    # import statsmodels.formula.api as smf
    #
    # mod = smf.ols(formula='Y ~ X1 * X2', data=df_test)
    # print(mod.fit().summary())
    # quit()


    # print(out.flatten())
    print(f'Time needed for interaction testing: {time() - t_st:.3f} s')

    out[np.diag_indices_from(out)] = np.nan
    n, _, _ = plt.hist(out.flatten(), bins=100)
    sd = np.nanstd(out.flatten())

    x = np.linspace(-sd*3, sd*3, 1000)
    y = stats.norm.pdf(x, loc=0, scale=sd)
    y *= np.max(n) / np.max(y)
    plt.plot(x, y, 'r-', lw=2, label='Normal Distribution')
    plt.show()

    t_st = time()
    out = pairwise_interaction_t_values_(Y, X)
    print(out.flatten())
    print(f'Time needed for interaction testing #2: {time() - t_st:.3f} s')



    quit()

    #
    # t_st = time()
    # out = test_interaction_effect(Y, X)
    # print(f'Time needed for interaction testing: {time() - t_st:.3f} s')
    # quit()