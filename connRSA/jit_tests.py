from time import time

import numpy as np
from numba import njit

from connRSA.jit_funcs import CACHE_NUMBA


@njit(fastmath=True, nopython=True, cache=CACHE_NUMBA)
def test_sum(rand):
    out = np.empty(rand.shape[0], dtype=np.float32)
    for i in range(rand.shape[0]):
        out[i] = rand[i, 0] - rand[i, 1]
    return out

@njit(fastmath=True, nopython=True)
def test_mult(rand):
    out = np.empty(rand.shape[0], dtype=np.float32)
    for i in range(rand.shape[0]):
        out[i] = rand[i, 0] * rand[i, 1]
    return out

import numba as nb
sig = nb.bool_[:](nb.int32[:, :])
@njit(sig, fastmath=True, nopython=True, cache=CACHE_NUMBA)
def test_greater(rand):
    out = np.empty(rand.shape[0], dtype=np.bool_)
    for i in range(rand.shape[0]):
        out[i] = rand[i, 0] > rand[i, 1]
    return out

@njit(sig, fastmath=True, nopython=True)
def test_greater_alt(rand):
    out = np.empty(rand.shape[0], dtype=np.bool_)
    # mask = (1 << 1) - 1
    for i in range(rand.shape[0]):
        out[i] = ((rand[i, 1] - rand[i, 0]) >> 31) & 1
    return out

if __name__ == '__main__':
    t_st = time()
    rand = np.random.normal(size=(100_000_000, 2), ).astype(np.float32)
    rand = (rand * 1000).astype(np.int32)
    print(f'Time taken to generate: {time() - t_st:.5f}')
    t_st = time()
    test_sum(rand)
    print(f'Time taken to sum: {time() - t_st:.5f}')
    t_st = time()
    test_sum(rand)
    print(f'Time taken to sum (2): {time() - t_st:.5f}')

    t_st = time()
    test_mult(rand)
    print(f'Time taken to mult: {time() - t_st:.5f}')
    t_st = time()
    test_mult(rand)
    print(f'Time taken to mult (2): {time() - t_st:.5f}')

    t_st = time()
    test_greater(rand)
    print(f'Time taken to greater: {time() - t_st:.5f}')
    t_st = time()
    a = test_greater(rand)
    print(f'Time taken to greater (2): {time() - t_st:.5f}')

    t_st = time()
    test_greater_alt(rand)
    print(f'Time taken to alt_greater: {time() - t_st:.5f}')
    t_st = time()
    b = test_greater_alt(rand)
    print(f'Time taken to alt_greater (2): {time() - t_st:.5f}')

    print(rand[:5])
    print(a[:10])
    print(b[:10])