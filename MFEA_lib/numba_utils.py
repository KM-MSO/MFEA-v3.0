import numpy as np
from numba import jit

@jit(nopython = True)
def numba_randomchoice_w_prob(a):
    assert np.abs(np.sum(a) - 1.0) < 1e-9
    rd = np.random.rand()
    res = 0
    sum_p = a[0]
    while rd > sum_p:
        res += 1
        sum_p += a[res]
    return res

@jit(nopython = True)
def numba_randomchoice(a, size= None, replace= True):
    return np.random.choice(a, size= size, replace= replace)

@jit(nopython = True)
def numba_linalgo_det(matrix):
    return np.linalg.det(matrix)

@jit(nopython = True)
def numba_linalgo_pinv(matrix):
    return np.linalg.pinv(matrix)

@jit(nopython = True)
def numba_dot(a, b):
    return np.dot(a, b)

@jit(nopython = True)
def numba_min(genes):
    return np.min(genes)

@jit(nopython = True)
def numba_max(genes):
    return np.max(genes)