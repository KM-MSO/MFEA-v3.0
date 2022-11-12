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