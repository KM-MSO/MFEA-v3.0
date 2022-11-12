import numpy as np

from ...EA import Individual
from . import AbstractMutation

from numba import jit

class PolynomialMutation(AbstractMutation):
    '''
    p in [0, 1]^n
    '''
    def __init__(self, nm = 15, pm = None, *arg, **kwargs):
        '''
        nm: parameters of PolynomialMutation
        pm: prob mutate of PolynomialMutation
        '''
        super().__init__(*arg, **kwargs)
        self.nm = nm
        self.pm = pm

    @staticmethod
    @jit(nopython = True)
    def _mutate(genes, dim_uss, pm, nm):
        idx_mutation = np.where(np.random.rand(dim_uss) <= pm)[0]

        u = np.zeros((dim_uss,)) + 0.5
        u[idx_mutation] = np.random.rand(len(idx_mutation))

        delta = np.where(u < 0.5,
            # delta_l
            (2*u)**(1/(nm + 1)) - 1,
            # delta_r
            1 - (2*(1-u))**(1/(nm + 1))
        )

        return np.where(delta < 0,
                    # delta_l: ind -> 0
                    # = genes * (delta + 1)
                    genes + delta * genes,
                    # delta_r: ind -> 1
                    # = genes (1 - delta) + delta
                    genes + delta * (1 - genes)
                )

    def __call__(self, ind: Individual, return_newInd:bool, *arg, **kwargs) -> Individual:
        if return_newInd:
            newInd = self.IndClass(
                genes= self.__class__._mutate(ind.genes, self.dim_uss, self.pm, self.nm),
                parent= ind
            )
            newInd.skill_factor = ind.skill_factor
            return newInd
        else:
            ind.genes = self.__class__._mutate(ind.genes, self.dim_uss, self.pm, self.nm)
            ind.fcost = None

            return ind
    