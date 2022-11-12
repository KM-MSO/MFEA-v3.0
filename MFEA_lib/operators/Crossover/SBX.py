import numpy as np
from typing import Tuple

from ...EA import Individual
from numba import jit
from . import AbstractCrossover

class SBX_Crossover(AbstractCrossover):
    '''
    pa, pb in [0, 1]^n
    '''
    def __init__(self, nc = 15, *args, **kwargs):
        self.nc = nc

    @staticmethod
    @jit(nopython = True)
    def _crossover(gene_pa, gene_pb, swap, dim_uss, nc):
        u = np.random.rand(dim_uss)
        beta = np.where(u < 0.5, (2*u)**(1/(nc +1)), (2 * (1 - u))**(-1 / (nc + 1)))

        #like pa
        gene_oa = np.clip(0.5*((1 + beta) * gene_pa + (1 - beta) * gene_pb), 0, 1)
        #like pb
        gene_ob = np.clip(0.5*((1 - beta) * gene_pa + (1 + beta) * gene_pb), 0, 1)

        #swap
        if swap:
            idx_swap = np.where(np.random.rand(dim_uss) < 0.5)[0]
            gene_oa[idx_swap], gene_ob[idx_swap] = gene_ob[idx_swap], gene_oa[idx_swap]
    
        return gene_oa, gene_ob
        
    def __call__(self, pa: Individual, pb: Individual, skf_oa= None, skf_ob= None, *args, **kwargs) -> Tuple[Individual, Individual]:
        gene_oa, gene_ob = self.__class__._crossover(pa.genes, pb.genes, pa.skill_factor == pb.skill_factor, self.dim_uss, self.nc)

        oa = self.IndClass(gene_oa)
        ob = self.IndClass(gene_ob)

        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob