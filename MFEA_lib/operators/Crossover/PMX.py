import numpy as np
from typing import Tuple
import numba as nb

from ...EA import Individual
from . import AbstractCrossover


@nb.njit
def pmx_func(p1, p2, t1, t2,  dim_uss):
    oa = np.empty_like(p1)
    ob = np.empty_like(p1)
    
    mid = np.copy(p2[t1:t2])
    mid_b = np.copy(p1[t1 : t2])
    
    added = np.zeros_like(p1)
    added_b = np.zeros_like(p2)
    
    added[mid] = 1
    added_b[mid_b] = 1
    
    redundant_idx = []
    redundant_idx_b = []
    
    
    for i in range(t1):
        if added[p1[i]]:
            redundant_idx.append(i)
        else:
            oa[i] = p1[i]
            added[oa[i]] = 1
            
        if added_b[p2[i]]:
            redundant_idx_b.append(i)
        else:
            ob[i] = p2[i]
            added_b[ob[i]] = 1
            
    for i in range(t2, dim_uss):
        
        if added[p1[i]]:
            redundant_idx.append(i)
        else:
            oa[i] = p1[i]
            added[oa[i]] = 1
            
        if added_b[p2[i]]:
            redundant_idx_b.append(i)
        else:
            ob[i] = p2[i]
            added_b[ob[i]] = 1
    
    redundant = np.empty(len(redundant_idx))
    redundant_b = np.empty(len(redundant_idx_b))
    
    cnt = 0
    cnt_b = 0
    
    for i in range(t1, t2):
        if added[p1[i]] == 0:
            redundant[cnt] = p1[i]
            cnt+=1
        if added_b[p2[i]] == 0:
            redundant_b[cnt_b] = p2[i]
            cnt_b+=1
    
    redundant_idx = np.array(redundant_idx)
    redundant_idx_b = np.array(redundant_idx_b)
    
    oa[redundant_idx] = redundant
    ob[redundant_idx_b] = redundant_b
    
    oa[t1:t2] = mid
    ob[t1:t2] = mid_b
    return oa, ob
    

class PMX_Crossover(AbstractCrossover):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

    def __call__(self, pa: Individual, pb: Individual, skf_oa=None, skf_ob=None, *args, **kwargs) -> Tuple[Individual, Individual]:
        t1 = np.random.randint(0, self.dim_uss + 1)
        t2 = np.random.randint(t1, self.dim_uss + 1)
        genes_oa, genes_ob = pmx_func(pa.genes, pb.genes, t1, t2, self.dim_uss)
        oa = self.IndClass(genes_oa)
        ob = self.IndClass(genes_ob)
        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob

        