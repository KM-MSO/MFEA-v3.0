from copy import deepcopy
from math import ceil
from typing import Type, List

import numpy as np
from ..EA import *

class AbstractSelection():
    def __init__(self, *args, **kwds) -> None:
        pass
    def __call__(self, population:Population, nb_inds_tasks:list, *args, **kwds) -> List[int]:
        pass
    def getInforTasks(self, tasks: List[AbstractTask], seed = None):
        self.dim_uss = max([t.dim for t in tasks])

        #seed
        np.random.seed(seed)
        pass
class ElitismSelection(AbstractSelection):
    def __init__(self, random_percent = 0, *args, **kwds) -> None:
        super().__init__(*args, **kwds)
        assert 0<= random_percent and random_percent <= 1
        self.random_percent = random_percent        
        
    def __call__(self, population:Population, nb_inds_tasks: list, *args, **kwds) -> List[int]:
        ls_idx_selected = []
        for idx_subpop, subpop in enumerate(population):
            N_i = min(nb_inds_tasks[idx_subpop], len(subpop))

            # on top
            N_elitism = ceil(N_i* (1 - self.random_percent))
            idx_selected_inds:list = np.where(subpop.scalar_fitness > 1/(N_elitism + 1))[0].tolist()
            
            #random
            remain_idx = np.where(subpop.scalar_fitness < 1/N_elitism)[0].tolist()
            idx_random = np.random.choice(remain_idx, size= (N_i - N_elitism, )).tolist()

            idx_selected_inds += idx_random

            subpop.select(idx_selected_inds)
            subpop.update_rank()

            ls_idx_selected.append(idx_selected_inds)

        return ls_idx_selected

class TournamentSelection(AbstractSelection):
    def __init__(self, k = 2) -> None:
        super().__init__()
        self.k = k
    def __call__(self, population: Population, nb_inds_tasks: list, *args, **kwds) -> List[int]:
        ls_idx_selected = []
        for idx_subpop, subpop in enumerate(population):
            N_i = min(nb_inds_tasks[idx_subpop], len(subpop))

            bool_selected_inds = np.zeros((len(subpop), ), dtype= int)
            while(np.sum(bool_selected_inds) < N_i):
                tournament = np.random.choice(len(subpop), p = (1 - bool_selected_inds)/np.sum(1 - bool_selected_inds), size = self.k)
                idx_best = tournament[0]
                for idx in tournament:
                    if subpop.scalar_fitness[idx] > subpop.scalar_fitness[idx_best]:
                        idx_best = idx
                bool_selected_inds[idx_best] = 1
            
            idx_selected_inds = np.where(bool_selected_inds)[0].tolist()
            subpop.select(idx_selected_inds)
            subpop.update_rank()

            ls_idx_selected.append(idx_selected_inds)
        return ls_idx_selected

