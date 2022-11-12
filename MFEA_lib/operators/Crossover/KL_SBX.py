import numpy as np
from typing import Tuple, Type, List

from ...tasks.task import AbstractTask
from ...EA import Individual, Population
from numba import jit
from . import AbstractCrossover
class KL_SBXCrossover(AbstractCrossover):
    '''
    pa, pb in [0, 1]^n
    '''
    def __init__(self, nc = 2, k = 1, conf_thres= 0.6, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nc = nc
        self.k = k
        self.conf_thres = conf_thres
    
    def getInforTasks(self, IndClass: Type[Individual], tasks: List[AbstractTask], seed=None):
        super().getInforTasks(IndClass, tasks, seed)
        # self.prob = 1 - KL_divergence
        self.prob = np.ones((self.nb_tasks, self.nb_tasks, self.dim_uss))

    @staticmethod
    @jit(nopython= True)
    def _updateProb(prob, k, dim_uss, nb_tasks, mean, std):
        for i in range(nb_tasks):
            for j in range(nb_tasks):
                kl = np.log((std[i] + 1e-50)/(std[j] + 1e-50)) + (std[j] ** 2 + (mean[j] - mean[i]) ** 2)/(2 * std[i] ** 2 + 1e-50) - 1/2
                # prob[i][j] = 1/(1 + kl/k)
                prob[i][j] = np.exp(-kl/k)

        return np.clip(prob, 1/dim_uss, 1)

    def update(self, population: Population, **kwargs) -> None:
        mean = np.zeros((self.nb_tasks, self.dim_uss))
        std = np.zeros((self.nb_tasks, self.dim_uss))
        for idx_subPop in range(self.nb_tasks):
            mean[idx_subPop] = population[idx_subPop].__meanInds__
            std[idx_subPop] = population[idx_subPop].__stdInds__

        self.prob = self.__class__._updateProb(self.prob, self.k, self.dim_uss, self.nb_tasks, mean, std)

    @staticmethod
    @jit(nopython = True)
    def _crossover(gene_pa, gene_pb, swap, conf_thres, dim_uss, nc, pcd, gene_p_of_oa, gene_p_of_ob):
        u = np.random.rand(dim_uss)
        beta = np.where(u < 0.5, (2*u)**(1/(nc +1)), (2 * (1 - u))**(-1 / (nc + 1)))

        idx_crossover = np.random.rand(dim_uss) < pcd

        if np.all(idx_crossover == 0) or np.all(gene_pa[idx_crossover] == gene_pb[idx_crossover]):
            # alway crossover -> new individual
            idx_notsame = np.where(gene_pa != gene_pb)[0]
            if len(idx_notsame) == 0:
                idx_crossover = np.ones((dim_uss, ), dtype= np.bool_)
            else:
                idx_crossover[np.random.choice(idx_notsame)] = True

        #like pa
        gene_oa = np.where(idx_crossover, np.clip(0.5*((1 + beta) * gene_pa + (1 - beta) * gene_pb), 0, 1), gene_p_of_oa)
        #like pb
        gene_ob = np.where(idx_crossover, np.clip(0.5*((1 - beta) * gene_pa + (1 + beta) * gene_pb), 0, 1), gene_p_of_ob)

        #swap
        # if swap:
            # idx_swap = np.where(np.logical_and(np.random.rand(dim_uss) < 0.5, idx_crossover))[0]
            # idx_swap = np.where(np.random.rand(dim_uss) < 0.5)[0]
        idx_swap = np.where(np.logical_and(np.random.rand(dim_uss) < 0.5, pcd >= conf_thres))[0]
        gene_oa[idx_swap], gene_ob[idx_swap] = gene_ob[idx_swap], gene_oa[idx_swap]
    
        return gene_oa, gene_ob
        
    def __call__(self, pa: Individual, pb: Individual, skf_oa=None, skf_ob=None, *args, **kwargs) -> Tuple[Individual, Individual]:
        if skf_oa == pa.skill_factor:
            p_of_oa = pa
        elif skf_oa == pb.skill_factor:
            p_of_oa = pb
        else:
            raise ValueError()
        if skf_ob == pb.skill_factor:
            p_of_ob = pb
        elif skf_ob == pa.skill_factor:
            p_of_ob = pa
        else:
            raise ValueError()

        # skf_pa == skf_pb => skf_oa == skf_ob
        # skf_pa != skf_pb => skf_oa == skf_ob || skf_oa != skf_ob
        gene_oa, gene_ob = self.__class__._crossover(pa.genes, pb.genes, skf_oa == skf_ob, self.conf_thres, self.dim_uss, self.nc, self.prob[pa.skill_factor][pb.skill_factor], p_of_oa.genes, p_of_ob.genes)
        # gene_oa, gene_ob = self.__class__._crossover(pa.genes, pb.genes, 
        #     ((pa.skill_factor == pb.skill_factor) + (skf_oa == skf_ob))/2, self.dim_uss, self.nc, self.prob[pa.skill_factor][pb.skill_factor], p_of_oa.genes, p_of_ob.genes)

        oa = self.IndClass(gene_oa)
        ob = self.IndClass(gene_ob)

        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob

        return oa, ob
