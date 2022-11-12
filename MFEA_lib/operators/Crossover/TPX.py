import numpy as np
from typing import Tuple

from ...EA import Individual
from . import AbstractCrossover


class TPX_Crossover(AbstractCrossover):
    def __call__(self, pa: Individual, pb: Individual, skf_oa=None, skf_ob=None, *args, **kwargs) -> Tuple[Individual, Individual]:
        t1, t2 = np.random.randint(0, self.dim_uss + 1, 2)
        if t1 > t2:
            t1, t2 = t2, t1

        genes_oa = np.copy(pa.genes)
        genes_ob = np.copy(pb.genes)

        genes_oa[t1:t2], genes_ob[t1:t2] = genes_ob[t1:t2], genes_oa[t1:t2]

        oa = self.IndClass(genes_oa)
        ob = self.IndClass(genes_ob)

        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob
