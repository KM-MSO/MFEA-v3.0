import numpy as np
from typing import Tuple, Type, List

from ...tasks.task import AbstractTask
from ...EA import Individual


class AbstractCrossover():
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, pa: Individual, pb: Individual, skf_oa= None, skf_ob= None, *args, **kwargs) -> Tuple[Individual, Individual]:
        pass
    def getInforTasks(self, IndClass: Type[Individual], tasks: List[AbstractTask], seed = None):
        self.dim_uss = max([t.dim for t in tasks])
        self.nb_tasks = len(tasks)
        self.tasks = tasks
        self.IndClass = IndClass
        #seed
        np.random.seed(seed)
        pass
    
    def update(self, *args, **kwargs) -> None:
        pass

class NoCrossover(AbstractCrossover):
    def __call__(self, pa: Individual, pb: Individual, skf_oa=None, skf_ob=None, *args, **kwargs) -> Tuple[Individual, Individual]:
        oa = self.IndClass(None, self.dim_uss)
        ob = self.IndClass(None, self.dim_uss)

        oa.skill_factor = skf_oa
        ob.skill_factor = skf_ob
        return oa, ob