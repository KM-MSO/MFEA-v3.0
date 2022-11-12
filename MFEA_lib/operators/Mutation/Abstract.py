from typing import Type, List
import numpy as np

from ...tasks.task import AbstractTask
from ...EA import Individual


class AbstractMutation():
    def __init__(self, *arg, **kwargs):
        self.pm = None
        pass
    def __call__(self, ind: Individual, return_newInd:bool, *arg, **kwargs) -> Individual:
        pass
    def getInforTasks(self, IndClass: Type[Individual], tasks: List[AbstractTask], seed = None):
        self.dim_uss = max([t.dim for t in tasks])
        self.nb_tasks = len(tasks)
        if self.pm is None:
            self.pm = 1/self.dim_uss
        self.tasks = tasks
        self.IndClass = IndClass
        #seed
        np.random.seed(seed)
        pass
    def update(self, *arg, **kwargs) -> None:
        pass
    def compute_accept_prob(self, new_fcost, min_fcost): 
        pass
class NoMutation(AbstractMutation):
    def __call__(self, ind: Individual, return_newInd:bool, *arg, **kwargs) -> Individual:
        if return_newInd:
            newInd =  self.IndClass(genes= np.copy(ind.genes), parent=ind)
            newInd.skill_factor = ind.skill_factor
            newInd.fcost = ind.fcost
            return newInd
        else:
            return ind