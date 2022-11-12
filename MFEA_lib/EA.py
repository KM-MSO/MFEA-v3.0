import random
import numpy as np
from typing import Type, List
from numba import jit
from .numba_utils import numba_randomchoice

from .tasks.task import AbstractTask

class Individual:
    '''
    a Individual include:\n
    + `genes`: numpy vector represent for the individual; If genes = None: random genes\n
    + `skill_factor`: skill factor of the individual\n
    + `fcost`: factorial cost of the individual for skill_factor
    '''
    def __init__(self, genes,  parent= None, dim= None, *args, **kwargs) -> None: 
        self.skill_factor: int = None
        self.fcost: float = None
        self.genes: np.ndarray = genes
        self.parent: Individual = parent

    def eval(self, task: AbstractTask) -> None:
        '''
        Evaluate Individual
        return factorial_cost after evaluate
        '''
        return task(self.genes)

    @property
    def shape(self) -> int:
        return self.genes.shape
    def __repr__(self) -> str:
        return 'Genes: {}\nSkill_factor: {}'.format(str(self.genes), str(self.skill_factor))
    def __str__(self) -> str:
        return str(self.genes)
    def __len__(self):
        return len(self.genes)
    def __getitem__(self, index):
        return self.genes[index]

    def __add__(self, other):
        ind = Individual(self[:] + other)
        ind.skill_factor = self.skill_factor
        return ind
    def __sub__(self, other):
        ind = Individual(self[:] - other)
        ind.skill_factor = self.skill_factor
        return ind
    def __mul__(self, other):
        ind = Individual(self[:] * other)
        ind.skill_factor = self.skill_factor
        return ind
    def __truediv__(self, other):
        ind = Individual(self[:] / other)
        ind.skill_factor = self.skill_factor
        return ind
    def __floordiv__(self, other):
        ind = Individual(self[:] // other)
        ind.skill_factor = self.skill_factor
        return ind
    def __mod__(self, other):
        ind = Individual(self[:] % other)
        ind.skill_factor = self.skill_factor
        return ind
    def __pow__(self, other):
        ind = Individual(self[:] ** other)
        ind.skill_factor = self.skill_factor
        return ind

    def __lt__(self, other) -> bool:
        if self.skill_factor == other.skill_factor:
            return self.fcost > other.fcost
        else:
            return False
    def __gt__(self, other) -> bool:
        if self.skill_factor == other.skill_factor:
            return self.fcost < other.fcost
        else:
            return False
    def __le__(self, other)-> bool:
        if self.skill_factor == other.skill_factor:
            return self.fcost >= other.fcost
        else:
            return False
    def __ge__(self, other)-> bool:
        if self.skill_factor == other.skill_factor:
            return self.fcost <= other.fcost
        else:
            return False
    def __eq__(self, other)-> bool:
        if self.skill_factor == other.skill_factor:
            return self.fcost == other.fcost
        else:
            return False
    def __ne__(self, other)-> bool:
        if self.skill_factor == other.skill_factor:
            return self.fcost != other.fcost
        else:
            return False
        
class SubPopulation:
    def __init__(self, IndClass: Type[Individual], skill_factor, dim, num_inds, task: AbstractTask = None) -> None:
        self.skill_factor = skill_factor
        self.task = task
        self.dim = dim
        self.ls_inds = [
            IndClass(genes= None, dim= self.dim)
            for i in range(num_inds)
        ]
        self.IndClass = IndClass
        for i in range(num_inds):
            self.ls_inds[i].skill_factor = skill_factor
            self.ls_inds[i].fcost = self.task(self.ls_inds[i].genes)

        self.factorial_rank: np.ndarray = None
        self.scalar_fitness: np.ndarray = None
        self.update_rank()
        
    def __len__(self): 
        return len(self.ls_inds)

    def __getitem__(self, index):
        try:
            return self.ls_inds[index]
        except:
            if type(index) == int:
                self.ls_inds[index]
            elif type(index) == list:
                return [self.ls_inds[i] for i in index]
            else:
                raise TypeError('Int, Slice or List[int], not ' + str(type(index)))
    
    def __getRandomItems__(self, size:int = None, replace:bool = False):
        if size == 0:
            return []
        if size == None:
            return self.ls_inds[numba_randomchoice(len(self), size= None, replace= replace)]
        return [self.ls_inds[idx] for idx in numba_randomchoice(len(self), size= size, replace= replace).tolist()]


    def __addIndividual__(self, individual: Individual, update_rank = False):
        if individual.fcost is None:
            individual.fcost = self.task(individual.genes)
        self.ls_inds.append(individual)
        if update_rank:
            self.update_rank()

    def __add__(self, other):
        assert self.task == other.task, 'Cannot add 2 sub-population do not have the same task'
        assert self.dim == other.dim, 'Cannot add 2 sub-population do not have the same dimensions'
        UnionSubPop = SubPopulation(
            IndClass= self.IndClass,
            skill_factor = self.skill_factor,
            dim= self.dim,
            num_inds= 0,
            task= self.task
        )
        UnionSubPop.ls_inds = self.ls_inds + other.ls_inds
        UnionSubPop.update_rank()
        return UnionSubPop

    @property 
    def __getBestIndividual__(self):
        return self.ls_inds[int(np.argmin(self.factorial_rank))]   
    @property 
    def __getWorstIndividual__(self):
        return self.ls_inds[int(np.argmax(self.factorial_rank))]
    
    @staticmethod
    @jit(nopython = True)
    def _numba_meanInds(ls_genes):
        res = [np.mean(ls_genes[:, i]) for i in range(ls_genes.shape[1])]
        return np.array(res)

    @property 
    def __meanInds__(self):
        # return self.__class__._numba_meanInds(np.array([ind.genes for ind in self.ls_inds]))
        return np.mean([ind.genes for ind in self.ls_inds], axis= 0)

    @staticmethod
    @jit(nopython = True)
    def _numba_stdInds(ls_genes):
        res = [np.std(ls_genes[:, i]) for i in range(ls_genes.shape[1])]
        return np.array(res)


    @property 
    def __stdInds__(self):
        # return self.__class__._numba_stdInds(np.array([ind.genes for ind in self.ls_inds]))
        return np.std([ind.genes for ind in self.ls_inds], axis= 0)

    @staticmethod
    @jit(nopython = True)
    def _sort_rank(ls_fcost):
        return np.argsort(np.argsort(ls_fcost)) + 1

    def update_rank(self):
        '''
        Update `factorial_rank` and `scalar_fitness`
        '''
        # self.factorial_rank = np.argsort(np.argsort([ind.fcost for ind in self.ls_inds])) + 1
        if len(self.ls_inds):
            self.factorial_rank = self.__class__._sort_rank(np.array([ind.fcost for ind in self.ls_inds]))
        else:
            self.factorial_rank = np.array([])
        self.scalar_fitness = 1/self.factorial_rank

    def select(self, index_selected_inds: list):
        self.ls_inds = [self.ls_inds[idx] for idx in index_selected_inds]

        self.factorial_rank = self.factorial_rank[index_selected_inds]
        self.scalar_fitness = self.scalar_fitness[index_selected_inds]
        
    def getSolveInd(self):
        return self.ls_inds[int(np.where(self.factorial_rank == 1)[0])]

    def index(self, ind: Individual):
        for idx, e in self.ls_inds:
            if e is ind:
                return idx
        raise ValueError(str(ind) + "is not in subPop")

class Population:
    def __init__(self, IndClass: Type[Individual], dim, nb_inds_tasks: List[int], list_tasks:List[AbstractTask] = [], 
        evaluate_initial_skillFactor = False) -> None:
        '''
        A Population include:\n
        + `nb_inds_tasks`: number individual of tasks; nb_inds_tasks[i] = num individual of task i
        + `dim`: dimension of unified search space
        + `bound`: [lower_bound, upper_bound] of unified search space
        + `evaluate_initial_skillFactor`:
            + if True: individuals are initialized with skill factor most fitness with them
            + else: randomize skill factors for individuals
        '''
        assert len(nb_inds_tasks) == len(list_tasks)

        # save params
        self.ls_tasks = list_tasks
        self.nb_tasks = len(list_tasks)
        self.dim_uss = dim
        self.IndClass = IndClass

        if evaluate_initial_skillFactor:
            # empty population
            self.ls_subPop: List[SubPopulation] = [
                SubPopulation(IndClass, skf, self.dim_uss, 0, list_tasks[skf]) for skf in range(len(nb_inds_tasks))
            ]

            # list individual (don't have skill factor)
            ls_inds = [
                IndClass(genes= None, dim= self.dim_uss)
                for i in range(np.sum(nb_inds_tasks))
            ]
            # matrix factorial cost and matrix rank
            matrix_cost = np.array([[ind.eval(t) for ind in ls_inds] for t in list_tasks]).T
            matrix_rank_pop = np.argsort(np.argsort(matrix_cost, axis = 0), axis = 0) 

            count_inds = np.zeros((len(list_tasks),))
            
            while not np.all(count_inds == nb_inds_tasks) :
                # random task do not have enough individual
                idx_task = numba_randomchoice(np.where(count_inds < nb_inds_tasks)[0])

                # get best individual of task
                idx_ind = np.argsort(matrix_rank_pop[:, idx_task])[0]

                # set skill factor
                ls_inds[idx_ind].skill_factor = idx_task
                # add individual
                self.__addIndividual__(ls_inds[idx_ind])

                # set worst rank for ind
                matrix_rank_pop[idx_ind] = len(ls_inds) + 1
                count_inds[idx_task] += 1

            for i in range(len(list_tasks)):
                self.ls_subPop[i].update_rank()
        else:
            self.ls_subPop: List[SubPopulation] = [
                SubPopulation(IndClass, skf, self.dim_uss, nb_inds_tasks[skf], list_tasks[skf]) for skf in range(self.nb_tasks)
            ]

    def __len__(self):
        return sum([len(subPop) for subPop in self.ls_subPop])

    def __getitem__(self, index):
        return self.ls_subPop[index]

    def __getIndsTask__(self, idx_task, size: int = None, replace: bool = False, type:str = 'random', tournament_size= 2,  
        p_ontop = None,
        *args, **kwargs) :
        '''
        `type`: 'random', 'tournament', 'ontop'\n
        `random`: random individual from subPop\n
        `tournament`: Deterministic Tournament: select random `tourament_size` individuals, and return top `size` individuals
        ``
        '''

        if size == 0:
            return []
        elif type == 'random':
            return self.ls_subPop[idx_task].__getRandomItems__(size, replace)
        elif type == 'tournament':
            ls_inds = self.ls_subPop[idx_task].__getRandomItems__(tournament_size, False)
            if size is None:
                idx = np.argmin([ind.fcost for ind in ls_inds])
                return ls_inds[idx]
            elif size == 1:
                idx = np.argmin([ind.fcost for ind in ls_inds])
                return [ls_inds[idx]]
            else:
                idx_inds = np.argsort([ind.fcost for ind in ls_inds])
                return [ls_inds[i] for i in idx_inds[:size]] 
                       
        elif type == 'ontop':
            idx_inds = np.where(self.ls_subPop[idx_task].factorial_rank <=  max(p_ontop * len(self[idx_task]),2) )[0]
            if size == None:
                return self.ls_subPop[idx_task].ls_inds[
                    numba_randomchoice(idx_inds, size = None, replace= replace)
                ]

            return [self.ls_subPop[idx_task].ls_inds[idx] for idx in idx_inds[numba_randomchoice(len(idx_inds), size = size, replace= replace)].tolist()]
        else:
            raise ValueError('`type` ==  random | tournament | ontop, not equal ' + type)
        

    def __getRandomInds__(self, size: int = None, replace: bool = False):
        if size == None:
            return self.ls_subPop[np.random.randint(0, self.nb_tasks)].__getRandomItems__(None, replace) 
        else:
            nb_randInds = [0] * self.nb_tasks
            for idx in numba_randomchoice(self.nb_tasks, size = size, replace= True).tolist():
                nb_randInds[idx] += 1

            res = []
            for idx, nb_inds in enumerate(nb_randInds):
                res += self.ls_subPop[idx].__getRandomItems__(size = nb_inds, replace= replace)

            return res
        
    def __addIndividual__(self, individual:Individual, update_rank = False):
        self.ls_subPop[individual.skill_factor].__addIndividual__(individual, update_rank)

    def get_solves(self):
        return [subPop.getSolveInd() for subPop in self.ls_subPop]

    def update_rank(self):
        for subPop in self:
            subPop.update_rank()
        return None

    def __add__(self, other):
        assert self.nb_tasks == other.nb_tasks
        newPop = Population(
            IndClass= self.IndClass,
            dim = self.dim_uss,
            nb_inds_tasks= [0] * self.nb_tasks,
            list_tasks= self.ls_tasks
        )
        newPop.ls_subPop = [
            self.ls_subPop[idx] + other.ls_subPop[idx]
            for idx in range(self.nb_tasks)
        ]
        return newPop

# class LSHADE_Population(Population): 
#     def __init__(self, IndClass: Type[Individual], dim, nb_inds_tasks: List[int], list_tasks:List[AbstractTask] = [], evaluate_initial_skillFactor = False) -> None:
#         super().__init__(IndClass, dim, nb_inds_tasks, list_tasks, evaluate_initial_skillFactor = False)
