from typing import Tuple, Type, List
import numpy as np
import scipy.stats

from ..tasks.task import AbstractTask
from ..EA import Individual, Population

class AbstractSearch():
    def __init__(self) -> None:
        pass
    def __call__(self, *args, **kwargs) -> Individual:
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

class SHADE(AbstractSearch):
    def __init__(self, len_mem = 30, p_best_type:str = 'ontop', p_ontop = 0.1, tournament_size = 2) -> None:
        '''
        `p_best_type`: `random` || `tournament` || `ontop`
        '''
        super().__init__()
        self.len_mem = len_mem
        self.p_best_type = p_best_type
        self.p_ontop = p_ontop
        self.tournament_size = tournament_size

    def getInforTasks(self, IndClass: Type[Individual], tasks: List[AbstractTask], seed = None):
        super().getInforTasks(IndClass, tasks, seed= seed)
        # memory of cr and F
        self.M_cr = np.zeros(shape = (self.nb_tasks, self.len_mem, ), dtype= float) + 0.5
        self.M_F = np.zeros(shape= (self.nb_tasks, self.len_mem, ), dtype = float) + 0.5
        self.index_update = [0] * self.nb_tasks

        # memory of cr and F in epoch
        self.epoch_M_cr:List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_F: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()

        # memory of delta fcost p and o in epoch
        self.epoch_M_w: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
    
    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        for skf in range(self.nb_tasks):
            new_cr = self.M_cr[skf][self.index_update[skf]]
            new_F = self.M_F[skf][self.index_update[skf]]

            new_index = (self.index_update[skf] + 1) % self.len_mem

            if len(self.epoch_M_cr) > 0:
                new_cr = np.sum(np.array(self.epoch_M_cr[skf]) * (np.array(self.epoch_M_w[skf]) / (np.sum(self.epoch_M_w[skf]) + 1e-50)))
                new_F = np.sum(np.array(self.epoch_M_w[skf]) * np.array(self.epoch_M_F[skf]) ** 2) / \
                    (np.sum(np.array(self.epoch_M_w[skf]) * np.array(self.epoch_M_F[skf])) + 1e-50)
            
            self.M_cr[skf][new_index] = new_cr
            self.M_F[skf][new_index] = new_F

            self.index_update[skf] = new_index

        # reset epoch mem
        self.epoch_M_cr:List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_F: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_w: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        
    def __call__(self, ind: Individual, population: Population, *args, **kwargs) -> Individual:
        super().__call__(*args, **kwargs)
        # random individual
        ind_ran1, ind_ran2 = population.__getIndsTask__(ind.skill_factor, size = 2, replace= False, type= 'random')


        if np.all(ind_ran1.genes == ind_ran2.genes):
            ind_ran2 = population[ind.skill_factor].__getWorstIndividual__

        # get best individual
        ind_best = population.__getIndsTask__(ind.skill_factor, type = self.p_best_type, p_ontop= self.p_ontop, tournament_size= self.tournament_size)
        while ind_best is ind:
            ind_best = population.__getIndsTask__(ind.skill_factor, type = self.p_best_type, p_ontop= self.p_ontop, tournament_size= self.tournament_size)


        k = np.random.choice(self.len_mem)
        cr = np.clip(np.random.normal(loc = self.M_cr[ind.skill_factor][k], scale = 0.1), 0, 1)

        F = 0
        while F <= 0 or F > 1:
            F = scipy.stats.cauchy.rvs(loc= self.M_F[ind.skill_factor][k], scale= 0.1) 
    
        u = (np.random.uniform(size = self.dim_uss) < cr)
        if np.sum(u) == 0:
            u = np.zeros(shape= (self.dim_uss,))
            u[np.random.choice(self.dim_uss)] = 1

        new_genes = np.where(u, 
            ind_best.genes + F * (ind_ran1.genes - ind_ran2.genes),
            ind.genes
        )
        new_genes = np.clip(new_genes, 0, 1)

        new_ind = self.IndClass(new_genes)
        new_ind.skill_factor = ind.skill_factor
        new_ind.fcost = new_ind.eval(self.tasks[new_ind.skill_factor])

        # save memory
        delta = ind.fcost - new_ind.fcost
        if delta > 0:
            self.epoch_M_cr[ind.skill_factor].append(cr)
            self.epoch_M_F[ind.skill_factor].append(F)
            self.epoch_M_w[ind.skill_factor].append(delta)

        return new_ind


class L_SHADE(AbstractSearch):
    def __init__(self, len_mem = 30, p_ontop = 0.1, tournament_size = 2) -> None:
        '''
        `p_best_type`: `random` || `tournament` || `ontop`
        '''
        super().__init__()
        self.len_mem = len_mem
        self.p_ontop = p_ontop
        self.tournament_size = tournament_size

    def getInforTasks(self, IndClass: Type[Individual], tasks: List[AbstractTask], seed = None):
        super().getInforTasks(IndClass, tasks, seed= seed)
        # memory of cr and F
        self.M_cr = np.zeros(shape = (self.nb_tasks, self.len_mem, ), dtype= float) + 0.5
        self.M_F = np.zeros(shape= (self.nb_tasks, self.len_mem, ), dtype = float) + 0.5
        self.index_update = [0] * self.nb_tasks

        # memory of cr and F in epoch
        self.epoch_M_cr:List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_F: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()

        # memory of delta fcost p and o in epoch
        self.epoch_M_w: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
    
    def update(self, *args, **kwargs) -> None:
        super().update(*args, **kwargs)
        for skf in range(self.nb_tasks):
            new_cr = self.M_cr[skf][self.index_update[skf]]
            new_F = self.M_F[skf][self.index_update[skf]]

            new_index = (self.index_update[skf] + 1) % self.len_mem

            if len(self.epoch_M_cr) > 0:
                new_cr = np.sum(np.array(self.epoch_M_cr[skf]) * (np.array(self.epoch_M_w[skf]) / (np.sum(self.epoch_M_w[skf]) + 1e-50)))
                new_F = np.sum(np.array(self.epoch_M_w[skf]) * np.array(self.epoch_M_F[skf]) ** 2) / \
                    (np.sum(np.array(self.epoch_M_w[skf]) * np.array(self.epoch_M_F[skf])) + 1e-50)
            
            self.M_cr[skf][new_index] = new_cr
            self.M_F[skf][new_index] = new_F

            self.index_update[skf] = new_index

        # reset epoch mem
        self.epoch_M_cr:List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_F: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_w: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        
    def __call__(self, ind: Individual, population: Population, *args, **kwargs) -> Individual:
        super().__call__(*args, **kwargs)
        # random individual
        ind_ran1, ind_ran2 = population.__getIndsTask__(ind.skill_factor, size = 2, replace= False, type= 'random')


        if np.all(ind_ran1.genes == ind_ran2.genes):
            ind_ran2 = population[ind.skill_factor].__getWorstIndividual__

        # get best individual
        ind_best = population.__getIndsTask__(ind.skill_factor, type = 'ontop', p_ontop= self.p_ontop)

        if ind_best is ind:
            if len(population[ind.skill_factor]) * self.p_ontop < 2:
                while ind_best is ind:
                    ind_best = population.__getIndsTask__(ind.skill_factor, type = 'tournament', tournament_size= self.tournament_size)

            else:
                while ind_best is ind:
                    ind_best = population.__getIndsTask__(ind.skill_factor, type = 'ontop', p_ontop= self.p_ontop)


        k = np.random.choice(self.len_mem)
        cr = np.clip(np.random.normal(loc = self.M_cr[ind.skill_factor][k], scale = 0.1), 0, 1)

        F = 0
        while F <= 0 or F > 1:
            F = scipy.stats.cauchy.rvs(loc= self.M_F[ind.skill_factor][k], scale= 0.1) 
    
        u = (np.random.uniform(size = self.dim_uss) < cr)
        if np.sum(u) == 0:
            u = np.zeros(shape= (self.dim_uss,))
            u[np.random.choice(self.dim_uss)] = 1

        new_genes = np.where(u, 
            ind.genes + F * (ind_best.genes - ind.genes + ind_ran1.genes - ind_ran2.genes),
            ind.genes
        )

        # u = np.random.rand(self.dim_uss)
        # tmp = ind.genes * u
        # new_genes = np.where(new_genes > 1,tmp + 1 - u, new_genes) 
        # new_genes = np.where(new_genes < 0, tmp, new_genes) 

        new_genes = np.where(new_genes > 1, (ind.genes + 1)/2, new_genes) 
        new_genes = np.where(new_genes < 0, ind.genes / 2, new_genes) 

        new_ind = self.IndClass(new_genes)
        new_ind.skill_factor = ind.skill_factor
        new_ind.fcost = new_ind.eval(self.tasks[new_ind.skill_factor])

        # save memory
        delta = ind.fcost - new_ind.fcost
        if delta > 0:
            self.epoch_M_cr[ind.skill_factor].append(cr)
            self.epoch_M_F[ind.skill_factor].append(F)
            self.epoch_M_w[ind.skill_factor].append(delta)

        return new_ind

class LocalSearch_DSCG(AbstractSearch):
    def __init__(self) -> None:
        super().__init__() 
        self.INIT_STEP_SIZE= 0.02
        self.EPSILON= 1e-8 
        self.EVALS_PER_LINE_SEARCH= 50 
    
    def getInforTasks(self, IndClass: Type[Individual], tasks: List[AbstractTask], seed=None):
        return super().getInforTasks(IndClass, tasks, seed)
    
    def update(self, *args, **kwargs) -> None:
        return super().update(*args, **kwargs)
    
    def search(self, start_point: Individual, fes: int, *args, **kwargs) -> Individual:
        s: float = self.INIT_STEP_SIZE 
        evals_per_linesarch= self.EVALS_PER_LINE_SEARCH 

        result = self.IndClass(start_point.genes, dim= self.dim_uss) 
        result.fcost = start_point.fcost 

        x: List[Individual] = [self.IndClass(genes = None, dim= self.dim_uss) for i in range(self.dim_uss + 2)]

        # x.append(self.IndClass(start_point.genes))
        x[0] = self.IndClass(start_point.genes)
        x[0].fcost = start_point.fcost 

        direct = 1
        evals= 0
        v = np.eye(N= self.dim_uss + 1, M = self.dim_uss)

        while True: 
            a = np.zeros(shape= (self.dim_uss, self.dim_uss)) 

            while (evals < fes - evals_per_linesarch) : 
                evals, x[direct] = self.lineSearch(x[direct-1], evals,  self.EVALS_PER_LINE_SEARCH, self.tasks[start_point.skill_factor], s, v[direct-1]) 
                # x[direct] = result

                for i in range(1, direct + 1, 1) : 
                    for j in range(self.dim_uss): 
                        a[i-1][j] += x[direct].genes[j] - x[direct-1].genes[j] 
                
                if result.fcost > x[direct].fcost : 
                    result.genes = np.copy(x[direct].genes) 
                    result.fcost = np.copy(x[direct].fcost) 
                
                if (direct < self.dim_uss): 
                    direct += 1 
                else: 
                    break 
                pass 
            
            if evals >= fes or direct < self.dim_uss : 
                break 
            
            z = np.zeros(shape= (self.dim_uss,)) 
            norm_z = 0 

            z = x[self.dim_uss].genes - x[0].genes 
            norm_z = np.sum(z ** 2) 

            norm_z = np.sqrt(norm_z) 

            if (norm_z == 0):
                x[self.dim_uss + 1].genes = np.copy(x[self.dim_uss].genes )
                x[self.dim_uss + 1].fcost = np.copy(x[self.dim_uss].fcost)

                s *= 0.1 
                if (s <= self.EPSILON):
                    break 
                else: 
                    direct = 1 

                    x[0].genes = np.copy(x[self.dim_uss + 1].genes) 
                    x[0].fcost = np.copy(x[self.dim_uss + 1].fcost) 

                    continue 
            else: 
                v[self.dim_uss] = z / norm_z 

                direct = self.dim_uss + 1 

                rest_eval = fes - evals 
                overall_ls_eval =0 
                
                if rest_eval < evals_per_linesarch: 
                    overall_ls_eval = rest_eval 
                else: 
                    overall_ls_eval = evals_per_linesarch 
                
                evals, x[direct] = self.lineSearch(x[direct-1], evals, overall_ls_eval, self.tasks[start_point.skill_factor], s, v[direct-1])

                if result.fcost > x[direct].fcost: 
                    result.fcost = np.copy(x[direct].fcost) 
                    result.genes = np.copy(x[direct].genes) 
                
                norm_z = 0 
                norm_z = np.sum((x[direct].genes - x[0].genes) ** 2) 
                norm_z = np.sqrt(norm_z)

                if norm_z < s: 
                    s *= 0.1 
                    if s <= self.EPSILON: 
                        break 
                    else: 
                        direct= 1 

                        x[0].genes = np.copy(x[self.dim_uss + 1].genes) 
                        x[0].fcost = np.copy(x[self.dim_uss + 1].fcost) 

                        continue 
                else: 
                    direct = 2 
                    x[0].genes = np.copy(x[self.dim_uss].genes) 
                    x[1].genes = np.copy(x[self.dim_uss + 1].genes)

                    x[0].fcost = np.copy(x[self.dim_uss].fcost) 
                    x[1].fcost = np.copy(x[self.dim_uss + 1].fcost)

                    continue 


        if result.fcost > x[self.dim_uss +1].fcost: 
            return evals, x[self.dim_uss + 1]
        
        return evals, result
    def lineSearch(self, start_point: Individual, eval: int,  fes: int, task: AbstractTask, step_size: int, v: np.array) :

        result: Individual = self.IndClass(genes = None, dim = self.dim_uss)

        evals= eval 
        s = step_size 
        change: bool = False 
        interpolation_flag = False 

        x0 = self.IndClass(start_point.genes)
        x0.fcost = np.copy(start_point.fcost)

        x = self.IndClass(x0.genes + s * v) 
        x.fcost = x.eval(task) 
        evals += 1 

        F = np.zeros(shape=(3,))
        interpolation_points = np.zeros(shape=(3, self.dim_uss))


        interpolation_points[0] = np.copy(x0.genes) 
        interpolation_points[1] = np.copy(x.genes)

        F[0] = x0.fcost 
        F[1] = x.fcost 

        if x.fcost > x0.fcost: 
            x.genes = x.genes - 2 * s * v 
            s = -s 

            x.fcost = x.eval(task) 
            evals += 1 

            if x.fcost <= x0.fcost: 
                change= True 
                interpolation_points[0] = np.copy(x0.genes) 
                interpolation_points[1] = np.copy(x.genes) 

                F[0] = x0.fcost 
                F[1] = x.fcost 
            else: 
                change= False 
                interpolation_flag = True 

                interpolation_points[2] = np.copy(interpolation_points[1]) 
                interpolation_points[1] = np.copy(interpolation_points[0]) 
                interpolation_points[0] = np.copy(x.genes) 

                F[2] = F[1] 
                F[1] = F[0] 
                F[0] = x.fcost 
        else: 
            change= True 
        
        while change: 
            s *= 2 

            x0.genes = np.copy(x.genes) 
            x0.fcost = np.copy(x.fcost) 

            x.genes = x0.genes + s * v
            x.fcost = x.eval(task) 
            evals +=1 

            if x.fcost < x0.fcost : 
                interpolation_points[0] = np.copy(x0.genes) 
                interpolation_points[1] = np.copy(x.genes) 

                F[0] = x0.fcost 
                F[1] = x.fcost 

            else: 
                change= False 
                interpolation_flag = True 

                interpolation_points[2] = np.copy(x.genes) 
                F[2] = x.fcost 

                # generate x = x0 + 0.5s 
                s *= 0.5 
                x.genes = x0.genes + s * v 
                x.fcost = x.eval(task) 
                evals += 1 

                if x.fcost > F[1] : 
                    interpolation_points[2] = np.copy(x.genes) 
                    F[2]= x.fcost 
                else: 
                    interpolation_points[0] = np.copy(interpolation_points[1])
                    interpolation_points[1] = np.copy(x.genes)
                    
                    F[0] = F[1] 
                    F[1] = x.fcost 

            if (evals >= fes -2): 
                change = False 


        if (interpolation_flag and ((F[0] - 2 * F[1] + F[2]) != 0)) : 
        
            x.genes = interpolation_points[1] + s * (F[0] - F[2]) / ( 2.0 * (F[0] - 2 * F[1] + F[2])) 
            x.fcost = x.eval(task) 
            evals += 1 

            if x.fcost < F[1] : 
                result.genes = np.copy(x.genes) 
                result.fcost = np.copy(x.fcost) 
            else: 
                result.genes = np.copy(interpolation_points[1]) 
                result.fcost = F[1] 
        else : 
            result.genes = np.copy(interpolation_points[1]) 
            result.fcost = F[1] 
        
        return evals, result 


class LSHADE_LSA21(AbstractSearch): 
    def __init__(self, len_mem = 30, p_ontop = 0.1) -> None:
        super().__init__()
        self.len_mem = len_mem 
        self.p_ontop = p_ontop 
        self.archive: List[List[Individual]] = None 
        self.arc_rate = 5 

        self.first_run = True 



    def getInforTasks(self, IndClass: Type[Individual], tasks: List[AbstractTask], seed=None):
        super().getInforTasks(IndClass, tasks, seed)

        # memory of cr and F
        self.M_cr = np.zeros(shape = (self.nb_tasks, self.len_mem, ), dtype= float) + 0.5
        self.M_F = np.zeros(shape= (self.nb_tasks, self.len_mem, ), dtype = float) + 0.5
        self.index_update = [0] * self.nb_tasks

        self.archive = np.empty(shape= (self.nb_tasks, 0)).tolist() 

        # memory of cr and F in epoch
        self.epoch_M_cr:List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_F: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()

        # memory of delta fcost p and o in epoch
        self.epoch_M_w: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
    
    def __call__(self,ind: Individual, population: Population, *args, **kwargs) -> Individual: 
        super().__call__(*args, **kwargs)

        k = np.random.choice(self.len_mem)
        cr = np.clip(np.random.normal(loc = self.M_cr[ind.skill_factor][k], scale = 0.1), 0, 1)

        F = 0
        while F <= 0:
            F = scipy.stats.cauchy.rvs(loc= self.M_F[ind.skill_factor][k], scale= 0.1) 
        
        if F >1: 
            F = 1 
    
        u = (np.random.uniform(size = self.dim_uss) < cr)
        if np.sum(u) == 0:
            u = np.zeros(shape= (self.dim_uss,))
            u[np.random.choice(self.dim_uss)] = 1

                # get best individual
        ind_best = population.__getIndsTask__(ind.skill_factor, p_ontop= self.p_ontop)
        while ind_best is ind:
            ind_best = population.__getIndsTask__(ind.skill_factor, p_ontop= self.p_ontop)
        
        ind1 = ind_best 
        while ind1 is ind_best or ind1 is ind : 
            ind1 = population.__getIndsTask__(ind.skill_factor, type='random') 
        

        if self.first_run is False and np.random.rand() < len(self.archive[ind.skill_factor]) / (len(self.archive[ind.skill_factor]) + len(population[ind.skill_factor])): 
            ind2 = self.archive[ind.skill_factor][np.random.choice(len(self.archive[ind.skill_factor]))]
        else: 
            ind2 = ind1 
            while ind2 is ind_best or ind2 is ind1 or ind2 is ind: 
                ind2 = population.__getIndsTask__(ind.skill_factor, type='random') 
        


        new_genes = np.where(u, 
            ind.genes + F * (ind_best.genes - ind.genes + ind1.genes - ind2.genes),
            ind.genes
        )
        new_genes = np.where(new_genes > 1, (ind.genes + 1)/2, new_genes) 
        new_genes = np.where(new_genes < 0, (ind.genes + 0)/2, new_genes) 

        new_ind = self.IndClass(new_genes)
        new_ind.skill_factor = ind.skill_factor
        new_ind.fcost = new_ind.eval(self.tasks[new_ind.skill_factor])

        # save memory 
        delta = ind.fcost - new_ind.fcost 
        if delta == 0 : 
            return new_ind 
        elif delta > 0: 
            self.epoch_M_cr[ind.skill_factor].append(cr)
            self.epoch_M_F[ind.skill_factor].append(F)
            self.epoch_M_w[ind.skill_factor].append(delta)

            if len(self.archive[ind.skill_factor]) < self.arc_rate * len(population[ind.skill_factor]): 
                self.archive[ind.skill_factor].append(ind)
            else: 
                del self.archive[ind.skill_factor][np.random.choice(len(self.archive[ind.skill_factor]))]
                self.archive[ind.skill_factor].append(ind)
            return new_ind 
        else: 
            return ind 


    def update(self, population, *args, **kwargs) -> None:
        self.first_run = False 
        for skf in range(self.nb_tasks): 
            if(len(self.epoch_M_cr[skf])) > 0: 
                sum_diff = np.sum(np.array(self.epoch_M_w[skf])) 
                w = np.array(self.epoch_M_w[skf]) / sum_diff 

                tmp_sum_cr = np.sum(w * np.array(self.epoch_M_cr[skf]))
                tmp_sum_f = np.sum(w * np.array(self.epoch_M_F[skf])) 


                self.M_F[skf][self.index_update[skf]] = np.sum(w * np.array(self.epoch_M_F[skf]) ** 2) / tmp_sum_f

                if (tmp_sum_cr == 0): 
                    self.M_cr[skf][self.index_update[skf]] = -1 
                else: 
                    self.M_cr[skf][self.index_update[skf]] = np.sum(w * np.array(self.epoch_M_cr[skf]) ** 2) / tmp_sum_cr 
                
                self.index_update[skf] = (self.index_update[skf] + 1) % self.len_mem
            
        
        # reset epoch mem
        self.epoch_M_cr:List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_F: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()
        self.epoch_M_w: List[list] = np.empty(shape = (self.nb_tasks, 0)).tolist()

        # update archive size
        for skf in range(self.nb_tasks): 
            while len(self.archive[skf]) > len(population[skf]) * self.arc_rate: 
                del self.archive[skf][np.random.choice(len(self.archive[skf]))]
