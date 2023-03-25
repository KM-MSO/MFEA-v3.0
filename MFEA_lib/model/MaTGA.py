import numpy as np
import copy
from MFEA_lib.model import AbstractModel
from MFEA_lib.operators import Crossover, Mutation, Selection
from MFEA_lib.tasks.task import AbstractTask
from MFEA_lib.numba_utils import *
from MFEA_lib.EA import *
import time

class model(AbstractModel.model):
    '''
    An Adaptive Archive-Based Evolutionary Framework for Many-Task Optimization.
    '''
    def compile(self, 
                IndClass: Type[Individual], 
                tasks: List[AbstractTask], 
                crossover: Crossover.AbstractCrossover, 
                mutation: Mutation.AbstractMutation, 
                selection: Selection.AbstractSelection, 
                *args, 
                **kwargs):
        super().compile(IndClass, tasks, crossover, mutation, selection, *args, **kwargs)
        
    def fit(self, 
            nb_generations, 
            rmp = 0.1, 
            nb_inds_each_task = 100, 
            ro = 0.8,
            shrink_rate = 0.8,
            max_archive_size = 300,
            replace_rate = 0.2,
            evaluate_initial_skillFactor = True, 
            *args, 
            **kwargs) -> List[Individual]:
        '''
        Arguments include:\n
        + `nb_generations`: number of generations
        + `rmp`: random mating probability
        + `nb_inds_tasks`: number of individual per task; nb_inds_tasks[i] = num individual of task i
        + `ro`: 
        + `shrink_rate`: 
        + `max_archive_size`: maximum size of each archive population
        + `replace_rate`: 
        + `evaluate_initial_skillFactor`:
            + if True: individuals are initialized with skill factor most fitness with them
            + else: randomize skill factors for individuals
        '''

        super().fit(*args, **kwargs)
        
        assert max_archive_size >= nb_inds_each_task

        self.max_archive_size = max_archive_size
        self.nb_inds_each_task = nb_inds_each_task
        self.replace_rate = replace_rate
        self.ro = ro
        
        # Initial population
        self.population = Population(
            self.IndClass,
            nb_inds_tasks = [nb_inds_each_task] * len(self.tasks), 
            dim = self.dim_uss,
            list_tasks = self.tasks,
            evaluate_initial_skillFactor = evaluate_initial_skillFactor
        )

        # Initial archive population
        self.archive_size = [nb_inds_each_task] * len(self.tasks)
        self.archive_population: List[SubPopulation] = [SubPopulation(self.IndClass, skill_factor=i, dim=self.dim_uss, num_inds=nb_inds_each_task, task=task) \
                                                        for i, task in enumerate(self.tasks)]

        # Update initial archives
        self.update_archives(init=True)
        
        # save history
        self.history_cost.append([ind.fcost for ind in self.population.get_solves()])
        
        self.render_process(0, ['Cost'], [self.history_cost[-1]], use_sys = True)
        
        # Reward
        self.R = np.ones((len(self.tasks), len(self.tasks)))

        # Roulette wheel probability
        self.probability = np.zeros((len(self.tasks), len(self.tasks)))
        
        for epoch in range(nb_generations):
            
            for t in range(len(self.tasks)):
                
                if (random.random() > rmp): 
                    # Intra-task transfer

                    # initial offspring_population of generation
                    offsprings = Population(
                        self.IndClass,
                        nb_inds_tasks = [0] * len(self.tasks), 
                        dim = self.dim_uss,
                        list_tasks= self.tasks,
                    )

                    # update operators
                    self.crossover.update(population = self.population)
                    self.mutation.update(population = self.population)

                    for i in range(int(nb_inds_each_task/2)):
                        pa, pb = self.population[t].__getRandomItems__(2)

                        if random.random() < 0.9:
                            oa, ob = self.crossover(pa, pb, t, t)           
                        else:  
                            oa, ob = pa, pb

                        # mutate
                        oa = self.mutation(oa, return_newInd = True)
                        oa.skill_factor = t

                        ob = self.mutation(ob, return_newInd = True)    
                        ob.skill_factor = t

                        offsprings.__addIndividual__(oa)
                        offsprings.__addIndividual__(ob)

                    self.population = self.population + offsprings

                    # Update rank
                    self.population.update_rank()

                    # selection
                    self.selection(self.population, [nb_inds_each_task] * len(self.tasks))
                
                else:
                    # Inter-task transfer
                    
                    # Find the most assisting task 
                    t_j = self.adaptive_choose(t)

                    assert t_j != t, "Something wrong"

                    s = self.population[t].__getBestIndividual__.fcost
                    tmpS = s

                    for i in range(len(self.population[t])):
                        genes = copy.copy(self.population[t][i].genes)

                        k = random.randint(0, self.dim_uss - 1)
                        r1 = numba_randomchoice(len(self.population[t_j]))
                        
                        idx = np.where(np.random.rand(self.dim_uss) < 0.9)[0]
                        
                        genes[idx] = copy.copy(self.population[t_j][r1].genes[idx])
                        genes[k] = copy.copy(self.population[t_j][r1].genes[k])
                        
                        fitness = self.tasks[t](genes)
                        tmpS = min(tmpS, fitness)
                        
                        if (fitness < self.population[t][i].fcost):
                            off = self.IndClass(genes = genes)
                            off.skill_factor = t
                            off.fcost = fitness
                            self.population[t][i] = off
                            
                    if (tmpS < s):
                        self.R[t][t_j] /= shrink_rate
                    else:
                        self.R[t][t_j] *= shrink_rate
            
            # Update rank
            self.population.update_rank()

            # Update archive population
            self.update_archives()

            # save history
            self.history_cost.append([ind.fcost for ind in self.population.get_solves()])

            #print
            self.render_process((epoch+1)/nb_generations, ['Cost'], [self.history_cost[-1]], use_sys= True)
        print('\nEND!')
        
        #solve
        self.last_pop = self.population
        return self.last_pop.get_solves() 

    def update_archives(self, init = True):
        '''
        Update every archive population
        '''

        if init == True:
            for t in range(len(self.tasks)):
                self.archive_population[t] = copy.copy(self.population[t])
        else:
            for t in range(len(self.tasks)):
                self.update_archive(t)
    
    def update_archive(self, task):
        '''
        Update one archive population
        '''
        
        remain_size = self.max_archive_size - self.archive_size[task]
        u = np.random.rand(self.nb_inds_each_task)
        idx = np.where(u < self.replace_rate)[0]
        
        assert len(self.archive_population[task]) == self.archive_size[task], "Wrong shape, got {} and {}".format(len(self.archive_population[task]), self.archive_size[task])

        if len(idx) <= remain_size:
            self.archive_population[task] += copy.copy(self.population[task][idx])
            self.archive_size[task] += len(idx)
        else:
            self.archive_population[task] += copy.copy(self.population[task][idx[:remain_size]])
            r = np.random.randint(0, self.max_archive_size, size = len(idx) - remain_size).tolist()
            self.archive_population[task][r] = copy.copy(self.population[task][idx[remain_size:]])
            self.archive_size[task] = self.max_archive_size

        assert len(self.archive_population[task]) == self.archive_size[task], "Wrong shape, got {} and {}".format(len(self.archive_population[task]), self.archive_size[task])
    
    def adaptive_choose(self, task):
        """"
        Adaptively choose another task
        """

        # The similarity vector
        sim_vec = np.array([self.find_sim(task, i) for i in range(len(self.tasks))])

        self.probability[task] = self.ro * self.probability[task] + self.R[task]/(1 + np.log(1 + sim_vec))

        self.probability[task][task] = 0

        return numba_randomchoice_w_prob(self.probability[task]/sum(self.probability[task]))
    
    def find_sim(self, task_a, task_b):
        """"
        Find similarity score
        """

        if (task_a == task_b):
            return 0
        
        subPop_a: SubPopulation = self.archive_population[task_a]
        subPop_b: SubPopulation = self.archive_population[task_b]
        D = min(self.tasks[task_a].dim, self.tasks[task_b].dim)

        genes_a = np.array([ind.genes[:D] for ind in subPop_a])
        genes_b = np.array([ind.genes[:D] for ind in subPop_b])
        
        similarity = self.KLD(genes_a, genes_b)
        assert not np.isnan(similarity)

        return similarity

    def KLD(self, genes_a, genes_b):
        """
        Find KLD(archive_a, archive_b)
        """
        
        gene_cov_a, gene_mean_a = np.cov(genes_a, rowvar = False), np.mean(genes_a, axis = 0)
        gene_cov_b, gene_mean_b = np.cov(genes_b, rowvar = False), np.mean(genes_b, axis = 0)

        assert gene_cov_b.shape[1] == gene_cov_a.shape[0], "imcompatible shape. Got {} and {}".format(gene_cov_b.shape, gene_cov_a.shape)

        inv_gene_cov_b = numba_linalgo_pinv(gene_cov_b)
        inv_gene_cov_a = numba_linalgo_pinv(gene_cov_a)

        D = gene_cov_a.shape[0]

        det_b = max(numba_linalgo_det(gene_cov_b), 0.001)
        det_a = max(numba_linalgo_det(gene_cov_a), 0.001)

        kld_a_b = np.trace(numba_dot(inv_gene_cov_b, gene_cov_a)) \
                + numba_dot(numba_dot(np.transpose(gene_mean_b - gene_mean_a), inv_gene_cov_b), (gene_mean_b - gene_mean_a)) - D\
                + np.log(det_b/det_a)
        
        kld_b_a = np.trace(numba_dot(inv_gene_cov_a, gene_cov_b)) \
                + numba_dot(numba_dot(np.transpose(gene_mean_a - gene_mean_b), inv_gene_cov_a), (gene_mean_a - gene_mean_b)) - D\
                + np.log(det_a/det_b)

        return 0.5 * (np.abs(0.5 * kld_a_b) + np.abs(0.5 * kld_b_a))