import numpy as np
from . import AbstractModel
from ..operators import Crossover, Mutation, Selection
from ..tasks.task import AbstractTask
from ..numba_utils import numba_randomchoice
from ..EA import *

class model(AbstractModel.model):
    def compile(self, 
        IndClass: Type[Individual],
        tasks: List[AbstractTask], 
        crossover: Crossover.SBX_Crossover, mutation: Mutation.GaussMutation, selection: Selection.ElitismSelection, 
        *args, **kwargs):
        return super().compile(IndClass, tasks, crossover, mutation, selection, *args, **kwargs)

    def fit(self,nb_generations, nb_inds_each_task: int, rmp = 0.3, evaluate_initial_skillFactor = False, *args, **kwargs):
        super().fit(*args, **kwargs)

        # initialize population
        population = Population(
            self.IndClass,
            nb_inds_tasks = [nb_inds_each_task] * len(self.tasks), 
            dim = self.dim_uss,
            list_tasks= self.tasks,
            evaluate_initial_skillFactor = evaluate_initial_skillFactor
        )
        
        # save history
        self.history_cost.append([ind.fcost for ind in population.get_solves()])

        self.render_process(0, ["Cost"], [self.history_cost[-1]], use_sys= True)

        for epoch in range(nb_generations):

            # initial offspring_population of generation
            offspring = Population(
                self.IndClass,
                nb_inds_tasks= [0] * len(self.tasks),
                dim =  self.dim_uss, 
                list_tasks= self.tasks,
            )

            # create offspring pop
            while len(offspring) < len(population): 
                # choose parent 
                pa, pb = population.__getRandomInds__(size= 2) 

                # crossover 
                if pa.skill_factor == pb.skill_factor or np.random.rand() < rmp: 
                    skf_oa, skf_ob = numba_randomchoice(np.array([pa.skill_factor, pb.skill_factor]), size= 2, replace= True)
                    oa, ob = self.crossover(pa, pb, skf_oa, skf_ob) 
                else: 
                    pa1 = population[pa.skill_factor].__getRandomItems__()
                    while pa1 is pa:
                        pa1 = population[pa.skill_factor].__getRandomItems__()
                    oa, _ = self.crossover(pa, pa1, pa.skill_factor, pa.skill_factor) 

                    pb1 = population[pb.skill_factor].__getRandomItems__()
                    while pb1 is pb:
                        pb1 = population[pb.skill_factor].__getRandomItems__()
                    ob, _ = self.crossover(pb, pb1, pb.skill_factor, pb.skill_factor) 
                
                # mutate
                oa = self.mutation(oa, return_newInd= False)
                ob = self.mutation(ob, return_newInd= False)    

                # eval and append # addIndividual already has eval  
                offspring.__addIndividual__(oa) 
                offspring.__addIndividual__(ob) 

            # merge and update rank
            population = population + offspring 
            population.update_rank()

            # selection 
            self.selection(population, [nb_inds_each_task] * len(self.tasks))

            # update operators
            self.crossover.update(population = population)
            self.mutation.update(population = population)
            
            # save history 
            self.history_cost.append([ind.fcost for ind in population.get_solves()])
            
            # # print 
            self.render_process((epoch + 1)/nb_generations, ["Cost"], [self.history_cost[-1]], use_sys= True)

        print('\nEND!')

        # solve 
        self.last_pop = population 
        return self.last_pop.get_solves()
