from typing import Tuple
import numpy as np
from ..operators import Crossover, Mutation, Selection
from ..tasks.task import AbstractTask
from ..EA import *
import sys
import matplotlib.pyplot as plt
import random
import time 
from IPython.display import display, clear_output

class model():
    def __init__(self, seed = None, percent_print = 0.5) -> None:
        # initial history of factorial cost
        self.history_cost: List[float] = []
        self.solve: List[Individual]  
        self.seed = None 
        self.seed = seed
        if seed is None:
            pass
        else:
            # not work
            np.random.seed(seed)
            random.seed(seed)
        self.seed = seed

        # Add list abstract 
        self.result = None 
        self.ls_attr_avg = ["history_cost"]
        
        self.generations = 100 # represent for 100% 
        self.display_time = True 
        self.clear_output = True 
        self.count_pre_line = 0
        self.printed_before_percent = -2
        self.percent_print = percent_print 
        

    def render_history(self, shape: Tuple[int, int] = None, min_cost = 1e-6,title = "", yscale = None, ylim: Tuple[float, float] = None, re_fig = False):
        if shape is None:
            shape = (int(np.ceil(len(self.tasks) / 3)), 3)
        else:
            assert shape[0] * shape[1] >= len(self.tasks)
        fig = plt.figure(figsize= (shape[1]* 6, shape[0] * 5))
        fig.suptitle(title, size = 20)
        fig.set_facecolor("white")

        np_his_cost = np.array(np.where(np.array(self.history_cost) >= min_cost, np.array(self.history_cost), 0))
        for i in range(np_his_cost.shape[1]):
            plt.subplot(shape[0], shape[1], i+1)

            plt.plot(np.arange(np_his_cost.shape[0]), np_his_cost[:, i])

            plt.title(self.tasks[i].name)
            plt.xlabel("Generations")
            plt.ylabel("Factorial Cost")
            
            if yscale is not None:
                plt.yscale(yscale)
            if ylim is not None:
                plt.ylim(bottom = ylim[0], top = ylim[1])
                
        plt.show()
        if re_fig:
            return fig
    
    def compile(self, 
        IndClass: Type[Individual],
        tasks: List[AbstractTask], 
        crossover: Crossover.AbstractCrossover, 
        mutation: Mutation.AbstractMutation, 
        selection: Selection.AbstractSelection,
        *args, **kwargs):
    
        self.IndClass = IndClass
        self.tasks = tasks
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        
        self.args = args 
        self.kwargs = kwargs

        # get info from tasks
        self.dim_uss = max([t.dim for t in tasks])
        self.crossover.getInforTasks(IndClass, tasks, seed = self.seed)
        self.mutation.getInforTasks(IndClass, tasks, seed = self.seed)
        self.selection.getInforTasks(tasks, seed = self.seed)
        
        # test Ind and task
        test_pop = Population(
            self.IndClass,
            nb_inds_tasks = [10] * len(self.tasks), 
            dim = self.dim_uss,
            list_tasks= self.tasks,
            evaluate_initial_skillFactor = True
        )

        # test crossover
        pa, pb = test_pop.__getRandomInds__(2)
        self.crossover(pa, pb, pa.skill_factor, pb.skill_factor)

        # test mutation
        self.mutation(pa, return_newInd= True)
        self.mutation(pa, return_newInd= False)

        # test selection
        # self.selection(test_pop, [5] * len(self.tasks))



    def render_process(self,curr_progress, list_desc, list_value, use_sys = False,print_format_e = True,  *args, **kwargs):
        percent = int(curr_progress * 100)
        if percent >= 100: 
            self.time_end = time.time() 
            percent = 100 
        else: 
            if percent - self.printed_before_percent >= self.percent_print:
                self.printed_before_percent = percent 
            else: 
                return 
                
        process_line = '%3s %% [%-20s]  '.format() % (percent, '=' * int(percent / 5) + ">")
        
        seconds = time.time() - self.time_begin  
        minutes = seconds // 60 
        seconds = seconds - minutes * 60 
        print_line = str("")
        if self.clear_output is True: 
            if use_sys is True: 
                # os.system("cls")
                pass
            else:
                clear_output(wait= True) 
        if self.display_time is True: 
            if use_sys is True: 
                # sys.stdout.write("\r" + "time: %02dm %.02fs  "%(minutes, seconds))
                # sys.stdout.write(process_line+ " ")
                seed_line= "Seed: " + str(self.seed) + " -- "
                print_line =seed_line + print_line + "Time: %02dm %2.02fs "%(minutes, seconds) + " " +process_line
                
            else: 
                display("Time: %02dm %2.02fs "%(minutes, seconds))
                display(process_line)
        for i in range(len(list_desc)):
            desc = str("")
            for value in range(len(list_value[i])):
                if print_format_e: 
                    desc = desc + str("%.2E " % (list_value[i][value])) + " "
                else: 
                    desc = desc + str(list_value[i][value]) + " "
            line = '{}: {},  '.format(list_desc[i], desc)
            if use_sys is True: 
                print_line = print_line + line 
            else: 
                display(line)
        if use_sys is True: 
            # sys.stdout.write("\033[K")
            sys.stdout.flush() 
            sys.stdout.write("\r" + print_line)
            sys.stdout.flush() 
    
    def render_process_terminal(self,curr_progress, list_desc, list_value, *args, **kwargs):
        percent = int(curr_progress * 100 +1)
        if percent >= 100: 
            self.time_end = time.time() 
            percent = 100 
        process_line = 'Epoch [{} / {}], [%-20s]'.format(percent, 100) % ('=' * (int(curr_progress * 20)+1) + ">")
        
        CURSOR_UP = '\033[F'
        ERASE_LINE = '\033[K'
        print(CURSOR_UP * int(self.count_pre_line+1))

        seconds = time.time() - self.time_begin  
        minutes = seconds // 60 
        seconds = seconds - minutes * 60 

        print(ERASE_LINE + process_line)
        print(ERASE_LINE + "time: %02dm %.02fs"%(minutes, seconds))
        if self.display_time is True: 
            pass
        for i in range(len(list_desc)):
            desc = str("")
            for value in range(len(list_value[i])):
                desc = desc + str("%.2E " % (list_value[i][value])) + " "
            line = '{}: {}'.format(list_desc[i], desc)
            print(ERASE_LINE+ line)
        
        self.count_pre_line = 2 + len(list_desc)
            

    def fit(self, *args, **kwargs) -> List[Individual] :
        self.time_begin = time.time()
        print('Checking...', end='\r')
        pass

