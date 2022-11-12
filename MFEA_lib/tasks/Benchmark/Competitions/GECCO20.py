from .Default_Ind import Individual_func
from .utils import *

from typing import Tuple, Type, List
import numpy as np
import os

path = os.path.dirname(os.path.realpath(__file__))


class GECCO20_benchmark_50tasks():
    task_size = 50
    dim = 50

    def get_choice_function(ID) -> List[int]:
        choice_functions = []
        if ID == 1:
            choice_functions = [1]
        elif ID == 2:
            choice_functions = [2]
        elif ID == 3:
            choice_functions = [4]
        elif ID == 4:
            choice_functions = [1, 2, 3]
        elif ID == 5:
            choice_functions = [4, 5, 6]
        elif ID == 6:
            choice_functions = [2, 5, 7]
        elif ID == 7:
            choice_functions = [3, 4, 6]
        elif ID == 8:
            choice_functions = [2, 3, 4, 5, 6]
        elif ID == 9:
            choice_functions = [2, 3, 4, 5, 6, 7]
        elif ID == 10:
            choice_functions = [3, 4, 5, 6, 7]
        else:
            raise ValueError("Invalid input: ID should be in [1,10]")
        return choice_functions

    def get_items(ID, fix = False) -> Tuple[List[AbstractFunc], Type[Individual_func]]:
        choice_functions = __class__.get_choice_function(ID)

        tasks = []

        for task_id in range(__class__.task_size):
            func_id = choice_functions[task_id % len(choice_functions)]
            file_dir = path + "/__references__/GECCO20/Tasks/benchmark_" + str(ID)
            shift_file = "/bias_" + str(task_id + 1)
            rotation_file = "/matrix_" + str(task_id + 1)
            matrix = np.loadtxt(file_dir + rotation_file)
            shift = np.loadtxt(file_dir + shift_file)

            if func_id == 1:
                tasks.append(
                    Sphere(__class__.dim, shift= shift, rotation_matrix= matrix,bound= [-100, 100])
                )
            elif func_id == 2:
                tasks.append(
                    Rosenbrock(__class__.dim, shift= shift, rotation_matrix= matrix, bound= [-50, 50])
                )
            elif func_id == 3:
                tasks.append(
                    Ackley(__class__.dim, shift= shift, rotation_matrix= matrix, bound= [-50, 50])
                )
            elif func_id == 4:
                tasks.append(
                    Rastrigin(__class__.dim, shift= shift, rotation_matrix= matrix, bound= [-50, 50])
                )
            elif func_id == 5:
                tasks.append(
                    Griewank(__class__.dim, shift= shift, rotation_matrix= matrix, bound = [-100, 100])
                )
            elif func_id == 6:
                tasks.append(
                    Weierstrass(__class__.dim, shift= shift, rotation_matrix= matrix, bound= [-0.5, 0.5])
                )
            elif func_id == 7:
                tasks.append(
                    Schwefel(__class__.dim, shift= shift, rotation_matrix= matrix, bound= [-500, 500], fixed= fix)
                )
        return tasks, Individual_func