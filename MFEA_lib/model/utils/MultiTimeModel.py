import pickle
import numpy as np
import traceback
import os
from typing import List
from pathlib import Path

from .. import AbstractModel
from . import MultiTimeModel

def get_model_name(model: AbstractModel.model):
    fullname = model.__module__
    index = None
    for i in range(len(fullname) - 1, -1, -1):
        if fullname[i] == '.':
            index = i
            break
    if index is None:
        return fullname
    return fullname[index + 1:]


class MultiTimeModel:
    def __init__(self, model: AbstractModel, list_attri_avg: list = None,  name=None) -> None:

        self.model = model.model

        if name is None:
            self.name = model.__name__
        else:
            self.name = name

        if list_attri_avg is None:
            self.list_attri_avg = None
        else:
            self.list_attri_avg = list_attri_avg

        self.ls_model: List[AbstractModel.model] = []
        self.ls_seed: List[int] = []
        self.total_time = 0

        # add inherit
        cls = self.__class__
        self.__class__ = cls.__class__(cls.__name__, (cls, self.model), {})

        # status of model run
        # self.status = 'NotRun' | 'Running' | 'Done'
        self.status = 'NotRun'

    def set_data(self, history_cost: np.ndarray):
        self.status = 'Done'
        self.history_cost = history_cost
        print('Set complete!')

    def set_attribute(self):
        # print avg
        if self.list_attri_avg is None:
            self.list_attri_avg = self.ls_model[0].ls_attr_avg
        for i in range(len(self.list_attri_avg)):
            try:
                result = [model.__getattribute__(
                    self.list_attri_avg[i]) for model in self.ls_model]
            except:
                print("cannot get attribute {}".format(self.list_attri_avg[i]))
                continue
            try:
                
            # min_dim1 = 1e10 
            # for idx1, array_seed in enumerate(result): 
            #     min_dim1 = min([len(array_seed), min_dim1])
            #     for idx2,k in enumerate(array_seed): 
                        
            #         for idx3, x in enumerate(k) : 
            #             if type(x) != float:
            #                 result[idx1][idx2][idx3] = float(x)
                    
            #         result[idx1][idx2]= np.array(result[idx1][idx2])
            #     result[idx1] = np.array(result[idx1])
            
            # for idx, array in enumerate(result):
            #     result[idx] = result[idx][:min_dim1]
            
                result = np.array(result[:][:min([len(his) for his in result])][:])
                result = np.average(result, axis=0)
                self.__setattr__(self.list_attri_avg[i], result)
            except:
                print(f'can not convert {self.list_attri_avg[i]} to np.array')
                continue

    def print_result(self, print_attr = [], print_time = True, print_name= True):
        # print time
        seconds = self.total_time
        minutes = seconds // 60
        seconds = seconds - minutes * 60
        if print_time: 
            print("total time: %02dm %.02fs" % (minutes, seconds))

        # print avg
        if self.list_attri_avg is None:
            self.list_attri_avg = self.ls_model[0].ls_attr_avg

        if len(print_attr) ==0 : 
            print_attr = self.list_attri_avg 
        
        for i in range(len(print_attr)):
            try:
                result = self.__getattribute__(print_attr[i])[-1]
                if print_name: 
                    print(f"{print_attr[i]} avg: ")
                np.set_printoptions(
                    formatter={'float': lambda x: format(x, '.2E')})
                print(result)
            except:
                try:
                    result = [model.__getattribute__(
                        print_attr[i]) for model in self.ls_model]
                    result = np.array(result)
                    result = np.sum(result, axis=0) / len(self.ls_model)
                    if print_name: 
                        print(f"{print_attr[i]} avg: ")
                    np.set_printoptions(
                        formatter={'float': lambda x: format(x, '.2E')})
                    print(result)
                except:
                    print(
                        f'can not convert {print_attr[i]} to np.array')

    def compile(self, **kwargs):
        self.compile_kwargs = kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)

    def fit(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        for name, value in kwargs.items():
            setattr(self, name, value)

    def run(self, nb_run: int = None, save_path: str = "./RESULTS/result/model.mso", seed_arr: list = None, random_seed: bool = False, replace_folder = True):
        print('Checking ...', end='\r')
        # folder
        idx = len(save_path) 
        while save_path[idx-1] != "/": 
            idx -= 1 

        if os.path.isdir(save_path[:idx]) is True:
            if replace_folder is True:
                pass
            else:
                raise Exception("Folder is existed")
        else:
            os.makedirs(save_path[:idx])

        if self.status == 'NotRun':
            if nb_run is None:
                self.nb_run = 1
            else:
                self.nb_run = nb_run

            if save_path is None:
                save_path = get_model_name(self.model) + '.mso'

            if seed_arr is not None:
                assert len(seed_arr) == nb_run
            elif random_seed:
                seed_arr = np.random.randint(
                    nb_run * 100, size=(nb_run, )).tolist()
            else:
                seed_arr = np.arange(nb_run).tolist()

            self.ls_seed = seed_arr
            index_start = 0
        elif self.status == 'Running':
            if nb_run is not None:
                assert self.nb_run == nb_run

            if save_path is None:
                save_path = get_model_name(self.model) + '.mso'

            if seed_arr is not None:
                assert np.all(
                    seed_arr == self.ls_seed), '`seed_arr` is not like `ls_seed`'

            index_start = len(self.ls_model)
        elif self.status == 'Done':
            print('Model has already completed before.')
            return
        else:
            raise ValueError('self.status is not NotRun | Running | Done')

        for idx_seed in range(index_start, len(self.ls_seed)):
            try:
                model = self.model(self.ls_seed[idx_seed])
                
                self.ls_model.append(model)
                
                model.compile(**self.compile_kwargs)
                model.fit(*self.args, **self.kwargs)

                self.total_time += model.time_end - model.time_begin

            except KeyboardInterrupt as e:
                self.status = 'Running'
                self.set_attribute()

                save_result = saveModel(self, save_path)
                print('\n\nKeyboardInterrupt: ' +
                      save_result + ' model, model is not Done')
                traceback.print_exc()
                break
        else:
            self.set_attribute()
            self.status = 'Done'
            print('DONE!')
            print(saveModel(self, save_path))


def saveModel(model: MultiTimeModel, PATH: str, remove_tasks=True):
    '''
    `.mso`
    '''
    assert model.__class__.__name__ == 'MultiTimeModel'
    assert type(PATH) == str

    # check tail
    path_tmp = Path(PATH)
    index_dot = None
    for i in range(len(path_tmp.name) - 1, -1, -1):
        if path_tmp.name[i] == '.':
            index_dot = i
            break

    if index_dot is None:
        PATH += '.mso'
    else:
        assert path_tmp.name[i:] == '.mso', 'Only save model with .mso, not ' + \
            path_tmp.name[i:]

    model.__class__ = MultiTimeModel

    tasks = model.tasks 

    if remove_tasks is True:
        model.tasks = None
        model.compile_kwargs['tasks'] = None
        for submodel in model.ls_model:
            submodel.tasks = None
            submodel.last_pop.ls_tasks = None
            for subpop in submodel.last_pop:
                subpop.task = None
            if 'attr_tasks' in submodel.kwargs.keys():
                for attribute in submodel.kwargs['attr_tasks']:
                    # setattr(submodel, getattr(subm, name), None)
                    setattr(getattr(submodel, attribute), 'tasks', None)
                    pass
            else:
                submodel.crossover.tasks = None
                submodel.mutation.tasks = None

    try:
        f = open(PATH, 'wb')
        pickle.dump(model, f)
        f.close()

    except:
        cls = model.__class__
        model.__class__ = cls.__class__(cls.__name__, (cls, model.model), {})

        if remove_tasks is True:
            model.tasks = tasks 
            model.compile_kwargs['tasks'] = None
            for submodel in model.ls_model:
                submodel.tasks = tasks
                submodel.last_pop.ls_tasks = tasks 
                for idx, subpop in enumerate(submodel.last_pop):
                    subpop.task = tasks[idx]
                if 'attr_tasks' in submodel.kwargs.keys():
                    for attribute in submodel.kwargs['attr_tasks']:
                        # setattr(submodel, getattr(subm, name), None)
                        setattr(getattr(submodel, attribute), 'tasks', tasks)
                        pass
                else:
                    submodel.crossover.tasks = tasks
                    submodel.mutation.tasks = tasks 
        return 'Cannot Saved'

    
    if remove_tasks is True:
        model.tasks = tasks 
        model.compile_kwargs['tasks'] = None
        for submodel in model.ls_model:
            submodel.tasks = tasks
            submodel.last_pop.ls_tasks = tasks 
            for idx, subpop in enumerate(submodel.last_pop):
                subpop.task = tasks[idx]
            if 'attr_tasks' in submodel.kwargs.keys():
                for attribute in submodel.kwargs['attr_tasks']:
                    # setattr(submodel, getattr(subm, name), None)
                    setattr(getattr(submodel, attribute), 'tasks', tasks)
                    pass
            else:
                submodel.crossover.tasks = tasks
                submodel.mutation.tasks = tasks 


    cls = model.__class__
    model.__class__ = cls.__class__(cls.__name__, (cls, model.model), {})

    return 'Saved'


def loadModel(PATH: str, ls_tasks=None, set_attribute=False) -> AbstractModel:
    '''
    `.mso`
    '''
    assert type(PATH) == str

    # check tail
    path_tmp = Path(PATH)
    index_dot = None
    for i in range(len(path_tmp.name)):
        if path_tmp.name[i] == '.':
            index_dot = i
            break

    if index_dot is None:
        PATH += '.mso'
    else:
        assert path_tmp.name[i:] == '.mso', 'Only load model with .mso, not ' + \
            path_tmp.name[i:]

    f = open(PATH, 'rb')
    model = pickle.load(f)
    f.close()

    cls = model.__class__
    model.__class__ = cls.__class__(cls.__name__, (cls, model.model), {})

    if model.tasks is None:
        model.tasks = ls_tasks
        if set_attribute is True:
            assert ls_tasks is not None, 'Pass ls_tasks plz!'
            model.compile_kwargs['tasks'] = ls_tasks
            for submodel in model.ls_model:
                submodel.tasks = ls_tasks
                submodel.last_pop.ls_tasks = ls_tasks
                for idx, subpop in enumerate(submodel.last_pop):
                    subpop.task = ls_tasks[idx]
                if 'attr_tasks' in submodel.kwargs.keys():
                    for attribute in submodel.kwargs['attr_tasks']:
                        # setattr(submodel, getattr(subm, name), None)
                        setattr(getattr(submodel, attribute),
                                'tasks', ls_tasks)
                        pass
                else:
                    submodel.crossover.tasks = ls_tasks
                    submodel.mutation.tasks = ls_tasks

                # submodel.search.tasks = ls_tasks
                # submodel.crossover.tasks = ls_tasks
                # submodel.mutation.tasks = ls_tasks

    if model.name.split('.')[-1] == 'AbstractModel':
        model.name = path_tmp.name.split('.')[0]
    return model