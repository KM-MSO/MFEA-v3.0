import numpy as np
from ...tasks.task import AbstractTask
from . import loadModel, MultiTimeModel, CompareModel
from .. import AbstractModel
import os
from typing import List

class TuningModel:
    def __init__(self, model_name, nb_run: int = 1, list_parameter: List[tuple] = []) -> None:
        self.best_compile_parameter = {}
        self.best_fit_parameter = {}
        self.model_name = model_name
        self.list_parameter: List[tuple(str, list)] = list_parameter
        self.nb_run = nb_run

        self.default_lsmodel= None 

    def compile(self, ls_benchmark=None, benchmark_weights=[], name_benchmark = [], ls_IndClass = [],  **kwargs):
        # if ls_benchmark is None:
        #     ls_benchmark.append(kwargs['tasks'])
        #     ls_IndClass.append(kwargs['IndClass'])
        #     name_benchmark.append("default")
        # else:
        #     if kwargs['tasks'] not in ls_benchmark and kwargs['tasks'] is not None:
        #         ls_benchmark.append(kwargs['tasks']) 
        #         ls_IndClass.append(kwargs['IndClass'])
        #         name_benchmark.append("default")

        assert len(ls_benchmark) == len(
            benchmark_weights), 'len of ls benchmark and benchmark_weights must be same'
        assert np.sum(np.array(benchmark_weights)
                      ) == 1, 'Benchmark weighs need sum up to 1'

        self.compile_kwargs = kwargs
        self.ls_benchmark: List[List[AbstractTask]] = ls_benchmark
        self.benchmark_weights = benchmark_weights
        self.name_benchmark = name_benchmark
        self.ls_IndClass = ls_IndClass

    def fit_multibenchmark(self, curr_fit_parameter, curr_compile_parameter, nb_run=1, save_path="./RESULTS/tuning_smp/k", name_model="model.mso"):
        ls_model = []
        for idx, benchmark in enumerate(self.ls_benchmark):
            model = MultiTimeModel(self.model_name)
            curr_compile_parameter['tasks'] = benchmark
            curr_compile_parameter['IndClass'] = self.ls_IndClass[idx] 
            model.compile(
                **curr_compile_parameter
            )
            model.fit(
                **curr_fit_parameter
            )

            if os.path.isdir(save_path) is False:
                os.makedirs(save_path)

            model.run(
                nb_run=self.nb_run,
                save_path=save_path + self.name_benchmark[idx]
            )
            model = loadModel(save_path + self.name_benchmark[idx] , ls_tasks=benchmark, set_attribute= True)

            import json
            file = open(save_path + self.name_benchmark[idx] + "_" + name_model.split('.')[0] + "_result.txt", 'w')
            file.write(json.dumps(dict(enumerate(model.history_cost[-1]))))
            file.close()   
            ls_model.append(model)
        
        if self.default_lsmodel is None: 
            self.default_lsmodel = ls_model 

        return ls_model 



    def fit(self, curr_fit_parameter, curr_compile_parameter, nb_run=1, save_path="./RESULTS/tuning_smp/", name_model="model.mso"):
        model = MultiTimeModel(self.model_name)
        model.compile(
            **curr_compile_parameter
        )
        model.fit(
            **curr_fit_parameter
        )
        if os.path.isdir(save_path) is False:
            os.makedirs(save_path)
        ls_tasks = model.tasks
        model.run(
            nb_run=self.nb_run,
            save_path=save_path + name_model
        )

        model = loadModel(save_path + name_model,
                          ls_tasks=ls_tasks, set_attribute=True)

        import json
        file = open(save_path + name_model.split('.')[0] + "_result.txt", 'w')
        file.write(json.dumps(dict(enumerate(model.history_cost[-1]))))
        file.close()

        return model
    
    def compare_between_2_ls_model(self, ls_model1: List[AbstractModel.model], ls_model2 : List[AbstractModel.model], min_value= 0, take_point = False):
        '''
        compare the result between models and return best model 
        [[model1_cec, model1_gecco], [model2_cec, model2_gecco]]
        '''
        point_model = np.zeros(shape= (2,))
        for benchmark in range(len(ls_model1)):
            result_model1 = np.where(ls_model1[benchmark].history_cost[-1] > min_value, ls_model1[benchmark].history_cost[-1], min_value)
            result_model2 = np.where(ls_model2[benchmark].history_cost[-1] > min_value, ls_model2[benchmark].history_cost[-1], min_value) 

            point1 =  np.sum(result_model1 < result_model2) / len(ls_model1[benchmark].tasks)
            point2 = np.sum(result_model2 < result_model1) / len(ls_model1[benchmark].tasks)

            point_model[0] += point1 * self.benchmark_weights[benchmark]
            point_model[1] += point2 * self.benchmark_weights[benchmark]
            point_model += (1 - np.sum(point_model))/2 
        if take_point is True: 
            return point_model
        else: 
            return np.argmax(point_model) 


    def take_idx_best_lsmodel(self, set_ls_model: List[List[AbstractModel.model]], min_value = 0, compare_default = True, take_point = False):
        if compare_default is True: 
            ls_point = [] 
            for idx, ls_model in enumerate(set_ls_model):
                ls_point += [(self.compare_between_2_ls_model(ls_model, self.default_lsmodel, min_value,take_point= True)[0])]
            if take_point is True:
                return ls_point 
            else: 
                best_idx = np.argmax(np.array(ls_point))
                return best_idx
            pass 
        else: 
            best_idx = 0  
            for idx, ls_model in enumerate(set_ls_model[1:],start= 1 ):
                better_idx = self.compare_between_2_ls_model(set_ls_model[best_idx], ls_model, min_value)
                if better_idx == 1: 
                    best_idx = idx 
        
        return best_idx 

    def take_idx_best_model(self, ls_model) -> int:
        compare = CompareModel(ls_model)
        end = compare.detail_compare_result()
        return np.argmax([float(point.split("/")[0]) for point in end.iloc[-1]])

    def run(self, path="./RESULTS/tuning", replace_folder=False,min_value = 0,  **kwargs):
        if path[-1] != "/":
            path += "/"
        path = path + self.model_name.__name__.split('.')[-1]
        curr_fit_parameter = kwargs.copy()
        curr_compile_parameter = self.compile_kwargs.copy()

        self.best_compile_parameter = self.compile_kwargs.copy()
        self.best_fit_parameter = kwargs.copy()
        result = self.list_parameter.copy()
        result = [list(result[i]) for i in range(len(result))]

        # folder
        if os.path.isdir(path) is True:
            if replace_folder is True:
                pass
            else:
                raise Exception("Folder is existed")
        else:
            os.makedirs(path)

        self.root_folder = os.path.join(path)

        # check value pass
        for name_arg, arg_pass in self.list_parameter:
            # name_args: 'crossover' , arg_pass: {'gamma': []}
            if name_arg in curr_compile_parameter.keys():
                if callable(curr_compile_parameter[name_arg]):
                    for name_para, para_value in arg_pass.items():
                        # name_para: 'gamma'
                        # para_value: [0.4, 0.5]
                        curr_para_value = getattr(
                            curr_compile_parameter[name_arg], name_para)
                        if curr_para_value is not None:
                            if curr_para_value in para_value:
                                para_value.remove(curr_para_value)
                                para_value.insert(0, curr_para_value)
                            else:
                                para_value.insert(0, curr_para_value)
                        else:
                            pass
                            setattr(
                                curr_compile_parameter[name_arg], name_para, para_value[0])
                else:  # if arg_pass in self.compile_kwargs is a str/number
                    if curr_compile_parameter[name_arg] is not None:
                        if curr_compile_parameter[name_arg] in arg_pass:
                            arg_pass.remove(curr_compile_parameter[name_arg])
                            arg_pass.insert(
                                0, curr_compile_parameter[name_arg])
                        else:
                            arg_pass.insert(
                                0, curr_compile_parameter[name_arg])
                    else:
                        curr_compile_parameter[name_arg] = arg_pass[0]

            elif name_arg in curr_fit_parameter.keys():
                if callable(curr_fit_parameter[name_arg]):
                    for name_para, para_value in arg_pass.items():
                        curr_para_value = getattr(
                            curr_fit_parameter[name_arg], name_para)
                        if curr_para_value is not None:
                            if curr_para_value in para_value:
                                para_value.remove(curr_para_value)
                                para_value.insert(0, curr_para_value)
                            else:
                                para_value.insert(0, curr_para_value)
                        else:
                            pass
                            setattr(
                                curr_fit_parameter[name_arg], name_para, para_value[0])

                else:
                    if curr_fit_parameter[name_arg] is not None:
                        if curr_fit_parameter[name_arg] in arg_pass:
                            arg_pass.remove(curr_fit_parameter[name_arg])
                            arg_pass.insert(0, curr_fit_parameter[name_arg])
                        else:
                            arg_pass.insert(0, curr_fit_parameter[name_arg])
                    else:
                        pass
                        curr_fit_parameter[name_arg] = arg_pass[0]

        
        # run many 
        curr_order_params = 0
        for name_arg, arg_value in self.list_parameter:
            curr_order_params += 1
            idx = self.list_parameter.index((name_arg, arg_value))

            if name_arg in self.compile_kwargs.keys():
                print("\n",name_arg)
                if callable(self.compile_kwargs[name_arg]):
                    for name_para, para_value in arg_value.items():
                        print("\n",name_para)
                        # take each parameter in function
                        # name_para: 'gamma'
                        # para_value: [0.4, 0.6]
                        # TODO
                        sub_folder = self.get_curr_path_folder(
                            curr_order_params) + "/" + str(name_para)
                        # root_folder/gamma

                        assert type(para_value) == list
                        curr_compile_parameter = self.best_compile_parameter.copy()
                        set_ls_model = []
                        for value in para_value:
                            value_folder_path = sub_folder + \
                                "/" + str(value) + "/"
                            # root_folder/gamma/0.4/
                            print(value)
                            setattr(curr_compile_parameter[name_arg], name_para, value)
                            # ls_model.append(self.fit(
                            #     self.best_fit_parameter, curr_compile_parameter, save_path=value_folder_path))
                            set_ls_model.append(self.fit_multibenchmark(self.best_fit_parameter, curr_compile_parameter, save_path=value_folder_path))
                            # set_ls_model.append(self.fit_multibenchmark(self.best_fit_parameter, curr_compile_parameter))

                        # TODO: take the best model and update best parameter
                        ls_point = self.take_idx_best_lsmodel(set_ls_model, min_value= min_value, take_point=True)
                        value = para_value[np.argmax(ls_point)]


                        import json 
                        # file= open(value_folder_path + "/result.txt" , 'w')
                        file = open(sub_folder + "/" + "result.txt", 'w')
                        file.write(name_arg+" ") 
                        file.write(name_para)
                        file.write(json.dumps(list(zip(arg_value[name_para], ls_point))))
                        file.close() 
                        
                        
                        
                        setattr(
                            self.best_compile_parameter[name_arg], name_para, value)

                        # save result

                        result[idx][1][name_para] = value

                else:
                    # if self.complile_kwargs[name_arg] is str/ number
                    assert type(arg_value) == list
                    curr_compile_parameter = self.best_compile_parameter.copy()
                    set_ls_model = []
                    sub_folder = self.get_curr_path_folder(
                        curr_order_params) + "/" + str(name_arg)
                    # root_folder/lr
                    for value in arg_value:
                        value_folder_path = sub_folder + "/" + str(value) + "/"
                        # root_folder/lr/0.1/
                        print(value)
                        curr_compile_parameter[name_arg] = value
                        # self.fit(self.best_fit_parameter, curr_compile_parameter)
                        set_ls_model.append(self.fit_multibenchmark(
                            self.best_fit_parameter, curr_compile_parameter, save_path=value_folder_path))
                    # TODO: take the best model and update best parameter
                    ls_point = self.take_idx_best_lsmodel(set_ls_model, min_value= min_value, take_point=True)
                    value = arg_value[np.argmax(ls_point)]

                    self.best_compile_parameter[name_arg] = value

                    import json 
                    # file = open(value_folder_path + "/result.txt", 'w')
                    file = open(sub_folder + "/" + "result.txt", 'w')
                    file.write(name_arg)
                    file.write(json.dumps(list(zip(arg_value, ls_point))))
                    file.close() 

                    # save result
                    result[idx][1] = value

            elif name_arg in curr_fit_parameter.keys():
                print("\n",name_arg)
                if callable(curr_fit_parameter[name_arg]):
                    for name_para, para_value in arg_value.items():
                        print("\n",name_para)

                        sub_folder = self.get_curr_path_folder(
                            curr_order_params) + "/" + str(name_para)

                        assert type(para_value) == list
                        set_ls_model = []
                        curr_fit_parameter = self.best_fit_parameter.copy()
                        for value in para_value:
                            value_folder_path = sub_folder + \
                                "/" + str(value) + "/"
                            print(value)
                            setattr(
                                curr_fit_parameter[name_arg], name_para, value)
                            set_ls_model.append(self.fit_multibenchmark(
                                curr_fit_parameter, self.best_compile_parameter, save_path=value_folder_path))
                        # TODO: take the best modle in update best parameter
                        ls_point = self.take_idx_best_lsmodel(set_ls_model, min_value= min_value, take_point=True)
                    
                        value = para_value[np.argmax(ls_point)]
                        setattr(
                            self.best_fit_parameter[name_arg], name_arg, value)

                        import json 
                        # file= open(value_folder_path + "/result.txt" , 'w')
                        file = open(sub_folder + "/" + "result.txt", 'w')
                        file.write(name_arg) 
                        file.write(name_para)
                        file.write(json.dumps(list(zip(arg_value[name_para], ls_point))))
                        file.close() 

                        # save result
                        result[idx][1][name_para] = value
                else:
                    assert type(arg_value) == list
                    curr_fit_parameter = self.best_fit_parameter.copy()
                    set_ls_model = []
                    sub_folder = self.get_curr_path_folder(
                        curr_order_params) + "/" + str(name_arg)
                    for value in arg_value:
                        print(value)
                        value_folder_path = sub_folder + "/" + str(value) + "/"
                        curr_fit_parameter[name_arg] = value
                        # self.fit(curr_fit_parameter, self.best_compile_parameter)
                        set_ls_model.append(self.fit_multibenchmark(
                            curr_fit_parameter, self.best_compile_parameter, save_path=value_folder_path))
                    # TODO: take the best model and update best fit parameter
                    ls_point = self.take_idx_best_lsmodel(set_ls_model, min_value= min_value, take_point=True)
                    value = arg_value[np.argmax(ls_point)]

                    self.best_fit_parameter[name_arg] = value

                    import json 
                    file = open(sub_folder + "/" + "result.txt", 'w')
                    file.write(name_arg)
                    file.write(json.dumps(list(zip(arg_value, ls_point))))
                    file.close() 


                    # save result
                    result[idx][1] = value

        import json
        file = open(self.root_folder + "/result.txt", 'w')
        file.write(json.dumps(result))
        file.close()

        return self.best_fit_parameter, self.best_compile_parameter, result

    def get_curr_path_folder(self, curr_order_params):
        if curr_order_params == 1:
            # path = root_path/gamma/
            path = self.root_folder
            pass
        else:
            # path = root_path/gamma/"value_gamma"
            path = self.root_folder
            index = 1
            while(curr_order_params > index):
                name_arg, arg_value = self.list_parameter[index-1]
                if type(arg_value) != list:
                    for key, _ in arg_value.items():
                        index += 1
                        # find value
                        if name_arg in self.best_compile_parameter.keys():
                            value = getattr(
                                self.best_compile_parameter[name_arg], key)
                        else:
                            value = getattr(
                                self.best_fit_parameter[name_arg], key)

                        # update path
                        path += "/" + str(key) + "/" + str(value)
                else:
                    if name_arg in self.best_compile_parameter.keys():
                        value = self.best_compile_parameter[name_arg]
                    else:
                        value = self.best_fit_parameter[name_arg]

                    # update path
                    path += "/" + str(name_arg) + "/" + str(value)
                    index += 1
        return path
