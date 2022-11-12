from . import MultiTimeModel
from .. import AbstractModel

class MultiBenchmark(): 
    def __init__(self, ls_benchmark = [], name_benchmark = [], ls_IndClass = [], model : AbstractModel= None) :
        self.ls_benchmark = ls_benchmark 
        self.ls_name_benchmark = name_benchmark 
        self.ls_IndClass = ls_IndClass 

        self.model = model  
        pass
    
    def compile(self, **kwargs): 
        self.compile_kwargs = kwargs 
    
    def fit(self, **kwargs):
        self.fit_kwargs = kwargs 
    
    def run(self,nb_run = 1, save_path = './RESULTS/result/'): 
        self.ls_model:list[MultiTimeModel] = [] 
        for idx, benchmark in enumerate(self.ls_benchmark): 
            self.compile_kwargs['tasks'] = benchmark 
            self.compile_kwargs['IndClass'] = self.ls_IndClass[idx] 

            model = MultiTimeModel(model = self.model) 
            model.compile(**self.compile_kwargs) 
            model.fit(**self.fit_kwargs) 
            model.run(nb_run = nb_run, save_path= save_path + str(self.ls_name_benchmark[idx]) + ".mso") 
            self.ls_model.append(model) 
    
    def print_result(self,print_attr = [],print_name = True, print_time = False, print_name_attr = False):
        
        for idx, model in enumerate(self.ls_model): 
            if print_name : 
                print(self.ls_name_benchmark[idx])
            model.print_result(print_attr, print_time= print_time, print_name= print_name_attr) 
