from ....EA import Individual
import numpy as np

class Individual_func(Individual):
    def __init__(self, genes, parent= None, dim= None, *args, **kwargs) -> None:
        super().__init__(genes, parent, dim)
        if genes is None:
            self.genes: np.ndarray = np.random.rand(dim)
