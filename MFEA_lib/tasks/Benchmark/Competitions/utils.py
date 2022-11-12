import numpy as np
from ...task import AbstractTask
from numba import jit
from numba.typed import Dict

class AbstractFunc(AbstractTask):
    limited_space = False
    bound = (None, None)
    global_optimal = 0

    def __init__(self, dim, shift: list = 0, rotation_matrix: np.ndarray = None, bound: tuple = None, *args, **kwargs):
        self.dim = dim

        if rotation_matrix is not None:
            assert np.all(np.array(rotation_matrix.shape) == dim)
            self.rotation_matrix = rotation_matrix
            self.inv_rotation_matrix = np.linalg.inv(self.rotation_matrix)
        else:
            self.rotation_matrix = np.identity(dim)
            self.inv_rotation_matrix = np.identity(dim)
        
        tmp = np.array(shift).reshape(-1, )
        assert dim % len(tmp) == 0
        self.shift = np.array([[i] * int(dim / len(tmp)) for i in tmp ]).reshape(-1, )

        self.global_optimal = self.encode(self.inv_rotation_matrix @ self.global_optimal + self.shift)
        
        if bound is not None:
            self.limited_space = True
            self.bound = bound
            self.name = self.__class__.__name__ + ': [' + str(self.bound[0]) + ', ' + str(self.bound[1]) + ']^' + str(dim)
        else:
            self.name = self.__class__.__name__ + ': R^' + str(dim)

    def __eq__(self, other: object) -> bool:
        if self.__repr__() == other.__repr__():
            return True
        return self.dim == other.dim and np.all(self.shift == other.shift) and self.bound == other.bound

    def encode(self, x):
        '''
        encode x to [0, 1]
        '''
        x_encode = x
        # x_encode = self.inv_rotation_matrix @ x_encode + self.shift
        if self.limited_space == True:
            x_encode = (x_encode - self.bound[0])/(self.bound[1] - self.bound[0])
        return x_encode 

    @staticmethod
    # @jit(nopython = True)
    def decode(x, dim, limited_space, bound, rotation_matrix, shift):
        '''
        decode x from [0, 1] to bound
        '''
        x_decode = x[:dim]
        if limited_space == True:
            x_decode = x_decode * (bound[1] - bound[0]) + bound[0]
        x_decode = rotation_matrix @ (x_decode - shift) 
        return x_decode 

    def __call__(self, x):
        x = self.__class__.decode(x, self.dim, self.limited_space, self.bound, self.rotation_matrix, self.shift)
        return self.__class__._func(x)

    @staticmethod
    @jit(nopython = True)
    def _func(x):
        pass

class Sphere(AbstractFunc):
    '''
    global optima = 0^d
    '''
    def __init__(self, dim, shift: list = 0, rotation_matrix: np.ndarray = None, bound: tuple = None):
        self.global_optimal = np.array([0] * dim)
        super().__init__(dim, shift, rotation_matrix, bound)

    @staticmethod
    @jit(nopython = True)
    def _func(x):
        '''
        Request: input x is encoded
        '''
        return np.sum(x**2, axis = 0)

class Weierstrass(AbstractFunc):
    '''
    global optima = 0^d
    '''    
    def __init__(self, dim, shift: list = 0, rotation_matrix: np.ndarray = None, bound: tuple = None):
        self.global_optimal = np.array([0] * dim)
        super().__init__(dim, shift, rotation_matrix, bound)
        self.params = Dict()
        self.params['a'] = 0.5
        self.params['b'] = 3
        self.params['k_max'] = 21

    @staticmethod
    @jit(nopython = True)
    def _func(x, dim, params: dict):
        '''
        Request: input x is encoded
        '''
        left = 0
        for i in range(dim):
            left += np.sum(params['a'] ** np.arange(params['k_max']) * \
                np.cos(2*np.pi * params['b'] ** np.arange(params['k_max']) * (x[i]  + 0.5)))
            
        right = dim * np.sum(params['a'] ** np.arange(params['k_max']) * \
            np.cos(2 * np.pi * params['b'] ** np.arange(params['k_max']) * 0.5)
        )
        return left - right

    def __call__(self, x):
        x = self.__class__.decode(x, self.dim, self.limited_space, self.bound, self.rotation_matrix, self.shift)
        return __class__._func(x,self.dim, self.params)

class Ackley(AbstractFunc):
    '''
    global optima = 0^d
    '''
    def __init__(self, dim, shift: list = 0, rotation_matrix: np.ndarray = None, bound: tuple = None, fixed = False):
        self.global_optimal = np.array([0] * dim)
        self.fixed = fixed
        super().__init__(dim, shift, rotation_matrix, bound)
        self.params = Dict()
        self.params['b'] = 0.2
        self.params['a'] = 20
        self.params['c'] = 2 * np.pi

    # @overload
    

    @staticmethod
    @jit(nopython = True)
    def _func(x, fixed, params):
        if fixed:
            return -params['a'] * np.exp(-params['b']*np.sqrt(np.mean(x**2)))\
            - np.exp(np.mean(np.cos(params['c'] * x)))\
            + params['a']\
            + np.exp(1)\
            - 4.440892098500626e-16
        else:
            return -params['a'] * np.exp(-params['b']*np.sqrt(np.mean(x**2)))\
            - np.exp(np.mean(np.cos(params['c'] * x)))\
            + params['a']\
            + np.exp(1)

    def __call__(self, x):
        x = self.__class__.decode(x, self.dim, self.limited_space, self.bound, self.rotation_matrix, self.shift)
        return __class__._func(x, self.fixed, self.params)

class Rosenbrock(AbstractFunc):
    '''
    global optima = 1^d
    '''
    def __init__(self, dim, shift: list = 0, rotation_matrix: np.ndarray = None, bound: tuple = None):
        self.global_optimal = np.array([1] * dim)
        super().__init__(dim, shift, rotation_matrix, bound)

    @staticmethod
    @jit(nopython = True)
    def _func(x):
        l = 100*np.sum((x[1:] - x[:-1]**2) ** 2)
        r = np.sum((x[:-1] - 1) ** 2)
        return l + r

class Schwefel(AbstractFunc):
    '''
    if fixed: 
        global optima = 420.968746^d
    else:
        global optima = 420.9687^d
    '''
    def __init__(self, dim, shift: list = 0, rotation_matrix: np.ndarray = None, bound: tuple = None, fixed = False):
        if fixed:
            self.global_optimal = np.array([420.968746] * dim)
        else:
            self.global_optimal = np.array([420.9687] * dim)
        super().__init__(dim, shift, rotation_matrix, bound)
        self.fixed = fixed

    @staticmethod
    @jit(nopython = True)
    def _func(x, dim, fixed):
        if fixed:
            return (418.9828872724336455576189785193) * dim - np.sum(x * np.sin(np.sqrt(np.abs(x)))) 
        else:
            return 418.9829 * dim - np.sum(x * np.sin(np.sqrt(np.abs(x)))) 
    
    def __call__(self, x):
        x =   self.__class__.decode(x, self.dim, self.limited_space, self.bound, self.rotation_matrix, self.shift)
        return __class__._func(x, self.dim, self.fixed)

class Griewank(AbstractFunc):
    ''' 
    global optima = [0] ^ d
    '''
    def __init__(self, dim, shift: list = 0, rotation_matrix: np.ndarray = None, bound: tuple = None):
        self.global_optimal = np.array([0] * dim)
        super().__init__(dim, shift, rotation_matrix, bound)
    
    @staticmethod
    @jit(nopython = True)
    def _func(x, dim):
        return np.sum(x**2) / 4000 \
            - np.prod(np.cos(x / np.sqrt((np.arange(dim) + 1))))\
            + 1

    def __call__(self, x):
        x =   self.__class__.decode(x, self.dim, self.limited_space, self.bound, self.rotation_matrix, self.shift)
        return __class__._func(x, self.dim)

class Rastrigin(AbstractFunc):
    ''' 
    global optima = 0 ^ d
    '''
    def __init__(self, dim, shift: list = 0, rotation_matrix: np.ndarray = None, bound: tuple = None):
        self.global_optimal = np.array([0] * dim)
        super().__init__(dim, shift, rotation_matrix, bound)
    
    @staticmethod
    @jit(nopython = True)
    def _func(x, dim):
        return 10 * dim + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
    
    def __call__(self, x):
        x =   self.__class__.decode(x, self.dim, self.limited_space, self.bound, self.rotation_matrix, self.shift)
        return __class__._func(x, self.dim)
        




