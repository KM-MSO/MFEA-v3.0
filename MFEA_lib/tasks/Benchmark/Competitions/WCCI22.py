from .Default_Ind import Individual_func
# from .utils import *
from .GECCO20 import GECCO20_benchmark_50tasks

from typing import Tuple, Type, List
import numpy as np
import os
from numba import jit


path = os.path.dirname(os.path.realpath(__file__))

@jit(nopython = True)
def Ellips_func(x,dim):
    f =0 
    for i in range(0,dim):
        f+=(10 ** (6*i/ (dim-1)))  * x[i]*x[i]
    return f
@jit(nopython = True)
def Ben_cigar_func(x,dim):
    f =x[0] * x[0] 
    for i in range(1,dim):
        f+=(10 ** 6)  * x[i]*x[i]
    return f
@jit(nopython = True)
def Discus_func(x,dim):
    return x[0]*x[0] *(10**6) + np.sum(x[1:] ** 2)
@jit(nopython = True)
def Rosenbrock_func(x,dim): 
    x[0] = x[0]+1
    f=0
    for i in range(dim-1):
        x[i+1] +=1
        tmp1  = x[i]*x[i]-x[i+1]
        tmp2 =x[i]-1
        f+=100*tmp1*tmp1+tmp2*tmp2
    return f
@jit(nopython = True)
def Ackley_func(x,dim):
    a = 20
    b= 0.2
    c= 2*np.pi
    return -a * np.exp(-b*np.sqrt(np.mean(x**2)))\
    - np.exp(np.mean(np.cos(c * x)))\
    + a\
    + np.exp(1)
@jit(nopython = True)
def Weierstrass_func(x,dim):
    left = 0
    a=0.5
    b=3
    k_max=21
    for i in range(dim):
        left += np.sum(a ** np.arange(k_max) * \
            np.cos(2*np.pi * b ** np.arange(k_max) * (x[i]  + 0.5)))
        
    right = dim * np.sum(a ** np.arange(k_max) * \
        np.cos(2 * np.pi * b ** np.arange(k_max) * 0.5)
    )
    return left - right
@jit(nopython = True)
def Griewank_func(x,dim):
    s=0
    p=1
    for i in range(dim):
        s+=x[i]*x[i]
        p*=np.cos(x[i]/np.sqrt(1+i))
    return 1+s/4000-p
@jit(nopython = True)
def Rastrigin_func(x,dim):
    return 10 * dim + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x))
@jit(nopython = True)
def Schwefel_func(x,dim):
    f=0
    for i in range(dim):
        x[i] += 4.209687462275036e+002
        if x[i] >500:
            f-= (500-(x[i] %500)) * np.sin((500-(x[i] % 500)) ** 0.5)
            tmp = (x[i]-500)/100
            f+=tmp * tmp / dim
        elif x[i] <-500:
            f-=(-500 + (np.abs(x[i])%500)) * np.sin((500 - np.abs(x[i])% 500) ** 0.5)
            tmp = (x[i]+500)/100
            f += tmp * tmp / dim
        else:
            f -= x[i] * np.sin(np.abs(x[i])** 0.5)
    f  = 4.189828872724338e+002 *dim+f
    return f
@jit(nopython = True)
def Katsuura_func(x,dim):
    f = 1.0
    tmp3= dim ** 1.2
    for i in range(dim):
        temp =0
        for j in range(1,33):
            tmp1 = 2 ** j 
            tmp2 = tmp1 *x[i]
            temp+=np.abs(tmp2- np.floor(tmp2+0.5))/ tmp1
        f*=(1+(i+1)*temp) **(10/ tmp3)
    tmp3 = 10/(dim*dim)    
    return f*tmp3-tmp3
@jit(nopython = True)
def HappyCat_func(x,dim):
    alpha =0.125
    sum =0 
    r2= 0 
    for i in range (dim):
        r2+=(x[i]-1) * (x[i]-1)
        sum +=x[i]-1
    return  np.abs(r2-dim) **(2*alpha) +(0.5*r2+sum)/dim+0.5
@jit(nopython = True)
def Hgbat_func(x,dim):
    r2= 0
    sum =0
    for i in range (dim):
        r2+=(x[i]-1) * (x[i]-1)
        sum +=x[i]-1
    return   np.abs(r2 ** 2- sum ** 2) ** 0.5 +(0.5*r2+sum)/dim+0.5
@jit(nopython = True)
def Grie_rosen_func(x,dim):
    f = 0
    x[0] +=  1 
    for i in range(dim-1):
        x[i+1] +=1
        tmp1 =x[i] *x[i]-x[i+1]
        tmp2 = x[i]-1
        temp = 100 * tmp1 * tmp1 + tmp2 *tmp2
        f+= (temp*temp)/4000 -np.cos(temp)+1
    tmp1 = x[dim-1] * x[dim-1]-x[0]
    tmp2 =x[dim-1] -1 
    temp =100 * tmp1 *tmp1 + tmp2 *tmp2 
    f+=(temp*temp)/4000 -np.cos(temp)+1
    return f 
@jit(nopython =True)
def Escaffer6_func(x,dim):
    f = 0
    for i in range(dim-1):
        f+= scaffer(x[i],x[i+1])
    f+= scaffer(x[dim-1],x[0])
    return f
@jit(nopython = True)
def scaffer(x,y):
    temp1 = np.sin((x ** 2 + y ** 2 ) ** 0.5)
    temp1 = temp1*temp1
    temp2 = 1+0.001*(x ** 2 +y**2)
    return 0.5+(temp1-0.5)/(temp2 ** 2)
@jit(nopython = True)
def hf01(x,dim):
    p=[15,15,20]
    g= [15,30,50]
    return Schwefel_func(x[:g[0]]*10,p[0]) + Rastrigin_func(x[g[0]:g[1]]*5.12/100,p[1]) + Ellips_func(x[g[1]:],p[2])
@jit(nopython = True)    
def hf02(x,dim):
    p=[15,15,20]
    g= [15,30,50]
    return Ben_cigar_func(x[:g[0]],p[0]) + Hgbat_func(x[g[0]:g[1]]*5/100,p[1]) + Rastrigin_func(x[g[1]:]*5.12/100,p[2])
@jit(nopython = True)  
def hf03(x,dim):
    p=[10,10,15,15]
    g= [10,20,35,50]
    return Grie_rosen_func(x[:g[0]]*5/100,p[0]) + Weierstrass_func(x[g[0]:g[1]]*0.5/100,p[1]) + Rosenbrock_func(x[g[1]:g[2]]*2.048 / 100.0,p[2]) + Escaffer6_func(x[g[2]:],p[3])
@jit(nopython = True)  
def hf04(x,dim):
    p=[10,10,15,15]
    g= [10,20,35,50]
    return Hgbat_func(x[:g[0]]*5/100,p[0]) + Discus_func(x[g[0]:g[1]],p[1]) + Grie_rosen_func(x[g[1]:g[2]]*5/100,p[2]) + Rastrigin_func(x[g[2]:]*5.12/100,p[3])
@jit(nopython = True)  
def hf05(x,dim):
    p=[5,10,10,10,15]
    g= [5,15,25,35,50]
    return Escaffer6_func(x[:g[0]],p[0]) + Hgbat_func(x[g[0]:g[1]]*5/100,p[1]) + Rosenbrock_func(x[g[1]:g[2]]*2.048 / 100.0,p[2]) + Schwefel_func(x[g[2]:g[3]]*10,p[3]) + Ellips_func(x[g[3]:],p[4])
@jit(nopython = True)  
def hf06(x,dim):
    p=[5,10,10,10,15]
    g= [5,15,25,35,50]
    return Katsuura_func(x[:g[0]]*5/100,p[0]) + HappyCat_func(x[g[0]:g[1]]*5/100,p[1]) + Grie_rosen_func(x[g[1]:g[2]]*5/100,p[2]) + Schwefel_func(x[g[2]:g[3]]*10,p[3]) + Ackley_func(x[g[3]:],p[4])



class WCCI22_benchmark():
    def get_50tasks_benchmark(ID):
        return GECCO20_benchmark_50tasks.get_items(ID)
    
    class Task():
        def __init__(self,dim,func_id,Ub,Lb, shift: list = 0, rotation_matrix: np.ndarray = None,SS:list =[]):
            self.dim = dim
            self.shift =shift
            self.rotation_matrix =rotation_matrix
            self.Ub =Ub
            self.Lb = Lb
            self.SS = SS
            self.func_id =func_id
            self.name = "func_id = " + str(func_id)
        def decode(self,x,sh_rate,rotation_matrix,shift):
            x_decode = x-shift
            x_decode = x_decode*sh_rate
            x_decode= rotation_matrix @ x_decode
            if len(self.SS) != 0:
                y= np.zeros(self.dim) 
                for i in range(self.dim):
                    y[i] = x_decode[self.SS[i]-1]
                return y
            return x_decode 
        @jit(nopython = True)
        def func(x,ID,dim):
            if ID == 1: 
                return Ellips_func(x,dim) + 100
            if ID == 2: 
                return Ben_cigar_func(x,dim) + 200
            if ID == 3: 
                return Discus_func(x,dim) + 300
            if ID == 4:
                return Rosenbrock_func(x,dim) + 400
            if ID == 5: 
                return Ackley_func(x,dim) + 500
            if ID == 6:
                return Weierstrass_func(x,dim) + 600
            if ID == 7:
                return Griewank_func(x,dim) + 700
            if ID == 8: 
                return Rastrigin_func(x,dim) + 800
            if ID == 9:
                return Rastrigin_func(x,dim) +900
            if ID == 10:
                return Schwefel_func(x,dim) + 1000
            if ID == 11:
                return Schwefel_func(x,dim) + 1100
            if ID == 12:
                return Katsuura_func(x,dim) + 1200
            if ID == 13: 
                return HappyCat_func(x,dim) + 1300
            if ID == 14:
                return Hgbat_func(x,dim) + 1400
            if ID == 15:
                return Grie_rosen_func(x,dim) + 1500
            if ID == 16:
                return Escaffer6_func(x,dim) + 1600
            if ID == 17:
                return hf01(x,dim) + 1700
            if ID == 18:
                return hf02(x,dim) + 1800
            if ID == 19:
                return hf03(x,dim) + 1900
            if ID == 20:
                return hf04(x,dim) + 2000
            if ID == 21: 
                return hf05(x,dim) + 2100
            if ID == 22:
                return hf06(x,dim) + 2200 

        def fnceval(self,x):
            if self.func_id < 4 or self.func_id == 5 or self.func_id >15:
                x = self.decode(x,1,self.rotation_matrix,self.shift)
            if self.func_id == 4:
                x= self.decode(x, 2.048 / 100.0,self.rotation_matrix,self.shift)
            if self.func_id == 6:
                x= self.decode(x,0.5/100,self.rotation_matrix,self.shift)
            if self.func_id == 7: 
                x=self.decode(x,600/100,self.rotation_matrix,self.shift)
            if self.func_id == 8:
                x= self.decode(x,5.12/100,np.identity,self.shift)
            if self.func_id == 9:
                x =self.decode(x,5.12/100,self.rotation_matrix,self.shift)
            if self.func_id == 10:
                x =self.decode(x,10,np.identity,self.shift)
            if self.func_id == 11:
                x = self.decode(x,10,self.rotation_matrix,self.shift)
            if self.func_id > 11 and self.func_id <16:
                x = self.decode(x,5/100,self.rotation_matrix,self.shift)
            return __class__.func(x,self.func_id,self.dim)

        def __call__(self,x):
            x= x*(self.Ub-self.Lb)+self.Lb
            return self.fnceval(x)

    def get_complex_benchmark(ID) -> List[Task]:
        dim =50  
        task =[]
        file_dir = path + "/__references__/WCCI2022/SO-Complex-Benchmarks/Tasks/benchmark_" + str(ID)
        for i in range(1,3):
            shift_file = "/bias_" + str(i)
            rotation_file = "/matrix_" + str(i)
            matrix = np.loadtxt(file_dir + rotation_file)
            shift = np.loadtxt(file_dir + shift_file)
            if ID == 1:
                task.append(__class__.Task(50,6,-100,100,shift,matrix))
            if ID == 2:
                task.append(__class__.Task(50,7,-100,100,shift,matrix))
            if ID == 3: 
                task.append(__class__.Task(50,17,-100,100,shift,matrix))
                f = path+'/__references__/WCCI2022/SO-Complex-Benchmarks/Tasks/shuffle/shuffle_data_' + str(17) + "_D" + str(dim)+'.txt'
                task[i-1].SS = np.loadtxt(f).astype(int)
            if ID == 4:
                task.append(__class__.Task(50,13,-100,100,shift,matrix))
            if ID == 5: 
                task.append(__class__.Task(50,15,-100,100,shift,matrix))
            if ID == 6:
                task.append(__class__.Task(50,21,-100,100,shift,matrix))
                f = path+'/__references__/WCCI2022/SO-Complex-Benchmarks/Tasks/shuffle/shuffle_data_' + str(21) + "_D" + str(dim)+'.txt'
                task[i-1].SS = np.loadtxt(f).astype(int)
            if ID == 7:
                task.append(__class__.Task(50,22,-100,100,shift,matrix))
                f = path+'/__references__/WCCI2022/SO-Complex-Benchmarks/Tasks/shuffle/shuffle_data_' + str(22) + "_D" + str(dim)+'.txt'
                task[i-1].SS = np.loadtxt(f).astype(int)
            if ID == 8:
                task.append(__class__.Task(50,5,-100,100,shift,matrix))
            if ID == 9:
                if i == 1:
                    task.append(__class__.Task(50,11,-100,100,shift,matrix))
                else :
                    task.append(__class__.Task(50,16,-100,100,shift,matrix))
            if ID == 10:
                if i == 1:
                    task.append(__class__.Task(50,20,-100,100,shift,matrix))
                    f = path+'/__references__/WCCI2022/SO-Complex-Benchmarks/Tasks/shuffle/shuffle_data_' + str(20) + "_D" + str(dim)+'.txt'
                    task[0].SS = np.loadtxt(f).astype(int)
                else:
                    task.append(__class__.Task(50,21,-100,100,shift,matrix))
                    f = path+'/__references__/WCCI2022/SO-Complex-Benchmarks/Tasks/shuffle/shuffle_data_' + str(21) + "_D" + str(dim)+'.txt'
                    task[1].SS = np.loadtxt(f).astype(int)
        return task, Individual_func