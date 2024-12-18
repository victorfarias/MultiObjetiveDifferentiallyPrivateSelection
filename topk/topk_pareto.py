from typing import Any

import numpy as np

from dp_mechanisms.exponential_mechanism import exponential_mechanism
from dp_mechanisms.local_dampening import local_dampening
from pareto.pareto_delta import delta_pareto_set_ls, gs_pareto
from pareto.pareto_score import pareto_set_score2d as pareto_set_score2d_
from utils.utils import cache_numpy_array

@cache_numpy_array(indexes=[0, 1])
def pareto_set_score2d(u1: np.ndarray, u2: np.ndarray):
    return np.array(pareto_set_score2d_(u1, u2))

class TopkPareto:
    def __init__(
        self,
        u1: np.ndarray,
        u2: np.ndarray,        
        k: int,
    ):
        self.k = k
        self.u = pareto_set_score2d(u1, u2)

    def execute(self):
        return np.argsort(self.u)[::-1][: self.k]


class TopKParetoExponential:
    def __init__(
        self,
        u1: np.ndarray,
        u2: np.ndarray,
        k: int,
        epsilon: float,
    ):
        self.k = k
        self.epsilon = epsilon
        self.u = pareto_set_score2d(u1, u2)   
        n = len(u1)     
        self.gs = gs_pareto(n)

    def execute(self):
        return exponential_mechanism(self.u, self.epsilon, self.gs, self.k)


# class TopKWeightedLocalDampeningGS:
#     def __init__(
#         self,
#         u1: np.ndarray,
#         u2: np.ndarray,        
#         gs1: float,
#         gs2: float,        
#         k: int,
#         epsilon: float,
#     ):
#         self.k = k
#         self.epsilon = epsilon        
#         self.u = pareto_set_score2d(u1, u2)           
#         self.u12 = np.c_[u1, u2]
#         self.gs1 = gs1
#         self.gs2 = gs2
    
#     def execute(self):
        
#         def Delta(i:int):            
#             return delta_pareto(self.u12, int(i), self.gs1, self.gs2)
        
        # return local_dampening(self.u, self.epsilon, Delta, k=self.k)

class TopKParetoLocalDampeningLS:
    def __init__(
        self,
        u1: np.ndarray,
        u2: np.ndarray,        
        Delta1: Any,
        Delta2: Any,
        k: int,
        epsilon: float,
    ):
        self.k = k
        self.epsilon = epsilon        
        self.u = pareto_set_score2d(u1, u2)           
        self.Delta1 = Delta1
        self.Delta2 = Delta2
        self.u1 = u1
        self.u2 = u2

    
    def execute(self):        
        
        def Delta(i:int):            
            return delta_pareto_set_ls(int(i), self.u1, self.u2, self.Delta1, self.Delta2)
        
        return local_dampening(self.u, self.epsilon, Delta, k=self.k)
