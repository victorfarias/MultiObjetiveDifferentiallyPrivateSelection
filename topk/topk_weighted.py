from typing import Any

import numpy as np

from dp_mechanisms.local_dampening import local_dampening
from dp_mechanisms.exponential_mechanism import exponential_mechanism
from dp_mechanisms.delta_weighted import DeltaWeighted


class TopkWeighted:
    def __init__(
        self,
        u1: np.ndarray,
        u2: np.ndarray,
        w1: float,
        w2: float,
        k: int,
    ):
        self.k = k
        self.u = u1 * w1 + u2 * w2

    def execute(self):
        return np.argsort(self.u)[::-1][: self.k]


class TopKWeightedExponential:
    def __init__(
        self,
        u1: np.ndarray,
        u2: np.ndarray,
        w1: float,
        w2: float,
        gs1: float,
        gs2: float,        
        k: int,
        epsilon: float,
    ):
        self.k = k
        self.epsilon = epsilon
        self.u = u1 * w1 + u2 * w2
        self.gs = gs1 * w1 + gs2 * w2

    def execute(self):
        return exponential_mechanism(self.u, self.epsilon, self.gs, self.k)


class TopKWeightedLocalDampening:
    def __init__(
        self,
        u1: np.ndarray,
        u2: np.ndarray,
        w1: float,
        w2: float,
        Delta1: Any,
        Delta2: Any,        
        k: int,
        epsilon: float,
    ):
        self.k = k
        self.epsilon = epsilon
        self.w1 = w1
        self.w2 = w2
        self.u = u1 * w1 + u2 * w2
        self.Delta1 = Delta1
        self.Delta2 = Delta2
    
    def execute(self):
        
        def Delta(i):
            delta1 = self.Delta1(i)
            delta2 = self.Delta2(i)
            return DeltaWeighted(delta1, delta2, self.w1, self.w2)
        
        return local_dampening(self.u, self.epsilon, Delta, k=self.k)


if __name__ == '__main__':
    from topk.topk_weighted import TopKWeightedLocalDampening
    from local_sensitivity.egocentric_density import delta_ed
    from local_sensitivity.degree import delta_degree as DeltaDegree

    from data.loader import load_enron

    g = load_enron()
    u_ego_density = g.ed()
    u_degrees = g.degrees().astype(np.float64)
    T = g.T().astype(np.float64)

    epsilon = 1000.0
    k = 10
    w_ed = 100
    w_deg = 1

    def DeltaEd(i):
        return delta_ed(T, u_degrees, int(i))

    TopKWeightedLocalDampening(u_ego_density, u_degrees, w_ed, w_deg, DeltaEd, DeltaDegree, k, epsilon).execute()