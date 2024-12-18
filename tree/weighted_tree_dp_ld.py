import numpy as np

from data.loader import Dataset, load_pimas
from dp_mechanisms.local_dampening import local_dampening
from tree.delta_tpr import DeltaTpr
from tree.delta_tnr import DeltaTnr
from dp_mechanisms.delta_weighted import DeltaWeighted

from tree.weighted_tree import WeightedTree

class WeightedTreeDPLD(WeightedTree):
    
    def __init__(self, 
                dataset:Dataset,
                n_population: int, 
                n_selection: int, 
                w_tpr: float, 
                w_tnr: float, 
                initial_depth: int, 
                max_depth: int, 
                n_iter: int, 
                epsilon: float,
                verbose=False) -> None:

                                
        super().__init__(dataset, n_population, 
                         n_selection, 
                         w_tpr, 
                         w_tnr, 
                         initial_depth, 
                         max_depth, 
                         n_iter, 
                         verbose)
        self.epsilon_iteration = epsilon / n_iter
        self.sensitivity = w_tpr+w_tnr

        # print(f"n_population = {self.n_population}")
        # print(f"n_selection = {self.n_selection}")
        # print(f"initial_depth = {self.initial_depth}")
        # print(f"max_depth = {self.max_depth}")
        # print(f"n_iter = {self.n_iter}")
        # print(f"w_tpr = {self.w_tpr}")
        # print(f"w_tnr = {self.w_tnr}")
        # print(f"epsilon = {epsilon}")

    def select(self, population, fitness, n_selection=None):
        if n_selection is None:
            n_selection = self.n_selection

        def Delta(i):
            delta_tpr = DeltaTpr(self.tps[i], self.ps[i])
            delta_tnr = DeltaTnr(self.tns[i], self.ns[i])
            return DeltaWeighted(delta_tpr, delta_tnr, self.w_tpr, self.w_tnr)
        
        return np.array(population)[
            local_dampening(fitness, self.epsilon_iteration, Delta, k=n_selection)
        ]

    
# if __name__ == '__main__':
#     # X = np.array([[0.5, 0.2], [0.3, 0.4], [0.1, 0.2], [0.2, 0.1], [0.4, 0.3], [0.6, 0.7], [0.8, 0.9], [0.9, 0.8], [0.7, 0.6], [0.5, 0.5]])
#     # y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

#     X, y = load_pimas()
    
#     gen = DPLDWeightedCostSensitiveTree(X, y, 30, 2, 2, 3, 4, 10, 2, n_iter=100, verbose=True)
#     gen.execute(verbose=True)
