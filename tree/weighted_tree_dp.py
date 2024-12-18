import numpy as np
from sklearn.metrics import confusion_matrix

from data.loader import Dataset, load_pimas
from dp_mechanisms.exponential_mechanism import exponential_mechanism
from tree.weighted_tree import WeightedTree

class WeightedTreeDP(WeightedTree):
    
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
                verbose: bool = False) -> None:        
                                
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
        return np.array(population)[
            exponential_mechanism(fitness, 
                                    self.epsilon_iteration, 
                                    self.sensitivity, 
                                    k=n_selection)
        ]
    

    
# if __name__ == '__main__':
    
#     X, y = load_pimas()
    
#     gen = WeightedTreeDP(X, y, 30, 2, 2, 3, 4, 10, 0.00001, n_iter=100, verbose=True)
#     gen.execute()
