import numpy as np

from data.loader import Dataset, load_pimas
from dp_mechanisms.exponential_mechanism import exponential_mechanism
from tree.pareto_tree import ParetoTree


class ParetoTreeDP(ParetoTree):
    
    def __init__(self, 
                dataset: Dataset,
                n_population: int, 
                n_selection: int, 
                initial_depth: int, 
                max_depth: int, 
                epsilon: float,
                n_iter: int = 50, 
                n_output = 1,
                verbose=False) -> None:


        super().__init__(dataset, n_population, 
                         n_selection,                          
                         initial_depth, 
                         max_depth, 
                         n_iter, 
                         n_output=n_output,
                         verbose=verbose)
        self.epsilon_iteration = epsilon / n_iter
        n = self.X.shape[0]
        self.sensitivity = n-1
        

    def select(self, population, fitness, m_selection=None):
        if m_selection is None:
            m_selection = self.m_selection        
        return np.array(population)[
            exponential_mechanism(fitness, 
                                    self.epsilon_iteration, 
                                    self.sensitivity, 
                                    k=m_selection)
        ]
    


# if __name__ == '__main__':
    # X = np.array([[0.5, 0.2], [0.3, 0.4], [0.1, 0.2], [0.2, 0.1], [0.4, 0.3], [0.6, 0.7], [0.8, 0.9], [0.9, 0.8], [0.7, 0.6], [0.5, 0.5]])
    # y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    # X, y = load_pimas()
    
    # gen = ParetoTreeDP(X, y, 30, 2, 4, 10, 1., n_iter=3, verbose=True)
    # gen.execute()