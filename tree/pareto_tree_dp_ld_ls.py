import numpy as np

from data.loader import Dataset, load_pimas
from dp_mechanisms.local_dampening import local_dampening
from tree.pareto_tree import ParetoTree
from tree.delta_tpr import DeltaTpr as DeltaTpr_
from tree.delta_tnr import DeltaTnr as DeltaTnr_
from pareto.pareto_delta import delta_pareto_set_ls


class ParetoTreeDPLDLS(ParetoTree):
    
    def __init__(self, 
                dataset: Dataset, 
                m_population: int, 
                m_selection: int, 
                initial_depth: int, 
                max_depth: int, 
                epsilon: float,
                n_iter: int = 50, 
                n_output = 1,
                verbose=False) -> None:


        super().__init__(dataset, m_population, 
                         m_selection,                          
                         initial_depth, 
                         max_depth, 
                         n_iter=n_iter, 
                         n_output=n_output,
                         verbose=verbose)
        self.epsilon_iteration = epsilon / n_iter

        

    def select(self, population, fitness, m_selection=None):
        if m_selection is None:
            m_selection = self.m_selection        

        def DeltaTpr(i):
            return DeltaTpr_(self.tps[i], self.ps[i])
            
        def DetlaTnr(i):
            return DeltaTnr_(self.tns[i], self.ns[i])
        
        def DeltaParetoSetLs(i):
            return delta_pareto_set_ls(int(i), np.array(self.tprs), np.array(self.tnrs), DeltaTpr, DetlaTnr)
        
        return np.array(population)[
            local_dampening(fitness, self.epsilon_iteration, DeltaParetoSetLs, k=m_selection)
        ]
        
        # def DeltaParetoSet(i):
        #     return delta_pareto(np.c_[self.tprs, self.tnrs], int(i), 1., 1.)

        # return local_dampening(fitness, self.epsilon_iteration, DeltaParetoSet, k=m_selection)



# if __name__ == '__main__':
    # X = np.array([[0.5, 0.2], [0.3, 0.4], [0.1, 0.2], [0.2, 0.1], [0.4, 0.3], [0.6, 0.7], [0.8, 0.9], [0.9, 0.8], [0.7, 0.6], [0.5, 0.5]])
    # y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

    # X, y = load_pimas()
    
    # gen = ParetoTreeDPLDLS(X, y, 30, 2, 4, 10, 1., n_iter=3, verbose=True)
    # gen.execute()