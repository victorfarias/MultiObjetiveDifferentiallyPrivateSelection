import numpy as np

from sklearn.metrics import confusion_matrix
from data.loader import Dataset, load_pimas
from pareto.pareto_score import pareto_set_score

from tree.decision_tree import DecisionTree


class ParetoTree():

    def __init__(self,
                dataset: Dataset,
                n_population:int,
                n_selection:int,
                initial_depth:int,
                max_depth:int,
                n_iter: int = 2,
                n_output : int = 1,
                verbose : bool = False,
                ) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.dataset = dataset
        self.X = dataset.X
        self.y = dataset.y
        self.n_population = n_population
        self.m_selection = n_selection
        self.initial_depth = initial_depth
        self.max_depth = max_depth
        self.n_classes = len(np.unique(self.y))
        self.n_attributes = self.X.shape[1]
        self.verbose = verbose
        self.fitness_history = []
        self.tpr_history = []
        self.tnr_history = []
        self.tree_sizes = []
        self.tree_depths = []   
        self.n_output = n_output 
        self.tprs = None
        self.tnrs = None
        self.tps = None
        self.tns = None
        self.ps = None
        self.ns = None

    def initialize(self):
        return [
            DecisionTree(self.dataset, self.initial_depth).fit() for _ in range(self.n_population)            
        ]
    
    def crossover(self, parent1: DecisionTree, parent2: DecisionTree) -> tuple[DecisionTree, DecisionTree]:
        
        child1 = parent1.clone()
        child2 = parent2.clone()

        node1 = child1.choose_random_node()
        node2 = child2.choose_random_node()

        if node1.parent is None and node2.parent is None:
            child1.root = node2
            child2.root = node1
        elif node1.parent is None:
            child1.root = node2
            if node2 == node2.parent.left:
                node2.parent.left = node1
            else:
                node2.parent.right = node1
            child1.root.parent = None
        elif node2.parent is None:
            child2.root = node1
            if node1 == node1.parent.left:
                node1.parent.left = node2
            else:
                node1.parent.right = node2
            child2.root.parent = None
        else:                
            if node1 == node1.parent.left and node2 == node2.parent.left:
                parent_node2 = node2.parent
                node1.parent.left = node2
                parent_node2.left = node1
            elif node1 == node1.parent.left and node2 == node2.parent.right:
                parent_node2 = node2.parent
                node1.parent.left = node2
                parent_node2.right = node1
            elif node1 == node1.parent.right and node2 == node2.parent.left:
                parent_node2 = node2.parent
                node1.parent.right = node2
                parent_node2.left = node1
            else:
                parent_node2 = node2.parent
                node1.parent.right = node2
                parent_node2.right = node1

        child1.prune_tree(self.max_depth, self.n_classes)
        child2.prune_tree(self.max_depth, self.n_classes)

        # child1.update_tree_size_depth()
        # child2.update_tree_size_depth()
        child1.assign_tree_size()
        child2.assign_tree_size()
        
        
        return child1, child2
                 
    
    def mutate(self, tree:DecisionTree):    
        node = tree.choose_random_node()
        subtree = tree.create_random_tree(tree.root.max_depth - node.depth + 1)
        if node.parent is None:
            tree.root = subtree
        elif node == node.parent.left:
            node.parent.left = subtree
        else:
            node.parent.right = subtree
        tree.assign_tree_size()        
        # tree.update_tree_size_depth()
    
    def fitness(self, population):
        self.tprs = []
        self.tnrs = []
        self.tps = []
        self.tns = []
        self.ps = []
        self.ns = []

        for tree in population:
            y_pred = tree.predict(self.X)
            tn, fp, fn, tp = confusion_matrix(self.y, y_pred).ravel()
            
            tpr = tp/(tp+fn)
            tnr = tn/(tn+fp)
            p = (tp+fn)
            n = (tn+fp)

            self.tprs.append(tpr)
            self.tnrs.append(tnr)
            self.tps.append(tp)
            self.tns.append(tn)
            self.ps.append(p)
            self.ns.append(n)

        # self.tprs = tprs
        # self.tnrs = tnrs
        # self.tps = tps
        # self.tns = tns
        # self.ps = ps
        # self.ns = ns        
        
        fitness =  pareto_set_score(np.c_[self.tprs, self.tnrs])
        return fitness
        
    def select(self, population, fitness, m_selection=None):
        if m_selection is None:
            m_selection = self.m_selection        
        return np.array(population)[np.argsort(fitness)[-m_selection:]]
    
    # def append_result(self, population, fitness):
    #     self.fitness_history.append(fitness)
    #     self.tree_depths.append([tree.root.max_depth for tree in population])
    #     self.tree_sizes.append([tree.root.subtree_size for tree in population])
    #     self.tpr_history.append(self.tprs)
    #     self.tnr_history.append(self.tnrs)

    # def save_results(self, path):
    #     np.savez(path, 
    #              fitness_history=self.fitness_history, 
    #              tree_depths=self.tree_depths, 
    #              tree_sizes=self.tree_sizes,                 
    #              m_population=self.n_population,
    #              m_selection=self.m_selection,
    #              initial_depth=self.initial_depth,
    #              max_depth=self.max_depth,
    #              n_iter=self.n_iter,
    #              n_classes=self.n_classes,
    #              n_attributes=self.n_attributes,
    #              tprs=self.tpr_history,
    #              tnrs=self.tnr_history)
        
    def execute(self):
        population = self.initialize()

        for i in range(self.n_iter-1):                    
            
            fitness = self.fitness(population)
            
            # self.report(population, i, fitness)
            # self.append_result(population, fitness)

            population = self.select(population, fitness)

            for _ in range(self.n_population//2):
                parent1, parent2 = np.random.choice(population, 2, replace=False)

                child1, child2 = self.crossover(parent1, parent2)
                # self.mutate(child1)
                self.mutate(child1)
                # self.mutate(child2)
                self.mutate(child2)

                population = np.append(population, child1)
                population = np.append(population, child2)
        
        fitness = self.fitness(population)
        best_trees = self.select(population, fitness, self.n_output)        
        fitness = self.fitness(best_trees)

        # self.report(best_trees, self.n_iter, fitness)
        # self.append_result(best_trees, fitness)

        return population, np.array(self.tprs), np.array(self.tnrs)
                
    # def report(self, population, i, fitness):
    #     if self.verbose:
    #         print(f'Iteration {i+1}')
    #         print(f'Fitness: {fitness}')
    #         print(f'Best fitness: {np.max(fitness)}')
    #         print(f'Worst fitness: {np.min(fitness)}')                
    #         print(f'Mean fitness: {np.mean(fitness)}')
    #         print(f'True positive rates: {self.tprs}')
    #         print(f'True negatives rates: {self.tnrs}')
    #         print(f'Tree depths: {[tree.root.max_depth for tree in population]}')
    #         print(f'Tree sizes: {[tree.root.subtree_size for tree in population]}')            
    #         print()


if __name__ == '__main__':
    # X = np.array([[0.5, 0.2], [0.3, 0.4], [0.1, 0.2], [0.2, 0.1], [0.4, 0.3], [0.6, 0.7], [0.8, 0.9], [0.9, 0.8], [0.7, 0.6], [0.5, 0.5]])
    # y = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])/

    X, y = load_pimas()
    
    gen = ParetoTree(X, y, 30, 2, 4, 10, n_iter=100, verbose=True)
    gen.execute()