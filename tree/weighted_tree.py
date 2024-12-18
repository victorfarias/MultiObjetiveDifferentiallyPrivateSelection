import numpy as np
from bigtree import BinaryNode
from sklearn.metrics import confusion_matrix
from data.loader import Dataset

from tree.decision_tree import DecisionTree


class WeightedTree():

    def __init__(self,
                dataset:Dataset,
                n_population:int,
                n_selection:int,
                w_tpr:float,
                w_tnr:float,
                initial_depth:int,
                max_depth:int,
                n_iter,
                verbose=False,
                ) -> None:
        super().__init__()
        self.n_iter = n_iter
        self.dataset = dataset
        self.X = dataset.X
        self.y = dataset.y
        self.n_population = n_population
        self.n_selection = n_selection
        self.initial_depth = initial_depth
        self.max_depth = max_depth
        self.n_classes = len(np.unique(self.y))
        self.n_attributes = self.X.shape[1]
        self.w_tpr = w_tpr
        self.w_tnr = w_tnr
        self.verbose = verbose
        # self.fitness_history = []
        # self.tpr_history = []
        # self.tnr_history = []
        # self.tree_sizes = []
        # self.tree_depths = []        
        # print(f"n_population = {self.n_population}")
        # print(f"n_selection = {self.n_selection}")
        # print(f"initial_depth = {self.initial_depth}")
        # print(f"max_depth = {self.max_depth}")
        # print(f"n_iter = {self.n_iter}")
        # print(f"w_tpr = {self.w_tpr}")
        # print(f"w_tnr = {self.w_tnr}")

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
    
    def fitness(self, population) -> np.ndarray:
        self.tprs = []
        self.tnrs = []
        self.tps = []
        self.tns = []
        self.ps = []
        self.ns = []

        for tree in population:
            y_pred = tree.predict(self.X)
            tn, fp, fn, tp = confusion_matrix(self.y, y_pred).ravel()
            
            p = tp+fn
            n = tn+fp
            tpr = tp/p
            tnr = tn/n

            self.tprs.append(tpr)
            self.tnrs.append(tnr)
            self.tps.append(tp)
            self.tns.append(tn)
            self.ps.append(p)
            self.ns.append(n)
        
        fitnesses = np.array(self.tprs)*self.w_tpr + np.array(self.tnrs)*self.w_tnr
        return fitnesses
        

    def select(self, population, fitness, n_selection=None):
        if n_selection is None:
            n_selection = self.n_selection        
        return np.array(population)[np.argsort(fitness)[-n_selection:]]
    
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
    #              w_tpr=self.w_tpr,
    #              w_tnr=self.w_tnr,
    #              m_population=self.m_population,
    #              m_selection=self.m_selection,
    #              initial_depth=self.initial_depth,
    #              max_depth=self.max_depth,
    #              n_iter=self.n_iter,
    #              n_classes=self.n_classes,
    #              n_attributes=self.n_attributes,
    #              tprs=self.tpr_history,
    #              tnrs=self.tnr_history)
        

    def execute(self) -> DecisionTree:
        population = self.initialize()

        for i in range(self.n_iter-1):                    
            
            fitness = self.fitness(population)
            
            # self.report(population, i, fitness)
            # self.append_result(population, fitness)

            population = self.select(population, fitness)

            for _ in range(self.n_population//2):
                parent1, parent2 = np.random.choice(population, 2, replace=False)

                child1, child2 = self.crossover(parent1, parent2)
                self.mutate(child1)
                self.mutate(child2)

                population = np.append(population, child1)
                population = np.append(population, child2)
        
        fitness = self.fitness(population)
        best_trees = self.select(population, fitness, 1)
        fitness = self.fitness(best_trees)

        # self.report(best_trees, self.n_iter, fitness)
        # self.append_result(best_trees, fitness)

        self.best_tree = best_trees[0]
        self.final_fitness = fitness[0]
        self.final_tpr = self.tprs[0]
        self.final_tnr = self.tnrs[0]
        self.final_accuracy = (self.tps[0]+self.tns[0])/(self.ps[0]+self.ns[0])

        return best_trees[0]
                

    def report(self, population, i, fitness):
        if self.verbose:
            print(f'Iteration {i+1}')
            print(f'Fitness: {fitness}')
            print(f'Best fitness: {np.max(fitness)}')
            print(f'Worst fitness: {np.min(fitness)}')                
            print(f'Mean fitness: {np.mean(fitness)}')
            print(f'True positive rates: {self.tprs}')
            print(f'True negatives rates: {self.tnrs}')
            print(f'Tree depths: {[tree.root.max_depth for tree in population]}')
            print(f'Tree sizes: {[tree.root.subtree_size for tree in population]}')            
            print()


if __name__ == '__main__':
    from data.loader import load_pimas
    from cProfile import Profile
    from pstats import SortKey, Stats

    dataset = load_pimas()
    X, y = dataset.X, dataset.y
    gen = WeightedTree(dataset, 100, 2, 2, 3, 4, 10, n_iter=3, verbose=False)
    
    with Profile() as profile:
        print(f"{gen.execute() = }")        
        (
            Stats(profile)
            .strip_dirs()
            .sort_stats(SortKey.CUMULATIVE)
            .print_stats()
        )

    
    