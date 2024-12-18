from typing import Optional

import numpy as np
from sklearn.metrics import confusion_matrix
from tqdm import trange

from data.loader import Dataset, split_dataset
from tree.decision_tree import DecisionTree
from tree.weighted_tree import WeightedTree
from tree.weighted_tree_dp import WeightedTreeDP
from tree.weighted_tree_dp_ld import WeightedTreeDPLD
from utils.logger import ExperimentLogger
from utils.utils import reset_seed

n_experiments = 500
epsilons = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0]
n_population = 100
n_selection = 20
initial_depth = 3
max_depth = 5
n_iter = 2
w_tpr = 3
w_tnr = 2


def open_logger(csv_path:str)->ExperimentLogger:
    return ExperimentLogger(csv_path, ['epsilon', 'method', 'fitness', 'accuracy', 'tpr', 'tnr'])


def experiment_tree_weighted_nodp(dataset:Dataset, logger:Optional[ExperimentLogger]=None):
    dataset_train, dataset_test = split_dataset(dataset)
    
    def nodp_method(_):
        return WeightedTree(dataset_train, n_population, n_selection, w_tpr, w_tnr, initial_depth, max_depth, n_iter)

    experiment(nodp_method, "nodp_weighted_tree", dataset_test, w_tpr, w_tnr, dp=False, logger=logger)


def experiment_tree_weighted_dp(dataset:Dataset, logger:Optional[ExperimentLogger]=None):
    dataset_train, dataset_test = split_dataset(dataset)
    
    def method2(epsilon):
        return WeightedTreeDP(dataset_train, n_population, n_selection, w_tpr, w_tnr, initial_depth, max_depth, n_iter, epsilon)

    experiment(method2, "weighted_tree_dp_exponential", dataset_test, w_tpr, w_tnr, dp=True, logger=logger)


def experiment_tree_weighted_dp_ld(dataset:Dataset, logger:Optional[ExperimentLogger]=None):
    dataset_train, dataset_test = split_dataset(dataset)
    
    def method3(epsilon):
        return WeightedTreeDPLD(dataset_train, n_population, n_selection, w_tpr, w_tnr, initial_depth, max_depth, n_iter, epsilon)

    experiment(method3, "weighted_tree_dp_ld", dataset_test, w_tpr, w_tnr, dp=True, logger=logger)



def experiment(Method, 
               method_name:str, 
               dataset_test: Dataset, 
               w_tpr:float, 
               w_tnr:float,
               dp:bool=True, 
               logger:Optional[ExperimentLogger]=None,
               epsilons:list[float]=epsilons
):      

    reset_seed()

    epss = epsilons if dp else [None]

    for epsilon in epss:

        fitnesses = np.zeros(n_experiments)    
        accuracies = np.zeros(n_experiments)    
        tprs = np.zeros(n_experiments)
        tnrs = np.zeros(n_experiments)
        
        for i in trange(n_experiments, desc=f"Experiments for {method_name} with epsilon={epsilon}"):

            method:WeightedTree = Method(epsilon)            

            tree:DecisionTree = method.execute()            

            y_pred = tree.predict(dataset_test.X)            
            tn, fp, fn, tp = confusion_matrix(dataset_test.y, y_pred).ravel()
            
            p = tp+fn
            n = tn+fp
            tpr = tp/p
            tnr = tn/n
            accuracy = (tp+tn)/(p+n)
            fitness = tpr*w_tpr + tnr*w_tnr
            
            fitnesses[i] = fitness
            accuracies[i] = accuracy
            tprs[i] = tpr
            tnrs[i] = tnr


        if logger is not None:    
            logger.log({
                'epsilon': epsilon,
                'method': method_name,
                'fitness': fitnesses.mean(),
                'accuracy': accuracies.mean(),
                'tpr': tprs.mean(),
                'tnr': tnrs.mean()
            })

        print("Mean fitness " + str(fitnesses.mean()))
        print("Mean accuracy " + str(accuracies.mean()))
        print("Mean tpr " + str(tprs.mean()))
        print("Mean tnr " + str(tnrs.mean()))
        print()