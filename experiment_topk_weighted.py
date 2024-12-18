import os
from typing import Optional

import numpy as np
from tqdm import trange

from local_sensitivity.degree import delta_degree as DeltaDegree
from local_sensitivity.egocentric_density import delta_ed
from topk.topk_weighted import (
    TopkWeighted,
    TopKWeightedExponential,
    TopKWeightedLocalDampening,
)
from utils.logger import ExperimentLogger
from utils.metrics import precision, recall
from utils.utils import reset_seed

if os.name != 'nt':
    from graph.graph import Graph

n_experiments = 500
epsilons = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20., 100., 1000., 10000., 100000.]
ks = [5]
w_ed = 100
w_deg = 1


def open_logger(csv_path:str)->ExperimentLogger:
    return ExperimentLogger(csv_path, ['iter', 'method', 'epsilon', 'k', 'recall', 'precision'])

if os.name != 'nt':
    def experiment_topk_weighted_exp(g:Graph, logger:Optional[ExperimentLogger]=None):
        u_ego_density = g.ed()
        u_degrees = g.degrees().astype(np.float64)

        def method_exponential(k, epsilon):
            return TopKWeightedExponential(u_ego_density, u_degrees, w_ed, w_deg, 1, 1, k, epsilon)
        
        experiment(u_ego_density, u_degrees, method_exponential, "TopKWeightedExponential", logger)


    def experiment_topk_weighted_ld(g:Graph, logger:Optional[ExperimentLogger]=None):
        u_ego_density = g.ed()
        u_degrees = g.degrees().astype(np.float64)
        T = g.T().astype(np.float64) 

        def DeltaEd(i):
            return delta_ed(T, u_degrees, int(i))

        def method_local_dampening(k, epsilon):
            return TopKWeightedLocalDampening(u_ego_density, u_degrees, w_ed, w_deg, DeltaEd, DeltaDegree, k, epsilon)

        experiment(u_ego_density, u_degrees, method_local_dampening, "TopKWeightedLocalDampening", logger)



def experiment(u_ego_density, u_degrees, Method, method_name, logger:Optional[ExperimentLogger]=None):

    # epss = epsilons if dp else [None]
    reset_seed()

    for k in ks:
        
        method_truth = TopkWeighted(u_ego_density, u_degrees, w_ed, w_deg, k)  
        true_topk = method_truth.execute()

        for epsilon in epsilons:

            recalls = np.zeros(n_experiments)    
            precisions = np.zeros(n_experiments)  
            
            method = Method(k, epsilon)   
                     
            for i in trange(n_experiments, desc=f"Running {method_name} with epsilon={epsilon} and k={k}"):

                retrieved_topk = method.execute()

                recalls[i] = recall(true_topk, retrieved_topk)
                precisions[i] = precision(true_topk, retrieved_topk)

                logger.log({
                    'iter': i,
                    'method': method_name,
                    'epsilon': epsilon,
                    'k': k,
                    'recall': recalls[i],
                    'precision': precisions[i]
                })
            
            print(f"Recall for {method_name} with epsilon={epsilon} and k={k}: {np.mean(recalls)}")
            print(f"Precision for {method_name} with epsilon={epsilon} and k={k}: {np.mean(precisions)}")
            print()