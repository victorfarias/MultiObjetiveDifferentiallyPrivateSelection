import os
from typing import Optional

import numpy as np
from tqdm import trange

from local_sensitivity.degree import delta_degree as DeltaDegree
from local_sensitivity.egocentric_density import delta_ed
from pareto.pareto_metrics import domination_metric
from topk.topk_pareto import (
    TopkPareto,
    TopKParetoExponential,
    TopKParetoLocalDampeningLS,
)
from utils.logger import ExperimentLogger
from utils.utils import reset_seed

if os.name != 'nt':
    from graph.graph import Graph

np.seterr(all='raise')


n_experiments = 500
epsilons = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20., 50.]
k = 3

def open_logger(csv_path:str)->ExperimentLogger:
    return ExperimentLogger(csv_path, ['iter', 'epsilon', 'method1', 'method2', 'method1_dominates_method2', 'method2_dominates_method1'])

if os.name != 'nt':
    def experiment_topk_pareto_exp(g:Graph, logger:Optional[ExperimentLogger]=None):    
        eds = g.ed()
        degrees = g.degrees().astype(np.float64)

        def method1(_):
            return TopkPareto(eds, degrees, k)
        
        def method2(epsilon): 
            return TopKParetoExponential(eds, degrees, k, epsilon)
        
        experiment(eds, degrees, method1, method2, "TopkParetoNoDP", "TopKParetoExponential", logger)


    def experiment_topk_pareto_ld(g:Graph, logger:Optional[ExperimentLogger]=None):

        eds = g.ed()
        degrees = g.degrees().astype(np.float64)
        T = g.T().astype(np.float64)

        def method1(_):
            return TopkPareto(eds, degrees, k)
        
        def DeltaED(i):
            return delta_ed(T, degrees, int(i))

        def method4(epsilon):
            return TopKParetoLocalDampeningLS(eds, degrees, DeltaED, DeltaDegree, k, epsilon)
        
        experiment(eds, degrees, method1, method4, "TopkParetoNoDP", "TopKWeightedLocalDampeningLS", logger)



def experiment(u1, u2, method1, method2, method1_name, method2_name, logger:Optional[ExperimentLogger]=None):

    reset_seed()

    for epsilon in epsilons:
        
        print(f"Running experiments for epsilon={epsilon}")

        domination_method1 = np.zeros(n_experiments)
        domination_method2 = np.zeros(n_experiments)

        met1 = method1(epsilon)
        met2 = method2(epsilon)
        
        for i in trange(n_experiments, desc=f"epsilon={epsilon} - {method1_name} vs {method2_name} - k={k}"):
            try:
                indexes1 = met1.execute()    
                indexes2 = met2.execute()
            except Exception as e:
                print('Skipping...')
                print(e)
                print(e.args)
                continue            

            retrieved_u1_method1 = u1[indexes1]
            retrieved_u2_method1 = u2[indexes1]
            retrieved_u1_method2 = u1[indexes2]
            retrieved_u2_method2 = u2[indexes2]

            domination_method1[i] = domination_metric(retrieved_u1_method1, retrieved_u2_method1, retrieved_u1_method2, retrieved_u2_method2)
            domination_method2[i] = domination_metric(retrieved_u1_method2, retrieved_u2_method2, retrieved_u1_method1, retrieved_u2_method1)

            if logger:
                logger.log({
                    'iter': i,
                    'epsilon': epsilon,
                    'method1': method1_name,
                    'method2': method2_name,
                    'method1_dominates_method2': domination_method1[i],
                    'method2_dominates_method1': domination_method2[i]
                })            
            

        print(f"Domination metric - How much {method1_name} dominates {method2_name}")
        print(domination_method1.mean())
        print(f"Domination metric - How much {method2_name} dominates {method1_name}")
        print(domination_method2.mean())
        print()