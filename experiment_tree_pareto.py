from typing import Optional
import numpy as np
import ray

from pareto.pareto_metrics import domination_metric
from tqdm import trange, tqdm
from utils.logger import ExperimentLogger
from utils.utils import reset_seed
from tree.pareto_tree import ParetoTree
from tree.pareto_tree_dp import ParetoTreeDP
from tree.pareto_tree_dp_ld_ls import ParetoTreeDPLDLS


n_experiments = 500
epsilons = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10., 20., 100., 1000., 10000., 100000.]
n_population = 30
n_selection = 3
initial_depth = 4
max_depth = 10
n_iter = 3
n_output = 3

ray.init()

def open_logger(csv_path:str)->ExperimentLogger:
    return ExperimentLogger(csv_path, ['iter', 'epsilon', 'method1', 'method2', 'method1_dominates_method2', 'method2_dominates_method1'])


def experiment_tree_pareto_exp(dataset, logger:ExperimentLogger):

    def Gen1(_):
        return ParetoTree(dataset, n_population, n_selection, initial_depth, max_depth, n_iter=n_iter, n_output=n_output)

    def Gen2(epsilon): 
        return ParetoTreeDP(dataset, n_population, n_selection, initial_depth, max_depth, epsilon, n_iter=n_iter, n_output=n_output)
    
    experiment(Gen1, Gen2, "no_privacy", "exp_mec", logger)



def experiment_tree_pareto_ld(dataset, logger:ExperimentLogger):

    def Gen1(_):
        return ParetoTree(dataset, n_population, n_selection, initial_depth, max_depth, n_iter=n_iter, n_output=n_output)

    def Gen4(epsilon):
        return ParetoTreeDPLDLS(dataset, n_population, n_selection, initial_depth, max_depth, epsilon, n_iter=n_iter, n_output=n_output)
    
    return experiment(Gen1, Gen4, "no_privacy", "local_dampening_local", logger)

@ray.remote
def run_experiment(i, Gen1, Gen2, epsilon):
    
    reset_seed(i)
    #Gen1 = ray.get(gen1_ref)
    #Gen2 = ray.get(gen2_ref)

    gen1 = Gen1(epsilon)
    gen2 = Gen2(epsilon)

    try:
        _, tprs1, tnrs1 = gen1.execute()    
        _, tprs2, tnrs2 = gen2.execute()
    except Exception as e:
        return _,_,e.args 

    dom1 = domination_metric(tprs1, tnrs1, tprs2, tnrs2)
    dom2 = domination_metric(tprs2, tnrs2, tprs1, tnrs1)
    return dom1, dom2, _


def experiment(Gen1, Gen2, gen1_name, gen2_name, logger:Optional[ExperimentLogger]=None,):

    gen1_ref = ray.put(Gen1)
    gen2_ref = ray.put(Gen2)

    for epsilon in epsilons:        

        tasks = [run_experiment.remote(i, gen1_ref, gen2_ref, epsilon) for i in range(n_experiments)]
        
        results = []
        with tqdm(total=len(tasks), desc=f"epsilon={epsilon} - {gen1_name} vs {gen2_name}") as pbar:
            while tasks:
                done, tasks = ray.wait(tasks)
                
                for future in done:
                    result = ray.get(future)
                    results.append(result)
                    pbar.update(1)

        
        domination_gen1 = np.zeros(n_experiments)
        domination_gen2 = np.zeros(n_experiments)
        for i, result in enumerate(results):

            dom1, dom2, e = result
            # print(dom1, dom2)

            if dom1 is None or dom2 is None:
                print('Skipped')
                print(e)                
                print(e.args)                
                continue
            
            domination_gen1[i] = dom1
            domination_gen2[i] = dom2        
        
            logger.log({
                'iter': i,
                'epsilon': epsilon,
                'method1': gen1_name,
                'method2': gen2_name,
                'method1_dominates_method2': domination_gen1[i],
                'method2_dominates_method1': domination_gen2[i]
            })            
        

        print(f"Domination metric - How much {gen1_name} dominates {gen2_name}")
        print(domination_gen1.mean())
        print(f"Domination metric - How much {gen2_name} dominates {gen1_name}")
        print(domination_gen2.mean())
        print()

if __name__ == '__main__':
    from data.loader import load_adult

    dataset = load_adult()

    dataset_name = 'Adult_teste'
    csv_path = f'./out/pareto_trees/{dataset_name.lower()}.csv'
    logger = open_logger(csv_path)

    experiment_tree_pareto_exp(dataset, logger)
