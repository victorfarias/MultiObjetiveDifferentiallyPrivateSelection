import numpy as np

def domination_metric(x1:np.ndarray , y1:np.ndarray, x2: np.ndarray, y2:np.ndarray):
    dominated_count = 0
    for x_, y_ in zip(x2, y2):
        dom = (x_ < x1) & (y_ < y1)
        if np.any(dom):
            dominated_count += 1
    return dominated_count/len(x2)
    


    