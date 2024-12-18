import numpy as np

def recall(true_topk, retrieved_topk):
    intersection = np.intersect1d(true_topk, retrieved_topk)
    positive_size = true_topk.shape[0]
    intersection_size = intersection.shape[0]    
    return intersection_size/positive_size

def precision(true_topk, retrieved_topk):
    intersection = np.intersect1d(true_topk, retrieved_topk)
    sample_size = retrieved_topk.shape[0]
    intersection_size = intersection.shape[0]
    return intersection_size/sample_size
