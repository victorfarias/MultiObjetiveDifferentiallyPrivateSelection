import numpy as np
import mpmath as mp

np.seterr(all='raise')

# from utils.utils import safe_exp, safe_normalize

# @cache_numpy_array
# def safe_exp(v):
#     try:
#         return np.exp(v)
#     except (FloatingPointError, RuntimeWarning):
#         exp = np.vectorize(mp.exp)
#         return exp(v)
    
def safe_exp(v):
    v = v - (np.amax(v) - 708.)

    # prevent underflow
    v[ v < -708] = -708
    exps = np.exp(v)
    return exps

# @cache_numpy_array
def safe_normalize(a):
    try:
        return a/np.sum(a)
    except (FloatingPointError, RuntimeWarning):
        a_ = mp.matrix(a)
        summ = mp.fsum(a_)
        normalized = a_/summ
        nparray = np.array(normalized.tolist(), dtype=np.float64)
        return nparray[:,0]


def exponential_mechanism(u, epsilon, sensitivity, k=None):    
    e_prime = epsilon if k is None else epsilon/k    
    u_ = (u*e_prime)/(2.0*sensitivity)

    # prevent overflow

    exps = safe_exp(u_)
    probs = safe_normalize(exps)    
    # probs[probs < np.finfo(np.float64).eps] = 0.

    if k is not None and np.count_nonzero(probs) >= k:
        try:
            return np.random.choice(len(u_), p=probs, replace=False, size=k)
        except Exception as e:
            e.args = (np.amax(probs), np.amin(probs), np.sum(probs), *e.args)
            raise e
    else:
        return np.argsort(u_)[-k:] if k is not None else np.argmax(u_)

    # if k is not None and np.count_nonzero(probs) >= k:
    #     try:
    #         return np.random.choice(len(u_), p=probs, replace=False, size=k)
    #     except Exception as e:
    #         print('Entrou aqui')
    #         print(np.amax(probs))
    #         print(np.amin(probs))
    #         print(e)
    #         return np.argsort(u)[-k:]
    # else:
    #     return np.argsort(u_)[-k:] if k is not None else np.argmax(u_)

    

# def exponential_mechanism(u, epsilon, sensitivity, k=None):
#     e_prime = epsilon if k is None else epsilon/k
#     u = u - (np.amax(u) + np.amin(u))/2
#     exps = safe_exp((u*e_prime)/(2*sensitivity))
#     probs = safe_normalize(exps)
    
#     pick = None
#     try:
#         pick = np.random.choice(len(u), p=probs, replace=False, size=k)        
#     except BaseException as e:
#         probs[probs < np.finfo(np.float64).eps] = np.finfo(np.float64).eps
#         probs = probs.astype(np.float64)
#         pick = np.random.choice(len(u), p=probs, replace=False, size=k)

#     return pick


# def exponential_mechanism(u, epsilon, sensitivity, k=None):
#     e_prime = epsilon if k is None else epsilon/k
#     # u_ = u - (np.amax(u) + np.amin(u))/2
#     u = (u*e_prime)/(2*sensitivity)

#     # prevent underflow/overflow
#     u = u - (np.amax(u) - 708.)
#     u[u < -0.] = -0.

#     # max_ = np.amax(u_)
#     # if max_ > 708.:
#     #     u_ = u_ - (708. - max_)
#     exps = np.exp(u)
#     probs = exps/np.sum(exps)
#     return np.random.choice(len(u), p=probs, replace=False, size=k)        
    