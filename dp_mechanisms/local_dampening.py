import numpy as np
# from tqdm import tqdm

from dp_mechanisms.exponential_mechanism import exponential_mechanism
from dp_mechanisms.local_dampening_cy import D_
from utils.utils import cache_numpy_array

@cache_numpy_array()
def D(u, Delta):
    nu = u.shape[0]
    du = np.empty(nu, dtype=np.float64)

    for i in range(nu):
        delta = Delta(i)        
        du[i] = D_(u[i], delta)
    return du


def local_dampening(u, epsilon, Delta, k=None):
    du = D(u, Delta)
    return exponential_mechanism(du, epsilon, 1, k=k)


# @cache_numpy_array()
# def D(u, Delta, gs=None, n=None):
#     shift = 0.
#     if (gs is not None) and (n is not None):
#         if gs > 0:
#             shift = - (gs*n + np.amax(u))
#         else:
#             shift = -gs*n - np.amin(u)
#     return D_cy(u + shift, Delta)
