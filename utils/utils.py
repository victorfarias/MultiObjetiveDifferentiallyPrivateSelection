import mpmath as mp
import numpy as np
import random

from lru import LRU

def cache_numpy_array(indexes=[0], other_indexes=[]):
    def _cache_numpy_array(f):
        # cache = {}
        # cache = LRU(int(1e8))
        cache = LRU(50)
        def cached_f(*args, **kwargs):                        
            key = ()
            for i in indexes:
                buffer = np.asarray(args[i]).data.tobytes()
                key += (buffer,)            
            for i in other_indexes:
                key += (args[i],)
            if key in cache:
                return cache[key]
            else:
                ret = f(*args, **kwargs)
                cache[key] = ret
                return np.copy(ret)
        return cached_f
    return _cache_numpy_array

def reset_seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)