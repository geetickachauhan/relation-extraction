'''
Author: Matthew McDermott
Taken from
https://github.com/mmcdermott/neural_hyperparameter_dists/blob/master/neural_hyperparameter_search/distributions.py
'''

from abc import ABC, abstractmethod

import random, numpy as np
from scipy.stats import poisson, uniform, beta, geom, dlaplace, rv_discrete, bernoulli

# TODO(mmd): Add an rv descriptor.

STATIC_TYPES = [
    type(None), bool, str, int, float, type(lambda:0)
] + list(set(np.typeDict.values()))
SEQ_TYPES = [list, tuple, np.ndarray]

def to_rv(x):
    if type(x) is type(uniform(0, 1)) or isinstance(x, Distribution): return x
    elif type(x) in SEQ_TYPES: return MixtureDistribution(x)
    elif type(x) is dict: return DictDistribution(x)
    else: return DeltaDistribution(x)
def make_rvs(d): return {k: to_rv(v) for k, v in d.items()}
def sample_dict(dict_of_rvs): return {k: v.rvs(1)[0] for k, v in make_rvs(dict_of_rvs).items()}

class Distribution(ABC):
    def __init__(self):
        """Base Distribution class"""

    def __seed(self, random_state):
        np.random.seed(random_state)
        random.seed(random_state)

    @abstractmethod
    def _sample(self):
        raise NotImplementedError

    def rvs(self, b=1, random_state=None):
        assert b >= 1, "Invalid b: %s" % str(b)
        if random_state is not None: self.__seed(random_state)

        num_samples = b if type(b) is int else b[0]
        return [self._sample() for _ in range(num_samples)]

class MarkovianGenerativeProcess(Distribution):
    def __init__(self, base_state):
        """Base Generative Process class"""
        self.prev_state = base_state

    @abstractmethod
    def _stop(self, prev_state, n):
        raise NotImplementedError

    @abstractmethod
    def _next(self, prev_state, n):
        raise NotImplementedError

    def _sample(self):
        generated_path = []
        while not self._stop(self.prev_state, len(generated_path)):
            generated_path.append(self.prev_state)
            self.prev_state = self._next(self.prev_state, len(generated_path))
        return generated_path

class TransformedRV(Distribution):
    def __init__(self, rv, fn, kwargs=True): self.rv, self.fn, self.kwargs = rv, fn, kwargs
    def _sample(self): return self.fn(**(self.rv.rvs(1)[0])) if self.kwargs else self.fn(self.rv.rvs(1))

def rv_int(rv): return TransformedRV(rv, int)

class MixtureDistribution(Distribution):
    def __init__(self, candidate_distributions, weights=None):
        self.candidate_distributions = [to_rv(x) for x in candidate_distributions]
        self.num_components = len(self.candidate_distributions)
        self.ws = weights if weights is not None else [1./self.num_components] * self.num_components
        self.distribution_selection = rv_discrete(
            name='mixture_components',
            values=(range(self.num_components), self.ws)
        )

    def _sample(self):
        # TODO(mmd): improve
        b = 1
        dists = list(self.distribution_selection.rvs(size=b))
        counts = [0] * self.num_components
        #print('Num components: ', self.num_components)
        #print('weights: ', self.ws)
        for dist in dists:
            #print(dist)
            counts[dist] += 1

        vals = [None] * self.num_components
        for i, dist, count in zip(range(self.num_components), self.candidate_distributions, counts):
            if count > 0:
                v = dist.rvs(count)
                #print(v)
                vals[i] = [v] if type(v) in [int, np.int32, np.int64, str] else list(v)

        samples = []
        for dist in dists:
            samples += [vals[dist].pop()]
        return samples[0]

class DictDistribution(Distribution):
    def __init__(self, dict_of_rvs):
        self.dict_of_rvs = make_rvs(dict_of_rvs)

    def _sample(self): return sample_dict(self.dict_of_rvs)

def dict_fcr(dict_of_fns):
    return lambda prev, layer_num: DictDistribution(
        {
            k: v(
                prev[k] if type(prev) is dict else prev, layer_num
            ) if type(v) is type(dict_fcr) else v for k, v in dict_of_fns.items()
        }
    )

class Censored(Distribution):
    def __init__(self, rv, high_limit = None, low_limit = None):
        self.rv, self.high_limit, self.low_limit = rv, high_limit, low_limit

    def _sample(self):
        x = self.rv.rvs(1)
        if type(x) is list or np.ndarray: x = x[0]
        if self.high_limit is not None and x > self.high_limit: return self.high_limit
        if self.low_limit is not None and x < self.low_limit: return self.low_limit
        return x

def build_mult_layer_fcr(
    base_rv, gamma=1, allow_expand=True, allow_decay=True, saturate_above=None, saturate_below=None,
    allow_noise=True,
):
    def layer_fn(prev, layer_num):
        if prev is None: return base_rv
        if type(prev) is list: prev = prev[0]

        high_limit = prev if saturate_above is True else saturate_above
        low_limit = prev if saturate_below is True else saturate_below
        cnsr = lambda x: Censored(x, high_limit, low_limit)

        if gamma == 1.0: return cnsr(base_rv)

        d = lambda x: poisson(x, loc=2) if allow_noise else DeltaDistribution(loc=x)

        opts = [d(prev)]
        if allow_expand: opts.append(d(prev / gamma))
        if allow_decay: opts.append(d(prev * gamma))

        return MixtureDistribution([cnsr(x) for x in opts])
    return layer_fn

class DeltaDistribution(Distribution):
    def __init__(self, loc=0):
        self.x = loc

    def _sample(self): return self.x

class SubsetDistribution(Distribution):
    def __init__(self, superset, p_inc): self.superset, self.p_inc = superset, p_inc
    def _sample(self): return [x for x in self.superset if Coin(self.p_inc).rvs(1)[0]]

class Coin(Distribution):
    def __init__(self, prob_true=0.5): self.true_rv = bernoulli(prob_true)
    def _sample(self): return self.true_rv.rvs(1)[0] == 1

# TODO(mmd): Maybe redundant
class CategoricalRV(MixtureDistribution):
    def __init__(self, options, weights=None):
        super(CategoricalRV, self).__init__([DeltaDistribution(x) for x in options], weights=weights)

class ListRV(MarkovianGenerativeProcess):
    def __init__(
        self,
        el_rv,
        stop_chance = 0.5,
        min_len     = 1,
        max_len     = 5,
    ):
        self.min_len, self.max_len = min_len, max_len
        self.el_rv = to_rv(el_rv)
        self.should_stop_rv = Coin(stop_chance)

        first_layer = self.el_rv.rvs(1)[0]
        super().__init__(first_layer)

    def _stop(self, prev_state, n):
        if n >= self.max_len: return True
        if n < self.min_len: return False

        return self.should_stop_rv._sample()

    def _next(self, prev_state, n):
        return self.el_rv.rvs(1)[0]
