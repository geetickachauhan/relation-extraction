'''
Author: Geeticka Chauhan
Idea from
https://github.com/mmcdermott/neural_hyperparameter_dists/blob/master/neural_hyperparameter_search/sklearn_distributions.py
'''

import scipy.stats as ss
import sys
sys.path.append('../..')
from relation_extraction.hyperparam_tuning.distributions_helpers import *

CNN_dist = DictDistribution({
    'num_epoches': ss.randint(70, 300),
    'learning_rate': ['constant', 'decay'],
    'learning_rate_init': ss.uniform(1e-5, 0.001),
    'filter_size': ['2,3', '2,3,4', '2,3,4,5', '3,4,5', '3,4,5,6'],
    'batch_size': ss.uniform(30, 70),
    'early_stop': [True, False],
})
