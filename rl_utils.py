import copy
import os
import torch
import json
import numpy as np

def load_pi(num_agents, topology):
    """Load the connectivity weight matrix."""
    wsize = num_agents
    if topology == 'dense':
        topo = 1
    elif topology == 'ring':
        topo = 2
    elif topology == 'bipartite':
        topo = 3

    with open('topology/connectivity/%s_%s.json' % (wsize, topo), 'r') as f:
        cdict = json.load(f)  # connectivity dict
    return cdict['pi']


def moving_average(a, window_size):
    """Move average for averaged returns."""
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))