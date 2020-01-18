'''
Created by: Kevin De Angeli
Email: kevindeangeli@utk.edu
Date: 2020-01-16

'''

import numpy as np


def sigmoid(x):
    x2=1 / (1 + np.exp(-x))
    return 1 / (1 + np.exp(-x))
