'''
Created by: Kevin De Angeli
Email: kevindeangeli@utk.edu
Date: 2020-01-16

'''

import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return 0
    #return sigmoid(z) * (1 - sigmoid(z))