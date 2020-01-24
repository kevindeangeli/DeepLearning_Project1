'''
Created by: Kevin De Angeli
Email: kevindeangeli@utk.edu
Date: 2020-01-16

'''

import numpy as np

def binary_cross_entropy_loss(prediction, output):
    return 0

def mse(prediction, output):
    if len(prediction) == 1:
        return (prediction[0]-output[0])**2
    else:
        return ((prediction - output)**2).mean(axis=0)

def mse_prime(output, target):
    return output-target




