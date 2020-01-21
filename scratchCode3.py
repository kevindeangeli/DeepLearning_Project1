'''
Created by: Kevin De Angeli
Email: kevindeangeli@utk.edu
Date: 2020-01-18

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sympy as sym
from errorFunctions import *

a= np.array([1,2,3,4,7])
b= np.array([3,3,3,3,3])

print(mse(a,b))

for i,k in enumerate(b):
    print(i)
    print(k)
    print(" ")