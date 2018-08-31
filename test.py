# -*- coding: utf-8 -*-
"""
Created on Fri Aug 31 15:47:23 2018

@author: shihao
"""
import numpy as np
import math
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from pyquaternion import Quaternion

Num = 1001                                      #size of the image to evaluate

#create a boundary object

#create mesh grid
c = np.linspace(-20, 20, Num)
n0 = 1.0
n1 = 0.8 - 0.1j           
term2 = np.exp( 1j * c * n0 / n1 * np.sin(np.pi/6))
term1 = np.exp(-c * np.sqrt(np.sin(np.pi/6)**2 * n0**2 / n1**2 - 1))
term3 = term1 * term2
plt.figure()
plt.plot(term2)

