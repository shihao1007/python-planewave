# -*- coding: utf-8 -*-
"""
Created on Thu May 24 11:13:17 2018

@author: shihao
"""

import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt

kDir = np.array([2, 1, 0])
E = np.array([1, 5, 0])

kDir = kDir / np.linalg.norm(kDir)
s = np.cross(kDir, E)
s = s / np.linalg.norm(s)
E0 = np.cross(s, kDir)
if np.abs(np.dot(E0, kDir)) < 1e-15:
    print("E field is orthogonal to k Vector!")
else:
    print("Error: E field is not orthogonal to k Vector!")