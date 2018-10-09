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
from scipy.io import loadmat

temp = loadmat(r'D:\irimages\irholography\oldQCL\bimsim_test\EsEi\EtYOZ.mat')
#d = loadmat(r'D:\irimages\irholography\bimsim_test\alpha2.mat')
EtYOZ= temp["E_t"]
