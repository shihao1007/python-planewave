# -*- coding: utf-8 -*-
"""
Created on Wed May 23 17:08:24 2018

@author: shihao
"""

# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:06:28 2018

@author: david
"""
import numpy as np
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt

class planewave:
    #implement all features of a plane wave
    #   k, E, frequency (or wavelength in a vacuum)--l
    #   try to enforce that E and k have to be orthogonal
    
    #initialization function that sets these parameters when a plane wave is created
    
    #function that renders the plane wave given a set of coordinates
    def evaluate(self, X, Y, Z):
        k_dot_r = self.k[0] * X + self.k[1] * Y + self.k[2] * Z
        ex = np.exp(1j * k_dot_r)
        return self.E.reshape((3, 1, 1)) * ex
    

    def __init__ (self, k, E):
        #add code to make sure that k and E are orthogonal - orthogonalize them (do it such that k is the same)
        self.k = k
        self.E = E
        
def planewaveinit (k, E, lambd, A, x, y, z):
    kHat = k / np.linalg.norm(k)
    kVec = kHat * 2 * np.pi / lambd
    
    s = np.cross(kHat, E)
    EHat = s / np.linalg.norm(s)
    E0 = np.cross(EHat, kVec)
    
    rVec = (x, y, z)
    phi = np.dot(kVec, rVec)

    Ep = A * E0 * np.exp(1j * phi)
    
    return Ep
    
#def evaluate(self):
l = 1                                      #specify the wavelength
kDir = np.array([0, 1, 0.5])                  #specify the k-vector direction
kDir = kDir / np.linalg.norm(kDir)
E = np.array([1, 0, 0])                     #specify the E vector

k = kDir * 2 * np.pi / l                 #calculate the k-vector from k direction and wavelength


p = planewave(k, E)                         #create a plane wave

N = 400                                      #size of the image to evaluate


c = np.linspace(-10, 10, N)
x = y = z = np.arange(-10,10,0.5)
[Y, Z] = np.meshgrid(c, c)
X = np.zeros(Y.shape)

Ep = p.evaluate(X, Y, Z)


plt.imshow(np.imag(Ep[0, :, :]))