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
    def __init__ (self, k, E):
    
#        k = k / np.linalg.norm(k)           #normalize k vector
        if np.abs(np.dot(E, k)) < 1e-15:    #if E and k vector is orthogonal
            self.k = k                      
            self.E = E                      #set their value
        else:                               #if they are not orthogonal
            s = np.cross(k, E)              #compute an orthogonal side vector
            s = s / np.linalg.norm(s)       #normalize it
            E = np.cross(s, k)              #compute new E vector which is orthogonal
            self.k = k
            self.E = E
    
    #function that renders the plane wave given a set of coordinates
    def evaluate(self, X, Y, Z):
        k_dot_r = self.k[0] * X + self.k[1] * Y + self.k[2] * Z     #phase term k*r
        ex = np.exp(1j * k_dot_r)       #E field equation  = E0 * exp (i * (k * r))
        return self.E.reshape((3, 1, 1)) * ex

l = 4                                      #specify the wavelength
kDir = np.array([0, 0, 1])                  #specify the k-vector direction
kDir = kDir / np.linalg.norm(kDir)
E = np.array([1, 0, 0])                     #specify the E vector

k = kDir * 2 * np.pi / l                 #calculate the k-vector from k direction and wavelength


p = planewave(k, E)                         #create a plane wave

N = 400                                      #size of the image to evaluate


c = np.linspace(-10, 10, N)
[Y, Z] = np.meshgrid(c, c)
X = np.zeros(Y.shape)

Ep = p.evaluate(X, Y, Z)


plt.imshow(np.real(Ep[0, :, :]))