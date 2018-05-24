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
from matplotlib import pyplot as plt

kVec = np.array([0, 0, 1])      #input k vector, field propogating direction
E = np.array([0, 1, 0])        #input E vector, E field isolating direction
lambd = 11                      #input lambda, wavelength in vaccumn
A = 1                           #input E field amplitude
N = 40                          #Number of samples for evaluating the field
c = np.linspace(-10, 10, N)     #mesh grid axis
x = y = z = np.arange(-10,10,0.5)   #x, y, z values for input r vector
[Y, Z] = np.meshgrid(c, c)      #create meshgrid
X = np.zeros(Y.shape)           #third axis


#class planewave:
    #implement all features of a plane wave
    #   k, E, frequency (or wavelength in a vacuum)--l
    #   try to enforce that E and k have to be orthogonal
    
    #initialization function that sets these parameters when a plane wave is created
    
    #function that renders the plane wave given a set of coordinates
    #evaluate(X, Y, Z):
    

#    def __init__ (self, k, E, lambd, A):
#        self.k = k
#        self.E = E
#        self.lambd = lambd
#        self.A = A

#create a field propogating along k direction
def planewaveinit (k, E, lambd, A, x, y, z):
    kHat = k / np.linalg.norm(k)            #normalize to get unit vector of k
    kVec = kHat * 2 * np.pi / lambd         #culculate k vector which includes wavelength
    
    s = np.cross(kHat, E)                   #compute an orthogonal side vector
    sHat = s / np.linalg.norm(s)            #normalize it
    E0 = np.cross(sHat, kHat)               #compute normalized E vector E0
    
    rVec = (x, y, z)                        #define r vector
    phi = np.dot(kVec, rVec)                #compute phase phi

    Ep = A * E0 * np.exp(1j * phi)          #compute plane wave field Ep
    
    return Ep
    
#def evaluate(self):
    
Epy = np.complex_(np.empty((np.size(x), np.size(y), np.size(z))))

#if E[0] != 0 && E[1] = 0 && E[2] = 0:
for i in range(0, np.size(x)):
    for j in range(0, np.size(y)):
        for h in range(0, np.size(z)):
            Epy[i, j, h] = planewaveinit(kVec, E, lambd, A, x[i], y[j], z[h])[1]
            
fig = plt.figure(3)
plt.plot(x, np.real(Epy[0,0,:]))
plt.grid(True)
plt.show

#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')        
#ax.plot_wireframe(Z, Y, np.real(Epy[0,:,:]), rstride=1, cstride=1,alpha=1)
#ax.set_xlabel('Z')
#ax.set_ylabel('Y')
#ax.set_zlabel('X')
#ax.plot_wireframe(Z, Y, np.imag(Epy[0,:,:]), rstride=1, cstride=1,alpha=0.3)
#plt.show()
        
#planewave1 = planewave(kVec, E, lambd, A)
#planewave1.evaluate()