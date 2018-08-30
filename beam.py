# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 16:22:17 2018

@author: david
"""

import planewave as pw
import numpy as np
import scipy as sp
import scipy.ndimage
import matplotlib.pyplot as plt

class beam:
    
    #take a focal point p and propagation direction d and assign it to the beam
    #d is the direction of the propagating field (direction of kz)
    #p is the focal point of the beam (where the Fourier transform is specified)
    #l is the free-space wavelength
    #sigma is the beam with specified such that 1 is a beam with standard deviation with pi/lambda
    def __init__(self, Ex, Ey, p, l=1, s=1):
        self.E = [Ex, Ey]
        self.p = p
        self.l = l
        self.s = s
        
    #calculates the maximum k-space frequency supported by this beam    
    def kmax(self):
        return 2 * np.pi / self.l
    
    def kspace(self, Kx, Ky):
        sigma = self.s * (np.pi / self.l)
        G = 1 / (2 * np.pi * sigma **2) * np.exp(- (0.5/(sigma**2)) * (Kx ** 2 + Ky ** 2))
        B = ((Kx ** 2 + Ky **2) > self.kmax() ** 2)
        
        #calculate Kz (required for calculating the Ez component of E)
        Kz = np.lib.scimath.sqrt(self.kmax() ** 2 - Kx**2 - Ky**2)
        
        Ex = self.E[0] * G
        Ey = self.E[1] * G
        Ez = -Kx/Kz * Ex
        
        #set all values outside of the band limit to zero
        Ex[B] = 0
        Ey[B] = 0
        Ez[B] = 0
        
        return [Ex, Ey, Ez]
    
    #return the k-space transform of the beam at p as an NxN array
    def kspace_image(self, N):
        #calculate the band limit [-2*pi/lambda, 2*pi/lambda]
        kx = np.linspace(-self.kmax(), self.kmax(), N)
        ky = kx
        
        #generate a meshgrid for the field
        [Kx, Ky] = np.meshgrid(kx, ky)

        #generate a Gaussian with a standard deviation of sigma * pi/lambda
        return self.kspace(Kx, Ky)
        
    #return a plane wave decomposition of the beam using N plane waves
    def decompose(self, N):
        PW = []                     #create an empty list of plane waves
        theta = np.random.uniform(0, 2*np.pi, N)
        r = np.random.uniform(0, self.kmax(), N)
        kx = np.sqrt(r) * np.cos(theta)
        ky = np.sqrt(r) * np.sin(theta)
        
        E0 = self.kspace(kx, ky)
        kz = np.sqrt(self.kmax() ** 2 - kx**2 - ky**2)
        
        for i in range(0, N):
            k = [kx[i], ky[i], kz[i]]
            E = [E0[0][i], E0[1][i], E0[2][i]]
            w = pw.planewave(k, E)
            PW.append(w)
            
        return PW