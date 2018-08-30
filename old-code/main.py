# -*- coding: utf-8 -*-
"""
Created on Mon May 21 10:03:35 2018

@author: shihao
"""

import numpy as np
#import scipy as sp
import random
import math
from pyquaternion import Quaternion
#import sympy.mpmath as mp

class SimuEf:
    
    def __init__ (self, wavlen, subThic, nReal, nImag, samples, orderEf):
        self.resolution = 128 			#detector resolution
        self.wavlen = wavlen			#wavelenth
        self.subThic = subThic
        self.nReal = nReal				#refractive index real part
        self.nImag = nImag				#refractive index imaginary part
        self.material = [wavlen, nReal, nImag]		#material properties list
        self.fov = np.zeros(1)			#simulation field of view
        self.spacing = 1				#
        self.poi = np.zeros(3)			#position of the interface in x,y,z
        self.samples = samples			#number of Monte Carlo samples
        self.orderEf = orderEf			#order of Legendre Polynomial and Bessel functions
        self.padding = 1            	#specify padding    
        self.E0 = 1						#amplitude of the field
        self.condenserNA_in = 0.34			#for lens, inner obscuration is 0
        self.condenserNA_out = 0.62		#NA for condenser lens
        self.theta = 1.5708
        self.phi = 0					#kVec direction from the y axis
        self.pf = np.zeros(3)			#focal point
#        self.legendre = np.zeros(1)
        
    def Legendre (self, orderEf, x):
        if np.isscalar(x):
            legendre = np.zeros((orderEf + 2, 1))
            legendre[0] = 1
            if orderEf  == 0 :
                return legendre
            legendre[1] = x
            for i in range(1, orderEf + 1):
                #error might exists here
                legendre[i + 1] = ((2 * i - 1) / i) * x * (legendre[i]) - ((i - 1) / i) * (legendre[i-1])
        if(np.size(x.shape) == 1):
            legendre = np.zeros((len(x), orderEf + 2))
            legendre[:,0] = 1
            if orderEf  == 0 :
                return legendre
            legendre[:,1] = x
            for i in range(1, orderEf + 1):
                legendre[:, i + 1] = ((2 * i - 1) / i) * x * legendre[:, i] - ((i - 1) / i) * legendre[:, i-1]
#        else:
#            self.legendre = np.zeros((np.shape(x), self.orderEf + 1))
#            self.legendre[:,:,0] = 1
#            if self.orderEf  == 0 :
#                return self.legendre
#            self.legendre[:,:,1] = x
#            for i in range(1, self.orderEf):
#                self.legender[:,:, i + 1] = ((2 * i - 1) / i) * x * self.legendre[:,:, i] - ((i - 1) / i) * self.legendre[:,:, i-1]
        return legendre
    
    def MonteCarlo (self, s, N, k, NAin, NAout):
        random.setstate(s)
        if np.size(k) == 1:
            k = np.transpose(k)
        
        cos_angle = np.dot (k, np.array([[0], [0], [1]]))
        R = np.eye(3);
        if cos_angle != 1.0:
            axis = np.cross(np.array([0, 0, 1]), k)
            axis /= np.linalg.norm(axis)
            angle = math.acos(cos_angle)
            q = np.zeros(4)
            q[0] = np.cos(angle / 2)
            q[1:4] = np.sin(angle / 2) * axis
            Q = Quaternion(q)
            R = Q.rotation_matrix
            
        inPhi = math.asin(NAin)
        outPhi = math.asin(NAout)
        inZ = np.cos(inPhi)
        outZ = np.cos(outPhi)
        rangeZ = inZ - outZ

        samples = np.zeros((3, N))
        for i in range(0, N):
            z = float(np.random.uniform(0,1,1)) + rangeZ + outZ
            theta = float(np.random.uniform(0,1,1)) * 2 * np.pi
            phi = math.acos(z)
            r = 1
            x = r * np.cos(theta) * np.sin(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(phi)
            cart = np.array([[x],[y],[z]])
            samples[:, i] = R * cart
            
        return samples
            

    def params (self, pixelsize):   #get parameters
        self.pixelsize = pixelsize  #set pixelsize on the detector
        
        if not self.fov:
            self.fov = int(self.resolution * self.pixelsize)    #field of view

        self.rad = np.arange(0, round(self.fov / 2), 1)
        self.s = random.getstate()  #control random number generation
        self.numRad = np.size(self.rad)
        self.simRes = self.resolution * (self.padding * 2 + 1) #spacial resolution of field plane
        self.alpha1 = math.asin(self.condenserNA_in)
        self.alpha2 = math.asin(self.condenserNA_out)
        self.ordVecEf = np.arange(0,self.orderEf + 1,1)
        self.ordVecEf.transpose()
        self.il = 1j ** self.ordVecEf   #the prefix term i to the power of l
        self.il = np.reshape(self.il, (1, 1, self.orderEf + 1))
        
        self.cropsize = self.padding * self.resolution  #define crop size on final image
        CartX = 1 * np.cos(self.phi) * np.cos(self.theta)
        CartY = 1 * np.cos(self.phi) * np.sin(self.theta)
        CartZ = 1 * np.sin(self.phi)                    #convert spherical coord to cartesian coord
        self.lightDirection = np.array([[CartX, CartY, CartZ]])   #vector of incident field
        
        halfGridSize = int(self.fov / 2) * (2 * self.padding + 1)
        self.gx = np.linspace(-halfGridSize, halfGridSize, self.simRes)
        self.gy = self.gx
        
        self.x, self.z =np.meshgrid(self.gx, self.gy)  #field slice on the x z plane
        self.y = np.ones((self.simRes, self.simRes)) * self.subThic
        
        self.rVec = np.zeros((self.simRes * self.simRes, 3))  #rvector of each point in the plane
        self.rVec[:,0] = np.reshape(self.z,(1, self.simRes * self.simRes))
        self.rVec[:,1] = np.reshape(self.y,(1, self.simRes * self.simRes))
        self.rVec[:,2] = np.reshape(self.x,(1, self.simRes * self.simRes))
        
        self.rVec_poi = self.rVec - self.poi    #rvectors of each point with respect to the center of the interface
        
        self.Pl_cosalpha1 = SimuEf.Legendre(self, self.orderEf + 1, np.cos(self.alpha1))
        self.Pl_cosalpha2 = SimuEf.Legendre(self, self.orderEf + 1, np.cos(self.alpha2))
        
        self.origRVecs = self.rVec
        self.NorRvecsPoi = self.rVec_poi / np.sqrt(np.sum(self.rVec_poi ** 2, axis = 1))[:,None]
        self.rPoi = np.reshape(np.sqrt(np.sum(self.rVec_poi ** 2, axis = 1)), (self.simRes, self.simRes))
        
        self.magKVec = 2 * np.pi / self.wavlen
        self.kVec = self.lightDirection * self.magKVec
        
        self.subA = 2 * np.pi * self.E0 * ((1 - np.cos(self.alpha2)) - (1 - np.cos(self.alpha1)))
        
        self.k_j = SimuEf.MonteCarlo(self, self.s, self.samples, self.kVec, self.condenserNA_in, self.condenserNA_out)
    
        
        return self

Simu1 = SimuEf(11, 200, 1.47634, 0.000258604, 1000, 100)
params1 = Simu1.params(5.5)

tempdict = params1.__dict__


                                    