# -*- coding: utf-8 -*-
"""
Created on Sat May 26 15:26:59 2018

@author: shihao
"""

import numpy as np
import math
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt

class plane:
    #define the interface to which the plane wave is hitting
    
    def __init__ (self, O, N, U, nr):
        #define a plane through a point, a normal vector and an arbitary vector on the plane
        self.O = O          #P is the query point, on origin
        self.N = N          #N is the normal vector, perpendicular to the plane
        self.U = U          #U is an arbitary vector lies on the plane
        self.nr = nr        #relative refractive index between the 2 sides of the plane
        
    def flip(self):                         #flip the plane front-to-back by set the opposite normal vector
        self.N = -self.N
        
    def face(self, v):                      #determine how a vector v intersects the plane
        dotpro = np.dot(self.N, v)
        if dotpro < 0:                      #if the angle between v and N is bigger than 90 degrees
            return 1                        #return 1 stands for v intersects front of the plane(opposite direction with N)
        if dotpro > 0:                      #if the angle between v and N is bigger than 90 degrees
            return -1                       #return -1 stands for v intersects back of the plane(same direction with N)
        else:                               #if dot product equals 0, v lies on the plane
            return 0
    
    def side(self, p):                      #determine which side of plane point p lies
        rv = p - self.O                     #get the vector from query point P to point p
        return self.face(rv)
    
    def perpendicular(self, v):             #calculate the component of v that is perpendicular to the plane
        return N * np.dot(N, v)
    
    def parallel(self, v):                  #compute the projection of v in the plane
        return v - self.perpendicular(v)
    
    def setU(self, v):                      #set the U vector by a given vector v
        vHat = v / np.linalg.norm(v)
        vPara = self.parallel(vHat)
        self.U = vPara / np.linalg.norm(vPara)
        
    def decompose(self, v):                 #get the parallel and perpendicular components of vector v w.r.t the plane
        self.vPerp = self.perpendicular(v)
        self.vPara = self.parallel(v)
        


class planewave():
    #implement all features of a plane wave
    #   k, E, frequency (or wavelength in a vacuum)--l
    #   try to enforce that E and k have to be orthogonal
    
    #initialization function that sets these parameters when a plane wave is created
    def __init__ (self, k, E, phi):
        
        self.phi = phi
        self.k = k                      
        self.E = E
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
        ex = np.exp(1j * k_dot_r) * np.exp(self.phi)       #E field equation  = E0 * exp (i * (k * r)) here we simply set amplitude as 1
        return self.E.reshape((3, 1, 1)) * ex
    
    #calculate the result of a plane wave hitting an interface between 2 refractive indices
    def singleSurface(self, P):
        
        facing = P.face(self.k)             #determine which direction the plane wave is coming in, -1(same as normal), 1(oppo), 0(in the plane)
        
        if (facing == -1):                  #if the plane wave hits the back of the plane, inverse the plane and nr
            P.flip()
            P.nr = 1 / P.nr
        
        cos_theta_i = np.dot(self.k, -P.N) / (np.linalg.norm(self.k) * np.linalg.norm(P.N))  #cos(theta) is the angle between kDir and opposite N
        theta_i = math.acos(cos_theta_i)
        sin_theta_i = np.sin(theta_i)
        sin_theta_t = (1 / P.nr) * np.sin(theta_i)            #compute the sin of theta t using Snell's Law
#        cos_theta_t = np.cos(math.asin(sin_theta_t))
        cos_theta_t = np.sqrt(1 - sin_theta_t**2)
        
        tir = False                         #set flag for total internal reflection
        if (sin_theta_t > 1):               #if sin theta t is bigger than 1
            tir = True                      #there is TIR, theta t is not needed and cant be computed, we only need sin theta t and cos theta t
            cos_theta_t = np.sqrt(1 - sin_theta_t**2 + 0j)  #compute cos theta t for later use for complex TIR
        else:
            theta_t = math.asin(sin_theta_t)    #compute theta t if no TIR
            
        z_hat = -P.N
        zHat = z_hat / np.linalg.norm(z_hat)    #compute normalized z vector
        y_hat = P.parallel(self.k)
        yHat = y_hat / np.linalg.norm(y_hat)    #compute normalized y vector
        x_hat = np.cross(yHat, zHat)
        xHat = x_hat / np.linalg.norm(x_hat)    #compute normalized x vector
        
        Ei_s = np.dot(self.E, xHat) + 0j        #compute the s-polarized component of incident field
        sgn = np.sign(np.dot(self.E, yHat))
        cxHat = xHat + 0j                       #make x vector complex
        Ei_p = np.linalg.norm(self.E - cxHat * Ei_s) * sgn      #compute the p-polarized component of incident field
        
        if (theta_i == 0):              #when the plane wave hits the interface head-on, s-components of the plane wave will be the same
            rp = (1 - nr) / (1 + nr)    #due to boundry conditions, so in this case we only compute rp and tp
            tp = 2 / (1 + nr)           #compute Fresnel Coefficients
            kr = -self.k                #reflected k vector is the inversed version of k vector
            kt = self.k * nr            #transmitted k vector is degenerated by nr
            Er = self.E * rp            #reflected field
            Et = self.E * tp            #transmitted field
            
        else:
            
            if (tir == True):
                ##handle total internal reflection here
                #will just be reflection, add evanescent wave later
                ininner = (sin_theta_i / P.nr ) ** 2 - 1 + 0j           #value inside square root
                inner_s = P.nr / cos_theta_i * np.sqrt(ininner)         #value before arctan
                rs = np.exp( -2 * 1j * math.atan(inner_s))              #compute rs
                ts = 0                                                  #set ts to 0 for now
                
                inner_p = 1 / (P.nr * cos_theta_i) * np.sqrt(ininner)   #same thing for rp and tp
                rp = np.exp( -2 * 1j * math.atan(inner_p))
                tp = 0
                
                kr = ( yHat * sin_theta_i - zHat * cos_theta_i) * np.linalg.norm(self.k)  #compute kr
                kt = 0
                
            else:
                ##handle normal situation from here
                rs = np.sin(theta_t - theta_i) / np.sin(theta_t + theta_i)
                rp = np.tan(theta_t - theta_i) / np.tan(theta_t + theta_i)
                
                tp = ( 2 * sin_theta_t * cos_theta_i ) / ( np.sin(theta_t + theta_i) * np.cos(theta_t - theta_i))
                ts = ( 2 * sin_theta_t * cos_theta_i ) / ( np.sin(theta_t + theta_i))
                
                kr = ( yHat * sin_theta_i - zHat * cos_theta_i) * np.linalg.norm(self.k)
                kt = ( yHat * sin_theta_t + zHat * cos_theta_t) * np.linalg.norm(self.k) * P.nr
                
            Er_s = Ei_s * rs        #compute s and p components for reflected and transmitted fields
            Er_p = Ei_p * rp
            Et_s = Ei_s * ts
            Et_p = Ei_p * tp
            
            Er = (yHat * cos_theta_i + zHat * sin_theta_i) * Er_p +cxHat * Er_s
            Et = (yHat * cos_theta_t - zHat * sin_theta_t) * Et_p +cxHat * Et_s
            
        phase_r = np.dot(P.O, self.k - kr)  #compute the phase offset
        phase_t = np.dot(P.O, self.k - kt)  #??? what is phase offset?
        
        self.R = planewave(kr, Er, phase_r)         #return the reflected and transmitted field
        self.T = planewave(kt, Et, phase_t)
        

# set plane wave attributes
l = 4                                        #specify the wavelength
kDir = np.array([0, 0, -1])                   #specify the k-vector direction
kDir = kDir / np.linalg.norm(kDir)
E = np.array([1, 0, 0])                      #specify the E vector
phi = 0

# set plane attributes
O = np.array([0, 0, 0])                     #specify the P point
N = np.array([0, 0, 1])                     #specify the normal vector
U = np.array([1, 0, 0])                     #specify U vector
nr = 1.5                                    #nr = nt / ni (n0 is the source material(incidental), n0 is the material after the interface(transmitted))
                                            #if nr > 1, no TIR, if nr < 1, TIR might happen

k = kDir * 2 * np.pi / l                 #calculate the k-vector from k direction and wavelength

P = plane(O, N, U, nr)
Ef = planewave(k, E, phi)                           #create a plane wave
#R = planewave(k, E, phi)                            #initialize reflected plane wave
#T = planewave(k, E, phi)                            #initialize transmitted plane wave
Ef.singleSurface(P)                                 #send the plane wave through a single interface

N = 200                                      #size of the image to evaluate

cm = np.linspace(-10, 0, N)
cp = np.linspace(0, 10, N)
[Yt, Zt] = np.meshgrid(cp, cp)
[Yi, Zi] = np.meshgrid(cm, cm)
Xi = np.zeros(Zi.shape)
Xt = np.zeros(Zt.shape)

Epi = Ef.evaluate(Xi, Yi, Zi) + Ef.R.evaluate(Xi, Yi, Zi)       #field in incidental side
Ept = Ef.T.evaluate(Xt, Yt, Zt)                                 #filed in transmitted side
Ep = np.concatenate((Epi, Ept), axis = 1)

Er = Ef.R.evaluate(Xi, Yi, Zi)
fig = plt.figure()
plt.imshow(np.real(Ep[0, :, :]))
#plt.imshow(np.real(Ept[0, :, :]))

