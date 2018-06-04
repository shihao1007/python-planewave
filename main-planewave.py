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
        
        #force E and k to be orthogonal
        if ( np.linalg.norm(k) > 1e-15 and np.linalg.norm(E) >1e-15):
            s = np.cross(k, E)              #compute an orthogonal side vector
            s = s / np.linalg.norm(s)       #normalize it
            Edir = np.cross(s, k)              #compute new E vector which is orthogonal
            self.k = k
            self.E = Edir / np.linalg.norm(Edir) * np.linalg.norm(E)
    
    def __str__(self):
        return str(self.k) + "\n" + str(self.E)     #for verify field vectors use print command

    #function that renders the plane wave given a set of coordinates
    def evaluate(self, X, Y, Z, n):
        k_dot_r = np.real(self.k[0]) * X + np.real(self.k[1]) * Y + np.real(self.k[2]) * Z     #phase term k*r
        k_dot_d = np.imag(n) * self.k[2] * Z                #decay scalar term k * d
        ex = np.exp(1j * k_dot_r) * np.exp(self.phi)       #E field equation  = E0 * exp (i * (k * r)) here we simply set amplitude as 1
        Ef = self.E.reshape((3, 1, 1)) * ex
        decay = np.exp( - k_dot_d)                          #decay mask
        return [Ef, decay]
    
#calculate the result of a plane wave hitting an interface between 2 refractive indices
def singleSurface(pw, P):
    
    facing = P.face(pw.k)             #determine which direction the plane wave is coming in, -1(same as normal), 1(oppo), 0(in the plane)
    
    if (facing == -1):                  #if the plane wave hits the back of the plane, inverse the plane and nr
        P.flip()
        P.nr = 1 / P.nr
    
    cos_theta_i = np.dot(pw.k, -P.N) / (np.linalg.norm(pw.k) * np.linalg.norm(P.N))  #cos(theta) is the angle between kDir and opposite N
    theta_i = math.acos(cos_theta_i)
    sin_theta_i = np.sin(theta_i)
    sin_theta_t = (1 / np.real(P.nr)) * np.sin(theta_i)            #compute the sin of theta t using Snell's Law
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
    y_hat = P.parallel(pw.k)
    yHat = y_hat / np.linalg.norm(y_hat)    #compute normalized y vector
    x_hat = np.cross(yHat, zHat)
    xHat = x_hat / np.linalg.norm(x_hat)    #compute normalized x vector
    
    Ei_s = np.dot(pw.E, xHat) + 0j        #compute the s-polarized component of incident field
    sgn = np.sign(np.dot(pw.E, yHat))
    cxHat = xHat + 0j                       #make x vector complex
    Ei_p = np.linalg.norm(pw.E - cxHat * Ei_s) * sgn      #compute the p-polarized component of incident field
    
    if (theta_i == 0):              #when the plane wave hits the interface head-on, s-components of the plane wave will be the same
        rp = (1 - nr) / (1 + nr)    #due to boundry conditions, so in this case we only compute rp and tp
        tp = 2 / (1 + nr)           #compute Fresnel Coefficients
        kr = -pw.k                #reflected k vector is the inversed version of k vector
        kt = pw.k * nr            #transmitted k vector is degenerated by nr
        Er = pw.E * rp            #reflected field
        Et = pw.E * tp            #transmitted field
        
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
            
            kr = ( yHat * sin_theta_i - zHat * cos_theta_i) * np.linalg.norm(pw.k)  #compute kr
            kt = ( yHat * sin_theta_i - zHat * cos_theta_i) * 0
            
        else:
            ##handle normal situation from here
            rs = np.sin(theta_t - theta_i) / np.sin(theta_t + theta_i)
            rp = np.tan(theta_t - theta_i) / np.tan(theta_t + theta_i)
            
            tp = ( 2 * sin_theta_t * cos_theta_i ) / ( np.sin(theta_t + theta_i) * np.cos(theta_t - theta_i))
            ts = ( 2 * sin_theta_t * cos_theta_i ) / ( np.sin(theta_t + theta_i))
            
            kr = ( yHat * sin_theta_i - zHat * cos_theta_i) * np.linalg.norm(pw.k)
            kt = ( yHat * sin_theta_t + zHat * cos_theta_t) * np.linalg.norm(pw.k) * np.real(P.nr)
            
        Er_s = Ei_s * rs        #compute s and p components for reflected and transmitted fields
        Er_p = Ei_p * rp
        Et_s = Ei_s * ts
        Et_p = Ei_p * tp
        
        Er = (yHat * cos_theta_i + zHat * sin_theta_i) * Er_p +cxHat * Er_s
        Et = (yHat * cos_theta_t - zHat * sin_theta_t) * Et_p +cxHat * Et_s
        
    phase_r = np.dot(P.O, pw.k - kr)  #compute the phase offset
    phase_t = np.dot(P.O, pw.k - kt)  #??? what is phase offset?
    
    R = planewave(kr, Er, phase_r)         #return the reflected and transmitted field
    T = planewave(kt, Et, phase_t)
    return [R, T]
        

# set plane wave attributes
l = 4                                        #specify the wavelength
kDir = np.array([0, -1, -1])                   #specify the k-vector direction
kDir = kDir / np.linalg.norm(kDir)
E = np.array([1, 0, 0])                      #specify the E vector
phi = 0

# set plane attributes
O = np.array([0, 0, 0])                     #specify the P point
N = np.array([0, 0, 1])                     #specify the normal vector
U = np.array([1, 0, 0])                     #specify U vector
n = 1.90 - 0.7j                                   #n = nt / ni (n0 is the source material(incidental), nt is the material after the interface(transmitted))
                                            #if n > 1, no TIR, if n < 1, TIR might happen
                                            #always assuming the incidental ni = 1

k = kDir * 2 * np.pi / l                 #calculate the k-vector from k direction and wavelength

P = plane(O, N, U, n)
Ef = planewave(k, E, phi)                           #create a plane wave
[Er, Et] = singleSurface(Ef, P)                                 #send the plane wave through a single interface

N = 1001                                      #size of the image to evaluate

#create mesh grid
c = np.linspace(-10, 10, N)
[Y, Z] = np.meshgrid(c, c)
X = np.zeros(Z.shape)

#create mask for incidental field and transmitted field
mask_in = np.zeros(Z.shape)
mask_tr = np.zeros(Z.shape)
mask_in[0:int(N/2)] = 1
mask_tr[int(N/2):N] = 1 

#Electric field of a plane wave
#Epi = (Ef.evaluate(X, Y, Z) + Er.evaluate(X, Y, Z)) * mask_in       #field in incidental side
[Fi, decayi] = Ef.evaluate(X, Y, Z, n)
[Ft, decayt] = Et.evaluate(X, Y, Z, n)
[Fr, decayr] = Er.evaluate(X, Y, Z, n)                             #filed in transmitted side
Fp = (Fi + Fr) * mask_in + Ft * decayt * mask_tr

#plt.imshow(decayt* mask_tr)
#plot the field
#fig = plt.figure()
plt.imshow(np.real(Fp[0, :, :]))

