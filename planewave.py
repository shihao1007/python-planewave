# -*- coding: utf-8 -*-
"""
Created on Thu Aug 30 11:13:18 2018

@author: shihao
"""

# -*- coding: utf-8 -*-
"""
Created on Sat May 26 15:26:59 2018

@author: shihao
"""

import numpy as np
import math
from mpl_toolkits.mplot3d import axes3d
from matplotlib import pyplot as plt
from pyquaternion import Quaternion

class boundary:
    #define the interface to which the plane wave is hitting
    
    def __init__ (self, p, N, n0, n1):
        #define a plane through a point p, a normal vector N
        self.p = p          #p is the query point, on origin
        self.N = N          #N is the normal vector, perpendicular to the plane
        self.n0 = n0
        self.n1 = n1
        
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
    
    def side(self, q):                      #determine which side of plane point q lies
        rv = q - self.p                     #get the vector from query point P to point q
        return self.face(rv)
    
    def perpendicular(self, v):             #calculate the component of v that is perpendicular to the plane
        return self.N * np.dot(self.N, v)
    
    def parallel(self, v):                  #compute the projection of v in the plane
        return v - self.perpendicular(v)
        
    def decompose(self, v):                 #get the parallel and perpendicular components of vector v w.r.t the plane
        self.vPerp = self.perpendicular(v)
        self.vPara = self.parallel(v)
        
    #calculate the result of a plane wave hitting an interface between 2 refractive indices
    def scatter(self, pw):
        
        facing = self.face(pw.k)             #determine which direction the plane wave is coming in, -1(same as normal), 1(oppo), 0(in the plane)
        
        if (facing == -1):                  #if the plane wave hits the back of the plane, inverse the plane and nr
            self.flip()
            temp = self.n0
            self.n0 = self.n1
            self.n1 = temp
        
        cos_theta_i = np.dot(pw.k, -self.N) / (np.linalg.norm(pw.k) * np.linalg.norm(self.N))  #cos(theta) is the angle between kDir and opposite N
        theta_i = math.acos(cos_theta_i)
        sin_theta_i = np.sin(theta_i)
        sin_theta_t = (1 / np.real(self.n1)) * np.sin(theta_i)            #compute the sin of theta t using Snell's Law
        #        cos_theta_t = np.cos(math.asin(sin_theta_t))
        cos_theta_t = np.sqrt(1 - sin_theta_t**2)
        
        tir = False                         #set flag for total internal reflection
        if (sin_theta_t > 1):               #if sin theta t is bigger than 1
            tir = True                      #there is TIR, theta t is not needed and cant be computed, we only need sin theta t and cos theta t
            cos_theta_t = np.sqrt(1 - sin_theta_t**2 + 0j)  #compute cos theta t for later use for complex TIR
        else:
            theta_t = math.asin(sin_theta_t)    #compute theta t if no TIR
                    
        if (theta_i == 0):              #when the plane wave hits the interface head-on, s-components of the plane wave will be the same
            rp = (1 - np.real(self.n1) / np.real(self.n0)) / (1 + np.real(self.n1) / np.real(self.n0))    #due to boundry conditions, so in this case we only compute rp and tp
            tp = 2 / (1 + np.real(self.n1) / np.real(self.n0))           #compute Fresnel Coefficients
            kr = -pw.k * np.real(self.n0)             #reflected k vector is the inversed version of k vector
            kt = pw.k * np.real(self.n1) / np.real(self.n0)            #transmitted k vector is degenerated by nr
            Er = pw.E * rp            #reflected field
            Et = pw.E * tp            #transmitted field
            
        else:
            
            z_hat = -self.N
            zHat = z_hat / np.linalg.norm(z_hat)    #compute normalized z vector
            y_hat = self.parallel(pw.k)
            yHat = y_hat / np.linalg.norm(y_hat)    #compute normalized y vector
            x_hat = np.cross(yHat, zHat)
            xHat = x_hat / np.linalg.norm(x_hat)    #compute normalized x vector
            
            Ei_s = np.dot(pw.E, xHat) + 0j        #compute the s-polarized component of incident field
            sgn = np.sign(np.dot(pw.E, yHat))
            cxHat = xHat + 0j                       #make x vector complex
            Ei_p = np.linalg.norm(pw.E - cxHat * Ei_s) * sgn      #compute the p-polarized component of incident field
            
            if (tir == True):
                ##handle total internal reflection here
                #will just be reflection, add evanescent wave later
                ininner = (sin_theta_i * self.n0 / self.n1 ) ** 2 - 1 + 0j           #value inside square root
                inner_s = self.n1 / (self.n0 * cos_theta_i) * np.sqrt(ininner)         #value before arctan
                rs = np.exp( -2 * 1j * math.atan(inner_s))              #compute rs
                ts = 0                                                  #set ts to 0 for now
                
                inner_p = self.n0 / (self.n1 * cos_theta_i) * np.sqrt(ininner)   #same thing for rp and tp
                rp = np.exp( -2 * 1j * math.atan(inner_p))
                tp = 0
                
                kr = ( yHat * sin_theta_i - zHat * cos_theta_i) * np.linalg.norm(pw.k) * np.real(self.n0)  #compute kr
                kt = ( yHat * sin_theta_i - zHat * cos_theta_i) * 0
                
            else:
                ##handle normal situation from here
                rs = np.sin(theta_t - theta_i) / np.sin(theta_t + theta_i)
                rp = np.tan(theta_t - theta_i) / np.tan(theta_t + theta_i)
                
                tp = ( 2 * sin_theta_t * cos_theta_i ) / ( np.sin(theta_t + theta_i) * np.cos(theta_t - theta_i))
                ts = ( 2 * sin_theta_t * cos_theta_i ) / ( np.sin(theta_t + theta_i))
                
                kr = ( yHat * sin_theta_i - zHat * cos_theta_i) * np.linalg.norm(pw.k) * np.real(self.n0)
                kt = ( yHat * sin_theta_t + zHat * cos_theta_t) * np.linalg.norm(pw.k) * np.real(self.n1)
                
            Er_s = Ei_s * rs        #compute s and p components for reflected and transmitted fields
            Er_p = Ei_p * rp
            Et_s = Ei_s * ts
            Et_p = Ei_p * tp
            
            Er = (yHat * cos_theta_i + zHat * sin_theta_i) * Er_p +cxHat * Er_s
            Et = (yHat * cos_theta_t - zHat * sin_theta_t) * Et_p +cxHat * Et_s
            
        phase_r = np.dot(self.p, pw.k - kr)  #compute the phase offset
        phase_t = np.dot(self.p, pw.k - kt)  #??? what is phase offset?
        
        return [planewave(kr, Er*np.exp(phase_r * 1j)), planewave(kt, Et*np.exp(phase_t * 1j))]
    
    def evaluate(self, X, Y, Z, pw):
        
        #calculate the reflected and transmitted waves
        [Wr, Wt] = self.scatter(pw)
        
        #calculate the coefficient for kt that is used to find the point on the plane through which the wave passes to get to r
        t =  -(np.dot(self.p, self.N) - ( X * self.N[0] + Y * self.N[1] + Z * self.N[2])) / np.dot( Wt.k, self.N)
        
        #calculate the decay for propagation through a material
        decay = np.exp( - np.imag(self.n1) * t )

        #apply the decay to the transmitted plane wave
        Ft = Wt.evaluate( X, Y, Z ) * decay
        Fi = pw.evaluate( X, Y, Z )
        Fr = Wr.evaluate( X, Y, Z )
        
        #calculate masks to separate the pixels that are inside and outside of the material
        mask_in = np.zeros(Z.shape)
        mask_tr = np.zeros(Z.shape)
        mask_in[0:int(Z.shape[0]/2)] = 1
        mask_tr[int(Z.shape[0]/2):Z.shape[0]] = 1
        
        #apply the masks
        Fp = (Fi + Fr) * mask_in + Ft * mask_tr
        
        #return the calculated field
        return Fp
        


class planewave():
    #implement all features of a plane wave
    #   k, E, frequency (or wavelength in a vacuum)--l
    #   try to enforce that E and k have to be orthogonal
    
    #initialization function that sets these parameters when a plane wave is created
    def __init__ (self, k, E):
        
        #self.phi = phi
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
    def evaluate(self, X, Y, Z):
        k_dot_r = self.k[0] * X + self.k[1] * Y + self.k[2] * Z     #phase term k*r
#        k_dot_d = np.imag(n) * self.k[2] * Z                #decay scalar term k * d
        ex = np.exp(1j * k_dot_r)       #E field equation  = E0 * exp (i * (k * r)) here we simply set amplitude as 1
        Ef = self.E.reshape((3, 1, 1)) * ex
#        decay = np.exp( - k_dot_d)                          #decay mask
        return Ef

#DAVID: rename to account for spherical coordinates so we can make a Monte-Carlo one later
#DAVID: return a plane wave
def focused_beam(NA, NumSample, kd):
    ### creat a list of planewaves sampled uniformly within NA range
    # NA: numberical aperture of the lens which planewaves are focused from
    # NumSample: number of samples(planewaves) along longitudinal and latitudinal axis
    # kd: center planewave of the focused beam

    CenterKd = [0, 0, -1]                                       #defualt planewave coming in perpendicular to the surface
    kd = kd / np.linalg.norm(kd)                                #normalize the new planewave
    r = np.sqrt(CenterKd[0] ** 2 + CenterKd[1] ** 2 + CenterKd[2] ** 2)             #radiance of the hemisphere where the k vectors are sampled from
    
    if(kd[0] == CenterKd[0] and kd[1] == CenterKd[1] and kd[2] == CenterKd[2]):     #if new planewave is at the same direction as the default plane wave
        rotateAxis = CenterKd                                   #set rotation axis as defualt k vector
        RoAngle = 0                                             #set rotation axis as 0 degrees
    else:                                                       #if new plane wave is at different direction as the defualt planewave, rotation is needed
        rotateAxis = np.cross(CenterKd, kd)                     #find a axis which is perpendicular to both vectors to be rotation axis
        RoAngle = math.asin(kd[2] / r)                          #calculate the rotation angle
    beamRotate = Quaternion(axis=rotateAxis, angle=RoAngle)     #create a quaternion for rotation
    
    Kd = np.zeros((3, NumSample, NumSample))                    #initialize the planewave list
    scaleFactor = np.zeros((NumSample, NumSample))              #initialize a list of scalefactors which are used to scale down the amplitude of planewaves later on along latitude domain
    
    #convert the axis from Cartesian to Spherical
    if(CenterKd[0] == 0 or CenterKd[1] == 0):                   #if the defualt planewave is at the direction of Z axis
        theta = 0                                               #set azimuthal angle theta as 0
    else:
        theta = math.atan(CenterKd[1] / CenterKd[0])            #if not calculate theta from X and Y coordinates
    
    pha = math.acos(CenterKd[2] / r)                            #calculate polar angle pha from Z coordinate
    
    phaM = math.asin(NA / n0)                                   #calculate sample range of pha from numerical aperture
    thetaM = 2* np.pi                                           #set sample range of theta as 2pi
    phaStep = phaM / NumSample                                  #set longitudinal sample resolution as maximal pha divided by number of samples
    thetaStep = thetaM / NumSample                              #set latitudinal sample resolution as maximal theta divided by number of samples
    for i in range(NumSample):                                  #sample along longitudinal (pha) domain
        for j in range(NumSample):                              #sample along latitudinal (theta) domain
            KdR = r                                             #sample hemisphere radiance will be all the same as r
            KdTheta = theta + thetaStep * j                     #sample theta at each step in the sample range
            KdPha = pha + phaStep * i                           #sample theta at each step in the sample range
            Kd[0,j,i] = KdR * np.cos(KdTheta) * np.sin(KdPha)   #convert coordinates from spherical to Cartesian
            Kd[1,j,i] = KdR * np.sin(KdTheta) * np.sin(KdPha)
            Kd[2,j,i] = KdR * np.cos(KdPha)
            Kd[:,j,i] = beamRotate.rotate(Kd[:,j,i])            #rotate k vectors by the quaternion generated
            scaleFactor[j,i] = np.sin(KdPha)                    #calculate the scalefactors by the current polar angle pha
    
    Kd = np.reshape(Kd, ((3, NumSample ** 2)))
    scaleFactor = np.reshape(scaleFactor, ((NumSample ** 2)))   #reshape list of k vectors and scalefactors to an one dimentional list

#DAVID: bake the scale factor into the plane wave    
    return Kd, scaleFactor
        

# set plane wave attributes
l = 4                                        #specify the wavelength
E = np.array([1, 0, 0])                      #specify the E vector

# set plane attributes
p = np.array([0, 0, 0])                     #specify the p point
N = np.array([0, 0, 1])                     #specify the normal vector
n0 = 1
n1 = 1.50 - 0.1j                                   #n = nt / ni (n0 is the source material(incidental), nt is the material after the interface(transmitted))
Num = 1001                                      #size of the image to evaluate

#create a boundary object
P = boundary(p, N, n0, n1)

#create mesh grid
c = np.linspace(-20, 20, Num)
[Y, Z] = np.meshgrid(c, c)
X = np.zeros(Z.shape)

kd0 = [0, 0, -1]

NA = 0.5
NumSample = 16
Kd, scaleFactor = focused_beam(NA, NumSample, kd0)

#allocatgge space for the field and initialize it to zero
Fp = np.zeros((3, X.shape[0], X.shape[1]), dtype=np.complex128)

for i in range(NumSample ** 2):
    #kDir = np.array([0, 0, -1])                   #specify the k-vector direction
    kDir = Kd[:,i] / np.linalg.norm(Kd[:,i])
    k = kDir * 2 * np.pi / l                 #calculate the k-vector from k direction and wavelength
    
    Ei = planewave(k, E * scaleFactor[i])          #create a plane wave
    Fp = Fp + P.evaluate(X, Y, Z, Ei)

#plot the field
fig = plt.figure()

plt.subplot(311)
plt.imshow(np.abs(Fp[0, :, :]))
plt.title('Absolute Value')

plt.subplot(312)
plt.imshow(np.real(Fp[0, :, :]))
plt.title('Real Part')

plt.subplot(313)
plt.imshow(np.imag(Fp[0, :, :]))
plt.title('Imaginary Part')

plt.suptitle('256 Samples 0 Degrees', fontsize = 15)
