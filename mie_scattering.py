# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 11:17:10 2018

@author: shihao
"""

import numpy as np
from matplotlib import pyplot as plt
from pyquaternion import Quaternion
import scipy as sp
import math
from scipy.io import loadmat
from matplotlib import animation as animation

def sampled_kvectors_spherical_coordinates(NA, NumSample, kd):
#sample multiple planewaves at different angle to do simulation as a focused beam
    # return a list of planewave direction vectors Kd
    # and the corresponding scale factor
    # NA: numberical aperture of the lens which planewaves are focused from
    # NumSample: number of samples(planewaves) along longitudinal and latitudinal axis
    # kd: center planewave of the focused beam
    
    #allocatgge space for the field and initialize it to zero
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
    
    phaM = math.asin(NA / np.real(n))                                   #calculate sample range of pha from numerical aperture
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
    
    return Kd, scaleFactor


def Legendre(order, x):
#calcula order l legendre polynomial
        #order: total order of the polynomial
        #x: array or vector or scalar for the polynomial
        #return an array or vector with all the orders calculated
        
    if np.isscalar(x):
    #if x is just a scalar value
    
        P = np.zeros((order+1, 1))
        P[0] = 1
        if order == 0:
            return P
        P[1] = x
        if order == 1:
            return P
        for j in range(1, order):
            P[j+1] = ((2*j+1)/(j+1)) *x *(P[j]) - ((j)/(j+1))*(P[j-1])
        return P
    
    elif np.asarray(x).ndim == 1:
    #if x is a vector
        P = np.zeros((len(x), order))
        P[:,0] = 1
        if order == 0:
            return P
        P[:, 1] = x
        if order == 1:
            return P
        for j in range(1, order):
            P[:,j+1] = ((2*j+1)/(j+1)) *x *(P[:, j]) - ((j)/(j+1))*(P[:, j-1])
        return P
    
    else:
    #if x is an array
        P = np.zeros((x.shape[0],x.shape[1], order+1))
        P[:,:,0] = 1
        if order == 0:
            return P
        P[:,:, 1] = x
        if order == 1:
            return P
        for j in range(1, order):
            P[:,:, j+1] = ((2*j+1)/(j+1)) *x *np.squeeze(P[:,:, j]) - ((j)/(j+1))*np.squeeze(P[:,:, j-1])
        return P
    
    
def sph2cart(az, el, r):
#convert coordinates from spherical to cartesian
        #az: azimuthal angle, horizontal angle with x axis
        #el: polar angle, vertical angle with z axis
        #r: radial distance with origin
        
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z


def sphbesselj(order, x, mode):
#calculate the spherical bessel function of the 1st kind with order specified
    #order: the order to be calculated
    #x: the variable to be calculated
    #mode: 1 stands for prime, -1 stands for derivative, 0 stands for nothing
        if np.isscalar(x):
            return np.sqrt(np.pi / (2*x)) * sp.special.jv(order + 0.5 + mode, x)
        
        elif np.asarray(x).ndim == 1:
            ans = np.zeros((len(x), len(order) + 1), dtype = np.complex128)
            for i in range(len(order)):
                ans[:,i] = np.sqrt(np.pi / (2*x)) * sp.special.jv(i + 0.5 + mode, x)
            return ans
        
        else:
            ans = np.zeros((x.shape[0],x.shape[1], len(order)), dtype = np.complex128)
            for i in range(len(order)):
                ans[:,:,i] = np.sqrt(np.pi / (2*x)) * sp.special.jv(i + 0.5 + mode, x)
            return ans
        
        
        
def sphhankel(order, x, mode):
#general form of calculating spherical hankel functions of the first kind at x
    
    if np.isscalar(x):
        return np.sqrt(np.pi / (2*x)) * (sp.special.jv(order + 0.5 + mode, x) + 1j * sp.special.yv(order + 0.5 + mode, x))
#
        
    elif np.asarray(x).ndim == 1:
        ans = np.zeros((len(x), len(order) + 1), dtype = np.complex128)
        for i in range(len(order)):
            ans[:,i] = np.sqrt(np.pi / (2*x)) * (sp.special.jv(i + 0.5 + mode, x) + 1j * sp.special.yv(i + 0.5 + mode, x))
        return ans
    else:
        ans = np.zeros((x.shape[0],x.shape[1], len(order)), dtype = np.complex128)
        for i in range(len(order)):
            ans[:,:,i] = np.sqrt(np.pi / (2*x)) * (sp.special.jv(i + 0.5 + mode, x) + 1j * sp.special.yv(i + 0.5 + mode, x))
        return ans
    

#derivative of the spherical bessel function of the first kind
def derivSphBes(order, x):
    js_n = np.zeros(order.shape, dtype = np.complex128)
    js_n_m_1 = np.zeros(order.shape, dtype = np.complex128)
    js_n_p_1 = np.zeros(order.shape, dtype = np.complex128)
    
    js_n = sphbesselj(order, x, 0)
    js_n_m_1 = sphbesselj(order, x, -1)
    js_n_p_1 = sphbesselj(order, x, 1)
    
    j_p = 1/2 * (js_n_m_1 - (js_n + x * js_n_p_1) / x)
    return j_p

#derivative of the spherical hankel function of the first kind
def derivSphHan(order, x):
    sh_n = np.zeros(order.shape, dtype = np.complex128)
    sh_n_m_1 = np.zeros(order.shape, dtype = np.complex128)
    sh_n_p_1 = np.zeros(order.shape, dtype = np.complex128)

    sh_n = sphhankel(order, x, 0)
    sh_n_m_1 = sphhankel(order, x, -1)
    sh_n_p_1 = sphhankel(order, x, 1)
    
    h_p = 1/2 * (sh_n_m_1 - (sh_n + x * sh_n_p_1) / x)
    return h_p

    
def calFocusedField():
#calculate a focused beam from the paramesters specified
    #the order of functions for calculating focused field
    orderEf = 100
    #il term
    ordVec = np.arange(0, orderEf+1, 1)
    il = 1j ** ordVec
    
    #legendre polynomial of the condenser
    plCosAlpha1 = Legendre(orderEf+1, np.cos(alpha1))
    plCosAlpha2 = Legendre(orderEf+1, np.cos(alpha2))
    
    #initialize magnitude of r vector at each pixel
    rMag = np.zeros((simRes, simRes))
    #initialize angle between k vector to each r vector 
    cosTheta = np.zeros((rMag.shape))
    #initialize normalized r vector
    rNorm = np.zeros((rVecs.shape))
    #normalized k vector 
    kNorm = kVec / magk
    #compute rMag and rNorm and cosTheta at each pixel
    for i in range(simRes):
        for j in range(simRes):
            rMag[i, j] = np.sqrt(rVecs_ps[i, j, 0]**2+rVecs_ps[i, j, 1]**2+rVecs_ps[i, j, 2]**2)
            rNorm[i, j, :] = rVecs_ps[i, j, :] / rMag[i,j]
            cosTheta[i, j] = np.dot(kNorm, rNorm[i, j, :])
    
    #compute spherical bessel function at kr
    jlkr= sphbesselj(ordVec, magk*rMag, 0)
    
    #compute legendre polynomial of all r vector
    plCosTheta = Legendre(orderEf, cosTheta)
    
    #product of them
    jlkrPlCosTheta = jlkr * plCosTheta
    
    il = il.reshape((1, 1, orderEf+1))
    iljlkrplcos = jlkrPlCosTheta * il
    
    order = 0
    iljlkrplcos[:,:,order] = iljlkrplcos[:,:,order]*(plCosAlpha1[order+1]-plCosAlpha2[order+1]-plCosAlpha1[0]+plCosAlpha2[0])
    
    order = 1
    iljlkrplcos[:,:,order] = iljlkrplcos[:,:,order]*(plCosAlpha1[order+1]-plCosAlpha2[order+1]-plCosAlpha1[0]+plCosAlpha2[0])
    
    for order in range(2, orderEf):
        iljlkrplcos[:,:,order] = iljlkrplcos[:,:,order]*(plCosAlpha1[order+1]-plCosAlpha2[order+1]-plCosAlpha1[order-1]+plCosAlpha2[order-1])
    
    #sum up all orders
    Ef = 2*np.pi*E0*np.sum(iljlkrplcos, axis = 2)
    
    return Ef

def scatterednInnerField():
#calculate and return a focused field and the corresponding scattering field and internal field
    #maximal number of orders used to calculate Es and Ei
    numOrd = math.ceil(2*np.pi * a / lambDa + 4 * (2 * np.pi * a / lambDa) ** (1/3) + 2)
    #create an order vector
    ordVec = np.arange(0, numOrd+1, 1)
    #calculate the prefix term (2l + 1) * i ** l
    twolplus1 = 2 * ordVec + 1
    il = 1j ** ordVec
    twolplus1_il = twolplus1 * il
    #compute the arguments for spherical bessel functions, hankel functions and thier derivatives
    ka = magk * a
    kna = magk * n * a
    #number of samples
    numSample = 25
    
    #evaluate the spherical bessel functions of the first kind at ka
    jl_ka = sphbesselj(ordVec, ka, 0)
    
    #evaluate the derivative of the spherical bessel functions of the first kind at kna
    jl_kna_p = derivSphBes(ordVec, kna)
    
    #evaluate the spherical bessel functions of the first kind at kna
    
    jl_kna = sphbesselj(ordVec, kna, 0)
    
    #evaluate the derivative of the spherical bessel functions of the first kind of ka
    jl_ka_p = derivSphBes(ordVec, ka)
    
    #compute the numerator for B coefficients
    numB = jl_ka * jl_kna_p * n - jl_kna * jl_ka_p
    
    #evaluate the hankel functions of the first kind at ka
    hl_ka = sphhankel(ordVec, ka, 0)
    
    #evaluate the derivative of the hankel functions of the first kind at ka
    hl_ka_p = derivSphHan(ordVec, ka)
    
    #compute the denominator for coefficient A and B
    denAB = jl_kna * hl_ka_p - hl_ka * jl_kna_p * n
    
    #compute B
    B = np.asarray(twolplus1_il * (numB / denAB), dtype = np.complex128)
    B = np.reshape(B, (1, 1, numOrd + 1))
    
    #compute the numerator of the scattering coefficient A
    numA = jl_ka * hl_ka_p - jl_ka_p * hl_ka
    
    #compute A
    A = np.asarray(twolplus1_il * (numA / denAB), dtype = np.complex128)
    A = np.reshape(A, (1, 1, numOrd + 1))
    
    #compute the distance between r vector and the sphere
    rMag = np.zeros((simRes, simRes))
    rNorm = np.zeros((rVecs.shape))
    for i in range(simRes):
        for j in range(simRes):
            rMag[i, j] = np.sqrt(rVecs_ps[i, j, 0]**2 + rVecs_ps[i, j, 1]**2 + rVecs_ps[i, j, 2]**2)
            rNorm[i, j, :] = rVecs_ps[i, j, :] / rMag[i,j]
    #computer k*r term
    kr = magk * rMag
    
    #compute the spherical hankel function of the first kind for kr
    hl_kr = sphhankel(ordVec, kr, 0)
    
    #computer k*n*r term
    knr = kr * n
    
    #compute the spherical bessel function of the first kind for knr
    jl_knr = sphbesselj(ordVec, knr, 0)
    
    #compute the distance from the center of the sphere to the focal point/ origin
    #used for calculating phase shift later
    c = ps - pf
    
    #initialize Ei and Es field
    Ei = np.zeros((simRes, simRes), dtype = np.complex128)
    Es = np.zeros((simRes, simRes), dtype = np.complex128)
    
    #a list of sampled k vectors
    k_j, scale = sampled_kvectors_spherical_coordinates(NA_out, numSample, lightdirection)
    
    for k_index in range(numSample ** 2):
        cos_theta = np.zeros((rMag.shape))
        for i in range(simRes):
            for j in range(simRes):
                cos_theta[i, j] = np.dot(k_j[:, k_index], rNorm[i, j, :])
        pl_costheta = Legendre(numOrd, cos_theta)
        hlkr_plcostheta = hl_kr * pl_costheta
        jlknr_plcostheta = jl_knr * pl_costheta
        
        phase = np.exp(1j * magk * np.dot(k_j[:, k_index], c))
        
        Es += phase * scale[k_index] * np.sum(hlkr_plcostheta * B, axis = 2)
    #    Ei += phase * scale[k_index] * np.sum(jlknr_plcostheta * a, axis = 2)
        Ei += phase * scale[k_index] * np.sum(jlknr_plcostheta * A, axis = 2)
    
    Es[rMag<a] = 0
    Ei[rMag>=a] = 0
    
    Ef = calFocusedField()
    
    
    Etot = np.zeros((simRes, simRes), dtype = np.complex128)
    Etot[rMag<a] = Ei[rMag<a]
    Etot[rMag>=a] = Es[rMag>=a] + Ef[rMag>=a]

#    plt.figure()
#    #plt.plot(np.abs(hl_ka))
#    plt.imshow(np.log10(np.abs(Etot)))
    return Ef, Etot

def BPF(halfgrid, simRes, NA_in, NA_out):
#create a bandpass filter
    #change coordinates into frequency domain    
    df = 1/(halfgrid*2)
    
    iv, iu = np.meshgrid(np.arange(0, simRes, 1), np.arange(0, simRes, 1))
    
    u = np.zeros(iu.shape)
    v = np.zeros(iv.shape)
    
    #initialize the filter as All Pass
    BPF = np.ones(iv.shape)
    
    idex1, idex2 = np.where(iu <= simRes/2)
    u[idex1, idex2] = iu[idex1, idex2]
    
    idex1, idex2 = np.where(iu > simRes/2)
    u[idex1, idex2] = iu[idex1, idex2] - simRes +1
    
    u *= df
    
    idex1, idex2 = np.where(iv <= simRes/2)
    v[idex1, idex2] = iv[idex1, idex2]
    
    idex1, idex2 = np.where(iv > simRes/2)
    v[idex1, idex2] = iv[idex1, idex2] - simRes +1
    
    v *= df
    
    magf = np.sqrt(u ** 2 + v ** 2)
    
    #block lower frequency
    idex1, idex2 = np.where(magf < NA_in / lambDa)
    BPF[idex1, idex2] = 0
    #block higher frequency
    idex1, idex2 = np.where(magf > NA_out / lambDa)
    BPF[idex1, idex2] = 0
    
    return BPF


#parameters used to calculate the fields
#resolution
res = 150
#position of the sphere
ps = np.asarray([0, 0, 0])
#position of the focal point
pf = np.asarray([0, 0, 0])

#padding for displaying the figure
padding = 1
#amplitude of the incoming electric field
E0 = 1
#in and out numerical aperture of the condenser
NA_in = 0
NA_out = 0.3
#theta and phi in spherical coordinate system
theta = 1.5708
phi = 0
#pixel size of the figure
pixel = 1.1

alpha1 = math.asin(NA_in)
alpha2 = math.asin(NA_out)

#convert coordinates to cartesian if necessary
x, y, z = sph2cart(theta, phi, 1)

#specify the direction of the incoming light
lightdirection = np.asarray([0, 0, -1])

#specify the wavelength of the incident light
lambDa = 2.8554
#refractive index of the sphere
n = 1.4763 + 0.000258604 * 1j
#radius of the sphere
radiusSphere = a = 15

#magnitude of the k vector
#wavenumber = magk = 2*np.pi/lambDa
wavenumber = magk = 2.2004
kVec = lightdirection * magk

#field of view, the size of the pic
fov = np.ceil(res*pixel)
#simulation resolution
simRes = res*(2*padding + 1)

#initialize a plane to evaluate the field
halfgrid = np.ceil(fov/2)*(2*padding +1)
gx = np.linspace(-halfgrid, +halfgrid-1, simRes)
gy = gx

#if it is a plane in []
[y, z] = np.meshgrid(gx, gy)

#make it a plane at 0 on the axis below
x = np.zeros((simRes, simRes,))

#initialize r vectors in the space
rVecs = np.zeros((simRes, simRes, 3))

rVecs[:,:,0] = x
rVecs[:,:,1] = y
rVecs[:,:,2] = z

#compute the rvector relative to the sphere
rVecs_ps = rVecs - ps

bpf = BPF(halfgrid, simRes, NA_in, NA_out)

Ef, Etot = scatterednInnerField()

#2D fft to the total field
Et_d = np.fft.fft2(Etot)
Ef_d = np.fft.fft2(Ef)

#apply bandpass filter to the fourier domain
Et_d *= bpf
Ef_d *= bpf

#invert FFT back to spatial domain
Et_bpf = np.fft.ifft2(Et_d)
Ef_bpf = np.fft.ifft2(Ef_d)

#initialize cropping
cropsize = padding * res
startIdx = int(np.fix(simRes /2 + 1) - np.floor(cropsize/2))
endIdx = int(startIdx + cropsize - 1)

#save the field
np.save(r'D:\irimages\irholography\New_QCL\BimSimPython\Et15YoZ.npy', Et_bpf)

#crop the image
D_Et = np.zeros((cropsize, cropsize), dtype = np.complex128)
D_Et = Et_bpf[startIdx:endIdx, startIdx:endIdx]

#plot the field
plt.figure()
plt.title("real")
plt.imshow(np.real(D_Et))
plt.colorbar()

plt.figure()
plt.title("imaginary")
plt.imshow(np.imag(D_Et))
plt.colorbar()

plt.figure()
plt.title("magnitude")
plt.imshow(np.abs(D_Et))
plt.colorbar()

#plt.title("YOZ Plane")
#plt.axis("off")
#plt.set_cmap('gnuplot2')
#plt.colorbar()
#


#temp = loadmat(r'D:\irimages\irholography\oldQCL\bimsim_test\EsEi\DEt.mat')
##temp = loadmat(r'D:\irimages\irholography\bimsim_test\DEt.mat')
#DEt= temp["D_Et"]
##print(np.amax(sh11 - hl_kr[:,:,0]))
##
##diff = np.zeros(hl_kr.shape, dtype = np.complex128)
##for i in range(numOrd):
##    diff[:,:,i] = cpu_hl_kr[:,:,i] - hl_kr[:,:,i]
##    print(np.amax(cpu_hl_kr[:,:,i] - hl_kr[:,:,i]))
##
###
####
#plt.figure()
###plt.plot(np.abs(hl_ka))
#plt.imshow(DEt)
#plt.plot(diff)
##plt.colorbar()
##plt.imshow(np.abs(Etot))
##
#_min, _max = np.amin(np.imag(hl_kr)), np.amax(np.imag(hl_kr))
#fig = plt.figure()
#
#img = []
#for i in range(23):
#    img.append([plt.imshow(np.imag(diff[:,:,i]), vmax = _max, vmin = _min)])
#
#ani = animation.ArtistAnimation(fig,img,interval=100)
#writer = animation.writers['ffmpeg'](fps=10)
#plt.colorbar()


#alpha2m= t["alpha2"]
####
####
#plt.figure()
#plt.subplot(121)
#plt.imshow(np.abs(iljlkr1[:,:,1]))
#plt.colorbar()
#plt.title("BimSIm")
#plt.subplot(122)
#plt.imshow(np.abs(iljlkrplcos[:,:,1]))
#plt.colorbar()
#plt.title("Shihao")
#
#plt.figure()
#plt.plot(alpha1m)
#plt.plot(plCosAlpha1)


