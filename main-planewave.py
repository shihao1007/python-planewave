# -*- coding: utf-8 -*-
"""
Created on Wed May 23 10:06:28 2018

@author: david
"""
import numpy

#class planewave:
    #implement all features of a plane wave
    #   k, E, frequency (or wavelength in a vacuum)
    #   try to enforce that E and k have to be orthogonal
    
    #initialization function that sets these parameters when a plane wave is created
    
    #function that renders the plane wave given a set of coordinates
    #evaluate(X, Y, Z):
    
k=(0, 0, 1)
E=(0, 1, 0)
l = 1

#planewave p(k, E, l)
N = 100
c = numpy.linspace(-10, 10, N)
[X, Y] = numpy.meshgrid(c, c)
Z = numpy.zeros(X.shape)