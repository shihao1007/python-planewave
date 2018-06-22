import planewave as pw
import beam as bm
import numpy as np
import matplotlib.pyplot as plt

#calculate the intensity of a 3 x X x Y x .... field
def intensity(E):
    I = np.zeros(E.shape[1:len(E.shape)])
    for i in range(0, 3):
        I = I + E[i] ** 2
    
    return np.abs(I)
        
N = 200
M = 800
Ex = 1
Ey = 0
p = [0, 0, 0]
B = bm.beam(Ex, Ey, p, s=0.2, l=0.5)
E = B.kspace_image(N)

plt.figure(1)
plt.subplot(1, 3, 1)
plt.imshow(np.abs(E[0]))
plt.colorbar()
plt.subplot(1, 3, 2)
plt.imshow(np.abs(E[1]))
plt.colorbar()
plt.subplot(1, 3, 3)
plt.imshow(np.abs(E[2]))
plt.colorbar()

PW = B.decompose(M)

D = 30
x = np.linspace(-D, D, N)
z = np.linspace(-D, D, N)

[X, Z] = np.meshgrid(x, z)
Y = np.zeros((N, N))

E = PW[0].evaluate(X, Y, Z)
for i in range(1, M):
    E = E + PW[i].evaluate(X, Y, Z)
I = intensity(E)

plt.figure(2)
plt.imshow(I)