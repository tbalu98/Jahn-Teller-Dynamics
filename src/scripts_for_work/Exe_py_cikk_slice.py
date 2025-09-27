import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
import jahn_teller_dynamics.math.maths as maths

print(maths.exp_func(12,21,1,3))

def y_from_pol(r, phi):
    return np.sin(phi)*r

def x_from_pol(r, phi):
    return np.cos(phi)*r


K=0.05903
F=0.041179
G=0.015442
G = 0.007

fig = plt.figure(figsize=(14,7))

plt.rcParams['font.size'] = 12


# Create the mesh in polar coordinates and compute corresponding Z.
r = np.linspace(-1.2, 2.5, 1000)

th = np.linspace(0, 2*np.pi, 1000)
R, P = np.meshgrid(r, th)

minus_r = -1*np.linspace(0,1.2)

Z_down = (0.5*K)*r**2-r*(F**2+G**2*r**2+(2*F*G)*r*np.cos(3*0))**0.5


Z_up = (0.5*K)*r**2+r*(F**2+G**2*r**2+(2*F*G)*r*np.cos(3*0))**0.5






E_JT = -F**2/(2*( K-2*G ))



n = 0
rho1 = F/(K-(-1)**n*2*G)
phi = n*np.pi/3
print('E_JT')
print(rho1)
print(phi)
print(E_JT)

print('x' + str( x_from_pol(rho1, phi) ))
print('y' + str( y_from_pol(rho1, phi) ))
print('E_JT: ' + str(-E_JT))


#Z = ((R**2 - 1)**2)

print('barrier')
#barrier energy
delta = 4*E_JT*G/(K+2*G)


n = 5
rhod = F/(K-(-1)**n*2*G)
phid = n*np.pi/3


delta_x = x_from_pol(rhod, phid)
delta_y = y_from_pol(rhod, phid)
print('x' + str( delta_x ))
print('y' + str( delta_y ))

print('delta: ' + str(-(E_JT-delta)))


print('rho: ' + str(rhod))
print('energy: ' + str(E_JT+delta))

plt.rcParams['font.size'] = 21

plt.plot(r, Z_down, 'k--',label = r'$\epsilon_{x}$')
plt.plot(r, Z_up, 'k-', label = r'$\epsilon_{y}$')

plt.plot([rho1],[E_JT],markersize = '12', marker = 'o', color = 'red', label = 'local minimum ' + r'$(E_{\text{min}})$' )
plt.plot([-rhod],[E_JT-delta], markersize = '12',marker = 'o' , color = 'red', label =  'saddle point ' + r'$(E_{\text{sadd}})$')
plt.plot([0.0],[0.0], markersize = '12',marker = 'o' , color = 'red', label = 'conical intersection ' + r'$(E_{\text{hs}})$')

#energies:




plt.xlim(-1.2, 2.0)
plt.ylim(-0.032, .032)
plt.xlabel(r'$\hat{X}$'+r" $\left(\sqrt{amu}\right)$ Å)")
#plt.xlabel(r'$\delta$' + '\t' + r'$E_{JT}')



plt.ylabel("energy (eV)")
#plt.legend()


plt.savefig('results/Exe.py_publication/figures/new_APES_slice.pdf')
plt.show()

# Express the mesh in the cartesian system.
