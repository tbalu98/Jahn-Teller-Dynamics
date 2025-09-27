import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def y_from_pol(r, phi):
    return np.sin(phi)*r

def x_from_pol(r, phi):
    return np.cos(phi)*r


K=0.05903
F=0.041179
#G=0.015442

G = 0.007

fig = plt.figure(figsize=(20,20))

plt.rcParams['font.size'] = 21
ax = fig.add_subplot(projection='3d')

# Create the mesh in polar coordinates and compute corresponding Z.
r = np.linspace(0, 1.2, 1000)
th = np.linspace(0, 2*np.pi, 1000)
R, P = np.meshgrid(r, th)

Z_down = (0.5*K)*R**2-R*(F**2+G**2*R**2+(2*F*G)*R*np.cos(3*P))**0.5

r2 = np.linspace(0, 0.2, 50)
th2 = np.linspace(0, 2*np.pi, 50)
R2, P2 = np.meshgrid(r2, th2)


Z_up = (0.5*K)*R2**2+R2*(F**2+G**2*R2**2+(2*F*G)*R2*np.cos(3*P2))**0.5

E_JT = F**2/(2*( K-2*G ))



n = 4
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


# Express the mesh in the cartesian system.
X, Y = R*np.cos(P), R*np.sin(P)
X2, Y2 = R2*np.cos(P2), R2*np.sin(P2)

# Plot the surface.
#ax.scatter(delta_x, delta_y,-(E_JT-delta)*0.5 , color='red')
ax.view_init(elev=20, azim=-120, roll=0)

ax.plot_surface(X, Y, Z_down,alpha = 1.0,label = 'surface_2', cmap=cm.coolwarm)
#ax.plot_surface(X2, Y2, Z_up,alpha = 1.0,label = 'surface_1', cmap=cm.coolwarm)


ax.grid(False)

# Tweak the limits and add latex math labels.

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.tick_params(axis='z', which='major', pad=20)

ax.set_zticks([-0.02,-0.015, -0.01,-0.005, 0.0, 0.005,0.01], [-2.0,-1.5, -1.0,-0.5, 0.0, 0.5,1.0])
ax.text(-0.8, 2.3, 0.00+0.005, r'$\times 10^{-2}$', zdir = None)
"""
ax.text(0.0, 0.0, 0.00+0.005, r"$E_{JT}$", zdir = None)
ax.text(0.0, 0.0, 0.00+0.01, r"$\delta_{b}$", zdir = None)
"""


"""
ax.text(-0.67, -0.97 , -0.032, r"$E_{min}$", zdir = None)
ax.text( delta_x+0.05, delta_y-0.05 , -(E_JT-delta)*0.75, r"$E_{sp}$", zdir = None)
ax.text( 0.0, 0.0 , 0.002, r"$E_{hs}$", zdir = None)

ax.text(0.0, 0.0, 0.03,r'$E_{JT}$', zdir = None)
ax.text(0.3, -0.3, 0.03,r'$\delta$', zdir = None)
"""

# Make the plot occupy more space by adjusting the axis limits
ax.set_xlim(-1.2, 1.2)  # Even tighter X limits
ax.set_ylim(-1.2, 1.2)  # Even tighter Y limits

# Adjust the position and size of the plot within the figure to take up more space
ax.set_position([-0.1, -0.05, 1.1, 1.1])  # Increased width and height, reduced margins

# Make the plot appear larger by adjusting the aspect ratio
ax.set_box_aspect([2, 2, 1.5])  # Reduced z-axis scaling for more dramatic effect


#ax.scatter(delta_x, delta_y,-(E_JT-delta)*0.9 , color='red')

ax.set_xlabel(r'$\hat{X}$'+r" $(\sqrt{amu}$ Å)", labelpad = 20)
ax.set_ylabel(r'$\hat{Y}$'+r" $(\sqrt{amu}$ Å)", labelpad = 20)
ax.set_zlabel(r'$E$ (eV)', labelpad = 40)
ax.set_zlim(-0.02 ,0.01)



#fig.colorbar(surf)
plt.show()