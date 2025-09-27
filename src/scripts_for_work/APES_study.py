import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import jahn_teller_dynamics.io.plotting as plotting
import jahn_teller_dynamics.math.maths as maths


#ffun = (0.5*K).*r.^2-r.*(F.^2.+G.^2.*r.^2+(2.*F.*G).*r.*cos(3.*th)).^0.5;


xs = np.linspace(-1.5, 1.5,500)
dx = abs(xs[1]-xs[0])

ys = np.linspace(-1.5, 1.5,500)
dy = abs(ys[1]-ys[0])

print('Try APES')

#Mine

K = 1348.4739751353143
F = 203.23751944854286
G = 13.22859065586539

#Not mine

K = 74.94681160585864
F = 47.91364181415197
G = 0.7352316099364676

#SiC problem:
F = 405.9937
G = 35.5848
K = 2018.1912

F = 39.1285
G = 1.9681
K = 28.8073

#TG scheme
K=0.05903
F=0.041179
G=0.015442
#G = 20



F = 116.4963 
G = 17.4457 
K = 255.3526

K=0.05903
F=0.041179
G=0.01544200

F = 39.1285
G = 1.9681
K = 28.8073

apes = maths.APES_second_order.from_pars(K, F , G )
#apes = APES_second_order.from_pars(K = 1, F = 2, G = 1)

#res_df = apes.calc_full_apes_negativ_Descartes(xs,ys)
res_df ,data = apes.calc_APES(xs,ys)



plt.rcParams['font.size'] = 20

#plt.title('My APES')
plt.title('My APES')
res_contour =  plotting.contour_data().from_df(res_df)



res_contour.create_contour()
plt.colorbar()




############################################################x
loc_mins =  maths.get_loc_min_indexes(res_df)

print('In loc_mins')
for loc_min in loc_mins:

    x = round(xs[loc_min[0]],4)
    y = round(ys[loc_min[1]],4)
    z = round(data[loc_min[0]][loc_min[1]],4)
    r = round((x**2 + y**2)**0.5,4)
    plt.plot([x],[y], 'x')
    print( f'x = {x} ,y = {y} ,r = {r}, z = {z}' )

plt.xlabel('x coordinate'+ r'$\left( angström \sqrt{ Dalton } \right)$')
plt.ylabel('y coordinate' + r'$\left( angström \sqrt{ Dalton } \right)$')
plt.show()


fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(projection='3d')

X,Y = np.meshgrid(xs,ys)

ax.plot_surface(X, Y, data, cmap=plt.cm.YlGnBu_r)

ax.grid(False)

# Tweak the limits and add latex math labels.

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.tick_params(axis='z', which='major', pad=10)


#ax.scatter(delta_x, delta_y,-(E_JT-delta)*0.9 , color='red')

ax.set_xlabel(r'$\hat{X}$')
ax.set_ylabel(r'$\hat{Y}$')
ax.set_zlabel(r'$E$', labelpad = 11)
ax.set_zlim(-0.03 ,0.03)




#fig.colorbar(surf)
plt.show()