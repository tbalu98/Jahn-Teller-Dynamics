import matplotlib.pyplot as plt
import numpy as np


plt.figure(figsize=(14,10))




x1 = 2.0
x2 = 5.0
def N3V_inactive(x):
    plt.plot([x,x+1],[0.5,0.5], 'k-', linewidth=2.5, zorder=10 )
    plt.plot([x,x+1],[0.8,0.8], 'k-', linewidth=2.5, zorder=10)
    plt.plot([x,x+1],[1.1,1.1], 'k-', linewidth=2.5, zorder=10)

    #plt.plot([2],[1], marker='^', color='black')

    plt.text(x+1.05, 0.47, '$A_1@N$', fontsize=30)
    plt.text(x+1.05, 0.77, '$E@N$', fontsize=30)
    plt.text(x+1.05, 1.07, '$A_1@C$', fontsize=30)

    plt.arrow(x+0.4, 0.38, 0.0, 0.2, head_length = 0.04, head_width = 0.05, width = 0.0125 , fc = 'black', edgecolor = 'black',zorder = 10)
    plt.arrow(x+.6, 0.62, 0.0, -0.2, head_length = 0.04, head_width = 0.05, width = 0.0125 , fc = 'black', edgecolor = 'black',zorder = 10)

    plt.arrow(x+.2, 0.68, 0.0, 0.2, head_length = 0.04, head_width = 0.05, width = 0.0125 , fc = 'black', edgecolor = 'black',zorder = 10)
    plt.arrow(x+.4, 0.92, 0.0, -0.2, head_length = 0.04, head_width = 0.05, width = 0.0125 , fc = 'black', edgecolor = 'black',zorder = 10)

    plt.arrow(x+.6, 0.68, 0.0, 0.2, head_length = 0.04, head_width = 0.05, width = 0.0125 , fc = 'black', edgecolor = 'black',zorder = 10)

    plt.arrow(x+.4, 0.98, 0.0, 0.2, head_length = 0.04, head_width = 0.05, width = 0.0125 , fc = 'black', edgecolor = 'black',zorder = 10)

N3V_inactive(x1)

plt.arrow(x1+.8, 0.92, 0.0, -0.2, head_length = 0.04, head_width = 0.05, width = 0.0125 , fc = 'black', edgecolor = 'black',zorder = 10)

plt.plot(x1+0.6, 1.1, marker='o', markeredgecolor='black', markerfacecolor='white', zorder=12, markersize=10)

N3V_inactive(x2)

plt.arrow(x2+.6, 1.22, 0.0, -0.2, head_length = 0.04, head_width = 0.05, width = 0.0125 , fc = 'black', edgecolor = 'black',zorder = 10)

plt.plot(x2+0.8, 0.8, marker='o', markeredgecolor='black', markerfacecolor='white', zorder=12, markersize=10)


plt.text(x1, 0.25, 'ground state', fontsize=30, color='blue')
plt.text(x2, 0.25, 'excited state', fontsize=30, color='red')





plt.axis('off')


plt.show()