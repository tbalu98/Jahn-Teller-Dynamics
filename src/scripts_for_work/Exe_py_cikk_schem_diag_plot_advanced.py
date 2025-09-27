import matplotlib.pyplot as plt


plt.figure(figsize=(10,6))
#ground state split plot
plt.xlim(1.0, 6.2)

plt.ylim(0,9)
"""
plt.arrow(4.3, 4.75, 0.0, -0.4, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = 'red', edgecolor = 'red')
plt.arrow(4.45, 4.75, 0.0, -0.9, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = 'red', edgecolor = 'red')
plt.arrow(4.6, 4.75, 0.0, -2.40, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = 'red', edgecolor = 'red')
plt.arrow(4.75, 4.75, 0.0, -2.90, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = 'red', edgecolor = 'red')
"""




#plt.arrow(5.125, 7.25, 0.0, -0.4, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = 'blue', edgecolor = 'blue')
#plt.arrow(5.125, 6.75, 0.0, 0.4, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = 'blue', edgecolor = 'blue')

plt.arrow(3.475, 6.9, 0.0, -1.8, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = 'blue', edgecolor = 'blue',zorder = 10)

plt.arrow(3.475, 5.1, 0.0, 1.8, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = 'blue', edgecolor = 'blue',zorder = 10)
plt.text(3.5, 5.8, r'$\lambda^{\text{ex}}_{\text{theory}}$', fontsize = 15)

plt.arrow(3.475, 2.9, 0.0, -1.8, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = 'blue', edgecolor = 'blue',zorder = 10)
plt.arrow(3.475, 1.1, 0.0, 1.8, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = 'blue', edgecolor = 'blue',zorder = 10)
plt.text(3.5, 1.8, r'$\lambda^{\text{gnd}}_{\text{theory}}$', fontsize = 15)


#DFT SOC:
plt.arrow(2.375, 7.4, 0.0, -2.8, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = 'blue', edgecolor = 'blue',zorder = 10)
plt.arrow(2.375, 4.6, 0.0, 2.8, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = 'blue', edgecolor = 'blue',zorder = 10)
plt.text(2.4, 5.8, r'$\lambda^{\text{ex}}_{\text{DFT}}$', fontsize = 15)

plt.arrow(2.375, 3.4, 0.0, -2.8, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = 'blue', edgecolor = 'blue',zorder = 10)
plt.arrow(2.375, 0.6, 0.0, 2.8, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = 'blue', edgecolor = 'blue',zorder = 10)
plt.text(2.4, 1.8, r'$\lambda^{\text{gnd}}_{\text{DFT}}$', fontsize = 15)



plt.plot([1.0, 1.75],[6.0, 6.0], 'k-')
plt.plot([1.75, 2.0], [6,7.5], 'k--')
plt.plot([2.0, 2.75], [7.5, 7.5] ,'k-')
plt.plot([2.75, 3.0],[7.5,7],'k--')
plt.plot([3.0,3.75],[7,7], 'k-')
plt.plot([3.75,4.0],[7,7.25], 'k--')
plt.plot([4.0,6.1],[7.25,7.25], 'k-')


plt.plot([3.75,4.0],[7,6.75], 'k--')
plt.plot([4.0, 6.1],[6.75, 6.75],'k-') 

#########################################

plt.plot([1.0, 1.75],[6.0, 6.0], 'k-')
plt.plot([1.75, 2.0], [6,4.5], 'k--')
plt.plot([2.0, 2.75], [4.5, 4.5] ,'k-')
plt.plot([2.75, 3.0],[4.5,5],'k--')
plt.plot([3.0,3.75],[5,5], 'k-')
plt.plot([3.75,4.0],[5,5.25], 'k--')
plt.plot([4.0,6.1],[5.25,5.25], 'k-')
plt.plot([3.75,4.0],[5,4.75], 'k--')
plt.plot([4.0,6.1],[4.75,4.75], 'k-')


#########################################
#Ground state


plt.plot([1.0, 1.75],[2.0, 2.0], 'k-')
plt.plot([1.75, 2.0], [2,3.5], 'k--')
plt.plot([2.0, 2.75], [3.5, 3.5] ,'k-')
plt.plot([2.75, 3.0],[3.5,3],'k--')
plt.plot([3.0,3.75],[3,3], 'k-')
plt.plot([3.75,4.0],[3,3.25], 'k--')
plt.plot([4.0,6.1],[3.25,3.25], 'k-')


plt.plot([3.75,4.0],[3,2.75], 'k--')
plt.plot([4.0, 6.1],[2.75, 2.75],'k-') 

#########################################

plt.plot([1.0, 1.75],[2.0, 2.0], 'k-')
plt.plot([1.75, 2.0], [2,0.5], 'k--')
plt.plot([2.0, 2.75], [0.5, 0.5] ,'k-')
plt.plot([2.75, 3.0],[0.5,1],'k--')
plt.plot([3.0,3.75],[1,1], 'k-')
plt.plot([3.75,4.0],[1,1.25], 'k--')
plt.plot([4.0,6.1],[1.25,1.25], 'k-')
plt.plot([3.75,4.0],[1,0.75], 'k--')
plt.plot([4.0,6.1],[0.75,0.75], 'k-')

"""

plt.plot( [ 1.0 ,2.0 ],[ 3.0, 3.0 ], 'k-' ) #sima
plt.plot([2,2.25],[3,4],'k--') #szaggatott
plt.plot([2.25, 4.0],[4.0, 4.0],'k-') #sima
plt.plot([4.0,4.25],[4.0,4.25],'k--') #szaggatott
plt.plot([4.25, 6.1],[4.25,4.25],'k-') #sima
plt.plot([4.0,4.25],[4.0,3.75],'k--') #szaggatott
plt.plot( [  4.25, 6.1 ],[  3.75, 3.75 ], 'k-' ) #sima

plt.plot( [ 1.0 ,2.0 ],[ 3.0, 3.0 ], 'k-' ) #sima
plt.plot([2,2.25],[3,2],'k--') #szaggatott
plt.plot([2.25, 4.0],[2.0, 2.0],'k-') #sima
plt.plot([4.0,4.25],[2.0,2.25],'k--') #szaggatott
plt.plot([4.25, 6.1],[2.25,2.25],'k-') #sima
plt.plot([4.0,4.25],[2.0,1.75],'k--') #szaggatott
plt.plot( [  4.25, 6.1 ],[  1.75, 1.75 ], 'k-' ) #sima
"""
#plt.plot( [ 1.0 ,2.0, 2.25, 4.0, 4.25, 6.1 ],[ 6.0, 6.0, 5.0, 5.0, 5.25, 5.25 ], 'k-' )
#plt.plot( [ 1.0 ,2.0, 2.25, 4.0, 4.25, 6.1 ],[ 6.0, 6.0, 5.0, 5.0, 4.75, 4.75 ], 'k-' )

#plt.plot( [ 1.0 ,2.0, 2.25, 4.0, 4.25, 6.1 ],[ 3.0, 3.0, 4.0, 4.0, 4.25, 4.25 ], 'k-' )
#plt.plot( [ 1.0 ,2.0, 2.25, 4.0, 4.25, 6.1 ],[ 3.0, 3.0, 4.0, 4.0, 3.75, 3.75 ], 'k-' )

#plt.plot( [ 1.0 ,2.0, 2.25, 4.0, 4.25, 6.1 ],[ 3.0, 3.0, 2.0, 2.0, 2.25, 2.25 ], 'k-' )
#plt.plot( [ 1.0 ,2.0, 2.25, 4.0, 4.25, 6.1 ],[ 3.0, 3.0, 2.0, 2.0, 1.75, 1.75 ], 'k-' )

#ax.annotate("", xy=(0.5, 0.5), xytext=(0, 0),arrowprops=dict(arrowstyle="->"))

#plt.arrow(3.5, 5.0, 0, 1.8, head_length = 0.2, head_width = 0.05, width = 0.0125, fc= 'black')

#plt.arrow(4.75, 6.75, 0, 0.3, head_length = 0.2, head_width = 0.05, width = 0.0125, fc= 'black')










#plt.text(2.275, 4.15, r'$\epsilon^{\text{gnd}}_{3}, \epsilon^{\text{gnd}}_{4}$', fontsize = 15, fontname = 'Arial')
#plt.text(2.275, 2.15, r'$\epsilon^{\text{gnd}}_{1}, \epsilon^{\text{gnd}}_{2}$', fontsize = 15, fontname = 'Arial')

plt.text(1.0, 6.25, r'$^2E^{\text{ex}}$', fontsize = 15, fontname = 'Arial')
plt.text(1.0, 2.25, r'$^2E^{\text{gnd}}$', fontsize = 15, fontname = 'Arial')




plt.text(2.0, 7.75, r'$E^{\text{ex}}_{ \frac{1}{2} }$', fontsize = 15, fontname = 'Arial')
plt.text(2.0, 4.75, r'$E^{\text{ex}}_{ \frac{3}{2} } $', fontsize = 15, fontname = 'Arial')



plt.text(2.0, 3.75, r'$E^{\text{gnd}}_{ \frac{1}{2} }$', fontsize = 15, fontname = 'Arial')
plt.text(2.0, 0.75, r'$E^{\text{gnd}}_{ \frac{3}{2} }$', fontsize = 15, fontname = 'Arial')




plt.text(3, 7.15, r'$\epsilon^{\text{ex}}_{3,4}$', fontsize = 15, fontname = 'Arial')
plt.text(3, 5.15, r'$\epsilon^{\text{ex}}_{1,2}$', fontsize = 15, fontname = 'Arial')

plt.text(3, 3.15, r'$\epsilon^{\text{gnd}}_{3,4}$', fontsize = 15, fontname = 'Arial')
plt.text(3, 1.15, r'$\epsilon^{\text{gnd}}_{1,2}$', fontsize = 15, fontname = 'Arial')


"""
plt.text(4.275, 4.4, r'$\epsilon_{4}$', fontsize = 15, fontname = 'Arial')
plt.text(4.275, 3.9, r'$\epsilon_{3}$', fontsize = 15, fontname = 'Arial')

plt.text(4.275, 2.4, r'$\epsilon_{2}$', fontsize = 15, fontname = 'Arial')
plt.text(4.275, 1.9, r'$\epsilon_{1}$', fontsize = 15, fontname = 'Arial')

plt.text(4.275, 5.4, r'$\epsilon_{2}$', fontsize = 15, fontname = 'Arial')
plt.text(4.275, 4.9, r'$\epsilon_{1}$', fontsize = 15, fontname = 'Arial')

plt.text(4.275, 7.4, r'$\epsilon_{4}$', fontsize = 15, fontname = 'Arial')
plt.text(4.275, 6.9, r'$\epsilon_{3}$', fontsize = 15, fontname = 'Arial')

"""

#plt.text(3.2, 8.5, "dynamic Jahn-Teller effect \n spin-orbit coupling", horizontalalignment = 'center', fontsize = 15, fontname = 'Arial')

plt.text(2.375, 8.5, "spin-orbit\ncoupling", horizontalalignment = 'center', fontsize = 15, fontname = 'Arial')

plt.text(3.5, 8.5, "dynamic "+"\n Jahn-Teller effect", horizontalalignment = 'center', fontsize = 15, fontname = 'Arial')


plt.text(5.1, 8.5, "Zeemann\nsplitting",horizontalalignment = 'center', fontsize = 15, fontname = 'Arial')

"""
plt.text(6.0, 1.75, '|4>', fontsize = 15)
plt.text(6.0, 2.25, '|3>', fontsize = 15)
plt.text(6.0, 3.75, '|2>', fontsize = 15)
plt.text(6.0, 4.25, '|1>', fontsize = 15)
"""

"""
plt.text(6.0, 7.25, r'$| E_{\frac{1}{2}} >$', fontsize = 15)
plt.text(6.0, 6.75, r'$| E_{-\frac{1}{2}} >$', fontsize = 15)
plt.text(6.0, 5.25, r'$| E_{\frac{3}{2}} >$', fontsize = 15)
plt.text(6.0, 4.75, r'$| E_{-\frac{3}{2}} >$', fontsize = 1plt.text(1.0, 6.2, r"$|excited\rangle$", fontsize = 15, fontname = 'Times New Roman')5)
"""
"""
plt.text(6.0, 7.25, r'$| E_{\frac{1}{2}} >$', fontsize = 15)
plt.text(6.0, 6.75, r'$| E_{-\frac{1}{2}} >$', fontsize = 15)
plt.text(6.0, 5.25, r'$| E_{\frac{3}{2}} >$', fontsize = 15)
plt.text(6.0, 4.75, r'$| E_{-\frac{3}{2}} >$', fontsize = 15)
"""

"""
plt.text(6.0, 7.25, r'$| A >$', fontsize = 15)
plt.text(6.0, 6.75, r'$| B >$', fontsize = 15)
plt.text(6.0, 5.25, r'$| C >$', fontsize = 15)
plt.text(6.0, 4.75, r'$| D >$', fontsize = 15)
"""

"""
plt.text(1.95, 6.9, r'$e_{ \frac{1}{2} }$', fontsize = 15)
plt.text(1.95, 5.1, r'$e_{ \frac{3}{2} }$', fontsize = 15)

plt.text(1.95, 3.9, r'$e_{ \frac{1}{2} }$', fontsize = 15)
plt.text(1.95, 2.1, r'$e_{ \frac{3}{2} }$', fontsize = 15)
"""

#plt.text(1.7, 8.72, r'$\hat{H} = \hat{H}_{osc} + \hat{H}_{DJT} + \hat{H}_{SOC} + \hat{H}_{ext} $', fontsize = 30)

plt.text(5.9, 5.6, 'D', fontsize  = 15)


arr_color = 'darkred'

plt.arrow(5.75, 4.75, 0.0, -1.4, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = arr_color, edgecolor = arr_color,zorder = 10)
plt.arrow(5.85, 4.75, 0.0, -1.9, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = arr_color, edgecolor = arr_color,zorder = 10)
plt.arrow(5.95, 5.25, 0.0, -1.9, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = arr_color, edgecolor = arr_color,zorder = 10)
plt.arrow(6.05, 5.25, 0.0, -2.4, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = arr_color, edgecolor = arr_color,zorder = 10)

plt.text(5.5, 5.6, 'C', fontsize  = 15)


arr_color = 'red'

plt.arrow(5.35, 4.75, 0.0, -3.4, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = arr_color, edgecolor = arr_color,zorder = 10)
plt.arrow(5.45, 4.75, 0.0, -3.9, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = arr_color, edgecolor = arr_color,zorder = 10)
plt.arrow(5.55, 5.25, 0.0, -3.9, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = arr_color, edgecolor = arr_color,zorder = 10)
plt.arrow(5.65, 5.25, 0.0, -4.4, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = arr_color, edgecolor = arr_color,zorder = 10)

plt.text(5.1, 7.6, 'B', fontsize  = 15)


arr_color = 'green'


plt.arrow(4.95, 6.75, 0.0, -3.4, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = arr_color, edgecolor = arr_color,zorder = 10)
plt.arrow(5.05, 6.75, 0.0, -3.9, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = arr_color, edgecolor = arr_color,zorder = 10)
plt.arrow(5.15, 7.25, 0.0, -3.9, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = arr_color, edgecolor = arr_color,zorder = 10)
plt.arrow(5.25, 7.25, 0.0, -4.4, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = arr_color, edgecolor = arr_color,zorder = 10)

plt.text(4.7, 7.6, 'A', fontsize  = 15)


arr_color = 'dodgerblue'

plt.arrow(4.55, 6.75, 0.0, -5.4, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = arr_color, edgecolor = arr_color,zorder = 10)
plt.arrow(4.65, 6.75, 0.0, -5.9, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = arr_color, edgecolor = arr_color,zorder = 10)
plt.arrow(4.75, 7.25, 0.0, -5.9, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = arr_color, edgecolor = arr_color,zorder = 10)
plt.arrow(4.85, 7.25, 0.0, -6.4, head_length = 0.1, head_width = 0.05, width = 0.0125 , fc = arr_color, edgecolor = arr_color,zorder = 10)



plt.axis('off')
dir = 'results/Exe.py_publication/'
plt.savefig(dir + 'schem_diag.pdf', dpi=4000)

plt.show()
