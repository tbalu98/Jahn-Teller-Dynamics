import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import sys
import os



import jahn_teller_dynamics.math.maths as maths




def updownarrow(ax,x1, y1, x2,y2, head_lenght  = 0.1, head_width = 0.05):
    dy = abs(y2-y1-2*head_lenght)
    dx = 0
    ax.arrow(x1, y1+head_lenght, dx, dy, head_length = head_lenght, head_width = head_width, width = 0.0125 , fc = 'blue', edgecolor = 'blue',zorder = 10)
    ax.arrow(x2, y2-head_lenght, dx, -dy, head_length = head_lenght, head_width = head_width, width = 0.0125 , fc = 'blue', edgecolor = 'blue',zorder = 10)

ylim_start = 8
ylim_fin = 17

threshold_energy_high = 13.0
threshold_energy_low = 9.7

# Create a figure with two subplots side by side
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

# Set font size for both subplots
plt.rcParams['font.size'] = 21

# Store the current axes to restore later
original_axes = plt.gca()




def create_electron_structure(data_ex, start_index, fin_index):

    sp_ch_1_energies = data_ex['spin_ch_1_energy'].values[start_index:fin_index].round(4)
    sp_ch_2_energies = data_ex['spin_ch_2_energy'].values[start_index:fin_index].round(4)  


    sp_ch_1_occ = data_ex['spin_ch_1_occ'].values[start_index:fin_index]
    sp_ch_2_occ = data_ex['spin_ch_2_occ'].values[start_index:fin_index]


    energy_structure_ch_1 = {}
    for energy, occ in zip(sp_ch_1_energies, sp_ch_1_occ):
        if energy not in energy_structure_ch_1:
            energy_structure_ch_1[energy] = [int(occ)]
        else:
            energy_structure_ch_1[energy].append(int(occ))

    energy_structure_ch_2 = {}
    for energy, occ in zip(sp_ch_2_energies, sp_ch_2_occ):
        if energy not in energy_structure_ch_2:
            energy_structure_ch_2[energy] = [int(occ)]
        else:
            energy_structure_ch_2[energy].append(int(occ))
    return energy_structure_ch_1, energy_structure_ch_2






def draw_line(x_start, y_start):
    ax1.plot([x_start-0.005,x_start+0.055],[y_start,y_start], 'k-', linewidth=1.0, zorder=10 )


folder = 'results/N3V_PBE/gamma_point/'

ex_filename = 'N3V_ex_eigen'
gnd_filename = 'N3V_gnd_eigen'

data_ex = pd.read_csv(folder + ex_filename, sep = '\s+', skiprows = 8, header = None, 
                   names=['row_number', 'spin_ch_1_energy', 'spin_ch_2_energy', 'spin_ch_1_occ', 'spin_ch_2_occ'])

data_gnd = pd.read_csv(folder + gnd_filename, sep = '\s+', skiprows = 8, header = None, 
                   names=['row_number', 'spin_ch_1_energy', 'spin_ch_2_energy', 'spin_ch_1_occ', 'spin_ch_2_occ'])


gnd_vb_index = 1020
gnd_start_index = 1000
gnd_fin_index = 1090
gnd_cb_index = 1025

ex_vb_index = 1017
ex_start_index = 1000
ex_fin_index = 1070
ex_cb_index = 1025

vb_energy = 9.616053
cb_energy = 14.999805


gnd_sp_1_el_str, gnd_sp_2_el_str = create_electron_structure(data_gnd, gnd_start_index, gnd_fin_index)
ex_sp_1_el_str, ex_sp_2_el_str = create_electron_structure(data_ex, ex_start_index, ex_fin_index)



def plot_gap_states(sp_1_el_str:dict,  x_place, marker,vb_energy, cb_energy, threshold_energy_high, threshold_energy_low):

    middle_x_place = x_place + 0.025
    for energy, occ in sp_1_el_str.items():
        if energy > cb_energy or energy < vb_energy:
            continue
        draw_line(x_place, energy)
        
        deg = len(occ)
        curr_x_place = (middle_x_place - 0.0125) if deg>1 else middle_x_place
        if energy>threshold_energy_high or energy<threshold_energy_low:
            continue
        for  o,i in zip(occ, range(deg)):
            curr_x_place = curr_x_place + i*0.025
            if o == 0:
                ax1.scatter(curr_x_place, energy, marker = marker, facecolors='white', edgecolors='black', s = 70, zorder=10)
            else:
                ax1.scatter(curr_x_place, energy, marker = marker, color = 'black', s = 70)

def plot_valence_band_structure(sp_1_el_str:dict,  x_place, valence_band_energy):

    for energy, occ in sp_1_el_str.items():
        if energy < valence_band_energy:
            draw_line(x_place, energy)

def plot_conduction_band_structure(sp_1_el_str:dict,  x_place, conduction_band_energy):

    for energy, occ in sp_1_el_str.items():
        if energy > conduction_band_energy:
            draw_line(x_place, energy)


#plot_band_structure(gnd_sp_1_el_str, -0.05, '^')
#plot_band_structure(gnd_sp_2_el_str, 0.05, 'v')


plot_gap_states(ex_sp_1_el_str, 0.15, '^', vb_energy, cb_energy, threshold_energy_high, threshold_energy_low)
plot_gap_states(ex_sp_2_el_str, 0.25, 'v', vb_energy, cb_energy, threshold_energy_high, threshold_energy_low)

plot_gap_states(gnd_sp_1_el_str, -0.05, '^', vb_energy, cb_energy, threshold_energy_high, threshold_energy_low)
plot_gap_states(gnd_sp_2_el_str, 0.05, 'v', vb_energy, cb_energy, threshold_energy_high, threshold_energy_low)

plot_valence_band_structure(gnd_sp_1_el_str, -0.05, vb_energy)
plot_valence_band_structure(gnd_sp_2_el_str, 0.05, vb_energy)

plot_valence_band_structure(ex_sp_1_el_str, 0.15, vb_energy)
plot_valence_band_structure(ex_sp_2_el_str, 0.25, vb_energy)

plot_conduction_band_structure(gnd_sp_1_el_str, -0.05, cb_energy)
plot_conduction_band_structure(gnd_sp_2_el_str, 0.05, cb_energy)

plot_conduction_band_structure(ex_sp_1_el_str, 0.15, cb_energy)
plot_conduction_band_structure(ex_sp_2_el_str, 0.25, cb_energy)

#Plot the VB and CB of the ground and excited states

ax1.axhspan(ylim_start,vb_energy, color='yellow', alpha=0.3, zorder=0)
ax1.axhspan(cb_energy, ylim_fin, color='yellow', alpha=0.3, zorder=0)

ax1.text(-0.005, 16.1, 'ground\n state', fontsize=20, color='blue')
ax1.text(0.195, 16.1, 'excited\n state', fontsize=20, color='red')

ax1.text(-0.095,ylim_start + (vb_energy-ylim_start)/2, 'VB', fontsize=20, color='blue')
ax1.text(-0.095, cb_energy + (ylim_fin-cb_energy)/2, 'CB', fontsize=20, color='blue')

ax1.text(0.15, 11.0, ' Jahn-Teller\nactive hole', fontsize=20, color='black')


ax1.set_ylim(ylim_start,ylim_fin)
ax1.set_xlim(-0.1, 0.35)

ax1.plot([0.125, 0.125], [ylim_start, ylim_fin], 'k-', linewidth=1.0, zorder=10)

ax1.set_ylabel('energy (eV)', fontsize=20)
ax1.tick_params(axis='both', which='major', labelsize=20)
ax1.set_xticks([])


#ax1.tight_layout()
plt.tight_layout()



#right side 

ax2.plot([2.25, 4.0],[7.0, 7.0],'k-') #sima
ax2.plot([4.0,4.25],[7.0,7.2],'k--') #szaggatott
ax2.plot([4.25, 5.025],[7.2,7.2],'k-') #sima
ax2.plot([4.0,4.25],[7.0,6.8],'k--') #szaggatott
ax2.plot( [  4.25, 5.025 ],[  6.8, 6.8 ], 'k-' ) #sima

ax2.plot([2.25, 5.025],[2.5, 2.5],'k-') #sima

ax2.set_xlim(1.5, 8)
ax2.set_ylim(1.0, 8.4)
ax2.set_xticks([])
ax2.set_yticks([])

ax2.text(1.85, 6.9, r'$^2\text{E}$', fontsize=20, color='black')

ax2.text(1.65, 6.4, r'$(e^3a^2)$')


ax2.text(1.76, 2.4, r'$^2\text{A}_1$', fontsize=20, color='black')

ax2.text(1.65, 1.9, r'$(e^4a^1)$')


updownarrow(ax2, 2.625, 2.5, 2.625,7.0, head_lenght = 0.2, head_width = 0.1)
updownarrow(ax2, 4.825, 6.8, 4.825, 7.2)

ax2.text(5.025, 6.85, r'$\lambda_{exp} = 0.59 \text{ meV}$')
ax2.text(5.65, 6.35, r'$\lambda_{theory} = p\lambda_{DFT}$')

#ax2.text(5.025, 6.35, r'$\lambda_{theory} = p\lambda_{DFT}$')
ax2.text(5.65, 5.85, r'$= 0.61 \text{ meV}$')


ax2.text(2.75, 4.75, r'$\text{ZPL} : 2.985 \text{ eV}' +'\t' +r'(415 \text{ nm})$') 

ax1.text(0.295 , 10.06, r'$e$')
ax1.text(0.08 , 12.30  , r'$a_1$')


plt.show()

print('fin')

