import matplotlib.pyplot as plt

import pandas as pd


A_tr_start = -18324.4485
D_tr_start = -17132.176

diff = A_tr_start - D_tr_start

center_shift = A_tr_start-diff/2

exp_data_folder = 'study_materials/articles/'

theory_data_folder  = 'pressure_dependence_publication_2024/SnV_0GPa_results/'
calculation_name = 'SnV_0GPa'

C_tr_theory_data_1_filename = calculation_name+'_ZPL_shift_C_transitions.csv'
D_tr_theory_data_1_filename = calculation_name+'_ZPL_shift_D_transitions.csv'
CD_th_fns = [C_tr_theory_data_1_filename, D_tr_theory_data_1_filename]

C_tr_exp_data_filename = 'C_transition_energies.csv'
D_tr_exp_data_filename = 'D_transition_energies.csv'
exp_fns = [C_tr_exp_data_filename, D_tr_exp_data_filename]


plt.rcParams['font.family'] = 'sans'
plt.rcParams['font.size'] = 21
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['lines.linewidth'] = 2.3



fig, axeses = plt.subplots(4, 1, sharex=True)
#fig.suptitle('SnV ZPL shift')
fig.set_figheight(6)
fig.set_figwidth(10)

AB_th_fns = [calculation_name+'_ZPL_shift_A_transitions.csv', calculation_name+'_ZPL_shift_B_transitions.csv']


i = 0
for th_fn in AB_th_fns:
    tr_theory_data =  pd.read_csv(theory_data_folder+th_fn)
    tr_th_offset = float(tr_theory_data['line_0'][0])
    
    print(tr_th_offset)

    axeses[i].plot(tr_theory_data['magnetic field (T)'], tr_theory_data['line_0']-center_shift,'b-')
    axeses[i].plot(tr_theory_data['magnetic field (T)'], tr_theory_data['line_1']-center_shift,'b-')
    axeses[i].plot(tr_theory_data['magnetic field (T)'], tr_theory_data['line_2']-center_shift,'b-')
    axeses[i].plot(tr_theory_data['magnetic field (T)'], tr_theory_data['line_3']-center_shift,'b-')
    i+=1

for exp_fn, th_fn in zip(exp_fns, CD_th_fns):

    tr_exp_data =  pd.read_csv(exp_data_folder+exp_fn)
    tr_theory_data =  pd.read_csv(theory_data_folder+th_fn)
    tr_th_offset = float(tr_theory_data['line_0'][0])
    
    print(tr_th_offset)
    labell = 'experimental data' if i==3 else None

    axeses[i].plot(tr_theory_data['magnetic field (T)'], tr_theory_data['line_0']-center_shift, 'b-')
    axeses[i].plot(tr_theory_data['magnetic field (T)'], tr_theory_data['line_1']-center_shift,'b-')
    axeses[i].plot(tr_theory_data['magnetic field (T)'], tr_theory_data['line_2']-center_shift,'b-', label = 'Exe.py')
    axeses[i].plot(tr_theory_data['magnetic field (T)'], tr_theory_data['line_3']-center_shift,'b-')
    axeses[i].plot( tr_exp_data['magnetic_field'], tr_exp_data['deltaE']+tr_th_offset-center_shift, 'ro', label = labell )
    i+=1

#plt.xlabel('magnetic field (T)')
#plt.ylabel('transition energy (GHz)')


d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)

axeses[0].plot([0, 1], [0, 0], transform=axeses[0].transAxes, **kwargs)
axeses[1].plot([0, 1], [1, 1], transform=axeses[1].transAxes, **kwargs)

axeses[1].plot([0, 1], [0, 0], transform=axeses[1].transAxes, **kwargs)
axeses[2].plot([0, 1], [1, 1], transform=axeses[2].transAxes, **kwargs)

axeses[2].plot([0, 1], [0, 0], transform=axeses[2].transAxes, **kwargs)
axeses[3].plot([0, 1], [1, 1], transform=axeses[3].transAxes, **kwargs)
axeses[0].set_title('SnV ZPL shift')



axeses[0].spines.bottom.set_visible(False)

axeses[1].spines.top.set_visible(False)
axeses[1].spines.bottom.set_visible(False)

axeses[2].spines.top.set_visible(False)
axeses[2].spines.bottom.set_visible(False)

axeses[3].spines.top.set_visible(False)

axeses[0].tick_params(labeltop=False,bottom = False)  # don't put tick labels at the top

axeses[1].tick_params(labeltop=False, bottom = False)  # don't put tick labels at the top

axeses[2].tick_params(labeltop=False, bottom = False)  # don't put tick labels at the top


axeses[3].tick_params(labeltop = False,bottom = False)  # don't put tick labels at the top

axeses[3].xaxis.tick_bottom()

#axeses[3].set_ylim(-3000, -1700)

"""
axeses[0].tick_params(axis='y', labelsize=10)
axeses[1].tick_params(axis='y', labelsize=10)
axeses[2].tick_params(axis='y', labelsize=10)
axeses[3].tick_params(axis='y', labelsize=10)

"""

for axes in axeses:
    axes.set_xlim(0,10.0)



#axeses[2].set_ylabel('ZPL split (GHz)')

#plt.text( -1 ,-1200, 'ZPL split (GHz)', fontsize=18)
#axeses[2].text(1, -1200, 'text', transform=axeses[2].transAxes, fontsize=8, va='top', ha='left')

#axeses[2].annotate('ZPL shift (GHz)', (-0.12, 0.45), xycoords='axes fraction', rotation=90)
axeses[3].set_xlabel('magnetic field (T)')
axeses[2].set_ylabel('               ZPL shift (GHz)')
#plt.rcParams['font.size'] = 16
#axeses[3].legend()


fig_res_folder = 'pressure_dependence_publication_2024/figs/'
fn_calc_name = calculation_name+'_ZPL_shift'
fig_filename = fig_res_folder + fn_calc_name + '.svg'
plt.rcParams['font.size'] = 15
#plt.legend(loc = 'lower left')
plt.savefig(fig_filename, bbox_inches='tight', dpi=700)
plt.show()
