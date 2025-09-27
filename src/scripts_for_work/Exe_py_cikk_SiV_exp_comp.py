import matplotlib.pyplot as plt

import pandas as pd

exp_data_folder = 'study_materials/articles/'

theory_data_folder  = 'pressure_dependence_publication_2024/SnV_0GPa_results/'
theory_data_folder  = 'results/SiV_results/'

calculation_name = 'SiV'

theory_data_D_filename = calculation_name+'_ZPL_shift_D_transitions.csv'
theory_data_C_filename = calculation_name+'_ZPL_shift_C_transitions.csv'
theory_data_B_filename = calculation_name+'_ZPL_shift_B_transitions.csv'
theory_data_A_filename = calculation_name+'_ZPL_shift_A_transitions.csv'


exp_data_filename = 'SiV_0GPa_experimental_results.csv'
exp_data =  pd.read_csv(exp_data_folder+exp_data_filename)

exp_data_offset = exp_data['transition_energy'][0]

theory_data_D =  pd.read_csv(theory_data_folder+theory_data_D_filename)
theory_data_C =  pd.read_csv(theory_data_folder+theory_data_C_filename)
theory_data_B =  pd.read_csv(theory_data_folder+theory_data_B_filename)
theory_data_A =  pd.read_csv(theory_data_folder+theory_data_A_filename)


D_tr_0 = -154.4612
C_tr_0 = -105.2275
B_tr_0 = 105.2331
A_tr_0 = 154.4674

"""
D_tr_0 = -136220.716020826

C_tr_0 = -136030.197285205

B_tr_0 = -134656.320831033
A_tr_0 = -134465.802095413
"""


D_tr_offset = D_tr_0-theory_data_D['line_0'][0]
C_tr_offset = C_tr_0-theory_data_C['line_0'][0]
B_tr_offset = B_tr_0-theory_data_B['line_0'][0]
A_tr_offset = A_tr_0-theory_data_A['line_0'][0]

offset_1 = float(theory_data_D['line_0'][0])
offset_1 = 0.0


zeroline=  float(theory_data_D['line_0'][0])
#zeroline = 12691.423680470529
print('zeroline ' + str(zeroline))

plt.figure(figsize=(10,6))
plt.rcParams['font.family'] = 'sans'
plt.rcParams['font.size'] = 21
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['lines.linewidth'] = 2.3

#plt.ylim(-300, 300)

plt.title('SiV ZPL shift')

line_names  =['line_0', 'line_1', 'line_2', 'line_3']
for line_name in line_names:
    plt.plot(theory_data_D['magnetic field (T)'], theory_data_D[line_name]+D_tr_offset, 'b-')
    plt.plot(theory_data_C['magnetic field (T)'], theory_data_C[line_name]+C_tr_offset, 'b-')
    plt.plot(theory_data_B['magnetic field (T)'], theory_data_B[line_name]+B_tr_offset, 'b-')
    
    labell = 'Exe.py' if line_name == 'line_0' else None
    
    
    plt.plot(theory_data_A['magnetic field (T)'], theory_data_A[line_name]+A_tr_offset, 'b-', label = labell)

plt.plot( exp_data['magnetic_field'], exp_data['transition_energy'], 'ro', label = 'experiment' )

plt.xlim(0,10.0)

plt.xlabel('magnetic field (T)')
plt.ylabel('ZPL shift (GHz)')
plt.rcParams['font.size'] = 15
plt.legend()


fig_res_folder = 'pressure_dependence_publication_2024/figs/'
fig_filename = fig_res_folder + calculation_name+'_ZPL_shift' + '.svg'
plt.savefig(fig_filename, bbox_inches='tight', dpi=700)
plt.show()
