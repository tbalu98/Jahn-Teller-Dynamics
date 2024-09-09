import utilities.matrix_formalism as mf
import numpy as np
import matplotlib.pyplot as plt
import utilities.matrix_formalism as mf
#import old_ones.OUTCAR_parsing as parsing
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)
import utilities.user_workflow as uw
import pandas as pd
import sys
import utilities.JT_config_file_parsing as  JT_cfg
import utilities.maths as maths

def meV_to_GHz(e):
    return e*241.798935



#Read config file data

#ZPL_config_file_name = 'ZPL_config.cfg'

ZPL_config_file_name = sys.argv[1]

ZPL_cfg_parser = JT_cfg.ZPL_config_parser(ZPL_config_file_name)

config_file_name_gnd = ZPL_cfg_parser.get_gnd_cfg_filename()
config_file_name_ex = ZPL_cfg_parser.get_ex_cfg_filename()

cfg_file_data_folder = ZPL_cfg_parser.get_cfg_data_folder()
results_folder = ZPL_cfg_parser.get_results_folder()

#config_file_name_gnd = 'PbV_JT_csv_config.cfg'

B_min = ZPL_cfg_parser.get_B_min()
B_max = ZPL_cfg_parser.get_B_max()
step_num = ZPL_cfg_parser.get_step_num()

Bs = np.linspace(B_min, B_max, step_num)

Bs_string = ''

for B in Bs:
    Bs_string +=str(round(B,4)) + ' T' + ', '

print('Magnetic fields:')

print(Bs_string)

print('Ground state:')

JT_int_1 =  uw.spin_orbit_JT_procedure( cfg_file_data_folder + config_file_name_gnd)




calculation_name = ZPL_cfg_parser.get_calculation_name()




JT_int_1_Es = [[],[],[],[]]


M = [[0.7071 , -0.7071 , 0],
      [0.4082 ,  0.4082,  -0.8165],
      [0.5774,   0.5774,   0.5774]]

M = np.matrix(M)

M = maths.Matrix(M)




for B in Bs:

    B_comp_vec = maths.col_vector(np.transpose(np.matrix( [0,0,B] )))
    
    B = float(B)

    B_field = M*B_comp_vec

    """
    Bx = 0.5*27.992*float(M[0,:]*B_comp_vec)*4.13567/1000
    By = 0.5*27.992*float(M[1,:]*B_comp_vec)*4.13567/1000
    Bz = 0.5*27.992*float(M[2,:]*B_comp_vec)*4.13567/1000
    """
    """
    Bx = float(np.matmul(M[0,:],B_comp_vec).tolist()[0][0])
    By = float((np.matmul(M[1,:],B_comp_vec).tolist()[0][0]))
    Bz = float((M[2,:]*B_comp_vec).tolist()[0][0])
    """


    JT_int_1.create_one_mode_DJT_hamiltonian()
    JT_int_1.add_spin_orbit_coupling()
    JT_int_1.add_magnetic_field(*B_field.tolist())
    #JT_int_1.add_magnetic_field(Bx, By, Bz)
    


    JT_int_1.H_int.calc_eigen_vals_vects()
    for i in range(0, 4):
        ket_i:mf.ket_vector =  JT_int_1.H_int.eigen_kets[i]
        JT_int_1_Es[i].append( meV_to_GHz( ket_i.eigen_val))



#config_file_name_ex = 'PbV_ex_JT_csv_config.cfg'
print('Excited state')
JT_int_2 =  uw.spin_orbit_JT_procedure( cfg_file_data_folder + config_file_name_ex)





JT_int_2_Es = [[],[],[],[]]



for B in Bs:
    B = float(B)


    #B_comp_vec = np.transpose(np.matrix( [0,0,B] ))
    B_comp_vec = maths.col_vector(np.transpose(np.matrix( [0,0,B] )))
    
    B = float(B)

    B_field = M*B_comp_vec



    B = float(B)
    
    """
    Bx = 0.5*27.992*float(M[0,:]*B_comp_vec)*4.13567/1000
    By = 0.5*27.992*float(M[1,:]*B_comp_vec)*4.13567/1000
    Bz = 0.5*27.992*float(M[2,:]*B_comp_vec)*4.13567/1000
    """
    """
    Bx = float(M[0,:]*B_comp_vec)
    By = float(M[1,:]*B_comp_vec)
    Bz = float(M[2,:]*B_comp_vec)
    """

    JT_int_2.create_one_mode_DJT_hamiltonian()
    JT_int_2.add_spin_orbit_coupling()  
    JT_int_2.add_magnetic_field(*B_field.tolist())

    JT_int_2.H_int.calc_eigen_vals_vects()
    for i in range(0, 4):
        ket_i:mf.ket_vector =  JT_int_2.H_int.eigen_kets[i]
        JT_int_2_Es[i].append( meV_to_GHz( ket_i.eigen_val))




res_data = np.zeros((16, len(Bs)),dtype = np.float64)

for i in range(0, len(Bs)):

    res_col = np.array([ JT_int_2_Es[j][i]-JT_int_1_Es[k][i] for k in range(0,4) for j in range(0,4)  ])

    res_data[:,i] = sorted(res_col)



energy_shift = abs(res_data[0][0]-res_data[15][0])/2

zeroline = res_data[15][0]-energy_shift




plt.rcParams['font.size'] = 20
fig, axeses = plt.subplots(4, 1, sharex=True)

fig.suptitle(calculation_name)

res_dict = {'split_0': [], 'split_1': [], 'split_2': [], 'split_3':[]}



for i in range(0,16):
    split_num = i//4
    
    res_row = [ j+abs(zeroline) for j in  res_data[i] ]
    

    res_dict['split_'+str(split_num) ].append(res_row)


split_names = ['split_0', 'split_1', 'split_2', 'split_3']


for split_name in split_names:
    split_res_dict = {}
    split_res_dict['magnetic_field'] = Bs
    split_res_dict['deltaE_1'] = res_dict[split_name][:][0]
    split_res_dict['deltaE_2'] = res_dict[split_name][:][1]
    split_res_dict['deltaE_3'] = res_dict[split_name][:][2]
    split_res_dict['deltaE_4'] = res_dict[split_name][:][3]

    split_res_df = pd.DataFrame(split_res_dict)
    split_res_df = split_res_df.set_index('magnetic_field')
    split_res_df.to_csv(results_folder+ split_name+'.csv')


k = 0
for key in res_dict:

    for i in range(0, 4):
        axeses[3-k].plot(Bs,res_dict[key][i], '-k')
    k+=1



axeses[0].tick_params(labeltop=False,bottom = False)  # don't put tick labels at the top

axeses[1].tick_params(labeltop=False, bottom = False)  # don't put tick labels at the top

axeses[2].tick_params(labeltop=False, bottom = False)  # don't put tick labels at the top


axeses[3].tick_params(labeltop = False,bottom = False)  # don't put tick labels at the top

axeses[3].xaxis.tick_bottom()


axeses[0].tick_params(axis='y', labelsize=10)
axeses[1].tick_params(axis='y', labelsize=10)
axeses[2].tick_params(axis='y', labelsize=10)
axeses[3].tick_params(axis='y', labelsize=10)


axeses[2].annotate('ZPL shift (GHz)', (-0.075, 0.45), xycoords='axes fraction', rotation=90)
axeses[3].set_xlabel('magnetic field (T)')
plt.show()


