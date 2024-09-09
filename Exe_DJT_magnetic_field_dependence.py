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

mag_field_label = 'magnetic field (T)'

eig_state_prefix = 'eigen_state_'

def meV_to_GHz(e):
    return e*241.798935



#Read config file data

#ZPL_config_file_name = 'ZPL_config.cfg'

ZPL_config_file_name = sys.argv[1]

ZPL_cfg_parser = JT_cfg.ZPL_config_parser(ZPL_config_file_name)

config_file_name_gnd = ZPL_cfg_parser.get_gnd_cfg_filename()

cfg_file_data_folder = ZPL_cfg_parser.get_cfg_data_folder()
results_folder = ZPL_cfg_parser.get_results_folder()

#config_file_name_gnd = 'PbV_JT_csv_config.cfg'

B_min = ZPL_cfg_parser.get_B_min()
B_max = ZPL_cfg_parser.get_B_max()
step_num = ZPL_cfg_parser.get_step_num()

Bs = np.round(np.linspace(B_min, B_max, step_num),4)

Bs_string = ''

for B in Bs:
    Bs_string +=str(round(B,4)) + ' T' + ', '

print('Magnetic fields:')

print(Bs_string)

print('Ground state:')

JT_int_1 =  uw.spin_orbit_JT_procedure( cfg_file_data_folder + config_file_name_gnd)




calculation_name = ZPL_cfg_parser.get_calculation_name()



num_of_eigen_vals =  JT_int_1.system_tree.root_node.dim



mag_field_res_dict = {  eig_state_prefix+str(i):[] for i in range( 1, num_of_eigen_vals+1 ) }
mag_field_res_dict[mag_field_label] = Bs


M = [[0.7071 , -0.7071 , 0],
      [0.4082 ,  0.4082,  -0.8165],
      [0.5774,   0.5774,   0.5774]]

M = np.matrix(M)

M = maths.Matrix(M)



for B in Bs:

    B_comp_vec = maths.col_vector(np.transpose(np.matrix( [0,0,B] )))
    
    B = float(B)

    B_field = M*B_comp_vec



    JT_int_1.create_one_mode_DJT_hamiltonian()
    JT_int_1.add_spin_orbit_coupling()
    JT_int_1.add_magnetic_field(*B_field.tolist())
    #JT_int_1.add_magnetic_field(Bx, By, Bz)
    


    JT_int_1.H_int.calc_eigen_vals_vects()


    for i in range(0, num_of_eigen_vals):
        ket_i:mf.ket_vector =  JT_int_1.H_int.eigen_kets[i]
        eig_val = round(ket_i.eigen_val,4)
        mag_field_res_dict[ eig_state_prefix + str(i+1) ].append(eig_val)


plt.rcParams['font.size'] = 20

for i in range(0, 2):

    plt.plot(Bs, mag_field_res_dict[eig_state_prefix+str(i+1)], 'x')


res_df = pd.DataFrame(mag_field_res_dict).set_index('magnetic field (T)')
res_df.to_csv(results_folder + calculation_name + '_magnetic_field_dependence.csv')

plt.xlabel('Magnetic field (T)')
plt.ylabel('energy (meV)')
plt.title( 'Splitting of the lowest eigen energy' )
plt.show()



