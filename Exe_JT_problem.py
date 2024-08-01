import utilities.jahn_teller_theory as jt
import utilities.maths as  maths
import utilities.matrix_formalism as mf
import numpy as np
import matplotlib.pyplot as plt
import utilities.quantum_physics as qmp
import utilities.matrix_formalism as mf
import utilities.braket_formalism as bf
import utilities.quantum_system as qs
import utilities.xml_parser
import sys
import utilities.JT_config_file_parsing as cfg_parser
from scipy.constants import physical_constants
import pandas as pd
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)

# Create the symm electron system

orbital_system = qs.quantum_system_node.create_2D_orbital_system_node()

electron_system = qs.quantum_system_node('electron_system', children=[orbital_system])


#Spin system ops

spin_sys = qs.quantum_system_node.create_spin_system_node()

spatial_dim = 2

JT_cfg_parser = cfg_parser.Jahn_Teller_config_parser(str(sys.argv[1]))
#JT_cfg_parser = cfg_parser.Jahn_Teller_config_parser('JT_csv_config.cfg')


# Get data from config file

order  = JT_cfg_parser.get_order()
calc_name = JT_cfg_parser.get_problem_name()
l  =  JT_cfg_parser.get_spin_orbit_coupling()
E_x, E_y = JT_cfg_parser.get_electric_field()

Bx, By,Bz = JT_cfg_parser.get_magnetic_field()

res_folder = JT_cfg_parser.get_res_folder_name()
data_folder = JT_cfg_parser.get_data_folder_name()

LzSz_calc_num = JT_cfg_parser.get_LzSz_exp_val_num()

gl_factor = JT_cfg_parser.get_gL_factor()

#order =12

#JT_theory = jt.Jahn_Teller_Theory()

filenames = None

arguments = sys.argv[1:]

save_raw_pars = False


if arguments[-1] == '-save_raw_pars':
    save_raw_pars = True
    arguments.pop()

JT_theory, symm_lattice, less_symm_lattice_1, less_symm_lattice_2 = JT_cfg_parser.build_JT_theory(data_folder)



if save_raw_pars == True:
    utilities.xml_parser.save_raw_data_from_xmls([symm_lattice, less_symm_lattice_1, less_symm_lattice_2], calc_name, data_folder)


    #Calculate the parameters of Jahn-Teller theory


print('Maximum number of energy quantums of vibrations in each direction:\n n = ' + str(order) )
print('Energy of spin-orbit coupling:\n ' +'lambda = ' + str(l)  )

print(JT_theory)


#Create quantum system tree graph:

mode_1 = qmp.one_mode_phonon_sys(JT_theory.hw,spatial_dim,order,['x','y'], 'mode_1', 'mode_1' )


nuclei = qs.quantum_system_node('nuclei')#, children=[mode_1])

point_defect_node = qs.quantum_system_node('point_defect', 
                                               children = [ nuclei,electron_system])

point_defect_tree = qs.quantum_system_tree(point_defect_node)

point_defect_tree.insert_node('nuclei', mode_1)

#point_defect_tree.insert_node('electron_system', spin_sys)


JT_int = qmp.Exe_tree(point_defect_tree, JT_theory)
JT_int.gL_factor = gl_factor

JT_int.create_one_mode_DJT_hamiltonian()
    
eig_vec_file_name = calc_name + '_eigen_vectors_only_DJT.csv'
eig_val_file_name = calc_name + '_eigen_values_only_DJT.csv'

JT_int.save_eigen_vals_vects_to_file( res_folder+ eig_vec_file_name,res_folder+ eig_val_file_name)


print('-------------------------------')

print('Eigen values of the Jahn-Teller interaction')

for x in JT_int.H_int.eigen_kets[0:10]:
    print(str(round(x.eigen_val.real,4)) + ' meV') 


print('-------------------------------')


#print( [  x.eigen_val for x in JT_int.H_int.eigen_kets ] )



ground_1 = JT_int.H_int.eigen_kets[0]

ground_2 = JT_int.H_int.eigen_kets[1]


    #2D degenrate ground state system spin-orbital perturbation

deg_sys = mf.degenerate_system_2D( [ground_1,ground_2] )



Sz = spin_sys.operators['Sz']

Lz = orbital_system.operators['Lz']

electron_system.operators['LzSz'] = Lz**Sz

pert_ham_Lz = JT_int.system_tree.create_operator(operator_id = 'Lz',operator_sys='orbital_system' )

pert_ham_LzSz = JT_int.system_tree.create_operator(operator_id = 'LzSz',operator_sys='electron_system')


print('Reduction factor from first order perturbation:')

deg_sys.add_perturbation(pert_ham_Lz)

print('p = '+ str( round(deg_sys.p_red_fact,4)) + ' meV')



orbital_system.operators['Lz_normal'] = mf.MatrixOperator.pauli_z_mx_op()


Lz_point_def = point_defect_tree.create_operator(operator_id='Lz', operator_sys='orbital_system')

JT_int.lambda_factor = l



if l!=0.0:

    print('spin orbit coupling lambda = ' + str(l))
    point_defect_tree.insert_node('electron_system', spin_sys)



    JT_int.create_one_mode_DJT_hamiltonian()

    JT_int.add_spin_orbit_coupling()


    JT_int.H_int.calc_eigen_vals_vects()

    E_32 = JT_int.H_int.eigen_kets[2]
    E_12 = JT_int.H_int.eigen_kets[0]

    print("K_JT expectation value:")

    K_JT_32 =  JT_int.H_int.calc_expected_val(E_32)
    K_JT_12 =  JT_int.H_int.calc_expected_val(E_12)
    JT_int.KJT_factor = K_JT_12-K_JT_32
    print(JT_int.KJT_factor)



    LzSz_op = JT_int.system_tree.create_operator('LzSz',subsys_id='point_defect', operator_sys='electron_system')

    


    p_32 = LzSz_op.calc_expected_val(JT_int.H_int.eigen_kets[2])
    p_12 = LzSz_op.calc_expected_val(JT_int.H_int.eigen_kets[0])


    print('p values after adding spin-orbit coupling to Hamiltonian')

    print( "p3/2 = " + str( p_32 ))
    print( "p1/2 = " + str( p_12 ))

    p = p_32+p_12
    delta = p_32-p_12

    JT_int.p_factor = p

    JT_int.f_factor = JT_int.gL_factor*JT_int.p_factor
    JT_int.delta_factor =delta

    print('C8 and C9 equation:'  )
    print("p = " + str( p ))
    print( "delta = " + str(delta) )
    
    if LzSz_calc_num>0:
        state_names = [ 'eigenstate_' + str(i) for i in range(0, LzSz_calc_num) ]
        eigen_energies = [ x.eigen_val for x in JT_int.H_int.eigen_kets[0:LzSz_calc_num]]

        LzSz_expected_vals= [ LzSz_op.calc_expected_val( eig_ket) for eig_ket in JT_int.H_int.eigen_kets[0:LzSz_calc_num] ]
        
        LzSz_res = { 'state_name': state_names , 'eigenenergy': eigen_energies, 'LzSz': LzSz_expected_vals }

        LzSz_res_df = pd.DataFrame(LzSz_res)
        LzSz_res_df = LzSz_res_df.set_index('state_name')

        LzSz_res_df.to_csv(res_folder+calc_name+'_LzSz_expected_values.csv')


        plt.rcParams['font.size'] = 18
        plt.title('PbV energy and spin-orbit coupling \n expectation value')
        plt.xlabel(r'$\left< L_{z} \otimes S_{z} \right>$')
        plt.ylabel(r'Energy (meV)')
        plt.plot( LzSz_expected_vals, eigen_energies , 'x')

        plt.plot([0.0,0.0], [eigen_energies[0], eigen_energies[-1]], '--'  )
        
        plt.grid()
        plt.show()
    
    print('-------------------------------')

    print('p Ham reduction factor')

    print( 'lambda_Ham = ' + str(round((JT_int.H_int.eigen_kets[2].eigen_val- JT_int.H_int.eigen_kets[0].eigen_val).real,4)) + ' meV')


    print('-------------------------------')



if l!=0.0 and(Bz!=0.0 or Bx!= 0.0 or By != 0.0):


    JT_int.add_magnetic_field(Bx,By,Bz)

    JT_int.H_int.calc_eigen_vals_vects()

    print('p Ham reduction factor')

    print( 'lambda_0 = ' + str(round((JT_int.H_int.eigen_kets[2].eigen_val- JT_int.H_int.eigen_kets[0].eigen_val).real,4)) + ' meV')




if E_x !=0.0 or E_y != 0.0:

    #JT_int.create_one_mode_hamiltonian()

    JT_int.add_electric_field(E_x, E_y)
    JT_int.H_int.calc_eigen_vals_vects()
    print('-------------------------------')

    print('q Ham reduction factor')

    if JT_int.system_tree.find_subsystem('spin_system') == None:
        print( 'lambda_0 = ' + str(round((JT_int.H_int.eigen_kets[1].eigen_val- JT_int.H_int.eigen_kets[0].eigen_val).real,4)) + ' meV')
    else:
        print( 'lambda_0 = ' + str(round((JT_int.H_int.eigen_kets[3].eigen_val- JT_int.H_int.eigen_kets[0].eigen_val).real,4)) + ' meV')


eig_vec_file_name = calc_name + '_eigen_vectors_interaction.csv'
eig_val_file_name = calc_name + '_eigen_values_interaction.csv'

JT_int.save_eigen_vals_vects_to_file(res_folder + eig_vec_file_name,res_folder+eig_val_file_name)




