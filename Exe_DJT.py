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
import utilities.user_workflow as uw
import sys
import utilities.JT_config_file_parsing as cfg_parser
from scipy.constants import physical_constants
import pandas as pd
import warnings

eig_state_prefix = 'eigen_state_'
mag_field_label = 'magnetic field (T)'
warnings.simplefilter("ignore", np.ComplexWarning)

# Create the symm electron system

orbital_system = qs.quantum_system_node.create_2D_orbital_system_node()



#Spin system ops

spin_sys = qs.quantum_system_node.create_spin_system_node()

#electron_system = qs.quantum_system_node('electron_system', children=[orbital_system, spin_sys])

electron_system = qs.quantum_system_node('electron_system', children=[orbital_system])


spatial_dim = 2

JT_cfg_parser = cfg_parser.Jahn_Teller_config_parser(str(sys.argv[1]))
#JT_cfg_parser = cfg_parser.Jahn_Teller_config_parser('JT_csv_config.cfg')


# Get data from config file


JT_cfg_parser.read_mag_field()

JT_cfg_parser.get_mag_field_dir()

order  = JT_cfg_parser.get_order()
calc_name = JT_cfg_parser.get_problem_name()
intrinsic_so_coupling  =  JT_cfg_parser.get_spin_orbit_coupling()
E_x, E_y = JT_cfg_parser.get_electric_field()

Bx, By,Bz = JT_cfg_parser.get_magnetic_field()

res_folder = JT_cfg_parser.get_res_folder_name()
data_folder = JT_cfg_parser.get_data_folder_name()

LzSz_calc_num = JT_cfg_parser.get_LzSz_exp_val_num()

gl_factor = JT_cfg_parser.get_gL_factor()




filenames = None

arguments = sys.argv[1:]

save_raw_pars = False


if arguments[-1] == '-save_raw_pars':
    save_raw_pars = True
    arguments.pop()


JT_theory, symm_lattice, less_symm_lattice_1, less_symm_lattice_2 = JT_cfg_parser.build_JT_theory(data_folder)



if save_raw_pars == True:
    utilities.xml_parser.save_raw_data_from_xmls([symm_lattice, less_symm_lattice_1, less_symm_lattice_2], calc_name, data_folder,JT_cfg_parser.config)


    #Calculate the parameters of Jahn-Teller theory



#print('Energy of spin-orbit coupling:\n ' +'lambda = ' + str(intrinsic_so_coupling) + str(' meV')  )



#Create quantum system tree graph:

mode_1 = qmp.one_mode_phonon_sys(JT_theory.hw_meV,spatial_dim,order,['x','y'], 'mode_1', 'mode_1' )


nuclei = qs.quantum_system_node('nuclei')#, children=[mode_1])

point_defect_node = qs.quantum_system_node('point_defect', 
                                               children = [ nuclei,electron_system])

point_defect_tree = qs.quantum_system_tree(point_defect_node)

point_defect_tree.insert_node('nuclei', mode_1)

#point_defect_tree.insert_node('electron_system', spin_sys)


JT_int = qmp.Exe_tree(point_defect_tree, JT_theory)
JT_int.gL_factor = gl_factor

orbital_system.operators['Lz_normal'] = mf.MatrixOperator.pauli_z_mx_op()


Lz_point_def = point_defect_tree.create_operator(operator_id='Lz', operator_sys='orbital_system')

JT_int.lambda_factor = intrinsic_so_coupling


print(JT_int.get_essential_input_string())

print('-------------------------------------------------')
print('Maximum number of energy quanta of vibrations in each direction = ' + str(order) )

print('-------------------------------------------------')
print(JT_theory)


if intrinsic_so_coupling==0.0:


    JT_int.create_one_mode_DJT_hamiltonian()

    eig_vec_file_name = calc_name + '_eigen_vectors_only_DJT.csv'
    eig_val_file_name = calc_name + '_eigen_values_only_DJT.csv'

    JT_int.save_eigen_vals_vects_to_file( res_folder+ eig_vec_file_name,res_folder+ eig_val_file_name)



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

    Ham_red_fact = deg_sys.p_red_fact

    print('p = '+ str( round(Ham_red_fact,4)) )

    JT_int.p_factor = Ham_red_fact

    
    



if intrinsic_so_coupling!=0.0:


    point_defect_tree.insert_node('electron_system', spin_sys)



    JT_int.create_one_mode_DJT_hamiltonian()

    JT_int.add_spin_orbit_coupling()


    JT_int.H_int.calc_eigen_vals_vects()

    #print("K_JT expectation value:")

    JT_int.calc_K_JT_factor()
    #print(JT_int.KJT_factor)


    LzSz_op = JT_int.system_tree.create_operator('LzSz',subsys_id='point_defect', operator_sys='electron_system')

    



    JT_int.calc_reduction_factors()

    print('-------------------------------------------------')
    print(JT_int.get_essential_theoretical_results_string())
    """
    print('p values after adding spin-orbit coupling to Hamiltonian')

    print( "p3/2 = " + str( JT_int.p_32 ))
    print( "p1/2 = " + str( JT_int.p_12 ))

    print('Ham reduction factor')
    print("p = " + str( JT_int.p_factor ))
    print( "delta_p = " + str(JT_int.delta_p_factor) )
    """
    if LzSz_calc_num>0:
        state_names = [ 'eigenstate_' + str(i) for i in range(0, LzSz_calc_num) ]
        eigen_energies = [ x.eigen_val for x in JT_int.H_int.eigen_kets[0:LzSz_calc_num]]

        LzSz_expected_vals= [ LzSz_op.calc_expected_val( eig_ket) for eig_ket in JT_int.H_int.eigen_kets[0:LzSz_calc_num] ]
        
        LzSz_res = { 'state_name': state_names , 'eigenenergy': eigen_energies, 'LzSz': LzSz_expected_vals }

        LzSz_res_df = pd.DataFrame(LzSz_res)
        LzSz_res_df = LzSz_res_df.set_index('state_name')

        LzSz_res_df.to_csv(res_folder+calc_name+'_LzSz_expected_values.csv')


        plt.rcParams['font.size'] = 18
        plt.title('Spin-orbit coupling \n expectation value')
        plt.xlabel(r'$\left< L_{z} \otimes S_{z} \right>$')
        plt.ylabel(r'Energy (meV)')
        plt.plot( LzSz_expected_vals, eigen_energies , 'x')

        plt.plot([0.0,0.0], [eigen_energies[0], eigen_energies[-1]], '--'  )
        
        plt.grid()
        plt.show()
    
    essential_data_res = JT_int.get_essential_theoretical_results()
    essential_data_res['calculation name'] = [calc_name]

    ess_data_df = pd.DataFrame(essential_data_res).set_index('calculation name')

    ess_data_df.to_csv(res_folder + calc_name+'_DJT_pars_and_theoretical_results.csv',sep = ';')

    """
    print('-------------------------------')



    print( 'lambda_Ham = ' + str(round((JT_int.H_int.eigen_kets[2].eigen_val- JT_int.H_int.eigen_kets[0].eigen_val).real,4)) + ' meV')


    print('-------------------------------')
    """


"""
if intrinsic_so_coupling!=0.0 and(Bz!=0.0 or Bx!= 0.0 or By != 0.0):


    JT_int.add_magnetic_field(Bx,By,Bz)

    JT_int.H_int.calc_eigen_vals_vects()

    JT_int.calc_reduction_factors()


"""

if intrinsic_so_coupling!=0.0 and JT_cfg_parser.mag_dir_vec!=None and len(JT_cfg_parser.mag_field_strengths)!=0:
    
    num_of_eigen_vals =  JT_int.system_tree.root_node.dim



    mag_field_res_dict = {  eig_state_prefix+str(i):[] for i in range( 1, num_of_eigen_vals+1 ) }
    mag_field_res_dict[mag_field_label] = JT_cfg_parser.mag_field_strengths
    
    dir_vec:maths.col_vector = JT_cfg_parser.mag_dir_vec
    
    dir_vec_in_bases = dir_vec.basis_trf(symm_lattice.basis_vecs)

    Bs= JT_cfg_parser.mag_field_strengths

    print('------------------------------------------------')
    print('Calculate magnetic field')
    print('magnetic field direction vector:')
    print(dir_vec)
    print('magnetic field strengths (T):')
    mag_field_str = ''
    for B in Bs:
        mag_field_str+=str(B) + ', '
    print( mag_field_str)

    for B in Bs:

        B_field = B*dir_vec

        JT_int.create_one_mode_DJT_hamiltonian()
        JT_int.add_spin_orbit_coupling()
        JT_int.add_magnetic_field(*B_field.tolist())
    


        JT_int.H_int.calc_eigen_vals_vects()


        for i in range(0, num_of_eigen_vals):
            ket_i:mf.ket_vector =  JT_int.H_int.eigen_kets[i]
            eig_val = round(ket_i.eigen_val,4)
            mag_field_res_dict[ eig_state_prefix + str(i+1) ].append(eig_val)


    plt.rcParams['font.size'] = 20

    for i in range(0, 2):

        plt.plot(Bs, mag_field_res_dict[eig_state_prefix+str(i+1)], 'x')


    res_df = pd.DataFrame(mag_field_res_dict).set_index('magnetic field (T)')
    res_df.to_csv(res_folder + calc_name + '_magnetic_field_dependence.csv')

    plt.xlabel('Magnetic field (T)')
    plt.ylabel('energy (meV)')
    plt.title( 'Splitting of the lowest eigen energy' )
    plt.show()



eig_vec_file_name = calc_name + '_eigen_vectors.csv'
eig_val_file_name = calc_name + '_eigen_values.csv'

JT_int.save_eigen_vals_vects_to_file(res_folder + eig_vec_file_name,res_folder+eig_val_file_name)


#Save essential data




input_data_res = JT_int.get_essential_input()


input_data_res['calculation name'] = [calc_name]


input_data_df = pd.DataFrame(input_data_res).set_index('calculation name')

input_data_df.to_csv(res_folder + calc_name +'_essential_input.csv')


