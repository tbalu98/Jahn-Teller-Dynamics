import utilities.jahn_teller_theory as jt
import utilities.maths as  maths
import utilities.matrix_formalism as mf
import numpy as np
import matplotlib.pyplot as plt
import utilities.quantum_physics as qmp
import utilities.matrix_formalism as mf
import utilities.braket_formalism as bf
import utilities.quantum_system as qs
#import old_ones.OUTCAR_parsing as parsing
import utilities.xml_parser
import sys
import utilities.JT_config_file_parsing as cfg_parser
from scipy.constants import physical_constants
from copy import deepcopy
import warnings
warnings.simplefilter("ignore", np.ComplexWarning)



def create_spin_orbit_coupled_DJT_int_from_file(config_file_name:str, save_raw_pars = False):

    spatial_dim = 2

# Create the symm electron system

    orbital_system = qs.quantum_system_node.create_2D_orbital_system_node()

    electron_system = qs.quantum_system_node('electron_system', children=[orbital_system])

    spin_sys = qs.quantum_system_node.create_spin_system_node()





#Extract data from config file

    JT_cfg_parser = cfg_parser.Jahn_Teller_config_parser(config_file_name)
    order  = JT_cfg_parser.get_order()
    calc_name = JT_cfg_parser.get_problem_name()
    l  =  JT_cfg_parser.get_spin_orbit_coupling()
    gL = JT_cfg_parser.get_gL_factor()

    data_folder = JT_cfg_parser.get_data_folder_name()





    #Create dynamic JT theory
    JT_theory, symm_lattice, less_symm_lattice_1, less_symm_lattice_2 = JT_cfg_parser.build_JT_theory(data_folder)



    if save_raw_pars == True:
        utilities.xml_parser.save_raw_data_from_xmls([symm_lattice, less_symm_lattice_1, less_symm_lattice_2], calc_name)


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

    point_defect_tree.insert_node('electron_system', spin_sys)


    JT_int = qmp.Exe_tree(point_defect_tree, JT_theory)
    JT_int.gL_factor = gL
    JT_int.lambda_factor = l

    JT_int.create_one_mode_DJT_hamiltonian()
    

    JT_int.add_spin_orbit_coupling()


    JT_int.H_int.calc_eigen_vals_vects()

    E_32 = JT_int.H_int.eigen_kets[2]
    E_12 = JT_int.H_int.eigen_kets[0]


    LzSz_op = JT_int.system_tree.create_operator('LzSz',subsys_id='point_defect', operator_sys='electron_system')

    LzSz_expected_vals= [ LzSz_op.calc_expected_val( eig_ket) for eig_ket in JT_int.H_int.eigen_kets[0:50] ]
    


    p_32 = LzSz_expected_vals[2]
    p_12 = LzSz_expected_vals[0]

    print('p values after adding SOC to Hamiltonian')

    print( "p3/2 = " + str( p_32 ))
    print( "p1/2 = " + str( p_12 ))

    JT_int.p_factor = p_32-p_12
    JT_int.delta_factor = p_32+p_12

    print('Based on C8 and C9 equation:'  )
    print("p = " + str( JT_int.p_factor ))
    print( "delta = " + str(JT_int.delta_factor) )

    print('-------------------------------')

    print('p Ham reduction factor')

    print( 'lambda_Ham = ' + str(round((JT_int.H_int.eigen_kets[2].eigen_val- JT_int.H_int.eigen_kets[0].eigen_val).real,4)) + ' meV')


    # K_JT reduction factor

    print("K_JT expectation value:")

    K_JT_32 =  JT_int.H_int.calc_expected_val(E_32)
    K_JT_12 =  JT_int.H_int.calc_expected_val(E_12)
    JT_int.KJT_factor = K_JT_12-K_JT_32
    print(JT_int.KJT_factor)

    JT_int.f_factor = JT_int.gL_factor*JT_int.p_factor

    return JT_int





def spin_orbit_JT_procedure(config_file_name:str, save_raw_pars = False):
    
    spatial_dim = 2

# Create the symm electron system

    orbital_system = qs.quantum_system_node.create_2D_orbital_system_node()

    electron_system = qs.quantum_system_node('electron_system', children=[orbital_system])

    spin_sys = qs.quantum_system_node.create_spin_system_node()





#Extract data from config file

    JT_cfg_parser = cfg_parser.Jahn_Teller_config_parser(config_file_name)
    order  = JT_cfg_parser.get_order()
    calc_name = JT_cfg_parser.get_problem_name()
    l  =  JT_cfg_parser.get_spin_orbit_coupling()
    gL = JT_cfg_parser.get_gL_factor()

    data_folder = JT_cfg_parser.get_data_folder_name()






    JT_theory, symm_lattice, less_symm_lattice_1, less_symm_lattice_2 = JT_cfg_parser.build_JT_theory(data_folder)



    if save_raw_pars == True:
        utilities.xml_parser.save_raw_data_from_xmls([symm_lattice, less_symm_lattice_1, less_symm_lattice_2], calc_name)


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

    point_defect_tree.insert_node('electron_system', spin_sys)


    JT_int = qmp.Exe_tree(point_defect_tree, JT_theory)
    JT_int.gL_factor = gL
    JT_int.lambda_factor = l

    JT_int.create_one_mode_DJT_hamiltonian()
    
#JT_int.save_eigen_vals_vects_to_file( res_folder+ eig_vec_file_name,res_folder+ eig_val_file_name)

    JT_int.H_int.calc_eigen_vals_vects()

    print('-------------------------------')

    print('Eigen values of the Jahn-Teller interaction')

    for x in JT_int.H_int.eigen_kets[0:10]:
        print(str(round(x.eigen_val.real,4)) + ' meV') 



#print( [  x.eigen_val for x in JT_int.H_int.eigen_kets ] )






    print('spin orbit coupling lambda = ' + str(JT_int.lambda_factor))


    JT_int.create_one_mode_DJT_hamiltonian()

    JT_int.add_spin_orbit_coupling()


    JT_int.H_int.calc_eigen_vals_vects()

    E_32 = JT_int.H_int.eigen_kets[2]
    E_12 = JT_int.H_int.eigen_kets[0]


    LzSz_op = JT_int.system_tree.create_operator('LzSz',subsys_id='point_defect', operator_sys='electron_system')

    LzSz_expected_vals= [ LzSz_op.calc_expected_val( eig_ket) for eig_ket in JT_int.H_int.eigen_kets[0:50] ]
    

    """
    p_32 = 2*LzSz_expected_vals[2]
    p_12 = -2*LzSz_expected_vals[0]
    """
    p_32 = 2*LzSz_op.calc_expected_val(JT_int.H_int.eigen_kets[2])
    p_12 = -2*LzSz_op.calc_expected_val(JT_int.H_int.eigen_kets[0])

    print('p values after adding SOC to Hamiltonian')

    print( "p3/2 = " + str(p_32 ))
    print( "p1/2 = " + str(p_12 ))

    JT_int.p_factor = (p_32+p_12)/2
    JT_int.delta_factor = (p_32-p_12)/2

    print('Based on C8 and C9 equation:'  )
    print("p = " + str( JT_int.p_factor ))
    print( "delta = " + str(JT_int.delta_factor) )

    print('-------------------------------')

    print('p Ham reduction factor')

    print( 'lambda_Ham = ' + str(round((JT_int.H_int.eigen_kets[2].eigen_val- JT_int.H_int.eigen_kets[0].eigen_val).real,4)) + ' meV')


    # K_JT reduction factor

    print("K_JT expectation value:")

    K_JT_32 =  JT_int.H_int.calc_expected_val(E_32)
    K_JT_12 =  JT_int.H_int.calc_expected_val(E_12)
    JT_int.KJT_factor = K_JT_12-K_JT_32
    print(JT_int.KJT_factor)

    JT_int.f_factor = JT_int.gL_factor*JT_int.p_factor

    return JT_int


