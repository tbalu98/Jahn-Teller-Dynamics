import utilities.JT_config_file_parsing
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
import pandas as pd
import os

gnd_sec = 'ground_state_parameters'

ex_sec = 'excited_state_parameters'
m_electron = 0.51e9
m_el_D = 0.0005485833


def create_spin_orbit_coupled_DJT_int_from_file(config_file_name:str, save_raw_pars = False):

    spatial_dim = 2


    orbital_system = qs.quantum_system_node.create_2D_orbital_system_node()

    electron_system = qs.quantum_system_node('electron_system', children=[orbital_system])

    spin_sys = qs.quantum_system_node.create_spin_system_node()



    JT_cfg_parser = cfg_parser.Jahn_Teller_config_parser(config_file_name)
    order  = JT_cfg_parser.get_order()
    calc_name = JT_cfg_parser.get_prefix_name()
    l  =  JT_cfg_parser.get_spin_orbit_coupling()
    gL = JT_cfg_parser.get_gL_factor()

    data_folder = JT_cfg_parser.get_data_folder_name()





    JT_theory, symm_lattice, less_symm_lattice_1, less_symm_lattice_2 = JT_cfg_parser.build_JT_theory(data_folder)



    if save_raw_pars == True:
        utilities.xml_parser.save_raw_data_from_xmls([symm_lattice, less_symm_lattice_1, less_symm_lattice_2], calc_name)




    print('Maximum number of energy quantums of vibrations in each direction:\n n = ' + str(order) )
    print('Energy of spin-orbit coupling:\n ' +'lambda = ' + str(l)  )

    print(JT_theory)



    mode_1 = qmp.one_mode_phonon_sys(JT_theory.hw_meV,spatial_dim,order,['x','y'], 'mode_1', 'mode_1' )


    nuclei = qs.quantum_system_node('nuclei')#, children=[mode_1])

    point_defect_node = qs.quantum_system_node('point_defect', 
                                               children = [ nuclei,electron_system])

    point_defect_tree = qs.quantum_system_tree(point_defect_node)

    point_defect_tree.insert_node('nuclei', mode_1)

    point_defect_tree.insert_node('electron_system', spin_sys)


    JT_int = qmp.Exe_tree(point_defect_tree, JT_theory)
    JT_int.orbital_red_fact = gL
    JT_int.intrinsic_soc = l

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
    JT_int.delta_p_factor = p_32+p_12

    print('Based on C8 and C9 equation:'  )
    print("p = " + str( JT_int.p_factor ))
    print( "delta = " + str(JT_int.delta_p_factor) )

    print('-------------------------------')

    print('p Ham reduction factor')

    print( 'lambda_Ham = ' + str(round((JT_int.H_int.eigen_kets[2].eigen_val- JT_int.H_int.eigen_kets[0].eigen_val).real,4)) + ' meV')



    print("K_JT expectation value:")

    K_JT_32 =  JT_int.H_int.calc_expected_val(E_32)
    K_JT_12 =  JT_int.H_int.calc_expected_val(E_12)
    JT_int.KJT_factor = K_JT_12-K_JT_32
    print(JT_int.KJT_factor)

    JT_int.f_factor = JT_int.orbital_red_fact*JT_int.p_factor

    return JT_int


def calc_and_save_magnetic_interaction(B_fields, JT_int:qmp.Exe_tree, JT_cfg_parser:cfg_parser.Jahn_Teller_config_parser, complex_trf = True):
    energy_labels = ['eigenstate_0', 'eigenstate_1', 'eigenstate_2', 'eigenstate_3']
    JT_int_Es_dict = { energy_labels[0]: [], energy_labels[1]: [], energy_labels[2]: [],energy_labels[3]: []}

    res_folder = JT_cfg_parser.get_res_folder_name()

    prefix_name = JT_cfg_parser.get_prefix_name()

    Bs = JT_cfg_parser.get_mag_field_strengths_list()



    for B_field,B in zip(B_fields, Bs):        

        B_field = B_field.in_new_basis(JT_int.JT_theory.symm_lattice.get_normalized_basis_vecs())
        JT_int.create_one_mode_DJT_hamiltonian()
        #JT_int.add_spin_orbit_coupling()  
        JT_int.H_int =  JT_int.create_DJT_SOC_mag_interaction(*B_field.tolist())

        fn_prefix = prefix_name + '_' + str(round(B, 4)) + 'T'

        if JT_cfg_parser.eigen_states_type=='complex':
            comp_eig_vecs = calc_transform_and_save_eigen_vals_vecs(JT_int,fn_prefix+'_complex',res_folder)
        elif JT_cfg_parser.eigen_states_type=='real':
            comp_eig_vecs = calc_and_save_eigen_vals_vecs(JT_int,fn_prefix+'_real', res_folder)

        for eig_ket, line_label in zip(comp_eig_vecs.eigen_kets, energy_labels):
            JT_int_Es_dict[line_label].append(maths.meV_to_GHz(eig_ket.eigen_val))
    
    JT_int_Es_dict[cfg_parser.mag_field_strength_csv_col] = Bs

    pd.DataFrame(JT_int_Es_dict).set_index(cfg_parser.mag_field_strength_csv_col).to_csv(res_folder + prefix_name + '_magnetic_field_dependence_of_energy_states.csv')

    return JT_int_Es_dict


def calc_magnetic_interaction(B_fields, JT_int:qmp.Exe_tree):
    energy_labels = ['E0', 'E1', 'E2', 'E3']
    JT_int_Es_dict = { 'E0': [], 'E1': [], 'E2': [],'E3': []}

    JT_int.create_one_mode_DJT_hamiltonian()
    for B_field in B_fields:        


        if JT_int.JT_theory!= None and JT_int.JT_theory.symm_lattice!=None:

            B_field:maths.col_vector = B_field.in_new_basis(JT_int.JT_theory.symm_lattice.get_normalized_basis_vecs())
        
        B_field = B_field.basis_trf(JT_int.get_normalized_basis_vecs())


        H_DJT_mag = JT_int.create_DJT_SOC_mag_interaction(*B_field.tolist())
   
        H_DJT_mag.calc_eigen_vals_vects()
        

        for eig_ket, line_label in zip(H_DJT_mag.eigen_kets, energy_labels):
            JT_int_Es_dict[line_label].append(maths.meV_to_GHz(eig_ket.eigen_val))

    return JT_int_Es_dict


def create_JT_int(JT_config_parser: cfg_parser.Jahn_Teller_config_parser , section_to_look_for = '',complex_trf=True ):
    order  = JT_config_parser.get_order()
    intrincis_soc  =  JT_config_parser.get_spin_orbit_coupling(section_to_look_for)
    orbital_red_fact = JT_config_parser.get_gL_factor(section_to_look_for)

    orientation_basis = JT_config_parser.get_system_orientation_basis()

    if JT_config_parser.is_from_model_Hamiltonian(section_to_look_for) is True:
        return JT_config_parser.create_minimal_Exe_tree_from_cfg(section_to_look_for)
        #JT_int = qmp.Exe_tree.

    JT_theory = JT_config_parser.create_Jahn_Teller_theory_from_cfg(section_to_look_for)



    JT_int = qmp.Exe_tree.create_electron_phonon_Exe_tree(JT_theory, order, intrincis_soc, orbital_red_fact, orientation_basis)
    if intrincis_soc!=0.0:


        JT_int.add_spin_system()


        JT_int.create_one_mode_DJT_hamiltonian()

        JT_int.add_spin_orbit_coupling()
    return JT_int    


def spin_orbit_JT_procedure_general( JT_config_parser: cfg_parser.Jahn_Teller_config_parser , section_to_look_for = '',save_raw_pars = False,complex_trf=True ):

    order  = JT_config_parser.get_order()

    LzSz_calc_num = JT_config_parser.get_calc_LzSz()


    intrincis_soc  =   JT_config_parser.get_spin_orbit_coupling(section_to_look_for)
    orbital_red_fact = JT_config_parser.get_gL_factor(section_to_look_for)


    res_folder = JT_config_parser.get_res_folder_name()
    calc_name = JT_config_parser.get_prefix_name()

    JT_theory = JT_config_parser.create_Jahn_Teller_theory_from_cfg(section_to_look_for)


    JT_int = qmp.Exe_tree.create_electron_phonon_Exe_tree(JT_theory, order, intrincis_soc, orbital_red_fact)

    print('-------------------------------------------------')
    print('Maximum number of energy quantums of vibrations in each direction = ' + str(order) )

    print('-------------------------------------------------')
    print(JT_int.JT_theory)

    if intrincis_soc!=0.0:


        JT_int.add_spin_system()



        JT_int.create_one_mode_DJT_hamiltonian()

        JT_int.add_spin_orbit_coupling()


        eigen_kets =  JT_int.H_int.calc_eigen_vals_vects()

        JT_int.calc_reduction_factors()
        JT_int.calc_K_JT_factor()
        print('-------------------------------------------------')
        print(JT_int.get_essential_theoretical_results_string())
        th_res_name = res_folder+ '/' + calc_name +  '_theoretical_results.csv'
    
        JT_int.save_essential_theoretical_results(th_res_name)

        if JT_config_parser.eigen_states_type=='real':
            calc_and_save_eigen_vals_vecs(JT_int,calc_name, res_folder )
        elif JT_config_parser.eigen_states_type=='complex':
            calc_transform_and_save_eigen_vals_vecs(JT_int, calc_name,res_folder)
        
        if LzSz_calc_num>0:
            LzSz_res_df =  LzSz_operation(JT_int, LzSz_calc_num)

            LzSz_res_df.to_csv(res_folder+calc_name+'_LzSz_expected_values.csv')

        B_fields = JT_config_parser.get_magnetic_field_vectors()
        if B_fields!= None and JT_config_parser.is_ZPL_calculation() == False:
            if JT_config_parser.is_use_model_hamiltonian()==True:
                JT_int = qmp.minimal_Exe_tree.from_Exe_tree(JT_int)
            
            calc_and_save_magnetic_interaction(B_fields, JT_int, JT_config_parser)


    elif intrincis_soc==0.0:
        JT_int.create_one_mode_DJT_hamiltonian()


        no_soc_operation(JT_int)
        th_res_name = res_folder+ '/' + calc_name +  '_gnd_theoretical_results.csv'
    
        JT_int.save_essential_theoretical_results(th_res_name)
        calc_and_save_eigen_vals_vecs(JT_int,calc_name+'_real', res_folder)
    JT_config_parser.save_raw_pars(JT_int)


    return JT_int

def calc_transform_and_save_eigen_vals_vecs(JT_int:qmp.Exe_tree, calc_name, res_folder):
    basis_trf_mx = JT_int.system_tree.create_operator(operator_id='C_tr', operator_sys='orbital_system')
    eig_vecs = JT_int.calc_eigen_vals_vects()
    comp_eig_vecs = eig_vecs.transform_vector_space(JT_int.H_int.quantum_state_bases ,basis_trf_mx)
    comp_eig_vec_file_name = calc_name + '_eigen_vectors.csv'
    comp_eig_val_file_name = calc_name + '_eigen_values.csv'
    comp_eig_vecs.save(res_folder+ comp_eig_vec_file_name,res_folder+comp_eig_val_file_name)
    return comp_eig_vecs
def calc_and_save_eigen_vals_vecs(JT_int:qmp.Exe_tree, calc_name, res_folder):

    eig_vecs = JT_int.calc_eigen_vals_vects()

    comp_eig_vec_file_name = calc_name + '_eigen_vectors.csv'
    comp_eig_val_file_name = calc_name + '_eigen_values.csv'
    eig_vecs.save(res_folder+ comp_eig_vec_file_name,res_folder+comp_eig_val_file_name)
    return eig_vecs

def no_soc_operation(JT_int: qmp.Exe_tree ):
    




    eigen_kets = JT_int.calc_eigen_vals_vects()

    ground_1 = eigen_kets[0]

    ground_2 = eigen_kets[1]



    deg_sys = mf.degenerate_system_2D( [ground_1,ground_2] )

    electron_system = JT_int.system_tree.find_subsystem('electron_system')

    orbital_system = JT_int.system_tree.find_subsystem('orbital_system')

    spin_sys = qs.quantum_system_node.create_spin_system_node()



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
    JT_int.lambda_Ham = None
    


def LzSz_operation(JT_int:qmp.Exe_tree,LzSz_calc_num:int):

    LzSz_op = JT_int.system_tree.create_operator('LzSz',subsys_id='point_defect', operator_sys='electron_system')
    state_names = [ 'eigenstate_' + str(i) for i in range(0, LzSz_calc_num) ]
    eigen_energies = [ x.eigen_val for x in JT_int.H_int.eigen_kets[0:LzSz_calc_num]]

    LzSz_expected_vals= [ LzSz_op.calc_expected_val( eig_ket) for eig_ket in JT_int.H_int.eigen_kets[0:LzSz_calc_num] ]
            
    LzSz_res = { 'state_name': state_names , 'eigenenergy': eigen_energies, 'LzSz': LzSz_expected_vals }



    plt.rcParams['font.size'] = 18
    plt.title('Spin-orbit coupling \n expectation value')
    plt.xlabel(r'$\left< L_{z} \otimes S_{z} \right>$')
    plt.ylabel(r'Energy (meV)')
    plt.plot( LzSz_expected_vals, eigen_energies , 'x')

    plt.plot([0.0,0.0], [eigen_energies[0], eigen_energies[-1]], '--'  )
            
    plt.grid()
    plt.show()
    LzSz_res_df = pd.DataFrame(LzSz_res)
    LzSz_res_df = LzSz_res_df.set_index('state_name')
    return LzSz_res_df
    


def spin_orbit_JT_procedure(config_file_name:str, save_raw_pars = False, section_to_look_for = ''):
    
    spatial_dim = 2


    orbital_system = qs.quantum_system_node.create_2D_orbital_system_node()

    electron_system = qs.quantum_system_node('electron_system', children=[orbital_system])

    spin_sys = qs.quantum_system_node.create_spin_system_node()






    JT_cfg_parser = cfg_parser.Jahn_Teller_config_parser(config_file_name)

    order  = JT_cfg_parser.get_order()
    calc_name = JT_cfg_parser.get_prefix_name()
    l  = JT_cfg_parser.get_spin_orbit_coupling(section_to_look_for)
    gL = JT_cfg_parser.get_gL_factor(section_to_look_for)

    data_folder = JT_cfg_parser.get_data_folder_name()






    JT_theory, symm_lattice, less_symm_lattice_1, less_symm_lattice_2 = JT_cfg_parser.build_JT_theory(data_folder, section_to_look_for)



    if save_raw_pars == True:
        utilities.xml_parser.save_raw_data_from_xmls([symm_lattice, less_symm_lattice_1, less_symm_lattice_2], calc_name)




    print('-------------------------------------------------')
    print('Maximum number of energy quantums of vibrations in each direction = ' + str(order) )

    print('-------------------------------------------------')
    print(JT_theory)




    mode_1 = qmp.one_mode_phonon_sys(JT_theory.hw_meV,spatial_dim,order,['x','y'], 'mode_1', 'mode_1' )


    nuclei = qs.quantum_system_node('nuclei')#, children=[mode_1])

    point_defect_node = qs.quantum_system_node('point_defect', 
                                               children = [ nuclei,electron_system])

    point_defect_tree = qs.quantum_system_tree(point_defect_node)

    point_defect_tree.insert_node('nuclei', mode_1)

    point_defect_tree.insert_node('electron_system', spin_sys)


    JT_int = qmp.Exe_tree(point_defect_tree, JT_theory)
    JT_int.orbital_red_fact = gL
    JT_int.intrinsic_soc = l

    JT_int.create_one_mode_DJT_hamiltonian()
    

    JT_int.H_int.calc_eigen_vals_vects()


    JT_int.create_one_mode_DJT_hamiltonian()

    JT_int.add_spin_orbit_coupling()


    JT_int.H_int.calc_eigen_vals_vects()

    JT_int.calc_reduction_factors()
    JT_int.calc_K_JT_factor()
    print('-------------------------------------------------')
    print(JT_int.get_essential_theoretical_results_string())

    return JT_int

def plot_essential_data( jt_data: qmp.jt.Jahn_Teller_Theory):
    
    if jt_data.order_flag==2:
        E_jt_osc = jt_data.hw_mG
        E_barr_osc = jt_data.hw_pG

        jt_dist = jt_data.JT_dist
        barr_dist = jt_data.barrier_dist

        symm_latt_en = jt_data.symm_lattice.energy*1000
        jt_latt_en = jt_data.JT_lattice.energy*1000
        barr_latt_en = jt_data.barrier_lattice.energy*1000


        k_jt = 0.5* E_jt_osc**2
        k_barr = 0.5*E_barr_osc**2

        x_from = -1.2*jt_dist
        x_to = 1.2*jt_dist
        
        xs = np.linspace( x_from, x_to,1000 )

        jt_osc_pot = list(map( lambda x: k_jt*(x-jt_dist)**2 + jt_latt_en, xs ))
        barr_osc_pot = list(map( lambda x: k_barr*(x+barr_dist)**2 + barr_latt_en , xs ))
        plt.plot(xs, jt_osc_pot)
        plt.plot(xs, barr_osc_pot)

        plt.plot([ jt_dist ], [ jt_latt_en ], 'x', label = 'Jahn-Teller distorted lattice energy')

        plt.plot([ -barr_dist ], [ barr_latt_en ],'x', label = 'barrier distorted lattice energy')

        plt.plot([ 0.0 ], [symm_latt_en], 'x')
        plt.xlabel('distance ' +r'$(angstrom*\sqrt{m})$')
        plt.ylabel('energy (eV)')
        plt.show()

def plot_3D_APES(jt_theory: qmp.jt.Jahn_Teller_Theory):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    F = jt_theory.F
    G = jt_theory.G
    
    K = jt_theory.K

    r = np.linspace(-1.0, 1.0, 1000)

    phi = np.linspace(0, 2*np.pi, 1000)

    R, Phi = np.meshgrid(r,phi)


    Z1 = 0.5*K*R**2 + R* ( F**2 + G**2*R**2 + 2*F*G*R*np.cos(3*Phi)  )**0.5
    Z2 = 0.5*K*R**2 - R* ( F**2 + G**2*R**2 + 2*F*G*R*np.cos(3*Phi)  )**0.5

    X, Y = R*np.cos(Phi), R*np.sin(Phi)

    ax.plot_surface(X, Y, Z1, cmap=plt.cm.YlGnBu_r)
    ax.plot_surface(X, Y, Z2, cmap=plt.cm.YlGnBu_r)


    ax.set_xlabel(r'$\phi_\mathrm{real}$')
    ax.set_ylabel(r'$\phi_\mathrm{im}$')
    ax.set_zlabel(r'$V(\phi)$')

    plt.show()

def plot_2D_APES(jt_theory: qmp.jt.Jahn_Teller_Theory):
    plt.rcParams['font.size'] = 14
    

    xs = np.linspace(-1,1, 1000)

    K = jt_theory.K
    F,G = jt_theory.F, jt_theory.G
    
    
    ys2 = 0.5*K*xs**2 + xs* ( F**2 + G**2*xs**2 + 2*F*G*xs*np.cos(3*0)  )**0.5

    ys1 = 0.5*K*xs**2 - xs* ( F**2 + G**2*xs**2 + 2*F*G*xs*np.cos(3*0)  )**0.5 
    
    print('K: ' + str(K))
    print('F: ' + str(F))
    print('G: ' + str(G))

    jt_dist_th = -F/(K+2*G)
    barr_dist_th = F/(K-2*G) 

    jt_dist = jt_theory.JT_dist

    print(f'JT dist: {jt_dist}')

    barr_dist = jt_theory.barrier_dist
    
    E_JT = -jt_theory.E_JT_meV

    E_barr_en_latt = E_JT-jt_theory.delta_meV




    print('')
    
    plt.plot([-jt_dist] , [ E_JT ], 'x', label = 'Jahn-Teller energy')
    plt.plot([barr_dist] , [ E_barr_en_latt ], 'x', label = "Jahn-Teller energy - Barrier energy ")
    
    plt.plot([jt_dist_th] , [ E_JT ], 'x', label = 'Jahn-Teller energy (Bersuker 3.28 eq.)')
    plt.plot([barr_dist_th] , [ E_barr_en_latt ], 'x', label = "Jahn-Teller energy - Barrier energy (Bersuker 3.28 eq.)")

    plt.xlabel('distance' + "(normal coordinates)" )

    plt.ylabel('energy (meV)')

    plt.plot(xs,ys1)
    plt.plot(xs,ys2)

    plt.legend()
    plt.show()


def plot_APES(F,G):
    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, y)

    def f(x,y):
        return F*(x+y) + G* ( (x**2-y**2) - 2*x*y )
    
    Z = f(X, Y)
 
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')

    ax.plot_surface(X, Y, Z, cmap='cool', alpha=0.8)
 
    ax.set_title('APES', fontsize=14)
    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('y', fontsize=12)
    ax.set_zlabel('z', fontsize=12)
 
    plt.show()


def calc_transition_energies(ex_Es:dict, gnd_Es:dict,ex_labels:list[str], gnd_labels:list[str], field_strengths:list):
    line_labels = ['line_0', 'line_1', 'line_2', 'line_3']
    transitions = {'magnetic field (T)': field_strengths,'line_0':[], 'line_1': [], 'line_2':[], 'line_3':[]}
    for i in range(0, len(field_strengths)):
        line_label_iter =  iter(line_labels)

        for j in range(0, len( ex_labels )):
            for k in range(0, len(gnd_labels)):
                line_label = next(line_label_iter)
                transitions[line_label].append( ex_Es[ex_labels[j]][i]-gnd_Es[gnd_labels[k]][i] )
    
    return transitions


def model_ZPL_procedure(JT_config_parser: cfg_parser.Jahn_Teller_config_parser):
    pass


def create_directory(directory_path):
    # Check if the directory exists
    if not os.path.exists(directory_path):
        # Create the directory
        os.makedirs(directory_path)
        


def ZPL_procedure(JT_config_parser:cfg_parser.Jahn_Teller_config_parser):

    gnd_from_model_hamilton = JT_config_parser.is_from_model_Hamiltonian(gnd_sec)
    ex_from_model_hamilton = JT_config_parser.is_from_model_Hamiltonian(ex_sec)


    calculation_name = JT_config_parser.get_prefix_name()

    save_raw_pars_from_cfg = JT_config_parser.is_save_raw_pars()

    results_folder = JT_config_parser.get_res_folder_name()

    create_directory(results_folder)


    Bs = JT_config_parser.get_mag_field_strengths_list()

    B_min = JT_config_parser.get_B_min()
    B_max = JT_config_parser.get_B_max()




    print('-------------------------------------------------')
    print('Ground state:')
    JT_int_gnd = create_JT_int(JT_config_parser, section_to_look_for=gnd_sec)
    print('Maximum number of energy quantums of vibrations in each direction = ' + str(JT_config_parser.max_vib_quant) )

    if gnd_from_model_hamilton is False:
        print(JT_int_gnd.JT_theory)
        JT_int_gnd.calc_eigen_vals_vects()

        JT_int_gnd.calc_reduction_factors()
        JT_int_gnd.calc_K_JT_factor()
        JT_int_gnd.calc_energy_splitting()
        th_res_name = results_folder+ '/' + calculation_name +  '_gnd_theoretical_results.csv'
    
        JT_int_gnd.save_essential_theoretical_results(th_res_name)

    print('-------------------------------------------------')
    print(JT_int_gnd.get_essential_theoretical_results_string())
    
    

    print('-------------------------------------------------')
    print('Excited state:')
    JT_int_ex = create_JT_int(JT_config_parser, section_to_look_for=ex_sec)

    print('Maximum number of energy quantums of vibrations in each direction = ' + str(JT_config_parser.max_vib_quant) )

    print(JT_int_ex.JT_theory)
    print('-------------------------------------------------')
    
    if ex_from_model_hamilton is False:
    
        JT_int_ex.calc_eigen_vals_vects()

        JT_int_ex.calc_reduction_factors()
        JT_int_ex.calc_K_JT_factor()
        JT_int_ex.calc_energy_splitting()
        th_res_name = results_folder+ '/' + calculation_name +  '_ex_theoretical_results.csv'
        
        JT_int_ex.save_essential_theoretical_results(th_res_name)
    
    print('-------------------------------------------------')
    print(JT_int_ex.get_essential_theoretical_results_string())
    print('-------------------------------------------------')


    if save_raw_pars_from_cfg==True:
        JT_config_parser.save_raw_pars_ZPL(JT_int_gnd, JT_int_ex)    


    if JT_config_parser.is_save_model_Hamiltonian_cfg() == True:
        JT_config_parser.save_raw_pars_ZPL_model(JT_int_gnd, JT_int_ex)

    line_labels = ['line_0', 'line_1', 'line_2', 'line_3']
    JT_int_gnd_Es_dict = { 'E0': [], 'E1': [], 'E2': [],'E3': []}

    B_fields = JT_config_parser.get_magnetic_field_vectors()

    if JT_config_parser.is_use_model_hamiltonian()==True:
        JT_int_gnd = qmp.minimal_Exe_tree.from_Exe_tree(JT_int_gnd)

    JT_int_gnd_Es_dict = calc_magnetic_interaction( B_fields, JT_int_gnd)

    if JT_config_parser.is_use_model_hamiltonian()==True:

        JT_int_ex = qmp.minimal_Exe_tree.from_Exe_tree(JT_int_ex)

    JT_int_ex_Es_dict = calc_magnetic_interaction(B_fields, JT_int_ex)








    D_transition = calc_transition_energies(JT_int_ex_Es_dict, JT_int_gnd_Es_dict,['E0','E1'], ['E2','E3'], Bs)
    C_transition = calc_transition_energies(JT_int_ex_Es_dict, JT_int_gnd_Es_dict,['E0','E1'], ['E0','E1'], Bs)

    B_transition = calc_transition_energies(JT_int_ex_Es_dict, JT_int_gnd_Es_dict,['E2','E3'], ['E2','E3'], Bs)
    A_transition = calc_transition_energies(JT_int_ex_Es_dict, JT_int_gnd_Es_dict,['E2','E3'], ['E0','E1'], Bs)



    plt.rcParams['font.size'] = 20
    fig, axeses = plt.subplots(4, 1, sharex=True)

    title_name = calculation_name.replace('_', ' ')

    fig.suptitle(title_name )
    fig.set_figheight(10)
    fig.set_figwidth(14)


    energy_shift = (A_transition[line_labels[0]][0]-D_transition[line_labels[0]][0])/2

    zeroline= abs(A_transition[line_labels[0]][0]-energy_shift)


    [axes.set_xlim(B_min,B_max) for axes in axeses]

    [axes.tick_params(labeltop=False, bottom = False,labelright=True, right = True) for axes in axeses]


    axeses[3].xaxis.tick_bottom()

    axeses[2].annotate('ZPL shift (GHz)', (-0.12, 0.45), xycoords='axes fraction', rotation=90)

    axeses[3].set_xlabel('magnetic field (T)')

    for line_label in line_labels:

        axeses[0].plot(Bs,A_transition[line_label]+zeroline, '-k')
        axeses[1].plot(Bs,B_transition[line_label]+zeroline, '-k')    
        axeses[2].plot(Bs,C_transition[line_label]+zeroline, '-k')
        axeses[3].plot(Bs,D_transition[line_label]+zeroline, '-k')

    fig_fn = calculation_name + "_ZPL_calculation.png"
    plt.savefig(results_folder + fig_fn , bbox_inches='tight', dpi=700)
    plt.show()

    calculation_name_fn = calculation_name.replace(' ', '_')
    pd.DataFrame(A_transition).set_index('magnetic field (T)').to_csv(results_folder + calculation_name_fn+'_A_transitions.csv')
    pd.DataFrame(B_transition).set_index('magnetic field (T)').to_csv(results_folder + calculation_name_fn+'_B_transitions.csv')

    pd.DataFrame(C_transition).set_index('magnetic field (T)').to_csv(results_folder + calculation_name_fn+'_C_transitions.csv')
    pd.DataFrame(D_transition).set_index('magnetic field (T)').to_csv(results_folder + calculation_name_fn+'_D_transitions.csv')