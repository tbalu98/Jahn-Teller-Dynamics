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

m_electron = 0.51e9
m_el_D = 0.0005485833
def meV_to_Hz(E):
    return E/(4.13566770e-12)

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

    mode_1 = qmp.one_mode_phonon_sys(JT_theory.hw_meV,spatial_dim,order,['x','y'], 'mode_1', 'mode_1' )


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
    JT_int.delta_p_factor = p_32+p_12

    print('Based on C8 and C9 equation:'  )
    print("p = " + str( JT_int.p_factor ))
    print( "delta = " + str(JT_int.delta_p_factor) )

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


    print('-------------------------------------------------')
    print('Maximum number of energy quantums of vibrations in each direction = ' + str(order) )

    print('-------------------------------------------------')
    print(JT_theory)



#Create quantum system tree graph:

    mode_1 = qmp.one_mode_phonon_sys(JT_theory.hw_meV,spatial_dim,order,['x','y'], 'mode_1', 'mode_1' )


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




#print( [  x.eigen_val for x in JT_int.H_int.eigen_kets ] )









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

        #jt_freq = meV_to_Hz(E_jt_osc)

        #barr_freq = meV_to_Hz(E_barr_osc)

        k_jt = 0.5* E_jt_osc**2
        k_barr = 0.5*E_barr_osc**2

        x_from = -1.2*jt_dist
        x_to = 1.2*jt_dist
        
        xs = np.linspace( x_from, x_to,1000 )

        jt_osc_pot = list(map( lambda x: k_jt*(x-jt_dist)**2 + jt_latt_en, xs ))
        #print(k_jt)
        barr_osc_pot = list(map( lambda x: k_barr*(x+barr_dist)**2 + barr_latt_en , xs ))
        #print(k_barr)
        plt.plot(xs, jt_osc_pot)
        plt.plot(xs, barr_osc_pot)

        plt.plot([ jt_dist ], [ jt_latt_en ], 'x', label = 'Jahn-Teller distorted lattice energy')

        plt.plot([ -barr_dist ], [ barr_latt_en ],'x', label = 'barrier distorted lattice energy')

        plt.plot([ 0.0 ], [symm_latt_en], 'x')
        plt.xlabel('distance ' +r'$(angstrom*\sqrt{m})$')
        plt.ylabel('energy (eV)')
        plt.show()

def plot_3D_APES(jt_theory: qmp.jt.Jahn_Teller_Theory):
    #plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    F = jt_theory.F
    G = jt_theory.G
    
    K = jt_theory.K

    r = np.linspace(-1.0, 1.0, 1000)

    phi = np.linspace(0, 2*np.pi, 1000)

    R, Phi = np.meshgrid(r,phi)

    #Z = ((R**2 - 1)**2)

    # Express the mesh in the cartesian system.
    Z1 = 0.5*K*R**2 + R* ( F**2 + G**2*R**2 + 2*F*G*R*np.cos(3*Phi)  )**0.5
    Z2 = 0.5*K*R**2 - R* ( F**2 + G**2*R**2 + 2*F*G*R*np.cos(3*Phi)  )**0.5

    # Express the mesh in the cartesian system.
    X, Y = R*np.cos(Phi), R*np.sin(Phi)

    # Plot the surface.
    ax.plot_surface(X, Y, Z1, cmap=plt.cm.YlGnBu_r)
    ax.plot_surface(X, Y, Z2, cmap=plt.cm.YlGnBu_r)

    #ax.plot_surface(X, Y, Z2, cmap=plt.cm.YlGnBu_r)

    # Tweak the limits and add latex math labels.
    #ax.set_zlim(0, 100000)
    ax.set_xlabel(r'$\phi_\mathrm{real}$')
    ax.set_ylabel(r'$\phi_\mathrm{im}$')
    ax.set_zlabel(r'$V(\phi)$')

    plt.show()

def plot_2D_APES(jt_theory: qmp.jt.Jahn_Teller_Theory):
    #plt.clf()
    plt.rcParams['font.size'] = 14
    

    xs = np.linspace(-1,1, 1000)

    K = jt_theory.K
    #K = jt_theory.K
    F,G = jt_theory.F, jt_theory.G
    
    # Express the mesh in the cartesian system.
    
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

    numerator = (1-2*jt_theory.delta_meV/(4*jt_theory.E_JT_meV-2*jt_theory.delta_meV))**2

    denominator = (1+2*jt_theory.delta_meV/(4*jt_theory.E_JT_meV-2*jt_theory.delta_meV))**2


    print('')
    
    plt.plot([-jt_dist] , [ E_JT ], 'x', label = 'Jahn-Teller energy')
    plt.plot([barr_dist] , [ E_barr_en_latt ], 'x', label = "Jahn-Teller energy - Barrier energy ")
    
    plt.plot([jt_dist_th] , [ E_JT ], 'x', label = 'Jahn-Teller energy (Bersuker 3.28 eq.)')
    plt.plot([barr_dist_th] , [ E_barr_en_latt ], 'x', label = "Jahn-Teller energy - Barrier energy (Bersuker 3.28 eq.)")

    plt.xlabel('distance' + "(normal coordinates)" )

    plt.ylabel('energy (meV)')

    # Plot the surface.
    plt.plot(xs,ys1)
    plt.plot(xs,ys2)

    plt.legend()
    plt.show()


def plot_APES(F,G):
    x = np.linspace(-100, 100, 100)
    y = np.linspace(-100, 100, 100)
    X, Y = np.meshgrid(x, y)

    def f(x,y):
        #return (x**2 + y**2)
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