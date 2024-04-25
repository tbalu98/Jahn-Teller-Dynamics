import utilities.jahn_teller_theory as jt
import utilities.maths as  maths
import utilities.matrix_formalism as mf
import numpy as np
import matplotlib.pyplot as plt
import utilities.quantum_physics as qmp
import utilities.matrix_formalism as mf
import utilities.braket_formalism as bf
import utilities.quantum_system as qs
import old_ones.OUTCAR_parsing as parsing
import utilities.xml_parser
import sys
import utilities.JT_config_file_parsing as cfg_parser
from scipy.constants import physical_constants

import warnings
warnings.simplefilter("ignore", np.ComplexWarning)





# Create the symm electron system

orbital_system = qs.quantum_system_node.create_2D_orbital_system_node()

electron_system = qs.quantum_system_node('electron_system', children=[orbital_system])


#Spin system ops

spin_sys = qs.quantum_system_node.create_spin_system_node()


#Geometries:
symm_types = ['symm_geom', 'less_symm_geom_1','less_symm_geom_2']



spatial_dim = 2


#JT_cfg_parser = cfg_parser.Jahn_Teller_config_parser(str(sys.argv[1]))
JT_cfg_parser = cfg_parser.Jahn_Teller_config_parser('JT_csv_config.cfg')



order  = JT_cfg_parser.get_order()
calc_name = JT_cfg_parser.get_problem_name()
l  =  JT_cfg_parser.get_spin_orbit_coupling()
E_x, E_y = JT_cfg_parser.get_electric_field()

Bx, By,Bz = JT_cfg_parser.get_magnetic_field()

res_folder = JT_cfg_parser.get_res_folder_name()+'/'
data_folder = JT_cfg_parser.get_data_folder_name()+'/'

#order =12

JT_theory = jt.Jahn_Teller_Theory()

filenames = None

arguments = sys.argv[1:]

save_raw_pars = False

"""
if arguments[-1] == '-save_raw_pars':
    save_raw_pars = True
    arguments.pop()
"""
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

#point_defect_tree.insert_node('electron_system', spin_sys)


JT_int = qmp.Exe_tree(point_defect_tree, JT_theory)


JT_int.create_one_mode_hamiltonian()
    
eig_vec_file_name = calc_name + '_eigen_vectors.csv'
eig_val_file_name = calc_name + '_eigen_values.csv'

JT_int.save_eigen_vals_vects_to_file( res_folder+ eig_vec_file_name,res_folder+ eig_val_file_name)


print('-------------------------------')

print('Eigen values of the Jahn-Teller interaction')

for x in JT_int.H_int.eigen_kets[0:10]:
    print(str(round(x.eigen_val.real,4)) + ' meV') 


X_op = JT_int.system_tree.create_operator('X_orb','point_defect','orbital_system')
Y_op = JT_int.system_tree.create_operator('Y_orb', 'point_defect', 'orbital_system')

X_vib_op = JT_int.system_tree.create_operator('X', 'point_defect', 'mode_1')
Y_vib_op = JT_int.system_tree.create_operator('Y', 'point_defect', 'mode_1')

print('Expected vals')

for i in range(0, len(JT_int.H_int.eigen_kets[0:10])):

    eigen_ket = JT_int.H_int.eigen_kets[i]

    print(str(i) + ' eigen ket:')
    print(X_op.calc_expected_val(eigen_ket))
    print(Y_op.calc_expected_val(eigen_ket))

    print(X_vib_op.calc_expected_val(eigen_ket))
    print(Y_vib_op.calc_expected_val(eigen_ket))

print('-------------------------------')


#print( [  x.eigen_val for x in JT_int.H_int.eigen_kets ] )



ground_1 = JT_int.H_int.eigen_kets[0]

ground_2 = JT_int.H_int.eigen_kets[1]


    #2D degenrate ground state system spin-orbital perturbation

deg_sys = mf.degenerate_system_2D( [ground_1,ground_2] )

#point_defect_tree.insert_node('electron_system', spin_sys)


Sz = spin_sys.operators['Sz']

Lz = orbital_system.operators['Lz']

electron_system.operators['LzSz'] = Lz**Sz

pert_ham_Lz = JT_int.system_tree.create_operator(operator_id = 'Lz',operator_sys='orbital_system' )

pert_ham_LzSz = JT_int.system_tree.create_operator(operator_id = 'LzSz',operator_sys='electron_system')

#pert_ham_Sz = JT_int.system_tree.create_operator(operator_id = 'Sz',operator_sys='spin_system' )

#pert_ham = JT_int.system_tree.create_operator(operator_id='LzSz', operator_sys='orbital_system')

print('Reduction factor from first order perturbation:')

deg_sys.add_perturbation(pert_ham_Lz)

print('p = '+ str( round(deg_sys.p_red_fact,4)) + ' meV')


b1 = mf.ket_vector( [ 1.0/2**0.5, 1.0/2**0.5 ] )
b2 = mf.ket_vector( [ complex(1.0, 0.0)/(-2)**0.5, complex(-1.0, 0.0)/(-2)**0.5 ] )
bs = [b1, b2]

basis_trf_matrix = mf.MatrixOperator.basis_trf_matrix(bs)

orbital_system.operators['normal_to_complex_basis'] = basis_trf_matrix

point_def_basis_trf = point_defect_tree.create_operator(operator_id='normal_to_complex_basis', operator_sys='orbital_system')

orbital_system.operators['Lz_normal'] = mf.MatrixOperator.pauli_z_mx_op()


Lz_point_def = point_defect_tree.create_operator(operator_id='Lz', operator_sys='orbital_system')

ground_1_complex = point_def_basis_trf*ground_1
ground_2_complex = point_def_basis_trf*ground_2



#deg_sys.to_complex_basis()




ket_state_1 = bf.ket_state( qm_nums = [ 0,0,'ex','up' ], amplitude=complex(1.0, 0.0) )
ket_state_2 = bf.ket_state( qm_nums = [ 0,0,'ey','up' ], amplitude=complex(-1.0, 0.0) )
    
ket_state_3 = bf.ket_state( qm_nums = [ 0,0,'ex','down' ], amplitude=complex(1.0, 0.0) )
ket_state_4 = bf.ket_state( qm_nums = [ 0,0,'ey','down' ], amplitude=complex(-1.0, 0.0) )
"""
kramers_12 = JT_int.get_base_state().create_ket_vector([  ket_state_1, ket_state_2])
kramers_32 = JT_int.get_base_state().create_ket_vector([  ket_state_3, ket_state_4])
"""

l = 34.6
if l!=0.0:

    point_defect_tree.insert_node('electron_system', spin_sys)

    #JT_int = qmp.Exe_tree(point_defect_tree, JT_theory)


    JT_int.create_one_mode_hamiltonian()

    JT_int.add_spin_orbit_coupling(l)


    JT_int.H_int.calc_eigen_vals_vects()


    LzSz_op = JT_int.system_tree.create_operator('LzSz',subsys_id='point_defect', operator_sys='electron_system')

    LzSz_expected_vals= [ LzSz_op.calc_expected_val( eig_ket) for eig_ket in JT_int.H_int.eigen_kets[0:50] ]
    
    eigen_energies = [ x.eigen_val for x in JT_int.H_int.eigen_kets[0:50]]
    """
    for i in range(0, 50):
        LzSz_exp = LzSz_op.calc_expected_val(JT_int.H_int.eigen_kets[i])
        #print(LS_exp)
        LzSz_expected_vals.append(LzSz_exp)
        #print(JT_int.H_int.calc_expected_val(JT_int.H_int.eigen_kets[i]))
    """

    p_32 = LzSz_expected_vals[2]
    p_12 = LzSz_expected_vals[0]

    print('p values after adding SOC to Hamiltonian')

    print( "p3/2 = " + str( p_32 ))
    print( "p1/2 = " + str( p_12 ))

    p = p_32+p_12
    delta = p_32-p_12

    print('C8 and C9 equation:'  )
    print("p = " + str( p ))
    print( "delta = " + str(delta) )
    plt.rcParams['font.size'] = 18
    plt.title('PbV energy and spin-orbit coupling \n expectation value')
    plt.xlabel(r'$\left< L_{z}S_{z} \right>$')
    plt.ylabel(r'Energy (meV)')
    plt.plot( LzSz_expected_vals, eigen_energies , 'x')
    plt.xlim(-0.5,0.5)
    plt.ylim(-50.0,250.0)

    
    plt.xticks([-0.5, -0.25,0.0, 0.25,  0.5],[ r'$ -\dfrac{1}{2} $' ,r'$ -\dfrac{1}{4} $',0, r'$ \dfrac{1}{4} $', r'$ \dfrac{1}{2} $' ])
    plt.plot([0.0,0.0], [-50.0, 300.0], '--'  )
    plt.grid()
    plt.show()
    
    print('-------------------------------')

    print('p Ham reduction factor')

    print( 'lambda_Ham = ' + str(round((JT_int.H_int.eigen_kets[2].eigen_val- JT_int.H_int.eigen_kets[0].eigen_val).real,4)) + ' meV')


    X_op = JT_int.system_tree.create_operator('Lz','point_defect','orbital_system')
    Y_op = JT_int.system_tree.create_operator('Y_orb', 'point_defect', 'orbital_system')

    X_vib_op = JT_int.system_tree.create_operator('X', 'point_defect', 'mode_1')

    Y_vib_op = JT_int.system_tree.create_operator('Y', 'point_defect', 'mode_1')

    spin_op = JT_int.system_tree.create_operator('Sz','point_defect', 'spin_system')

    print('X position expected val')

    orb_pos_x_vals = []
    spin_z_vals = []

    for i in range(0, len(JT_int.H_int.eigen_kets[0:10])):
    #for i in range(0, 10):

        eigen_ket = JT_int.H_int.eigen_kets[i]

        orb_pos_x_val = X_op.calc_expected_val(eigen_ket)
        orb_pos_y_val = Y_op.calc_expected_val(eigen_ket)

        nuc_x_val = X_vib_op.calc_expected_val(eigen_ket)
        nuc_y_val = Y_vib_op.calc_expected_val(eigen_ket)

        orb_pos_x_vals.append(orb_pos_x_val)

        spin_z_val = spin_op.calc_expected_val(eigen_ket)

        spin_z_vals.append(spin_z_val)

        print(str(i) + ' eigen ket:')


        print( 'orbital x position ' +  str(orb_pos_x_val))
        print( 'orbital y position ' +  str(orb_pos_y_val))

        print( 'spin z ' + str(spin_z_val))
        print( 'nuclei x position ' + str( nuc_x_val))
        print( 'nuclei y position ' + str( nuc_y_val))

        print('-------------------------------')


    
    base_ket_vec:mf.ket_vector = JT_int.system_tree.root_node.base_states.create_ket_vector( [ ket_state_1, ket_state_2, ket_state_3, ket_state_4   ]  )

    base_ket_vec = base_ket_vec.normalize()

    eigen_energies = []

    vals = []

    for i in range(0, len( JT_int.H_int.eigen_kets[0:10])):
        eig_ket_vec:mf.ket_vector = JT_int.H_int.eigen_kets[i]
        eigen_energies.append( eig_ket_vec.eigen_val.real)
        
        coeff = (base_ket_vec.to_bra_vector()*eig_ket_vec)
        val = abs(coeff)**2
        vals.append(val)
        spin_z_vals[i] = spin_z_vals[i]*val

    """
    plt.plot(eigen_energies, vals, 'x')
    plt.show()
    print(base_ket_vec)
    """
"""
plt.plot(orb_pos_x_vals,spin_z_vals, 'x')
plt.show()
"""
delta = 1.0
f = 1.0


if Bz!=0.0 or Bx!= 0.0 or By != 0.0:
    JT_int.system_tree.insert_node('electron_system', spin_sys)

    JT_int.create_one_mode_hamiltonian()

    JT_int.add_spin_orbit_coupling(l) if l>0.0 else None

    JT_int.add_magnetic_field(Bx,By,Bz, delta, f)

    JT_int.H_int.calc_eigen_vals_vects()

    print('p Ham reduction factor')

    print( 'lambda_0 = ' + str(round((JT_int.H_int.eigen_kets[2].eigen_val- JT_int.H_int.eigen_kets[0].eigen_val).real,4)) + ' meV')




if E_x !=0.0 or E_y != 0.0:

    JT_int.create_one_mode_hamiltonian()

    JT_int.add_electric_field(E_x, E_y)
    JT_int.H_int.calc_eigen_vals_vects()
    print('-------------------------------')

    print('q Ham reduction factor')

    if JT_int.system_tree.find_subsystem('spin_system') == None:
        print( 'lambda_0 = ' + str(round((JT_int.H_int.eigen_kets[1].eigen_val- JT_int.H_int.eigen_kets[0].eigen_val).real,4)) + ' meV')
    else:
        print( 'lambda_0 = ' + str(round((JT_int.H_int.eigen_kets[3].eigen_val- JT_int.H_int.eigen_kets[0].eigen_val).real,4)) + ' meV')


JT_int.save_eigen_vals_vects_to_file(res_folder + calc_name+'_eigen_vecs.csv',res_folder+ calc_name+'_eigen_vals.csv')




