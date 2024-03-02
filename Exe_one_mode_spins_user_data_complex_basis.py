import pandas as pd
import utilities.VASP as VASP
import math
import utilities.jahn_teller_theory as jt
import utilities.maths as  maths
import utilities.matrix_formalism as mf
import numpy as np
import scipy as sc
from scipy.sparse.linalg import eigs
from numpy import linalg as LA
import matplotlib.pyplot as plt
import utilities.quantum_physics as qmp
import utilities.matrix_formalism as mf
import utilities.braket_formalism as bf
import utilities.quantum_system as qs
control_file_path = 'data/'

control_data = pd.read_csv( control_file_path + 'control.csv' , index_col='case')

# Create the symm electron system
#Symmetry operations
sz = np.matrix([[1,0],[0,-1]], dtype= np.float64)
sx = np.matrix([[0,1],[1,0]], dtype= np.float64)
s0 = np.matrix([[1,0],[0,1]], dtype= np.float64)



el_sys_ops = {}

el_sys_ops['sz'] = mf.MatrixOperator(maths.Matrix(sz), name = 'sz')
el_sys_ops['sx'] = mf.MatrixOperator(maths.Matrix(sx), name = 'sx')
el_sys_ops['s0'] = mf.MatrixOperator(maths.Matrix(s0), name = 's0')
el_sys_ops['Lz'] = mf.MatrixOperator.create_Lz_op()


spin_sys_ops = {}
Sz = np.matrix([[0.5,0],[0,-0.5]], dtype= np.float64)

spin_sys_ops['Sz'] = mf.MatrixOperator(maths.Matrix(Sz), name = 'Sz')



spin_sys = qs.quantum_system_node('spin_system', mf.hilber_space_bases().from_qm_nums_list([['up'], ['down']] , qm_nums_names=['spin']) , operators=spin_sys_ops)

orbital_system = qs.quantum_system_node('orbital_system', base_states=mf.hilber_space_bases().from_qm_nums_list([ ['ex'],[ 'ey']],qm_nums_names=['orbital'])  ,
                                         operators=el_sys_ops, dim= 2)



#electron_system = qs.quantum_system_node('electron_system', children=[orbital_system, spin_sys])
electron_system = qs.quantum_system_node('electron_system', children=[orbital_system])


#Geometries:
symm_types = ['symm_geom', 'less_symm_geom_1','less_symm_geom_2']

l = 1
# Read cases from control file

E_JT = 91.6
delta = 12.3
hw = 78.6
for case_name in control_data.index:

    control = control_data.loc[case_name]

    # Read geometries

    #Calculate the parameters of Jahn-Teller theory
    JT_theory = jt.Jahn_Teller_Theory().from_parameters(E_JT, delta, hw)
    #print(JT_theory)



    spatial_dim = 2
    order  = 12

    mode_1 = qmp.one_mode_phonon_sys(JT_theory.hw,spatial_dim,order,['x','y'], 'mode_1', 'mode_1' )


    #Create quantum system:



    nuclei = qs.quantum_system_node('nuclei')#, children=[mode_1])

    point_defect_node = qs.quantum_system_node('point_defect', 
                                               children = [ nuclei,electron_system])

    point_defect_tree = qs.quantum_system_tree(point_defect_node)

    point_defect_tree.insert_node('nuclei', mode_1)

    point_defect_node.base_states.savetxt('states.txt')

    #JT_theory.E_JT = 41.8
    #JT_theory.delta  = 9.1

    JT_int = qmp.Exe_tree(point_defect_tree, JT_theory)


    JT_int.create_one_mode_hamiltonian()



    

    print('fin')
    
    
    JT_int.H_int.calc_eigen_vals_vects()

    res_df = JT_int.H_int.create_eigen_kets_vals_table(JT_int.system_tree.root_node.base_states)
    
    res_df.to_csv('one_mode_spin_Exe_problem_NO_so_coupling.csv',sep = ';')

    

    ground_1 = JT_int.H_int.eigen_kets[0]

    ground_2 = JT_int.H_int.eigen_kets[1]


    #ground_1_sq = ground_1*ground_1
    #ground_2_sq = ground_2*ground_2


    print('Square differences of the two ground state coefficients')

    #print(ground_1_sq-ground_2_sq)



    """

    #New electron eigen vectors

    eminus_el_sys = mf.ket_vector([complex(1.0,0.0), complex(0.0, -1.0) ])/(2.0**0.5)
    eplus_el_sys = mf.ket_vector([complex(1.0,0.0), complex(0.0, 1.0) ])/(2.0**0.5)

    #eminus_el_sys = mf.ket_vector([complex(1.0,0.0), complex(1.0, 0.0) ])/(2.0**0.5)
    #eplus_el_sys = mf.ket_vector([complex(  0.0,-1.0), complex(  0.0,-1.0) ])/(2.0**0.5)


    orbital_system.base_states.base_vectors['complex_basis'] = [ eminus_el_sys, eplus_el_sys ]
    # basis transformations
    #ph_sys_basis_trf = mode_1.create_complex_basis_trf()

    mode_1.base_states.base_vectors['complex_basis'] = mode_1.generate_new_bases()

    """


    deg_sys = mf.degenerate_system_2D( [ground_1,ground_2] )

    Lz_op = mf.MatrixOperator.create_Lz_op()

    orbital_system.operators['Lz'] = Lz_op




    pert_ham = JT_int.system_tree.create_operator(operator_id = 'Lz',operator_sys='orbital_system' )
    pert_ham.save('or_pert_ham.csv')

    """
    pert_ham = Lz_op**id_op
    pert_ham.save('my_pert_ham.csv')
    """
    print(pert_ham)
    
    

    print('red factors')

    print(deg_sys.add_perturbation(pert_ham))
    

    #Red factor from basis trf-ed states

    ground_1 = JT_int.H_int.eigen_kets[0]
    ground_2 = JT_int.H_int.eigen_kets[1]



    eminus_el_sys = mf.ket_vector([complex(1.0,0.0), complex(0.0, -1.0) ])/(2.0**0.5)
    eplus_el_sys = mf.ket_vector([complex(1.0,0.0), complex(0.0, 1.0) ])/(2.0**0.5)


    orbital_system.base_states.base_vectors['complex_basis+-'] = [  eplus_el_sys,eminus_el_sys ]

    orb_base_trf = mf.MatrixOperator.basis_trf_matrix(orbital_system.base_states.base_vectors['complex_basis+-'])

    el_osc_sys_trf = mf.MatrixOperator.create_id_matrix_op(dim = mode_1.h_sp_dim)**orb_base_trf

    ground_1_trf:mf.ket_vector = el_osc_sys_trf*ground_1
    ground_2_trf = el_osc_sys_trf*ground_2

    ground_1_trf_even = ground_1_trf.coeffs.tolist()[::2]
    ground_1_trf_odd = ground_1_trf.coeffs.tolist()[1::2]


    c_sq = sum( [ abs(x[0])**2 for x in ground_1_trf_even ] )


    d_sq = sum( [ abs(x[0])**2 for x in ground_1_trf_odd ] )

    print(d_sq-c_sq)

    ground_2_trf_df = ground_2.to_dataframe(JT_int.system_tree.root_node.base_states)

    ground_2_trf_df.to_csv('ground_2.csv')
    

    ground_2_trf_df = ground_2_trf.to_dataframe(JT_int.system_tree.root_node.base_states)

    ground_2_trf_df.to_csv('ground_2_trf.csv')

    Psip = (1/0.5**2)*( ground_1 + ground_2  )

    Psip_trf = el_osc_sys_trf*Psip

    #Psip_trf.to_  

    #Add spin
    #electron_system.add_child(spin_sys)

    point_defect_tree.insert_node('electron_system', spin_sys)

    JT_int = qmp.Exe_tree(point_defect_tree, JT_theory)


    JT_int.create_one_mode_hamiltonian()





    
    JT_int.add_spin_orbit_coupling(l)
    JT_int.H_int.calc_eigen_vals_vects()
    res_df = JT_int.H_int.create_eigen_kets_vals_table(JT_int.system_tree.root_node.base_states)
    
    res_df.to_csv('one_mode_spin_Exe_problem_so_coupling.csv',sep = ';')


    print('red factor by Hamiltonian')

    print( JT_int.H_int.eigen_kets[3].eigen_val- JT_int.H_int.eigen_kets[0].eigen_val)

    print('-------------------------------')


    
    ground_1 = JT_int.H_int.eigen_kets[0]
    ground_2 = JT_int.H_int.eigen_kets[2]



    eminus_el_sys = mf.ket_vector([complex(1.0,0.0), complex(0.0, -1.0) ])/(2.0**0.5)
    eplus_el_sys = mf.ket_vector([complex(1.0,0.0), complex(0.0, 1.0) ])/(2.0**0.5)


    orbital_system.base_states.base_vectors['complex_basis+-'] = [  eplus_el_sys,eminus_el_sys ]

    orb_base_trf = mf.MatrixOperator.basis_trf_matrix(orbital_system.base_states.base_vectors['complex_basis+-'])

    electron_basis_trf = orb_base_trf**mf.MatrixOperator.create_id_matrix_op(dim = 2)

    sys_basis_trf_matrix_pm = electron_basis_trf**mf.MatrixOperator.create_id_matrix_op(dim = mode_1.h_sp_dim)



    print('transformation')
    ground_1_trf:mf.ket_vector = sys_basis_trf_matrix_pm*ground_1
    ground_2_trf:mf.ket_vector = sys_basis_trf_matrix_pm*ground_2

    ground_2_trf_sq = ground_2_trf.calc_abs_square()
    ground_1_trf_sq = ground_1_trf.calc_abs_square()
    
    print(ground_1_trf_sq-ground_2_trf_sq)

    """
    print('ground_trf_1')
    print(ground_1_trf)
    print('ground_trf_2')

    print(ground_2_trf)
    """
    print('sq_diff')
    #print(ground_2_trf*ground_2_trf - ground_1_trf*ground_1_trf )


    #res_df = res_df.set_index('states')

    #JT_eigen_states = qm.eigen_vect.from_vals_vects( JT_int.H_int.eigen_vals, JT_int.H_int.eigen_vects)

    



    """
    new_trf_matrix = JT_int.system_tree.create_basis_trf_matrix('complex_basis')
    deg_sys.to_complex_basis(new_trf_matrix)

    eminus_prob = deg_sys.complex_deg_ket_vectors[0]
    eplus_prob = deg_sys.complex_deg_ket_vectors[1]

    op = mf.MatrixOperator.from_ket_vectors([ eminus_prob, eplus_prob ])
    """


    
    Psiplus:mf.ket_vector = (ground_1+complex(0.0,1.0)*ground_2)/(2.0)**0.5
    Psiminus:mf.ket_vector = (ground_1-complex(0.0,1.0)*ground_2)/(2.0)**0.5

    print('psi_lus_psi_minus')

    Psi_plus_square = Psiplus.calc_abs_square()
    Psi_minus_square = Psiminus.calc_abs_square()

    print(Psi_minus_square-Psi_plus_square)

    #print(Psiplus*Psiplus-Psiminus*Psiminus)

    """
    # basis transformations
    ph_sys_basis_trf = mode_1.create_complex_basis_trf()

    mode_1.base_states.base_vectors['complex_bases'] = mode_1.generate_new_bases()

    mode_1.operators['basis_trf'] = ph_sys_basis_trf

    el_basis_trf = mf.MatrixOperator.basis_trf_matrix([ eminus_el_sys, eplus_el_sys  ])

    orbital_system.operators['basis_trf'] = el_basis_trf





    Psiplus_ph_el_sys_trfed = new_trf_matrix*Psiplus
    Psiplus_df = Psiplus_ph_el_sys_trfed.to_dataframe(JT_int.system_tree.root_node.base_states)
    
    Psiplus_df.to_csv('Psi_plus.csv',sep = ';')

    Psiminus_ph_el_sys_trfed = new_trf_matrix*Psiminus



    Psiminus_df = Psiminus_ph_el_sys_trfed.to_dataframe(JT_int.system_tree.root_node.base_states)
    
    Psiminus_df.to_csv('Psi_minus.csv',sep = ';')

    op = mf.MatrixOperator.from_ket_vectors([  Psiminus_ph_el_sys_trfed, Psiplus_ph_el_sys_trfed ])
    """