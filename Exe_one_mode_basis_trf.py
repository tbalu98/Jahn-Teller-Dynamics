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

def read_lattice(control, symm_type):
    latt = VASP.POSCAR_data( control.data_folder + control[symm_type+'_path']).lattice
    latt.get_dope_ions().m = control.dopant_m
    latt.get_maj_ions().m = control.maj_m
    latt.energy = control[symm_type+'_energy']

    return latt

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


electron_system = qs.quantum_system_node('electron_system', base_states=mf.hilber_space_bases().from_qm_nums_list([ ['e-'],[ 'e+']],qm_nums_names=['orbital'])  ,
                                         operators=el_sys_ops, dim= 2)



#Geometries:
symm_types = ['symm_geom', 'less_symm_geom_1','less_symm_geom_2']


# Read cases from control file
for case_name in control_data.index:

    control = control_data.loc[case_name]

    # Read geometries
    symm_lattice = read_lattice(control, 'symm_geom')

    less_symm_lattice_1 = read_lattice(control, 'less_symm_geom_1')

    less_symm_lattice_2 = read_lattice(control, 'less_symm_geom_2')

    #Calculate the parameters of Jahn-Teller theory
    JT_theory = jt.Jahn_Teller_Theory(symm_lattice, less_symm_lattice_1, less_symm_lattice_2)

    print(JT_theory)

    mode = 77.6

    mode2 = 78.8

    spatial_dim = 2
    order  = 12
    order = 12

    mode_1 = qmp.one_mode_phonon_sys(mode,spatial_dim,order,['x','y'], 'mode_1', 'mode_1' )

    #Create quantum system:



    nuclei = qs.quantum_system_node('nuclei')#, children=[mode_1])

    point_defect_node = qs.quantum_system_node('point_defect', 
                                               children = [ nuclei,electron_system])

    point_defect_tree = qs.quantum_system_tree(point_defect_node)

    point_defect_tree.insert_node('nuclei', mode_1)

    point_defect_node.base_states.savetxt('states.txt')

    JT_theory.E_JT = 41.8
    JT_theory.delta  = 9.1

    JT_int = qmp.Exe_tree(point_defect_tree, JT_theory)


    #JT_int.create_multi_mode_hamiltonian()

    JT_int.create_one_mode_hamiltonian()

    

    print('fin')
    
    JT_int.H_int.save('H_int_tree.csv')
    
    JT_int.H_int.calc_eigen_vals_vects()

    res_df = JT_int.H_int.create_eigen_kets_vals_table(JT_int.system_tree.root_node.base_states)

    #res_df = res_df.set_index('states')

    res_df.to_csv('one_mode_Exe_problem.csv',sep = ';')
    #JT_eigen_states = qm.eigen_vect.from_vals_vects( JT_int.H_int.eigen_vals, JT_int.H_int.eigen_vects)

    print('Eigen values of the Jahn-Teller interaction')
    print( [  x.eigen_val for x in JT_int.H_int.eigen_kets ] )

    print(len(JT_int.H_int.eigen_kets))

    ground_1 = JT_int.H_int.eigen_kets[0]

    ground_2 = JT_int.H_int.eigen_kets[1]

    #New electron eigen vectors

    eminus_el_sys = mf.ket_vector([complex(1.0,0.0), complex(0.0, -1.0) ])/(2.0**0.5)
    eplus_el_sys = mf.ket_vector([complex(1.0,0.0), complex(0.0, 1.0) ])/(2.0**0.5)

    #eminus_el_sys = mf.ket_vector([complex(1.0,0.0), complex(1.0, 0.0) ])/(2.0**0.5)
    #eplus_el_sys = mf.ket_vector([complex(  0.0,-1.0), complex(  0.0,-1.0) ])/(2.0**0.5)


    electron_system.base_states.base_vectors['complex_basis'] = [ eminus_el_sys, eplus_el_sys ]
    # basis transformations
    #ph_sys_basis_trf = mode_1.create_complex_basis_trf()

    mode_1.base_states.base_vectors['complex_basis'] = mode_1.generate_new_bases()

    new_trf_matrix = JT_int.system_tree.create_basis_trf_matrix('complex_basis')



    deg_sys = mf.degenerate_system_2D( [ground_1,ground_2] )

    Lz_op = mf.MatrixOperator.create_Lz_op()

    electron_system.operators['Lz'] = Lz_op




    pert_ham = JT_int.system_tree.create_operator(operator_id = 'Lz',operator_sys='electron_system' )
    pert_ham.save('or_pert_ham.csv')

    """
    pert_ham = Lz_op**id_op
    pert_ham.save('my_pert_ham.csv')
    """
    print(pert_ham)
    
    

    print('red factors')

    print(deg_sys.add_perturbation(pert_ham))
    deg_sys.to_complex_basis(new_trf_matrix)

    eminus_prob = deg_sys.complex_deg_ket_vectors[0]
    eplus_prob = deg_sys.complex_deg_ket_vectors[1]

    op = mf.MatrixOperator.from_ket_vectors([ eminus_prob, eplus_prob ])


    
    Psiplus:mf.MatrixOperator = (ground_1+complex(0.0,1.0)*ground_2)/(2.0)**0.5
    Psiminus:mf.MatrixOperator = (ground_1-complex(0.0,1.0)*ground_2)/(2.0)**0.5

    # basis transformations
    ph_sys_basis_trf = mode_1.create_complex_basis_trf()

    mode_1.base_states.base_vectors['complex_bases'] = mode_1.generate_new_bases()

    mode_1.operators['basis_trf'] = ph_sys_basis_trf

    el_basis_trf = mf.MatrixOperator.basis_trf_matrix([ eminus_el_sys, eplus_el_sys  ])

    electron_system.operators['basis_trf'] = el_basis_trf





    Psiplus_ph_el_sys_trfed = new_trf_matrix*Psiplus
    Psiplus_df = Psiplus_ph_el_sys_trfed.to_dataframe(JT_int.system_tree.root_node.base_states)
    
    Psiplus_df.to_csv('Psi_plus.csv',sep = ';')

    Psiminus_ph_el_sys_trfed = new_trf_matrix*Psiminus



    Psiminus_df = Psiminus_ph_el_sys_trfed.to_dataframe(JT_int.system_tree.root_node.base_states)
    
    Psiminus_df.to_csv('Psi_minus.csv',sep = ';')

    op = mf.MatrixOperator.from_ket_vectors([  Psiminus_ph_el_sys_trfed, Psiplus_ph_el_sys_trfed ])
    