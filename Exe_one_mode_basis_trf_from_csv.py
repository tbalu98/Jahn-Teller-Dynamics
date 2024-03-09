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
import utilities.OUTCAR_parsing as parsing
import utilities.xml_parser
import sys



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


orbital_system = qs.quantum_system_node('orbital_system', base_states=mf.hilber_space_bases().from_qm_nums_list([ ['ex'],[ 'ey']],qm_nums_names=['orbital'])  ,
                                         operators=el_sys_ops, dim= 2)



#electron_system = qs.quantum_system_node('electron_system', children=[orbital_system, spin_sys])
electron_system = qs.quantum_system_node('electron_system', children=[orbital_system])

spin_sys_ops = {}
Sz = np.matrix([[0.5,0],[0,-0.5]], dtype= np.float64)

spin_sys_ops['Sz'] = mf.MatrixOperator(maths.Matrix(Sz), name = 'Sz')



spin_sys = qs.quantum_system_node('spin_system', mf.hilber_space_bases().from_qm_nums_list([['up'], ['down']] , qm_nums_names=['spin']) , operators=spin_sys_ops)



#Geometries:
symm_types = ['symm_geom', 'less_symm_geom_1','less_symm_geom_2']



spatial_dim = 2


order = int(sys.argv[1])
filename = sys.argv[2]
theory_data = pd.read_csv(filename, index_col='case_name')

JT_theory = jt.Jahn_Teller_Theory().from_df(theory_data)
#order = 12
print(JT_theory)


#Create quantum system tree graph:

mode_1 = qmp.one_mode_phonon_sys(JT_theory.hw,spatial_dim,order,['x','y'], 'mode_1', 'mode_1' )


nuclei = qs.quantum_system_node('nuclei')#, children=[mode_1])

point_defect_node = qs.quantum_system_node('point_defect', 
                                               children = [ nuclei,electron_system])

point_defect_tree = qs.quantum_system_tree(point_defect_node)

point_defect_tree.insert_node('nuclei', mode_1)


JT_int = qmp.Exe_tree(point_defect_tree, JT_theory)


JT_int.create_one_mode_hamiltonian()
    
JT_int.H_int.calc_eigen_vals_vects()

res_df = JT_int.H_int.create_eigen_kets_vals_table(JT_int.system_tree.root_node.base_states)

res_df.to_csv('one_mode_Exe_problem_NO_spin_orbit_coupling.csv',sep = ';')

print('Eigen values of the Jahn-Teller interaction')
print( [  x.eigen_val for x in JT_int.H_int.eigen_kets ] )



ground_1 = JT_int.H_int.eigen_kets[0]

ground_2 = JT_int.H_int.eigen_kets[1]

    #New electron eigen vectors

eminus_el_sys = mf.ket_vector([complex(1.0,0.0), complex(0.0, -1.0) ])/(2.0**0.5)
eplus_el_sys = mf.ket_vector([complex(1.0,0.0), complex(0.0, 1.0) ])/(2.0**0.5)

electron_system.base_states.base_vectors['complex_basis'] = ([ eminus_el_sys, eplus_el_sys ], mf.hilber_space_bases().from_qm_nums_list([['e+'],['e-']],['complex_orbital']))


    #New harmonic oscillator eigen vectors
mode_1.base_states.base_vectors['complex_basis'] = mode_1.generate_new_hilbert_space_and_bases()


    #Transformation matrix for the whole system
new_trf_matrix, new_hilbert_space = JT_int.system_tree.create_basis_trf_matrix('complex_basis')

    #2D degenrate ground state system spin-orbital perturbation

deg_sys = mf.degenerate_system_2D( [ground_1,ground_2] )

pert_ham = JT_int.system_tree.create_operator(operator_id = 'Lz',operator_sys='orbital_system' )
    
print('red factor:')

print(deg_sys.add_perturbation(pert_ham))

deg_sys.to_complex_basis(new_trf_matrix)

eminus_prob = deg_sys.complex_deg_ket_vectors[0]
eplus_prob = deg_sys.complex_deg_ket_vectors[1]

    #op = mf.MatrixOperator.from_ket_vectors([ eminus_prob, eplus_prob ])


    
Psiplus:mf.ket_vector = (ground_1+complex(0.0,1.0)*ground_2)/(2.0)**0.5
Psiminus:mf.ket_vector = (ground_1-complex(0.0,1.0)*ground_2)/(2.0)**0.5


Psiplus_ph_el_sys_trfed = new_trf_matrix*Psiplus


"""
Psiplus_df = Psiplus_ph_el_sys_trfed.to_dataframe(new_hilbert_space)
    
Psiplus_df.to_csv('Psi_plus_new.csv',sep = ';')
"""
Psiminus_ph_el_sys_trfed = new_trf_matrix*Psiminus


"""
Psiminus_df = Psiminus_ph_el_sys_trfed.to_dataframe(new_hilbert_space)
    
Psiminus_df.to_csv('Psi_minus_new.csv',sep = ';')
"""


point_defect_tree.insert_node('electron_system', spin_sys)

JT_int = qmp.Exe_tree(point_defect_tree, JT_theory)


JT_int.create_one_mode_hamiltonian()




l  =1.0
JT_int.add_spin_orbit_coupling(l)
JT_int.H_int.calc_eigen_vals_vects()
res_df = JT_int.H_int.create_eigen_kets_vals_table(JT_int.system_tree.root_node.base_states)
    
res_df.to_csv('one_mode_spin_Exe_problem_so_coupling.csv',sep = ';')


print('-------------------------------')

print('red factor by Hamiltonian')

print( JT_int.H_int.eigen_kets[3].eigen_val- JT_int.H_int.eigen_kets[0].eigen_val)

print('-------------------------------')

    #op = mf.MatrixOperator.from_ket_vectors([  Psiminus_ph_el_sys_trfed, Psiplus_ph_el_sys_trfed ])
    