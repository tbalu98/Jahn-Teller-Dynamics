import pandas as pd
import utilities.VASP as VASP
import math
import utilities.quantum_mechanics as qm
import numpy as np
import scipy as sc
from scipy.sparse.linalg import eigs
from numpy import linalg as LA
control_file_path = 'data/'

control_data = pd.read_csv( control_file_path + 'control.csv' , index_col='case')

def read_lattice(control, symm_type):
    latt = VASP.POSCAR_data( control.data_folder + control[symm_type+'_path']).lattice
    latt.get_dope_ions().m = control.dopant_m
    latt.get_maj_ions().m = control.maj_m
    latt.energy = control[symm_type+'_energy']

    return latt

# Create the symm electron system

sz = np.matrix([[1,0],[0,-1]], dtype= np.float64)
sx = np.matrix([[0,1],[1,0]], dtype= np.float64)
s0 = np.matrix([[1,0],[0,1]], dtype= np.float64)

symm_ops = {}

symm_ops['sz'] = qm.MatrixOperator(sz, name = 'sz')
symm_ops['sx'] = qm.MatrixOperator(sx, name = 'sx')
symm_ops['s0'] = qm.MatrixOperator(s0, name = 's0')



el_states = qm.symmetric_electron_system(symm_ops)


symm_types = ['symm_geom', 'less_symm_geom_1','less_symm_geom_2']


"""
[[ 0.        +6.09296912e-18j  0.01573528-1.18673920e-01j]
 [-0.01573528+1.18673920e-01j  0.        -9.31775486e-19j]]
[ 0.11867392+0.01573528j -0.11867392-0.01573528j]
0.11867391966166024

"""


for case_name in control_data.index:

    control = control_data.loc[case_name]

    symm_lattice = read_lattice(control, 'symm_geom')

    less_symm_lattice_1 = read_lattice(control, 'less_symm_geom_1')

    less_symm_lattice_2 = read_lattice(control, 'less_symm_geom_2')

    JT_theory = qm.Jahn_Teller_Theory(symm_lattice, less_symm_lattice_1, less_symm_lattice_2)

    print(JT_theory)

    JT_theory.E_b = 21.6
    JT_theory.calc_E_b = 1.6
    JT_theory.quantum = 78.8

    #el_states = qm.symmetric_electron_system()


    fonon_sys = qm.CoupledHarmOscOperator(5)
    fonon_sys = qm.AbsCoupledHarmOscOperator(2,4)
    print(fonon_sys.eig_states._states)
    

    JT_int = qm.Jahn_Teller_interaction(JT_theory.JT_pars,el_states,fonon_sys)


    np.savetxt('Hpy.csv',JT_int.H_int.matrix)



    #vals, vecs = LA.eig(JT_int.H_int.matrix,)
    vals, vecs = eigs(JT_int.H_int.matrix, which='SM', k=30)

    JT_eigen_states = qm.eigen_state.from_vals_vects(vals, vecs)

    

    print(vals)

    print(sorted(vals))



    
    Lz_op = qm.MatrixOperator.create_Lz_op()

    id_op = qm.MatrixOperator.create_id_op(len(fonon_sys.get_ham_op()))

    pert_ham = id_op.interaction_with(Lz_op)
    print(pert_ham.matrix)

    Lz_perturbation = qm.FirstOrderPerturbation([ JT_eigen_states[0], JT_eigen_states[1] ], pert_ham)
    print('eigen_state 0: ' + str(JT_eigen_states[0]))
    print('eigen_state 1: ' + str(JT_eigen_states[1]))

    print(Lz_perturbation.pert_op.matrix)
    np.savetxt('LzFULL_python.csv', pert_ham.matrix,delimiter=',')

    print(Lz_perturbation.pert_op.eigen_vals)

    print(Lz_perturbation.get_reduction_factor())

