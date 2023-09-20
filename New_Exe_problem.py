import pandas as pd
import utilities.VASP as VASP
import math
import utilities.matrix_quantum_mechanics as qm
import numpy as np
import scipy as sc
from scipy.sparse.linalg import eigs
from numpy import linalg as LA
import utilities.maths as maths
from utilities.maths import Matrix
import utilities.jahn_teller_theory as jt
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
sz = Matrix(sz)
sx = np.matrix([[0,1],[1,0]], dtype= np.float64)
sx = Matrix(sx)
s0 = np.matrix([[1,0],[0,1]], dtype= np.float64)
s0 = Matrix(s0)


symm_ops = {}

symm_ops['sz'] = qm.MatrixOperator(sz, name = 'sz')
symm_ops['sx'] = qm.MatrixOperator(sx, name = 'sx')
symm_ops['s0'] = qm.MatrixOperator(s0, name = 's0')



el_states = qm.symmetric_electron_system(symm_ops)


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

    #Create the phonon system
    fonon_sys = qm.n_dim_harm_osc(2,4)
    print('Eigen states of the phonon system')
    print(fonon_sys.eig_states._states)
    
    #Calculate the fonon electron interaction's hamiltonian
    JT_int = jt.Exe_JT_int(JT_theory.JT_pars,el_states,fonon_sys)

    JT_int.H_int.calc_eigen_vals_vects()
    #vals, vecs = eigs(JT_int.H_int.matrix, which='SM', k=30)

    JT_eigen_states = qm.eigen_vect.from_vals_vects(JT_int.H_int.eigen_vals , JT_int.H_int.eigen_vects)

    print('Eigen values of the Jahn-Teller interaction')
    print(sorted(JT_int.H_int.eigen_vals))


    
    #Add the spin orbit coupling interaction
    Lz_op = qm.MatrixOperator.create_Lz_op(maths.Matrix)

    id_op = qm.MatrixOperator.create_id_matrix_op( len(fonon_sys.get_ham_op()),maths.Matrix )

    pert_ham = id_op**(Lz_op)

    pert_ham.save('pert_ham_new.csv')

    print('-----------------')
    print(JT_eigen_states[0])
    print(JT_eigen_states[1])
    print('-----------------')


    Lz_perturbation = qm.FirstOrderPerturbation([ JT_eigen_states[0], JT_eigen_states[1] ], pert_ham)

    print(Lz_perturbation.pert_op.matrix.matrix)
    print('Eigen values of the perturbation')
    print(Lz_perturbation.pert_op.eigen_vals)
    print('reduction factor:')
    print(Lz_perturbation.get_reduction_factor())


