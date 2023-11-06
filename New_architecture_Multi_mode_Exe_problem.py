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

symm_ops = {}

symm_ops['sz'] = mf.MatrixOperator(maths.Matrix(sz), name = 'sz')
symm_ops['sx'] = mf.MatrixOperator(maths.Matrix(sx), name = 'sx')
symm_ops['s0'] = mf.MatrixOperator(maths.Matrix(s0), name = 's0')



complex_basis = [ mf.ket_vector([ 1.0, complex(0.0, 1.0) ]), mf.ket_vector([1.0, complex(0.0, -1.0)]) ]

el_sys = qmp.electron_system([ 'e-', 'e+' ], symm_ops,complex_basis)



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

    #mode = 78.8

    spatial_dim = 2
    order  = 12
    #order = 4

    ph_sys_78 = qmp.one_mode_phonon_sys(mode,spatial_dim,order,['x','y'], 'mode1' )
    

    mm_ph_sys = qmp.multi_mode_phonon_system([ph_sys_78])

    JT_theory.E_JT = 41.8
    JT_theory.delta  = 9.1

    JT_int = qmp.Exe_phonon_electron_system(mm_ph_sys,el_sys,JT_theory)

    
    JT_int.H_int.save('H_int_JT.csv')
    
    JT_int.H_int.calc_eigen_vals_vects()

    #JT_eigen_states = qm.eigen_vect.from_vals_vects( JT_int.H_int.eigen_vals, JT_int.H_int.eigen_vects)

    print('Eigen values of the Jahn-Teller interaction')
    print( [  x.eigen_val for x in JT_int.H_int.eigen_kets ] )


    ground_1 = JT_int.H_int.eigen_kets[0]

    ground_2 = JT_int.H_int.eigen_kets[1]

    #New electron eigen vectors

    eminus_el_sys = mf.ket_vector([complex(1.0,0.0), complex(0.0, -1.0) ])/(2.0**0.5)
    eplus_el_sys = mf.ket_vector([complex(1.0,0.0), complex(0.0, 1.0) ])/(2.0**0.5)

    #eminus_el_sys = mf.ket_vector([complex(1.0,0.0), complex(1.0, 0.0) ])/(2.0**0.5)
    #eplus_el_sys = mf.ket_vector([complex(  0.0,-1.0), complex(  0.0,-1.0) ])/(2.0**0.5)

    # basis transformations
    ph_sys_basis_trf = ph_sys_78.create_complex_basis_trf()

    ph_sys_basis_trf.save('ph_sys_complex_trf_matrix.csv')

    el_basis_trf = mf.MatrixOperator.basis_trf_matrix([ eminus_el_sys, eplus_el_sys  ])



    el_basis_trf.save('el_complex_trf_matrix.csv')

    ph_el_sys_bases_trf = ph_sys_basis_trf**el_basis_trf

    ph_el_sys_bases_trf.save('ph_el_complex_trf_matrix.csv')


    #2D degenrate ground state system

    deg_sys = mf.degenerate_system_2D( [ground_1,ground_2] )

    Lz_op = mf.MatrixOperator.create_Lz_op()

    id_op = mf.MatrixOperator.create_id_matrix_op( ph_sys_78.h_sp_dim )

    pert_ham = id_op**(Lz_op)

    
    """

    print('red factors')

    print(deg_sys.add_perturbation(pert_ham))
    deg_sys.to_complex_basis(ph_el_sys_bases_trf)

    eminus_prob = deg_sys.complex_deg_ket_vectors[0]
    eplus_prob = deg_sys.complex_deg_ket_vectors[1]

    op = mf.MatrixOperator.from_ket_vectors([ eminus_prob, eplus_prob ])

    op.save('psi_minus_plus.csv')

    """
    Psiplus = (ground_1+complex(0.0,1.0)*ground_2)/(2.0)**0.5
    Psiminus = (ground_1-complex(0.0,1.0)*ground_2)/(2.0)**0.5

    # basis transformations
    ph_sys_basis_trf = ph_sys_78.create_complex_basis_trf()

    ph_sys_basis_trf.save('ph_sys_complex_trf_matrix.csv')

    el_basis_trf = mf.MatrixOperator.basis_trf_matrix([ eminus_el_sys, eplus_el_sys  ])

    #el_basis_trf = mf.MatrixOperator( maths.Matrix(el_basis_trf.matrix.matrix.transpose()))




    el_basis_trf_matrix = mf.MatrixOperator.create_id_matrix_op(dim = ph_sys_78.h_sp_dim ) **el_basis_trf

    ph_sys_basis_trf_matrix = ph_sys_basis_trf**mf.MatrixOperator.create_id_matrix_op(dim = 2)

    #Psiplus_ph_el_sys_trfed = ph_sys_basis_trf_matrix * Psiplus

    Psiplus_ph_el_sys_trfed = el_basis_trf_matrix*ph_sys_basis_trf_matrix*Psiplus
    Psiminus_ph_el_sys_trfed = el_basis_trf_matrix*ph_sys_basis_trf_matrix*Psiminus

    op = mf.MatrixOperator.from_ket_vectors([  Psiminus_ph_el_sys_trfed, Psiplus_ph_el_sys_trfed ])
    op.save('psi_minus_plus.csv')
