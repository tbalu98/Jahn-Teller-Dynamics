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

symm_ops = {}

symm_ops['sz'] = mf.MatrixOperator(maths.Matrix(sz), name = 'sz')
symm_ops['sx'] = mf.MatrixOperator(maths.Matrix(sx), name = 'sx')
symm_ops['s0'] = mf.MatrixOperator(maths.Matrix(s0), name = 's0')



complex_basis = [ mf.ket_vector([ 1.0, complex(0.0, 1.0) ]), mf.ket_vector([1.0, complex(0.0, -1.0)]) ]

#el_sys = qmp.electron_system([ 'e-', 'e+' ], symm_ops,complex_basis)

electron_system = qs.quantum_system_node('electron_system',base_states=mf.hilber_space_bases().from_qm_nums_list([ ['e-'],[ 'e+']],qm_nums_names=['orbital'])  ,
                                         operators=symm_ops, dim= 2)



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
    order = 4

    mode_1 = qmp.one_mode_phonon_sys(mode,spatial_dim,order,['x','y'], 'mode_1', 'mode_1' )
    mode_2 = qmp.one_mode_phonon_sys(mode,spatial_dim,order,['x','y'], 'mode_2', 'mode_2' )

    #Create quantum system:



    nuclei = qs.quantum_system_node('nuclei', children=[mode_1,mode_2])

    point_defect_node = qs.quantum_system_node('point_defect', 
                                               children = [electron_system, nuclei])

    point_defect_tree = qs.quantum_system_tree(point_defect_node)

    #point_defect_tree.insert_node('nuclei', mode_1)

    point_defect_node.base_states.savetxt('states.txt')

    JT_theory.E_JT = 41.8
    JT_theory.delta  = 9.1

    JT_int = qmp.Exe_tree(point_defect_tree, JT_theory)


    JT_int.create_multi_mode_hamiltonian()

    #JT_int.create_one_mode_hamiltonian()

    

    print('fin')
    
    
    JT_int.H_int.calc_eigen_vals_vects()

    #JT_eigen_states = qm.eigen_vect.from_vals_vects( JT_int.H_int.eigen_vals, JT_int.H_int.eigen_vects)

    print('Eigen values of the Jahn-Teller interaction')
    print( [  x.eigen_val for x in JT_int.H_int.eigen_kets ] )

    print(len(JT_int.H_int.eigen_kets))


