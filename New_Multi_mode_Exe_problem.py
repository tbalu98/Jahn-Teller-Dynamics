import pandas as pd
import utilities.VASP as VASP
import math
import utilities.jahn_teller_theory as jt
import utilities.maths as  maths
import utilities.new_quantum_mechanics as qm
import numpy as np
import scipy as sc
from scipy.sparse.linalg import eigs
from numpy import linalg as LA
import matplotlib.pyplot as plt
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

symm_ops['sz'] = qm.MatrixOperator(maths.Matrix(sz), name = 'sz')
symm_ops['sx'] = qm.MatrixOperator(maths.Matrix(sx), name = 'sx')
symm_ops['s0'] = qm.MatrixOperator(maths.Matrix(s0), name = 's0')



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

    sec_fonon_modes = np.linspace(60, 80, 21)
    #sec_fonon_modes = [78.8]
    lowest_eig_energies = []

    mode = 78.79527995410930

    print(mode)
    #Create the phonon system
    fonon_sys78 = qm.n_dim_harm_osc(2,4,energy = 78.79527995410929)
    
    #Create the phonon system
    fonon_sys781 = qm.n_dim_harm_osc(2,4,energy = mode)

    mm_fon_sys = qm.fast_multimode_fonon_sys({ 78.79527995410929: fonon_sys78, mode: fonon_sys781 })

    #Calculate the fonon electron interaction's hamiltonian
    JT_int = jt.multi_mode_Exe_jt_int(JT_theory,el_states,mm_fon_sys)


    vals, vecs = eigs(JT_int.H_int.matrix.matrix, which='SM', k=len(JT_int.H_int))

    JT_eigen_states = qm.eigen_vect.from_vals_vects(vals, vecs)

    print('Eigen values of the Jahn-Teller interaction')
    print(sorted(vals))



    
    