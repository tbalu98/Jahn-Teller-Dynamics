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
from scipy.constants import physical_constants

import warnings
warnings.simplefilter("ignore", np.ComplexWarning)





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





order  = int(sys.argv[1])

l  =  float(sys.argv[2])

#order =12

JT_theory = jt.Jahn_Teller_Theory()

filenames = None

arguments = sys.argv[1:]

save_raw_pars = False

if arguments[-1] == '-save_raw_pars':
    save_raw_pars = True
    arguments.pop()

if len(arguments) == 5:# or len(sys.argv) ==7:
    if maths.isFloat(arguments[2]) and maths.isFloat(arguments[3]) and maths.isFloat(arguments[4]):
        JT_theory = jt.Jahn_Teller_Theory().from_Taylor_coeffs(hw=float(arguments[2]), F = float(arguments[3]), G = float(arguments[4]))
    else:
        filenames = [arguments[2],arguments[3],arguments[4] ]

elif len(arguments) == 6:

    JT_theory = jt.build_jt_theory_from_csv([arguments[2],arguments[3],arguments[4],arguments[5]])

elif len(arguments)==4:
    filenames = [arguments[2],arguments[3]]

if filenames !=None:

    if filenames[0][-3:]=="xml":
        symm_lattice = utilities.xml_parser.xml_parser(filenames[0]).lattice
        less_symm_lattice_1 = utilities.xml_parser.xml_parser(filenames[1]).lattice
        less_symm_lattice_2 = utilities.xml_parser.xml_parser(filenames[2]).lattice if len(filenames)==3 else None
        JT_theory = jt.Jahn_Teller_Theory(symm_lattice, less_symm_lattice_1, less_symm_lattice_2)

    elif filenames[0][-3:]=="csv":
        JT_theory = jt.build_jt_theory_from_csv(filenames)



    if save_raw_pars == True:
        utilities.xml_parser.save_raw_data_from_xmls([symm_lattice, less_symm_lattice_1, less_symm_lattice_2])


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


JT_int = qmp.Exe_tree(point_defect_tree, JT_theory)


JT_int.create_one_mode_hamiltonian()
    
JT_int.H_int.calc_eigen_vals_vects()

res_df = JT_int.H_int.create_eigen_kets_vals_table(JT_int.system_tree.root_node.base_states)

res_df.to_csv('Eigen_vectors_no_so_coupling.csv',sep = ';')
print('-------------------------------')

print('Eigen values of the Jahn-Teller interaction without spin-orbit coupling')

for x in JT_int.H_int.eigen_kets[0:10]:
    print(str(round(x.eigen_val.real,4)) + ' meV') 


#print( [  x.eigen_val for x in JT_int.H_int.eigen_kets ] )



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

print('Reduction factor:')

print('p = '+ str( round(deg_sys.add_perturbation(pert_ham),4)) + ' meV')

    


if l>0.0 or l < 0.0:

    point_defect_tree.insert_node('electron_system', spin_sys)

    JT_int = qmp.Exe_tree(point_defect_tree, JT_theory)


    JT_int.create_one_mode_hamiltonian()




    JT_int.add_spin_orbit_coupling(l)
    JT_int.H_int.calc_eigen_vals_vects()
    res_df = JT_int.H_int.create_eigen_kets_vals_table(JT_int.system_tree.root_node.base_states)

    res_df.to_csv('Eigen_vectors_so_coupling.csv',sep = ';')
    print('-------------------------------')

    print('Eigen values of the Jahn-Teller interaction with spin-orbit coupling')

    for x in JT_int.H_int.eigen_kets[0:10]:
        print(str(round(x.eigen_val.real,4)) + ' meV') 



    print('-------------------------------')

    print('p Ham reduction factor')

    print( 'lambda_Ham = ' + str(round((JT_int.H_int.eigen_kets[3].eigen_val- JT_int.H_int.eigen_kets[0].eigen_val).real,4)) + ' meV')

