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


filenames = [sys.argv[1], sys.argv[2], sys.argv[3]]

#filenames = ['SnV_D3d_vasprun.xml', 'SnV_C2h_JT_vasprun.xml','SnV_C2h_barrier_vasprun.xml']


res_data = {}


symm_lattice = utilities.xml_parser.xml_parser(filenames[0]).lattice


less_symm_lattice_1 = utilities.xml_parser.xml_parser(filenames[1]).lattice
less_symm_lattice_2 = utilities.xml_parser.xml_parser(filenames[2]).lattice if filenames[2]!=None else None

    #Calculate the parameters of Jahn-Teller theory
JT_theory = jt.Jahn_Teller_Theory(symm_lattice, less_symm_lattice_1, less_symm_lattice_2)

print(JT_theory)

res_data['hw'] = [JT_theory.hw]
res_data['JT energy'] = [JT_theory.E_JT]
res_data['barrier energy'] = [JT_theory.E_b]


res_data['dopant atom'] = [ symm_lattice.ions_arr[0].name if len(symm_lattice.ions_arr[0]._vecs)<len(symm_lattice.ions_arr[1]._vecs) else symm_lattice.ions_arr[1].name]

res_data['majority atom'] = [ symm_lattice.ions_arr[0].name if len(symm_lattice.ions_arr[0]._vecs)>len(symm_lattice.ions_arr[1]._vecs) else symm_lattice.ions_arr[1].name]


#Calculate distances

print("Symmetric lattice, Jahn Teller lattice distances")
print(symm_lattice.ions_arr[0].name)
print( symm_lattice.ions_arr[0].calc_dist_sq(less_symm_lattice_1.ions_arr[0]))

res_data['symm-JT distance ' + symm_lattice.ions_arr[0].name + ' atoms'] = [math.sqrt(symm_lattice.ions_arr[0].calc_dist_sq(less_symm_lattice_1.ions_arr[0]))] 

print(symm_lattice.ions_arr[1].name)
print( symm_lattice.ions_arr[1].calc_dist_sq(less_symm_lattice_1.ions_arr[1]))


res_data['symm-JT distance ' + symm_lattice.ions_arr[1].name + ' atoms'] = [math.sqrt(symm_lattice.ions_arr[1].calc_dist_sq(less_symm_lattice_1.ions_arr[1]))] 


print("Symmetric lattice, Barrier lattice distances")
print(symm_lattice.ions_arr[0].name)
print( symm_lattice.ions_arr[0].calc_dist_sq(less_symm_lattice_2.ions_arr[0]))

res_data['symm-barrier distance ' + symm_lattice.ions_arr[0].name + ' atoms'] = [math.sqrt(symm_lattice.ions_arr[0].calc_dist_sq(less_symm_lattice_2.ions_arr[0]))] 



print(symm_lattice.ions_arr[1].name)
print( symm_lattice.ions_arr[1].calc_dist_sq(less_symm_lattice_2.ions_arr[1]))

res_data['symm-barrier distance ' + symm_lattice.ions_arr[1].name + ' atoms'] = [math.sqrt(symm_lattice.ions_arr[1].calc_dist_sq(less_symm_lattice_2.ions_arr[1]))] 

res_data['case_name'] = ['case_1']

res_data[symm_lattice.ions_arr[0].name+' mass'] = [symm_lattice.ions_arr[0].m]

res_data[symm_lattice.ions_arr[1].name+' mass'] = [symm_lattice.ions_arr[1].m]


import pandas as pd

res_df = pd.DataFrame.from_dict(res_data)
res_df = res_df.set_index('case_name')
res_df.to_csv('JT_data_results.csv')
