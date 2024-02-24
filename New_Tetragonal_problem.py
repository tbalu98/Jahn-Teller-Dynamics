import utilities.matrix_formalism as mf
import numpy as np
import utilities.jahn_teller_theory as jt
import utilities.maths as maths
res_folder = 'hamiltonians/'

# Parameters


omega_T = 73.060
F_T = 132.64
F_E = 144.30

#system of quantum harmonic oscillators

dim = 5
order = 5

fonon_sys = mf.n_dim_harm_osc(dim, order)

print(fonon_sys.over_est_int_op.matrix)

#tetragonal symmetric electron system

# symmetrical operators
#orbital operators

O0= mf.MatrixOperator(maths.Matrix( np.matrix([[1, 0, 0 ], [0, 1 ,0],[0, 0, 1]  ],dtype= np.float64)),'O0')

Ox=mf.MatrixOperator(maths.Matrix(np.matrix([[0, 0, 0 ], [0, 0 ,1],[0, 1, 0]  ],dtype= np.float64)), 'Ox')
Oy=mf.MatrixOperator(maths.Matrix(np.matrix([[0, 0, 1 ], [0, 0 ,0],[1, 0, 0]  ],dtype= np.float64)),'Oy')
Oz= mf.MatrixOperator(maths.Matrix( np.matrix([[0, 1, 0 ], [1, 0 ,0],[0, 0, 0]  ],dtype= np.float64)), 'Oz')

Ov= mf.MatrixOperator(maths.Matrix(0.5* np.matrix([[1, 0, 0 ], [0, 1 ,0],[0, 0, -2]  ],dtype= np.float64)), 'Ov')

Ow= mf.MatrixOperator(maths.Matrix(((3**0.5)/2)* np.matrix([[-1, 0, 0 ], [0, 1 ,0],[0, 0, 0]  ],dtype= np.float64)), 'Ow')

symm_ops = {str:mf.MatrixOperator}

symm_ops['O0'] = O0
symm_ops['Ox'] = Ox
symm_ops['Oy'] = Oy
symm_ops['Oz'] = Oz
symm_ops['Ov'] = Ov
symm_ops['Ow'] = Ow




el_sys = mf.symmetric_electron_system(symm_ops)


tet_jt = jt.Tet_JT_int(F_T, F_E,omega_T, el_sys, fonon_sys)

H = tet_jt.H_int
H.calc_eigen_vals_vects()
print(sorted( [  x.eigen_val for x in H.eigen_kets ] ))