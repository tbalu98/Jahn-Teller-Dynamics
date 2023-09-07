import utilities.quantum_mechanics as qm
import numpy as np


res_folder = 'hamiltonians/'

# Parameters


omega_T = 73.060
F_T = 132.64
F_E = 144.30

#system of quantum harmonic oscillators

dim = 5
order = 5

fonon_sys = qm.n_dim_harm_osc(dim, order)

print(fonon_sys.over_est_int_op.matrix)

#tetragonal symmetric electron system

# symmetrical operators
#orbital operators

O0= qm.MatrixOperator( np.matrix([[1, 0, 0 ], [0, 1 ,0],[0, 0, 1]  ]),'O0')

Ox=qm.MatrixOperator(np.matrix([[0, 0, 0 ], [0, 0 ,1],[0, 1, 0]  ]), 'Ox')
Oy=qm.MatrixOperator(np.matrix([[0, 0, 1 ], [0, 0 ,0],[1, 0, 0]  ]),'Oy')
Oz= qm.MatrixOperator( np.matrix([[0, 1, 0 ], [1, 0 ,0],[0, 0, 0]  ]), 'Oz')

Ov= qm.MatrixOperator(0.5* np.matrix([[1, 0, 0 ], [0, 1 ,0],[0, 0, -2]  ]), 'Ov')

Ow= qm.MatrixOperator(((3**0.5)/2)* np.matrix([[-1, 0, 0 ], [0, 1 ,0],[0, 0, 0]  ]), 'Ow')

symm_ops = {str:qm.MatrixOperator}

symm_ops['O0'] = O0
symm_ops['Ox'] = Ox
symm_ops['Oy'] = Oy
symm_ops['Oz'] = Oz
symm_ops['Ov'] = Ov
symm_ops['Ow'] = Ow




el_sys = qm.symmetric_electron_system(symm_ops)


#H=omega_T*kron(K+0*2.5*eye(l),O0)-F_T*(kron(X,Ox)+kron(Y,Oy)+kron(Z,Oz))  +F_E*(kron(V,Ov)+kron(W,Ow));

#H=omega_T*kron(K+0*2.5*eye(l),O0)-F_T*(kron(X,Ox)+kron(Y,Oy)+kron(Z,Oz))  +F_E*(kron(V,Ov)+kron(W,Ow));

#np.savetxt('X_python.csv', fonon_sys.pos_i_ops[0].matrix)
K = fonon_sys.get_ham_op().matrix
K_over = fonon_sys.over_est_int_op.matrix


X =  fonon_sys.get_pos_i_op(0).matrix
Y =  fonon_sys.get_pos_i_op(1).matrix
Z =  fonon_sys.get_pos_i_op(2).matrix
V =  fonon_sys.get_pos_i_op(3).matrix
W =  fonon_sys.get_pos_i_op(4).matrix


H = omega_T* np.kron(K, el_sys.symm_ops['O0'].matrix) - F_T*  (np.kron( X, el_sys.symm_ops['Ox'].matrix ) + np.kron(Y , el_sys.symm_ops['Oy'].matrix)  + np.kron(Z, el_sys.symm_ops['Oz'].matrix )) + F_E* (  np.kron(V , el_sys.symm_ops['Ov'].matrix)  + np.kron(W , el_sys.symm_ops['Ow'].matrix ) ) 





H = qm.MatrixOperator(H)
H.eigen_vals.sort()
np.savetxt('eig_val.csv', H.eigen_vals)
print(H.eigen_vals)

