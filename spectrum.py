import numpy as np
import collections
import utilities.quantum_mechanics as qm
from numpy import linalg as LA
#Folders
op_folder = 'results/operators/'

#Test
#Jahn-Teller parameters:

EJT = 50
delta = 10
omega = 60


order = 4

F = ( 2*EJT*omega*(1-delta/(2*EJT-delta)) )**0.5
G = omega*delta/(4*EJT - 2*delta)


harm_osc_2D = qm.CoupledHarmOscOperator(4)


sz = np.matrix([[1,0],[0,-1]], dtype= np.float64)
sx = np.matrix([[0,1],[1,0]], dtype= np.float64)
s0 = np.matrix([[1,0],[0,1]], dtype= np.float64)

H = omega* np.kron(harm_osc_2D.int_op.matrix, s0) + F*( np.kron(harm_osc_2D.X.matrix,sz) + np.kron(harm_osc_2D.Y.matrix, sx)) + 1.0*G*(np.kron((harm_osc_2D.XX.matrix-harm_osc_2D.YY.matrix) ,sz) - np.kron(harm_osc_2D.XY.matrix + harm_osc_2D.YX.matrix, sx))

# The system's Hamilton operator

H_op =  qm.MatrixOperator(H)

print(H_op.eigen_vals)
print(H_op.eigen_vects)

Psi1 = H_op.get_eigen_vect(0)
Psi2 = H_op.get_eigen_vect(1)

print(H_op.calc_sandwich(Psi1, Psi2))

np.savetxt('Ham.csv', H)

#Spin angular momentum interaction
print( qm.buildSpinAngIntMO(len(harm_osc_2D.X.matrix), Psi1, Psi2))
