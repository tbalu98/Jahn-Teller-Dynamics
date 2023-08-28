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


#Create the possible eigen states of the 2D harmonic oscillator
eig_states = qm.CoupledHarmOscEigStates(order)


# Build the basic creator and annihilator operators
op_builder = qm.Operator_builder(eig_states)

creator_x = op_builder.create_operator(qm.creator_x_sandwich)
annil_x = op_builder.create_operator(qm.annil_x_sandwich)
  
creator_y = op_builder.create_operator(qm.creator_y_sandwich)
annil_y = op_builder.create_operator(qm.annil_y_sandwich)



# 2D coupled oscillator's hamiltonian

H_osc_x = np.matmul(creator_x,annil_x)

H_osc_y = np.matmul(creator_y,annil_y)

K = np.round(H_osc_x + H_osc_y, 0)

# Operators for the whole system's hamiltonin

X = (creator_x + annil_x)/(2**0.5)

Y = (creator_y + annil_y)/(2**0.5)

np.savetxt(op_folder+  'X.csv', X)

np.savetxt(op_folder+  'Y.csv', Y)
XY = np.matmul(X,Y)
YX = np.matmul(Y,X)
XX = np.matmul(X,X)

YY = np.matmul(Y,Y)


sz = np.matrix([[1,0],[0,-1]], dtype= np.float64)
sx = np.matrix([[0,1],[1,0]], dtype= np.float64)
s0 = np.matrix([[1,0],[0,1]], dtype= np.float64)

H = omega* np.kron(K, s0) + F*( np.kron(X,sz) + np.kron(Y, sx)) + 1.0*G*(np.kron((XX-YY) ,sz) - np.kron(XY + YX, sx))

# The system's Hamilton operator

H_op =  qm.MatrixOperator(H)

print(H_op.eigen_vals)
print(H_op.eigen_vects)

Psi1 = H_op.get_eigen_vect(0)
Psi2 = H_op.get_eigen_vect(1)

print(H_op.calc_sandwich(Psi1, Psi2))

np.savetxt('Ham.csv', H)

#Spin angular momentum interaction
print( qm.buildSpinAngIntMO(len(X), Psi1, Psi2))



#eigen_vals, eigen_vects = LA.eig(H)
#print(eigen_vals)
#print(eigen_vects)