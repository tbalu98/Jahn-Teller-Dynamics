import numpy as np
import collections
from numpy import linalg as LA
import math
import utilities.VASP as VASP
import collections
import copy
from scipy.sparse.linalg import eigs

#Data structures
Eigen_state_2D = collections.namedtuple('Eigen_state',  'x_fonon y_fonon' )

Jahn_Teller_Pars = collections.namedtuple('Jahn_Teller_Pars',  'E_JT E_b hwpG hwmG hw F G ' )


class eigen_state:
     def from_vals_vects(vals, vecs):
          return [eigen_state( val, vec ) for val, vec in zip( vals, vecs )]
     
     def __init__(self, eig_val,coeffs):
          self.eig_val = eig_val
          self._coeffs = coeffs


     def __getitem__(self,key):
          return self._coeffs[key]
     def __len__(self):
          return len(self._coeffs)
     
     def __repr__(self) -> str:
          #return str(self.eig_val)
          return str(self.eig_val)  + ': ' + str(self._coeffs)




     
class abs_coupled_harm_ocs_eigen_states:
     def __init__(self,coeffs):
          self._coeffs = coeffs
     
     def __getitem__(self,key):
          return self._coeffs[key]
     def __len__(self):
          return len(self._coeffs)
     
     def __repr__(self) -> str:
          return str(self._coeffs)
     
     def equals_but(self, other,i:int):
          for j in range(0,len(self._coeffs)):
               if i==j:
                    continue
               else:
                    if self[j]!=other[j]:
                         return False
          
          return True
     
     def diff_in_coeff(self,other, i):
          return self[i] - other[i]
     
     def creator_i_sandwich(self, other, i: int):
          if self.equals_but(other, i) and self[i]==(other[i] + 1):
               return (other[i] + 1)**0.5
          else:
               return 0.0


     def annil_i_sandwich(self, other, i: int):
          if self.equals_but(other, i) and self[i]==(other[i] - 1):
               return (other[i])**0.5
          else:
               return 0.0





     def get_order(self):
          return sum(self)

class AbsCoupledHarmOscEigStates:
     def __init__(self, dim,order):
          self._states = []
          self.create_oscillators(dim,order, [])
          self._states.sort(key = lambda x: x.get_order())


     def create_oscillators(self, dim, order, curr_osc_coeffs: list):
          if len(curr_osc_coeffs)<dim:

               for i in range(0,order+1):
                    temp_curr_osc_coeffs = copy.deepcopy(curr_osc_coeffs)
                    temp_curr_osc_coeffs.append(i)
                    if sum(temp_curr_osc_coeffs)>order:
                         return
                    else:
                         self.create_oscillators(dim, order,temp_curr_osc_coeffs )
          elif len(curr_osc_coeffs) == dim and sum(curr_osc_coeffs)<=order:
               return self._states.append( abs_coupled_harm_ocs_eigen_states( curr_osc_coeffs))
          else:
               return


     def __len__(self):
          return len(self._states)
     
     def __getitem__(self, position):
          return self._states[position]
     
     def order(self):
          return sum(self)



class CoupledHarmOscEigStates:
     def __init__(self, order):
          self._states = []
          for E in range(0, order+1):
               for y_fonon in range(0,E+1):
                    x_fonon = E-y_fonon
                    self._states.append(Eigen_state_2D(x_fonon, y_fonon))
          print(self._states)
     def __len__(self):
          return len(self._states)
     
     def __getitem__(self, position):
          return self._states[position]

#Create operators

class AbsOperator_builder:
     def __init__(self, eig_states):
          self.eig_states = eig_states

     def create_operator(self, sandwich_fun):
          dim = len(self.eig_states)
          operator = np.zeros((dim, dim), dtype = np.float64)
          for i in range(0,len(self.eig_states)):
               for j in range(0,len(self.eig_states)):
                    operator[i][j] = sandwich_fun(self.eig_states[i], self.eig_states[j])
          return MatrixOperator(operator)
     


class Operator_builder:
     def __init__(self, eig_states: CoupledHarmOscEigStates):
          self.eig_states = eig_states

     def create_operator(self, sandwich_fun):
          dim = len(self.eig_states)
          operator = np.zeros((dim, dim), dtype = np.float64)
          for i in range(0,len(self.eig_states)):
               for j in range(0,len(self.eig_states)):
                    operator[i][j] = sandwich_fun(self.eig_states[i], self.eig_states[j])
          return operator



#Sandwich functions:
#<x_fonon', y_fonon'|ax-|x_fonon, y_fononon>
def annil_x_sandwich(eigen_state_1:Eigen_state_2D, eigen_state_2:Eigen_state_2D):
     if (eigen_state_1.x_fonon == eigen_state_2.x_fonon-1  and
         eigen_state_1.y_fonon==eigen_state_2.y_fonon ) is True:
          return (eigen_state_2.x_fonon)**0.5
     else:
          return 0


#<x_fonon', y_fonon'|ax+|x_fonon, y_fononon>
def creator_x_sandwich(eigen_state_1: Eigen_state_2D, eigen_state_2:Eigen_state_2D):
     if (eigen_state_1.x_fonon == eigen_state_2.x_fonon+1 and
         eigen_state_1.y_fonon==eigen_state_2.y_fonon )  is True:
          return (eigen_state_2.x_fonon+1)**0.5
     else:
          return 0

def annil_y_sandwich(eigen_state_1:Eigen_state_2D, eigen_state_2:Eigen_state_2D):
     if (eigen_state_1.y_fonon == eigen_state_2.y_fonon-1  and
         eigen_state_1.x_fonon==eigen_state_2.x_fonon ) is True:
          return (eigen_state_2.y_fonon)**0.5
     else:
          return 0

def creator_y_sandwich(eigen_state_1: Eigen_state_2D, eigen_state_2:Eigen_state_2D):
     if (eigen_state_1.y_fonon == eigen_state_2.y_fonon+1 and
         eigen_state_1.x_fonon==eigen_state_2.x_fonon )  is True:
          return (eigen_state_2.y_fonon+1)**0.5
     else:
          return 0





#Quantummechanical operator:
class MatrixOperator:

     

     def __init__(self, matrix, name = ""):
          self.name = name
          self.matrix = matrix
     
          self.calc_eigen_vals_vects()

     def __len__(self):
          return len(self.matrix)

     def create_Lz_op():
          Lz_mat = np.matrix([[0, complex(0,1)], [complex(0,-1), 0]], dtype=np.complex64)

          return MatrixOperator(Lz_mat)

     def create_id_op(n:int):
          return MatrixOperator(np.eye(n))

     def __get_item__(self,key):
          return self.matrix[key]

     def calc_eigen_vals_vects(self):
          self.eigen_vals, self.eigen_vects =  eigs(self.matrix, k = len(self.matrix), which = 'SM') #LA.eig(self.matrix)
          

     def get_eigen_vect(self, i):
          return np.array(self.eigen_vects[i])
     
     def get_eigen_val(self, i):
          return self.eigen_vals[i]
     
     def calc_sandwich(self, Phi1: np.array, Phi2: np.array):
          Phi1_tr = np.transpose(Phi1)
          
          return complex(np.matmul( Phi2, np.matmul( self.matrix, Phi1_tr ) ))
     
     def interaction_with(self, other):
          return MatrixOperator(np.kron( self.matrix, other.matrix ) )

class SpinAngMomOp(MatrixOperator):
     def __init__(self, dim, Ps1: np.array, Ps2: np.array):
          Lz = np.matrix([[0, complex(0,1)], [complex(0,-1), 0]], dtype=np.complex64)
     
          LzFull = np.kron(np.eye(dim), Lz )
     
          left = np.matrix( [ Ps1.flatten() , Ps2.flatten()] )
          right = np.transpose(left)

          mat= np.matmul( left,  np.matmul( LzFull, right ))

          super().__init__(mat)

def buildSpinAngIntMO(dim, Ps1: np.array, Ps2: np.array):
     Lz = np.matrix([[0, complex(0,1)], [complex(0,-1), 0]], dtype=np.complex64)
     
     LzFull = np.kron(np.eye(dim), Lz )
     
     left = np.matrix( [ Ps1.flatten() , Ps2.flatten()] )
     right = np.transpose(left)

     return np.matmul( left,  np.matmul( LzFull, right ))

class FirstOrderPerturbation:
     
     def __init__(self,deg_eigen_vecs: list[eigen_state], ham_comma: MatrixOperator):
          self.deg_eigen_vecs = deg_eigen_vecs
          self.ham_comma = ham_comma
          self.create_pert_op()

     
     def create_pert_op(self):
          left = np.matrix( [ x._coeffs.flatten() for x in self.deg_eigen_vecs] )
          right = np.transpose(left)

          raw_pert_mat = np.matmul(left, np.matmul( self.ham_comma.matrix, right ))



          self.pert_op = MatrixOperator(raw_pert_mat)

          self.pert_eigen_vals = self.pert_op.eigen_vals

          return self.pert_op

     def get_reduction_factor(self):
          return abs( (self.pert_eigen_vals[1] - self.pert_eigen_vals[0]).real/2 )


class AbsCoupledHarmOscOperator:
     def __init__(self,  dim,order):
          self.dim = dim
          self.order = order
          self.eig_states = AbsCoupledHarmOscEigStates(self.dim,self.order)
          self.op_builder = AbsOperator_builder(self.eig_states)
          self.build_creator_ops()
          self.build_annil_ops()
          self.build_H_i_ops()
          self.build_whole_sys_op()
          print(self.int_op)
          self.create_pos_i_ops()
          self.create_pos_i_sq_ops()

          
     def build_creator_ops(self):
          self.creator_ops = []
          for i in range(0,self.dim):
               creator_i_op = self.op_builder.create_operator( lambda x,y: x.creator_i_sandwich(y,i) )
               self.creator_ops.append(creator_i_op)
     
     def build_annil_ops(self):

          self.annil_ops = []
          for i in range(0,self.dim):
               annil_i_op = self.op_builder.create_operator( lambda x,y: x.annil_i_sandwich(y,i) )
               self.annil_ops.append(annil_i_op)
     
     def build_H_i_ops(self):
          self.H_i_ops = []
          for i in range(0, self.dim):
               H_i = np.matmul(self.creator_ops[i].matrix, self.annil_ops[i].matrix)
               self.H_i_ops.append(MatrixOperator(H_i))
               np.savetxt('H_osc_'+str(i)+'_new.csv', H_i)


     def build_whole_sys_op(self):
               
          int_op = np.matrix( np.round( sum( [x.matrix for x in self.H_i_ops ] ) ), dtype=np.int16 )
          
          self.int_op = MatrixOperator(int_op)

     def create_pos_i_ops(self):
          self.pos_i_ops = []
          for i in range(0,self.dim):
               pos_i_mat = (self.creator_ops[i].matrix + self.annil_ops[i].matrix)/(2**0.5)
               self.pos_i_ops.append( MatrixOperator( pos_i_mat ) )

     def create_pos_i_sq_ops(self):
          self.pos_i_sq_ops = []
          for i in range(0,self.dim):
               pos_i_sq_op = np.matmul(self.pos_i_ops[i].matrix, self.pos_i_ops[i].matrix)
               self.pos_i_sq_ops.append( MatrixOperator(pos_i_sq_op))



class CoupledHarmOscOperator:
     def __init__(self, order):
          eig_states = CoupledHarmOscEigStates(order)
          op_builder = Operator_builder(eig_states)

          self.creator_x = op_builder.create_operator(creator_x_sandwich)
          self.annil_x = op_builder.create_operator(annil_x_sandwich)
  
          self.creator_y = op_builder.create_operator(creator_y_sandwich)
          self.annil_y = op_builder.create_operator(annil_y_sandwich)

          H_osc_x = np.matmul(self.creator_x,self.annil_x)

          H_osc_y = np.matmul(self.creator_y,self.annil_y)

          K = np.matrix(np.round(H_osc_x + H_osc_y, 0))

          self.int_op = MatrixOperator(K)
          self.create_essention_operators()
     
     def __len__(self):
          return len(self.int_op.matrix)

     def create_essention_operators(self):

          X = (self.creator_x + self.annil_x)/(2**0.5)
          self.X = MatrixOperator(X)
          Y = (self.creator_y + self.annil_y)/(2**0.5)
          self.Y = MatrixOperator(Y)

          XY = np.matmul(X,Y)
          #XY = np.matmul(Y,X)
          
          self.XY=MatrixOperator(XY)
          

          YX =  np.matmul(Y,X)
          self.YX = MatrixOperator(YX)

          
          XX = np.matmul(X,X)
          self.XX = MatrixOperator(XX)
          YY = np.matmul(Y,Y)
          self.YY  =MatrixOperator(YY)




class Jahn_Teller_Theory:


     def __init__(self, symm_lattice: VASP.Lattice, less_symm_lattice_1: VASP.Lattice, less_symm_lattice_2:VASP.Lattice):
          self.symm_lattice = symm_lattice
          self.JT_lattice = less_symm_lattice_1 if less_symm_lattice_1.energy< less_symm_lattice_2.energy else less_symm_lattice_2
          self.barrier_lattice = less_symm_lattice_1 if less_symm_lattice_1.energy> less_symm_lattice_2.energy else less_symm_lattice_2
          self.calc_paramters()



     def __repr__(self) -> str:
          return 'Jahn-Teller energy: ' + str(self.E_JT) + '\n' + 'Barrier energy: '  + str(self.E_b) + '\n' + 'hw+G: ' + str(self.quantum_pG) + '\n' + 'hw-G: ' + str(self.quantum_mG) + '\n' + 'hw: '+ str(self.quantum) 

     def calc_dists(self):
          self.JT_dist = self.symm_lattice.calc_dist(self.JT_lattice)
          self.barrier_dist = self.symm_lattice.calc_dist(self.barrier_lattice)

     def calc_E_JT(self):
          self.E_JT = abs(self.JT_lattice.energy - self.symm_lattice.energy)*1000
     def calc_E_b(self):
          self.E_b = abs( self.JT_lattice.energy - self.barrier_lattice.energy)*1000


     def calc_paramters(self):
          self.calc_dists()
          c = 64.654148236
          self.calc_E_JT()
          self.calc_E_b()
          self.delta = self.E_JT - self.E_b
          #self.E_JT = abs(self.JT_lattice.energy - self.symm_lattice.energy)*1000
          
          #self.E_b = abs( self.JT_lattice.energy - self.barrier_lattice.energy)*1000

          self.quantum_mG = c*( 2*(-abs( self.E_b/1000 ) + abs(self.E_JT/1000) ) / self.JT_dist**2  )**0.5

          self.quantum_pG = c*( 2*(abs( self.E_JT/1000 ) ) / self.barrier_dist**2 )**0.5
          self.quantum = (self.quantum_mG + self.quantum_pG)/2
          self.calc_Taylor_coeffs()
          self.JT_pars= Jahn_Teller_Pars(self.E_JT, self.E_b, self.quantum_pG, self.quantum_mG, self.quantum, self.F, self.G)

     
     def calc_Taylor_coeffs(self):
          self.F = ( 2*self.E_JT*self.quantum*(1-self.delta/(2*self.E_JT-self.delta)) )**0.5
          self.G = self.quantum*self.delta/(4*self.E_JT - 2*self.delta)




class symmetric_electron_system:
     def __init__(self, symm_ops: dict):
          self.symm_ops = symm_ops
          """
     def __init__(self):
          self.sz = np.matrix([[1,0],[0,-1]], dtype= np.float64)
          self.sx = np.matrix([[0,1],[1,0]], dtype= np.float64)
          self.s0 = np.matrix([[1,0],[0,1]], dtype= np.float64)
"""

class Jahn_Teller_interaction:
     def __init__(self,Jahn_Teller_pars: Jahn_Teller_Pars, el_states: symmetric_electron_system, fonon_system: AbsCoupledHarmOscOperator):
          self.JT_pars = Jahn_Teller_pars
          self.el_states = el_states
          self.fonon_system = fonon_system
          self.create_hamiltonian_op()
     
     def create_hamiltonian_op(self):
          
          
          X = self.fonon_system.pos_i_ops[0].matrix
          Y = self.fonon_system.pos_i_ops[1].matrix
          XX = self.fonon_system.pos_i_sq_ops[0].matrix
          YY = self.fonon_system.pos_i_sq_ops[1].matrix     
          XY = np.matmul(X,Y)
          K = self.fonon_system.int_op.matrix          
          """

          X = self.fonon_system.X.matrix
          Y = self.fonon_system.Y.matrix
          XX = self.fonon_system.XX.matrix
          YY = self.fonon_system.YY.matrix
          XY = self.fonon_system.XY.matrix
          K = self.fonon_system.int_op.matrix
          """

          H_int_mat = self.JT_pars.hw * np.kron(K, self.el_states.symm_ops['s0'].matrix) + self.JT_pars.F*( np.kron(X,self.el_states.symm_ops['sz'].matrix) + np.kron(Y, self.el_states.symm_ops['sx'].matrix)) + 1.0*self.JT_pars.G*(np.kron((XX-YY) ,self.el_states.symm_ops['sz'].matrix) - np.kron(2*XY , self.el_states.symm_ops['sx'].matrix))

          np.savetxt('H_int.csv', H_int_mat)


          self.H_int = MatrixOperator(H_int_mat)





