import numpy as np
import collections
from numpy import linalg as LA
import math
import utilities.VASP as VASP
import collections
import copy
from scipy.sparse.linalg import eigs
import itertools
#Data structures
Eigen_state_2D = collections.namedtuple('Eigen_state',  'x_fonon y_fonon' )

Jahn_Teller_Pars = collections.namedtuple('Jahn_Teller_Pars',  'E_JT E_b hwpG hwmG hw F G ' )


class eigen_vect:
     def from_vals_vects(vals, vecs):
          res = []
          for i in range(0,len(vecs)):
               eig_state = eigen_vect( vals[i], vecs[:,i] )
               res.append( eig_state)
          
          return sorted(res, key=lambda x: x.eig_val)
     
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




     
class harm_osc_eigen_vect:
     def __init__(self,coeffs, energy = None):
          self._coeffs = coeffs
          self.energy = energy
     
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

class harm_osc_eigen_vects:
     def __init__(self, dim,order):
          self._states = []
          self.create_oscillators(dim,order, [])
          #self._states = sorted(self._states, key=lambda x: (x.get_order(), *x._coeffs) )
          self._states = sorted(self._states, key=lambda x: x.get_order() )
          self.h_space_dim = len(self)
          #self._states = sorted(self._states, key=lambda x: (x.get_order(),x._coeffs[1]) )


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
               return self._states.append( harm_osc_eigen_vect( curr_osc_coeffs))
          else:
               return


     def __len__(self):
          return len(self._states)
     
     def __getitem__(self, position):
          return self._states[position]
     
     def order(self):
          return sum(self)



class operator_builder:
     def __init__(self, eig_states):
          self.eig_states = eig_states

     def create_operator(self, sandwich_fun):
          dim = len(self.eig_states)
          operator = np.zeros((dim, dim), dtype = np.float64)
          for i in range(0,len(self.eig_states)):
               for j in range(0,len(self.eig_states)):
                    operator[i][j] = sandwich_fun(self.eig_states[i], self.eig_states[j])
          return MatrixOperator(operator)
     







#Quantummechanical operator:
class MatrixOperator:

     def from_sandwich_fun(self, states, sandwich_fun):
          pass

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

     def multiply(self, other):
          matrix1 = self.matrix
          matrix2 = other.matrix
          return MatrixOperator(np.matmul(matrix1,matrix2,dtype=np.complex64))

     def truncate_matrix(self, trunc_num):
          dim = len(self.matrix)
          self.matrix = self.matrix[0:dim-trunc_num, 0: dim-trunc_num]
          return self
     
     def get_dim(self):
          return len(self.matrix)



class FirstOrderPerturbation:
     
     def __init__(self,deg_eigen_vecs: list[eigen_vect], ham_comma: MatrixOperator):
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


class n_dim_harm_osc:
     def __init__(self,  dim,order, energy = None):
          self.spatial_dim = dim
          self.order = order
          self.energy = energy
          self.calc_order = order+1
          #self.trunc_num = self.calc_order +1
          self.eig_states = harm_osc_eigen_vects(self.spatial_dim,self.calc_order)
          self.op_builder = operator_builder(self.eig_states)
          self.build_creator_ops()
          self.build_annil_ops()
          self.build_H_i_ops()
          self.build_whole_sys_op()
          self.calc_trunc_num()
          print(self.over_est_int_op)
          self.over_est_pos_i_ops()
          self.create_pos_i_sq_ops()
     
     def get_h_space_dim(self):
          return self.eig_states.h_space_dim

          
     def build_creator_ops(self):
          self.creator_ops = []
          for i in range(0,self.spatial_dim):
               creator_i_op = self.op_builder.create_operator( lambda x,y: x.creator_i_sandwich(y,i) )
               self.creator_ops.append(creator_i_op)
     
     def build_annil_ops(self):

          self.annil_ops = []
          for i in range(0,self.spatial_dim):
               annil_i_op = self.op_builder.create_operator( lambda x,y: x.annil_i_sandwich(y,i) )
               self.annil_ops.append(annil_i_op)
     
     def build_H_i_ops(self):
          self.H_i_ops = []
          for i in range(0, self.spatial_dim):
               H_i = np.matmul(self.creator_ops[i].matrix, self.annil_ops[i].matrix)
               self.H_i_ops.append(MatrixOperator(H_i))
               #np.savetxt('H_osc_'+str(i)+'_new.csv', H_i)

     def calc_trunc_num(self):
          self.trunc_num = np.count_nonzero(self.over_est_int_op.matrix == self.calc_order)

     def build_whole_sys_op(self):
               
          int_op = np.matrix( np.round( sum( [x.matrix for x in self.H_i_ops ] ) ), dtype=np.int16 )
          
          self.over_est_int_op = MatrixOperator(int_op)
     
     def get_ham_op(self):
          return MatrixOperator(self.over_est_int_op.matrix).truncate_matrix(self.trunc_num)

     def over_est_pos_i_ops(self):
          self.pos_i_ops = []
          for i in range(0,self.spatial_dim):
               self.pos_i_ops.append( self.over_est_pos_i_op(i))

     def over_est_pos_i_op(self, i):
          pos_i_mat = (self.creator_ops[i].matrix + self.annil_ops[i].matrix)/(2**0.5)
          return MatrixOperator( pos_i_mat )

     def over_est_pos_i_j_op(self, i,j):
          pos_i_op = self.over_est_pos_i_op(i)
          pos_j_op = self.over_est_pos_i_op(j)
          return pos_i_op.multiply(pos_j_op)

     def get_pos_i_op(self, i):
          return self.over_est_pos_i_op(i).truncate_matrix(self.trunc_num)

     def get_pos_i_j_op(self, i,j):
          return self.over_est_pos_i_j_op(i,j).truncate_matrix(self.trunc_num)

     def over_est_pos_i_i_op(self, i):
          return self.over_est_pos_i_j_op(i,i)
     
     def get_pos_i_i_op(self, i):
          return self.over_est_pos_i_i_op(i).truncate_matrix(self.trunc_num)


     def create_pos_i_sq_ops(self):
          self.pos_i_sq_ops = []
          for i in range(0,self.spatial_dim):
               pos_i_sq_op = np.matmul(self.pos_i_ops[i].matrix, self.pos_i_ops[i].matrix)
               self.pos_i_sq_ops.append( MatrixOperator(pos_i_sq_op))



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
         


class Exe_JT_int:
     def __init__(self,Jahn_Teller_pars: Jahn_Teller_Pars, el_states: symmetric_electron_system, fonon_system: n_dim_harm_osc):
          self.JT_pars = Jahn_Teller_pars
          self.el_states = el_states
          self.fonon_system = fonon_system
          self.create_hamiltonian_op()
     
     def create_hamiltonian_op(self):
          
          
          X = self.fonon_system.get_pos_i_op(0).matrix
          Y = self.fonon_system.get_pos_i_op(1).matrix



          XX = self.fonon_system.get_pos_i_i_op(0).matrix

          XY = self.fonon_system.get_pos_i_j_op(0,1).matrix
          YX = self.fonon_system.get_pos_i_j_op(1,0).matrix
          
          YY = self.fonon_system.get_pos_i_i_op(1).matrix
          

          K = self.fonon_system.get_ham_op().matrix    
          


          H_int_mat = self.JT_pars.hw * np.kron(K, self.el_states.symm_ops['s0'].matrix) + self.JT_pars.F*( np.kron(X,self.el_states.symm_ops['sz'].matrix) + np.kron(Y, self.el_states.symm_ops['sx'].matrix)) + 1.0*self.JT_pars.G*(np.kron((XX-YY) ,self.el_states.symm_ops['sz'].matrix) - np.kron(XY + YX, self.el_states.symm_ops['sx'].matrix))

          np.savetxt('H_int.csv', H_int_mat)


          self.H_int = MatrixOperator(H_int_mat)

class fast_multimode_fonon_sys:
     def __init__(self,harm_osc_syss:dict[float,n_dim_harm_osc]):
          self.fonon_syss = harm_osc_syss
     
     def calc_pos_operator(self,energy, i:int ):

          #init res
          j = 0
          for el_energy in self.fonon_syss.keys():
               
               if j == 0:
                    op_dim = self.fonon_syss[el_energy].get_h_space_dim()
                    res = self.fonon_syss[el_energy].pos_i_ops[i].matrix if energy==el_energy else np.identity(op_dim)
               else:
               
                    if el_energy != energy:

                         res = np.kron(res,  self.fonon_syss[el_energy].pos_i_ops[i].matrix)
               
                    else:
                         res = np.kron(res, np.identity( self.fonon_syss[el_energy].spatial_dim ))
               j=j+1
          return MatrixOperator(res)
     
     def get_pos_op(h_osc: n_dim_harm_osc) -> MatrixOperator:
          return h_osc.get_pos_i_op()


     def calc_multi_mode_op(self, energy, op_getter):
          ops_to_kron = []
          


          for el_energy in self.fonon_syss.keys():
               if el_energy == energy:
                    ops_to_kron.append( op_getter(self.fonon_syss[energy]).matrix)
               else:
                    ops_to_kron.append(np.identity(self.fonon_syss[energy].get_h_space_dim()))

          return list(itertools.accumulate(ops_to_kron,np.kron))[-1]


     

class multi_mode_Exe_jt_int:
     def __init__(self,Jahn_Teller_pars: Jahn_Teller_Pars, el_states: symmetric_electron_system, fonon_systems: fast_multimode_fonon_sys):
          self.JT_pars = Jahn_Teller_pars
          self.el_states = el_states
          self.fonon_systems = fonon_systems
          
          Hs = []

          for mode in fonon_systems.fonon_syss.keys():
               Hs.append( self.create_ham_op_one_mode(mode))
          all_mode_ham = sum(Hs)
          
          self.H_int = MatrixOperator(all_mode_ham)

     
     def create_ham_op_one_mode(self,mode):
          
          fonon_system = self.fonon_systems

     

          X = fonon_system.calc_multi_mode_op(mode, lambda x: x.get_pos_i_op(0) )
          Y = fonon_system.calc_multi_mode_op(mode, lambda x: x.get_pos_i_op(1) )

          #Y = fonon_system.calc_pos_operator(mode, 1)

          XX = fonon_system.calc_multi_mode_op(mode, lambda x: x.get_pos_i_j_op(0,0) )
          YY = fonon_system.calc_multi_mode_op(mode, lambda x: x.get_pos_i_j_op(1,1) )

          XY = fonon_system.calc_multi_mode_op(mode, lambda x: x.get_pos_i_j_op(0,1) )
          YX = fonon_system.calc_multi_mode_op(mode, lambda x: x.get_pos_i_j_op(1,0) )

          


          K = fonon_system.calc_multi_mode_op(mode, lambda x: x.get_ham_op())

          hw = mode

          H_int_mat = hw * np.kron(K, self.el_states.symm_ops['s0'].matrix) + self.JT_pars.F*( np.kron(X,self.el_states.symm_ops['sz'].matrix) + np.kron(Y, self.el_states.symm_ops['sx'].matrix)) + 1.0*self.JT_pars.G*(np.kron((XX-YY) ,self.el_states.symm_ops['sz'].matrix) - np.kron(XY + YX, self.el_states.symm_ops['sx'].matrix))

          np.savetxt('H_int.csv', H_int_mat)


          #self.H_int = MatrixOperator(H_int_mat)
          return H_int_mat
