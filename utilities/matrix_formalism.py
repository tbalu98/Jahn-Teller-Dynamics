import itertools
import numpy as np
import itertools
from utilities.braket_formalism import hilber_space_bases, operator
import utilities.maths as  maths
import copy
import math

import utilities.braket_formalism as  bf
#import utilities.jahn_teller_theory as  jt
#Data structures

from collections import namedtuple







class ket_vector:
     def __init__(self, coeffs:maths.col_vector, eigen_val = None):
          self.eigen_val = eigen_val
          if type(coeffs) is maths.col_vector:
               self.amplitudo = complex(1.0,0)
               self.coeffs = coeffs
               self.dim = len(self.coeffs.coeffs)
          elif type(coeffs) is list:
               self.amplitudo = complex(1.0,0)

               self.coeffs = maths.col_vector(np.matrix( [  [num] for num in coeffs ] ))
               self.dim = len(self.coeffs.coeffs)
     
     def __repr__(self):
          rep_str = ""
          if self.eigen_val == None:
               return str(self.coeffs)
          else:
               return str('eigen_val: ') +str(self.eigen_val) + '\n'  +str(self.coeffs)
     
     def __getitem__(self,key):
          return self.coeffs[key]
     def set_val(self, index, val):
          self.coeffs.set_val(index,val)

     def to_bra_vector(self):
          return bra_vector( self.coeffs.to_row_vector() )
      
     def __add__(self, other):
          if isinstance(other,ket_vector):
               return ket_vector(self.coeffs + other.coeffs)

     def __sub__(self, other):
          if isinstance(other,ket_vector):
               return ket_vector(self.coeffs - other.coeffs)
     def __rsub__(self, other):
          return self-other

     def __mul__(self, other):
          return ket_vector(self.coeffs* other)
     
     def __rmul__(self, other):
          return ket_vector(self.coeffs.__rmul__(other))

     def __truediv__(self,other):
          return ket_vector(self.coeffs/other)
     
     def __abs__(self):
          return abs(self.coeffs)

class bra_vector:
     def __init__(self, coeffs:maths.row_vector):
          if type(coeffs) is maths.row_vector:
               self.vector = coeffs
               self.dim = len(self.vector.coeffs[0])
          elif type(coeffs) is list:
               self.vector = maths.row_vector(np.matrix( [  [num for num in coeffs ]  ] ) )
               self.dim = len(self.vector.coeffs[0])
          self.amplitudo = complex(1.0,0)
          
          

     def __mul__(self, other:ket_vector):
          if type(other) is ket_vector:
               return complex(self.vector*other.coeffs)
          elif type(other) is MatrixOperator:
               return bra_vector(self.vector*other.matrix)


     def __rmul__(self, other:ket_vector):
          return MatrixOperator(other.coeffs*self.vector)
     def __repr__(self):
          return str(self.vector)
     def __getitem__(self,key):
          return self.vector[key]
     def set_val(self, index, val):
          self.vector.set_val(index,val)
          

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
     def __init__(self,coeffs, quantum_energy = None):
          self._coeffs = coeffs
          self.quantum_energy = quantum_energy
     
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
          self._states = sorted(self._states, key=lambda x: x.get_order() )
          print(self._states)
          self.h_space_dim = len(self)


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
          #self.mxtype = mxtype

     def create_operator(self, sandwich_fun):
          dim = len(self.eig_states)
          operator = np.zeros((dim, dim), dtype = np.complex64)
          for i in range(0,len(self.eig_states)):
               for j in range(0,len(self.eig_states)):
                    operator[i][j] = sandwich_fun(self.eig_states[i], self.eig_states[j])
          #if self.mxtype == 'ordinary':
          return MatrixOperator(maths.Matrix(operator))
          #elif self.mxtype == 'sparse':
          #     return MatrixOperator( maths.SparseMatrix( operator ) )


class QuantumState:
    def __init__(self, matrix :maths.Matrix):
        self.matrix = matrix





#Quantummechanical operator:
class MatrixOperator:

     def as_part_of_a_system(self, qm_sys_sign: bf.quantum_system_signature ):
          if qm_sys_sign ==None:
               return self
          else:
               dim_before, dim_after = qm_sys_sign.get_dim_before_and_after(self.subsys_name)
               id_0 = MatrixOperator.create_id_matrix_op(dim_before)
               id_1 = MatrixOperator.create_id_matrix_op(dim_after)

               return id_0**self**id_1
          


     def accumulate_operators(mx_ops, fun):
          return list( itertools.accumulate(mx_ops, fun) )[-1]

     def drop_base_states(self, indexes):
          
          np.delete(self.matrix,indexes,axis=0)
          np.delete(self.matrix,indexes,axis=1)



     def save(self,filename):
          self.matrix.save(filename)
     #matrix:maths.SparseMatrix
     def round(self, dig):
          return MatrixOperator(self.matrix.round(dig), name = self.name, subsys_name=self.subsys_name)
     def change_type(self, dtype):
          return MatrixOperator(self.matrix.change_type(dtype), name = self.name, subsys_name=self.subsys_name)
     def __add__(self,other):
          return MatrixOperator(self.matrix+ other.matrix,name= self.name, subsys_name=self.subsys_name)
     
     def __radd__(self, other):
          if type(other)==int:
               return self #+ MatrixOperator(maths.Matrix.create_eye(self.dim))
          else:
               return self + other
          #return MatrixOperator(self.matrix + other)
     
     def __sub__(self, other):
          return MatrixOperator(self.matrix-other.matrix,name= self.name, subsys_name=self.subsys_name)
     
     def __mul__(self, other):
          if type(other) is MatrixOperator:
               return MatrixOperator(self.matrix.__mul__(other.matrix),name= self.name, subsys_name=self.subsys_name)
          elif type(other) is ket_vector:
               return ket_vector(self.matrix.__mul__(other.coeffs),name= self.name, subsys_name=self.subsys_name)
     
     def __rmul__(self, other):
          return MatrixOperator(self.matrix.__rmul__(other),name= self.name, subsys_name=self.subsys_name)
     

     def __truediv__(self, other):
          return MatrixOperator(self.matrix.__truediv__(other),name= self.name, subsys_name=self.subsys_name)
     
     def __pow__(self, other):
          return MatrixOperator(self.matrix**other.matrix,name = self.name, subsys_name=self.subsys_name)
     
     def __repr__(self):
          return self.matrix.__repr__()

     def from_sandwich_fun(self, states, sandwich_fun):
          pass

     def __init__(self, matrix:maths.Matrix, name = "", subsys_name = ""):
          self.name = name
          self.subsys_name = subsys_name
          self.matrix = matrix
          #self.dim = self.matrix.dim
          #self.calc_eigen_vals_vects() !!!
          self.matrix_class = type(matrix)
     
     def create_id_matrix_op(dim, matrix_type=maths.Matrix):
          return MatrixOperator(matrix_type.create_eye(dim))

     def __len__(self):
          return len(self.matrix)

     def create_Lz_op(matrix_type=maths.Matrix):
          #Lz_mat = np.matrix([[0, complex(0,1)], [complex(0,-1), 0]], dtype=np.complex64)

          return MatrixOperator(matrix_type.create_Lz_mx())

     #def create_id_op(n:int):
     #     return MatrixOperator(maths.Matrix(np.eye(n)))

     def __getitem__(self,key):
          return self.matrix[key]

     def calc_eigen_vals_vects_old(self, num_of_vals = None, ordering_type = None):
        #self.eigen_vals, self.eigen_vects =  eigs(self.matrix, k = len(self.matrix), which = 'SM')
          eigen_vals, eigen_vects =  self.matrix.get_eig_vals(num_of_vals, ordering_type)

          self.eigen_kets = []

          for (eigen_val, eigen_vect) in zip( eigen_vals, eigen_vects  ):
               self.eigen_kets.append( ket_vector(maths.col_vector(np.transpose(np.matrix([eigen_vect]))),eigen_val) )
             
          self.eigen_kets = sorted(self.eigen_kets, key =lambda x: x.eigen_val)

     def calc_eigen_vals_vects(self, num_of_vals = None, ordering_type = None):
        #self.eigen_vals, self.eigen_vects =  eigs(self.matrix, k = len(self.matrix), which = 'SM')
          eigen_vals, eigen_vects =  self.matrix.get_eig_vals(num_of_vals, ordering_type)
          
          self.eigen_kets = []

          for i in range(0, len(eigen_vals)):
               
               self.eigen_kets.append( ket_vector(maths.col_vector(np.transpose(np.matrix([eigen_vects[:,i]]))),eigen_vals[i]) )

          self.eigen_kets = sorted(self.eigen_kets, key =lambda x: x.eigen_val)






     def get_eigen_vect(self, i):
        #return np.array(self.eigen_vects[i])
        return np.array(self.eigen_vects[:,i])
     
     def get_eigen_val(self, i):
        return self.eigen_vals[i]
     
     def calc_sandwich(self, Phi1: QuantumState, Phi2: QuantumState):
        #Phi1_tr = np.transpose(Phi1)
          
        #return complex(np.matmul( Phi2, np.matmul( self.matrix, Phi1_tr ) ))
        Phi1_tr_matrix = Phi1.matrix.transpose()
        Phi2_matrix = Phi2.matrix

        return complex( Phi2_matrix.multiply( self.matrix.multiply(Phi1_tr_matrix) ) )
     
     def interaction_with(self, other):
          return MatrixOperator( self.matrix.kron(other.matrix) )
          #return MatrixOperator(np.kron( self.matrix, other.matrix ) )

     def multiply(self, other):
          #return MatrixOperator(np.matmul(matrix1,matrix2,dtype=np.complex64))
        return MatrixOperator(self.matrix.multiply(other.matrix))

     def truncate_matrix(self, trunc_num):
          dim = len(self.matrix)
          #self.matrix = self.matrix[0:dim-trunc_num, 0: dim-trunc_num]
          #return MatrixOperator(self.matrix.truncate(trunc_num, trunc_num))
          return MatrixOperator(maths.Matrix(self.matrix[0:dim-trunc_num, 0: dim-trunc_num]),self.name, self.subsys_name)
     
     def get_dim(self):
          return len(self.matrix)



class FirstOrderPerturbation:
     
     def __init__(self,deg_eigen_vecs: list[ket_vector], ham_comma: MatrixOperator):
          self.deg_eigen_vecs = deg_eigen_vecs
          self.ham_comma = ham_comma
          self.create_pert_op_old()



     def create_pert_op_old(self):
          left = np.matrix( [ x.coeffs.get_coeffs_list() for x in self.deg_eigen_vecs] )
          right = np.transpose(left)

          raw_pert_mat = np.matmul(left, np.matmul( self.ham_comma.matrix.matrix, right ))



          self.pert_op = MatrixOperator(maths.Matrix(raw_pert_mat))
          
          self.pert_op.calc_eigen_vals_vects()
          self.pert_eigen_vals = [ ket_vec.eigen_val for ket_vec in self.pert_op.eigen_kets] 

          return self.pert_op




     def create_pert_op(self):
          left = np.matrix( [ x.coeffs.coeffs.flatten() for x in self.deg_eigen_vecs] )
          bra_vec = self.deg_eigen_vecs[0].to_bra_vector()
          ket_vec = self.deg_eigen_vecs[1]

          self.pert_op = bra_vec*self.ham_comma
          right = np.transpose(left)

          raw_pert_mat = np.matmul(left, np.matmul( self.ham_comma.matrix.matrix, right ))



          self.pert_op = MatrixOperator(maths.Matrix(raw_pert_mat))

          self.pert_op.calc_eigen_vals_vects()
          self.pert_eigen_vals = self.pert_op.eigen_vals

          return self.pert_op



     def get_reduction_factor(self):
          return abs( (self.pert_eigen_vals[1] - self.pert_eigen_vals[0]).real/2 )


class n_dim_harm_osc:
     def __init__(self,  dim,order, quantum_energy = None):
          self.spatial_dim = dim
          self.order = order
          self.quantum_energy = quantum_energy
          self.calc_order = order+1
          self.eig_states = harm_osc_eigen_vects(self.spatial_dim,self.calc_order)
          self.op_builder = operator_builder(self.eig_states)
          self.build_creator_ops()
          self.build_annil_ops()
          self.build_H_i_ops()
          self.build_whole_sys_op()
          self.calc_trunc_num()
          print(self.over_est_int_op)
          self.over_est_pos_i_ops()
          #self.create_pos_i_sq_ops()
     
     def get_h_space_dim(self):
          return self.eig_states.h_space_dim

          
     def build_creator_ops(self):
          self.creator_ops = []
          for i in range(0,self.spatial_dim):
               creator_i_op = self.op_builder.create_operator( lambda x,y: x.creator_i_sandwich(y,i) )
               self.creator_ops.append(creator_i_op)
               creator_i_op.save('old_creator' + str(i) + '.csv')
          

     def build_annil_ops(self):

          self.annil_ops = []
          for i in range(0,self.spatial_dim):
               annil_i_op = self.op_builder.create_operator( lambda x,y: x.annil_i_sandwich(y,i) )
               self.annil_ops.append(annil_i_op)
     
     def build_H_i_ops(self):
          self.H_i_ops = []
          for i in range(0, self.spatial_dim):
               
               op_ov = self.creator_ops[i]*self.annil_ops[i]
               print(op_ov)
               
               self.H_i_ops.append(op_ov)
               #H_i = np.matmul(self.creator_ops[i].matrix, self.annil_ops[i].matrix)
               
               #self.H_i_ops.append( MatrixOperator( maths.Matrix(H_i)))
               

     def calc_trunc_num(self):
          #self.trunc_num = np.count_nonzero(self.over_est_int_op.matrix == self.calc_order)
          self.trunc_num = self.over_est_int_op.matrix.count_occurrences(self.calc_order)
          print('calc trunc num')

     def build_whole_sys_op(self):
               
          #int_op_raw_matrix = sum( [ x.matrix.matrix for x in self.H_i_ops ] )
          #int_op_matrix = maths.Matrix(int_op_raw_matrix).round(0).change_type(np.int16)
          #self.over_est_int_op = MatrixOperator(int_op_matrix)

          int_op = sum(self.H_i_ops)
          self.over_est_int_op = int_op.round(0).change_type(np.int16)

     
     def get_ham_op(self):
          #return MatrixOperator(self.over_est_int_op.matrix).truncate_matrix(self.trunc_num)
          return self.over_est_int_op.truncate_matrix(self.trunc_num)

     def over_est_pos_i_ops(self):
          self.pos_i_ops = []
          for i in range(0,self.spatial_dim):
               self.pos_i_ops.append( self.over_est_pos_i_op(i))
               

     def over_est_pos_i_op(self, i):
          #pos_i_mat = (self.creator_ops[i].matrix + self.annil_ops[i].matrix)/(2**0.5)
          #return MatrixOperator( maths.Matrix(pos_i_mat))
          return (self.creator_ops[i] + self.annil_ops[i])/(2**0.5)
     
     def over_est_pos_i_j_op(self, i,j):
          pos_i_op = self.over_est_pos_i_op(i)
          pos_j_op = self.over_est_pos_i_op(j)
          
          return pos_i_op*(pos_j_op)

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

               self.pos_i_sq_ops.append(self.pos_i_sq_ops[i]* self.pos_i_sq_ops[i])

     def create_pos_operator_as_sub_sys(self, pos_index:int):
          pass


class multi_mode_fonon_sys:
     def __init__(self,harm_osc_syss:dict[float,n_dim_harm_osc]):
          self.fonon_syss = harm_osc_syss



class symmetric_electron_system:
     def __init__(self, symm_ops: dict[str,maths.Matrix]):
          self.symm_ops = symm_ops
         


class fast_multimode_fonon_sys:
     def __init__(self,harm_osc_syss:dict[float,n_dim_harm_osc]):
          self.fonon_syss = harm_osc_syss
     
     def calc_pos_operator(self,energy, i:int ):
          

          ops_to_kron = []

          for el_energy in self.fonon_syss.keys():
               if el_energy == energy:
                    ops_to_kron.append(self.fonon_syss[el_energy].pos_i_ops[i])
               else:
                    ops_to_kron.append(MatrixOperator.create_id_matrix_op(self.fonon_syss[el_energy].spatial_dim))

          return list( itertools.accumulate(ops_to_kron, lambda x,y: x**y) )[-1]
         
     


     
     def get_pos_op(h_osc: n_dim_harm_osc) -> MatrixOperator:
          return h_osc.get_pos_i_op()


     def calc_multi_mode_op(self, energy, op_getter):
          ops_to_kron = []
          


          for el_energy in self.fonon_syss.keys():
               if el_energy == energy:
                    ops_to_kron.append( op_getter(self.fonon_syss[energy]))
               else:
                    ops_to_kron.append(MatrixOperator.create_id_matrix_op(self.fonon_syss[energy].get_h_space_dim()))

          return list(itertools.accumulate(ops_to_kron,lambda x,y: x**y))[-1]


class degenerate_system:
     def __init__(self,deg_ket_vectors:list):
          self.deg_ket_vectors = deg_ket_vectors
          self.eigen_val = deg_ket_vectors[0].eigen_val

     def __getitem__(self,key):
          return self.deg_ket_vectors[key]
     
class degenerate_system_2D(degenerate_system):
     def __init__(self, deg_ket_vectors: list):
          if len(deg_ket_vectors)==2:
               super().__init__(deg_ket_vectors)
          else:
               return None
     
     def to_complex_basis(self):
          phix = self.deg_ket_vectors[0]
          phiy = self.deg_ket_vectors[1]
          phiplus = (phix+complex(0.0,1.0)*phiy)/(2**0.5)
          phiminus = (phix-complex(0.0,1.0)*phiy)/(2**0.5)
          self.deg_ket_vectors[0] = phiminus
          self.deg_ket_vectors[1] = phiplus


class braket_to_mx_operator_builder:
     def __init__(self, eig_states:hilber_space_bases):
          self.eig_states = eig_states


     def create_MatrixOperator(self, op: operator,name = '', subsys_name = ''):
          dim = len(self.eig_states)
          mx_op = np.zeros((dim, dim), dtype = np.complex64)
          for i in range(0,len(self.eig_states._ket_states)):
               for j in range(0,len(self.eig_states._bra_states)):
                    bra = self.eig_states._bra_states[j]
                    ket = self.eig_states._ket_states[i]


                    mx_op[i][j] = bra*op*ket
          #if self.mxtype == 'ordinary':
          return MatrixOperator(maths.Matrix(mx_op), name = name,subsys_name=subsys_name)
