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
     def __init__(self, coeffs:maths.col_vector, eigen_val = None, name = '', subsys_name  = ''):
          self.name = name
          self.subsys_name = subsys_name
          self.eigen_val = eigen_val
          if type(coeffs) is maths.col_vector:
               self.amplitudo = complex(1.0,0)
               self.coeffs = coeffs
               self.dim = len(self.coeffs.coeffs)
          elif type(coeffs) is list:
               self.amplitudo = complex(1.0,0)

               self.coeffs = maths.col_vector(np.matrix( [  [num] for num in coeffs ] ))
               self.dim = len(self.coeffs.coeffs)
     
     def map(self, fun):
          return ket_vector(self.coeffs.map(fun))

     def tolist(self):
          return self.coeffs.tolist()

     def abs_sq(self):
          return sum( [ abs(complex(coeff))**2 for coeff in self.coeffs ] )

     def normalize(self):
          norm_factor = self.abs_sq()**0.5
          return self/norm_factor

     def set_item(self, index, item):
          return ket_vector(self.coeffs.set_item(index, item))

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

     def __eq__(self,other):
          return self.name == other.name and self.subsys_name == other.subsys_name and self.coeffs == other.coeffs and self.eigen_val == other.eigen_val

     def round(self, dig):
          return ket_vector(coeffs=self.coeffs.round(dig), name = self.name, subsys_name = self.subsys_name, eigen_val = self.eigen_val)

class bases_system:
     def __init__(self, bases_kets:list[ket_vector]):
          self.bases_kets = bases_kets



class bra_vector:
     def __init__(self, coeffs:maths.row_vector):
          if type(coeffs) is maths.row_vector:
               self.coeffs = coeffs
               self.dim = len(self.coeffs.coeffs[0])
          elif type(coeffs) is list:
               self.coeffs = maths.row_vector(np.matrix( [  [num for num in coeffs ]  ] ) )
               self.dim = len(self.coeffs.coeffs[0])
          self.amplitudo = complex(1.0,0)
          
          

     def __mul__(self, other:ket_vector):
          if type(other) is ket_vector:
               return complex(self.coeffs*other.coeffs)
          elif type(other) is MatrixOperator:
               return bra_vector(self.coeffs*other.matrix)


     def __rmul__(self, other:ket_vector):
          return MatrixOperator(other.coeffs*self.coeffs)
     def __repr__(self):
          return str(self.coeffs)
     def __getitem__(self,key):
          return self.coeffs[key]
     def set_val(self, index, val):
          self.coeffs.set_val(index,val)
          





     



class QuantumState:
    def __init__(self, matrix :maths.Matrix):
        self.matrix = matrix





#Quantummechanical operator:
class MatrixOperator:

     def basis_trf_matrix( kets:list[ket_vector]):
          return MatrixOperator(maths.Matrix.from_col_vectors([ ket.coeffs for ket in kets ]).transpose())



     def from_ket_vectors(kets: list[ket_vector]):
          return MatrixOperator(maths.Matrix.from_col_vectors([ ket.coeffs for ket in kets ]))

     def from_bra_vectors(bras: list[ket_vector]):
          return MatrixOperator(maths.Matrix.from_row_vectors([ bra.coeffs for bra in bras ]))


     def new_basis_system(self, bases: list[ket_vector]):
          new_bases_matrix = self.matrix.to_new_bases(list(map(lambda x: x.coeffs  ,bases )))
          return MatrixOperator(new_bases_matrix.transpose(), name  = self.name, subsys_name=self.subsys_name)


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




  


class degenerate_system:
     def __init__(self,deg_ket_vectors:list):
          self.deg_ket_vectors = deg_ket_vectors
          self.deg_bra_vectors = [ ket.to_bra_vector() for ket in self.deg_ket_vectors ]
          self.eigen_val = deg_ket_vectors[0].eigen_val

     def __getitem__(self,key):
          return self.deg_ket_vectors[key]
     
     def add_perturbation(self, perturbation:MatrixOperator):
          left_op = MatrixOperator.from_bra_vectors(self.deg_bra_vectors)
          right_op = MatrixOperator.from_ket_vectors(self.deg_ket_vectors)

          return left_op*perturbation*right_op
     


class degenerate_system_2D(degenerate_system):
     def __init__(self, deg_ket_vectors: list):
          if len(deg_ket_vectors)==2:
               super().__init__(deg_ket_vectors)
          else:
               return None
     
     def to_complex_basis(self, basis_trf_matrix:MatrixOperator):
          
          phix = self.deg_ket_vectors[0]
          phiy = self.deg_ket_vectors[1]
          phiplus = basis_trf_matrix*(phix+complex(0.0,1.0)*phiy)/(2**0.5)
          phiminus = basis_trf_matrix*(phix-complex(0.0,1.0)*phiy)/(2**0.5)
          self.complex_deg_ket_vectors = [phiminus, phiplus]
          
     def add_perturbation(self, perturbation: MatrixOperator):
          pert_sys_mat =  super().add_perturbation(perturbation)
          pert_sys_mat.calc_eigen_vals_vects()


          red_fact= abs(pert_sys_mat.eigen_kets[0].eigen_val-pert_sys_mat.eigen_kets[1].eigen_val)/2

          return red_fact


class braket_to_matrix_formalism:
     def __init__(self, eig_states:hilber_space_bases):
          self.eig_states = eig_states


     def create_new_basis(self, gen_ops:list[MatrixOperator], generating_order:int)->list[ket_vector]:

        raise_x_op = bf.raise_index_operator(0)

        raise_y_op = bf.raise_index_operator(1)

        raise_x_mx_op = self.create_MatrixOperator(raise_x_op)
        
        raise_y_mx_op = self.create_MatrixOperator(raise_y_op)


        #creator_x =  self.create_mx_ops[self.qm_nums_names[0]]#.truncate_matrix(self.trunc_num)
        #creator_y = self.create_mx_ops[self.qm_nums_names[1]]#.truncate_matrix(self.trunc_num)
        bases_vectors = []

        plus_gen_op = 1/2**0.5*(raise_x_mx_op+complex(0.0,1.0)*raise_y_mx_op)
        minus_gen_op = 1/2**0.5*(raise_x_mx_op+complex(0.0,-1.0)*raise_y_mx_op)

        gen_ops = [plus_gen_op, minus_gen_op]

        base_0 = ket_vector( maths.col_vector.zeros(self.eig_states.dim) )
        base_0.set_item(0,complex(1.0,0.0))

        bases_vectors.append(base_0)

        for i in range(0,generating_order):
            new_bases_vectors = []
            for base in bases_vectors:
            
                for gen_op in gen_ops:
            
                    new_base = gen_op*base

                    if (new_base not in bases_vectors) and (new_base not in new_bases_vectors) :
                        new_bases_vectors.append(new_base)
            bases_vectors+=new_bases_vectors
        return [ base_vec.normalize() for base_vec in bases_vectors ]                


     def create_basis_trf(self, gen_ops:list[MatrixOperator], generation_order:int)->MatrixOperator:
          basis =  self.create_new_basis(gen_ops, generation_order)
          return MatrixOperator.basis_trf_matrix(basis)



     def create_MatrixOperator(self, op: operator,name = '', subsys_name = ''):
          dim = len(self.eig_states)
          mx_op = np.zeros((dim, dim), dtype = np.complex128)
          for i in range(0,len(self.eig_states._ket_states)):
               for j in range(0,len(self.eig_states._bra_states)):
                    bra = self.eig_states._bra_states[j]
                    ket = self.eig_states._ket_states[i]


                    mx_op[i][j] = bra*op*ket
          #if self.mxtype == 'ordinary':
          return MatrixOperator(maths.Matrix(mx_op), name = name,subsys_name=subsys_name)
