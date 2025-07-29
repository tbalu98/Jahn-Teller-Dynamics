import itertools
import numpy as np
import itertools
from jahn_teller_dynamics.math.braket_formalism import  operator
import jahn_teller_dynamics.math.maths as  maths
import copy
import math
import pandas as pd
import jahn_teller_dynamics.math.braket_formalism as  bf

from collections import namedtuple




dtype = np.complex64


class ket_vector:


     def round(self, dig_num):
          return ket_vector(self.coeffs.round(dig_num),self.eigen_val)


     def from_str_list(str_list: list[str], eigen_energy:float =None):
          coeffs = []
          for coeff_str in str_list:
               coeff = complex(coeff_str)
               coeffs.append(coeff)

          return ket_vector(coeffs, eigen_energy)

     def normalize(self):
          length = self.abs_sq()
          return self/length



     def __init__(self, coeffs, eigen_val = None, name = '', subsys_name  = ''):
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
     
     def to_dataframe(self, bases)->pd.DataFrame:
          ket_dict = {}
          index_col_name = str(bases.qm_nums_names)
          ket_dict[index_col_name] = list( map( lambda x: str(x), bases._ket_states ) )

          ket_dict['coefficients'] = self.coeffs.tolist()

          ket_dataframe = pd.DataFrame.from_dict(ket_dict)

          ket_dataframe = ket_dataframe.set_index(index_col_name)

          return ket_dataframe
     
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
          if self.eigen_val == None:
               return str(self.coeffs)
          else:
               return str('eigen value: ') +str(self.eigen_val) + '\n'  +str(self.coeffs)
     
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

     def calc_abs_square(self):
          return  ket_vector(self.coeffs.calc_abs_square())

class bases_system:
     def __init__(self, bases_kets:list[ket_vector]):
          self.bases_kets = bases_kets



class bra_vector:


     def from_str_list(str_list: list[str], eigen_energy:float =None):
          coeffs = []
          for coeff_str in str_list:
               coeff = complex(coeff_str)
               coeffs.append(coeff)

          return bra_vector(coeffs, eigen_energy)

     def __init__(self, coeffs, eigen_val:float = None):
          self.eigen_val = eigen_val
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
          


class hilber_space_bases:


    def create_ket_vector(self, ket_states:list[bf.ket_state]):
          

          indexes = []
          coefficients = []

          ket_vec = ket_vector(maths.col_vector(np.zeros((len(self._ket_states), 1), dtype=np.complex128  )))

          for ket_state in ket_states:
               ket_state_index = self.get_ket_state_index(ket_state)
               indexes.append(ket_state_index)
               coeff = ket_state.qm_state.amplitude
               coefficients.append(coeff)
               ket_vec.set_val(ket_state_index,coeff )

          return ket_vec
               



    def get_ket_index(self, find_ket_state):
          for (i, ket_state) in  zip( range(0, self._ket_states) ,self._ket_states):
               if find_ket_state == ket_state:
                    return i
          return None



    def create_trf_op(self, basis_name:str):
         return MatrixOperator.basis_trf_matrix(self.base_vectors[basis_name][0])


    def kron_hilber_spaces(hilber_spaces:list):
        return list( itertools.accumulate(hilber_spaces , lambda x,y: x**y) )[-1]



    def create_hosc_eigen_states(self, dim, order, curr_osc_coeffs: list):
          if len(curr_osc_coeffs)<dim:

               for i in range(0,order+1):
                    temp_curr_osc_coeffs = copy.deepcopy(curr_osc_coeffs)
                    temp_curr_osc_coeffs.append(i)
                    if sum(temp_curr_osc_coeffs)>order:
                         return
                    else:
                         self.create_hosc_eigen_states(dim, order,temp_curr_osc_coeffs )

          elif len(curr_osc_coeffs) == dim and sum(curr_osc_coeffs)<=order:
               
               self._bra_states.append( bf.bra_state(qm_nums = curr_osc_coeffs))
               self._ket_states.append( bf.ket_state(qm_nums =  curr_osc_coeffs))
               return
          
          else:
               return


    def harm_osc_sys(self, dim,order, qm_nums_names:list):
        curr_osc_coeffs = []
        self._bra_states = []
        self._ket_states = []

        self.create_hosc_eigen_states(dim, order, curr_osc_coeffs)
        self._bra_states = sorted(self._bra_states,key=lambda x:(x.calc_order(), *x.qm_state.qm_nums))
        self._ket_states = sorted(self._ket_states,key=lambda x:(x.calc_order() , *x.qm_state.qm_nums ))
        self.dim = len(self._bra_states)
        self.qm_nums_names = qm_nums_names
        return self


    def from_qm_nums_list(self, qm_nums_list, qm_nums_names :list = None):

        self._bra_states = []
        self._ket_states = []
        for qm_nums in qm_nums_list:
            self._bra_states.append(bf.bra_state( qm_nums=qm_nums ))
            self._ket_states.append(bf.ket_state( qm_nums=qm_nums ))
        self.dim = len(self._bra_states)
        self.qm_nums_names = qm_nums_names
        return self



    def __init__(self, bra_states = None, ket_states = None, names = None):
        

        self._ket_states = []
        self._bra_states = []
        self.base_vectors = {}

        if bra_states == None:
            self._bra_states = []
        else:
            self._bra_states = bra_states

        if ket_states == None:
            self._ket_states = []
        else:
            self._ket_states = ket_states
        if names == None:
            self.qm_nums_names = []
        else:
            self.qm_nums_names = names
        self.dim = len(self._bra_states)

    def savetxt(self, filename):
        txt = ''


        txt+=str(self.qm_nums_names) + "\n"

        for ket in self._ket_states:
            txt+= str(ket) + '\n'

        text_file = open(filename, "w")

        text_file.write(txt)
        text_file.close()

    def reduce_space(self, new_dim):
        return hilber_space_bases(self._bra_states[0:new_dim], self._ket_states[0:new_dim],names = self.qm_nums_names)

    def get_ket_state_index(self, ks:bf.ket_state):

        exam_ket_states = np.array(self._ket_states)

        return int(np.where(exam_ket_states == ks)[0])


    def __len__(self):
        return len(self._ket_states)
     
    def __getitem__(self, position):
        return self._ket_states[position]

    def __pow__(self, other):
        if isinstance(other, hilber_space_bases):
            
            new_bra_states = [ bra_a**bra_b for bra_a in self._bra_states 
                                            for bra_b in other._bra_states ]
            new_ket_states = [ ket_a**ket_b for ket_a in self._ket_states 
                                            for ket_b in other._ket_states ]
            
            new_qm_nums_names = self.qm_nums_names + other.qm_nums_names
            
            
            return hilber_space_bases(new_bra_states, new_ket_states,names = new_qm_nums_names)
        
        else:
            return None

class MatrixOperator:

     def tolist(self):
          return self.matrix.tolist()

     def pauli_x_mx_op():
          mx = np.matrix( [ [ complex(0.0, 0.0), complex(1.0, 0.0)]  , [ complex( 1.0, 0.0 ), complex( 0.0, 0.0)] ] , dtype=dtype)
          return MatrixOperator(maths.Matrix(mx))

     def pauli_y_mx_op():
          mx = np.matrix( [ [ complex(0.0, 0.0), complex(0.0, -1.0)]  , [ complex( 0.0, 1.0 ), complex( 0.0, 0.0)] ], dtype=dtype )
          return MatrixOperator(maths.Matrix(mx))
     
     def pauli_z_mx_op():
          mx = np.matrix( [ [ complex(1.0, 0.0), complex(0.0, 0.0)]  , [ complex( 0.0, 0.0 ), complex( -1.0, 0.0)] ], dtype=dtype )
          return MatrixOperator(maths.Matrix(mx))

     def save_eigen_vals_vects_to_file(self, bases_states:hilber_space_bases, eig_vec_fn, eig_vals_fn):
          eigen_vectors:eigen_vector_space = self.calc_eigen_vals_vects()
          eigen_vectors.save(eig_vec_fn, eig_vals_fn)
          eig_vec_df, eig_val_df  = self.create_eigen_kets_vals_table(bases_states)

          eig_vec_df.to_csv( eig_vec_fn,sep = ';')
          eig_val_df.to_csv( eig_vals_fn,sep = ';')


     def basis_trf_matrix( kets:list[ket_vector]):
          return MatrixOperator(maths.Matrix.from_col_vectors([ ket.coeffs for ket in kets ]).transpose())


     def calc_expected_val(self, ket:ket_vector)->float:
          return (ket.to_bra_vector()*self*ket).real

     def from_ket_vectors(kets: list[ket_vector]):
          return MatrixOperator(maths.Matrix.from_col_vectors([ ket.coeffs for ket in kets ]))

     def from_bra_vectors(bras: list[ket_vector]):
          return MatrixOperator(maths.Matrix.from_row_vectors([ bra.coeffs for bra in bras ]))


     def to_new_basis(self, basis_kets:list[ket_vector]):
          return MatrixOperator(self.matrix.to_new_bases( [ ket.coeffs for ket in basis_kets ]))

     def new_basis_system(self, bases: list[ket_vector]):
          new_bases_matrix = self.matrix.to_new_bases(list(map(lambda x: x.coeffs  ,bases )))
          return MatrixOperator(new_bases_matrix.transpose(), name  = self.name, subsys_name=self.subsys_name)

     def accumulate_operators(mx_ops, fun):
          return list( itertools.accumulate(mx_ops, fun) )[-1]

     def drop_base_states(self, indexes):
          
          np.delete(self.matrix,indexes,axis=0)
          np.delete(self.matrix,indexes,axis=1)



     def save(self,filename):
          self.matrix.save(filename)
     def round(self, dig):
          return MatrixOperator(self.matrix.round(dig), name = self.name, subsys_name=self.subsys_name)
     def change_type(self, dtype):
          return MatrixOperator(self.matrix.change_type(dtype), name = self.name, subsys_name=self.subsys_name)
     def __add__(self,other):
          return MatrixOperator(self.matrix+ other.matrix,name= self.name, subsys_name=self.subsys_name)
     
     def __radd__(self, other):
          if type(other)==int:
               return self
          else:
               return self + other

     
     def __sub__(self, other):
          return MatrixOperator(self.matrix-other.matrix,name= self.name, subsys_name=self.subsys_name)
     
     def __mul__(self, other):
          if type(other) is MatrixOperator:
               return MatrixOperator(self.matrix.__mul__(other.matrix),name= self.name, subsys_name=self.subsys_name)
          elif type(other) is ket_vector:
               return ket_vector(self.matrix.__mul__(other.coeffs),eigen_val=other.eigen_val,name= self.name, subsys_name=self.subsys_name)

     
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
          self.matrix_class = type(matrix)
          self.quantum_state_bases:hilber_space_bases = None
     def set_quantum_states(self, quantum_states:hilber_space_bases):
          self.quantum_state_bases = quantum_states
     def create_id_matrix_op(dim, matrix_type=maths.Matrix):
          return MatrixOperator(matrix_type.create_eye(dim))

     def create_null_matrix_op(dim, matrix_type=maths.Matrix):
          return MatrixOperator(matrix_type.create_zeros(dim))

     def __len__(self):
          return len(self.matrix)

     def create_Lz_op(matrix_type=maths.Matrix):

          return MatrixOperator(matrix_type.create_Lz_mx())


     def __getitem__(self,key):
          return self.matrix[key]

     def calc_eigen_vals_vects_old(self, num_of_vals = None, ordering_type = None):
          eigen_vals, eigen_vects =  self.matrix.get_eig_vals(num_of_vals, ordering_type)

          self.eigen_kets = []

          for (eigen_val, eigen_vect) in zip( eigen_vals, eigen_vects  ):
               self.eigen_kets.append( ket_vector(maths.col_vector(np.transpose(np.matrix([eigen_vect]))),eigen_val) )
             
          self.eigen_kets = sorted(self.eigen_kets, key =lambda x: x.eigen_val)

     def calc_eigen_vals_vects(self, num_of_vals = None, ordering_type = None,quantum_states_bases:hilber_space_bases = None):
          if quantum_states_bases!=None:
               self.quantum_state_bases = quantum_states_bases
          eigen_vals, eigen_vects =  self.matrix.get_eig_vals(num_of_vals, ordering_type)
          
          self.eigen_kets = []

          for i in range(0, len(eigen_vals)):
               
               self.eigen_kets.append( ket_vector(maths.col_vector(np.transpose(np.round(np.matrix([eigen_vects[:,i]]),10))),round(eigen_vals[i].real, 10)) )

          self.eigen_kets = sorted(self.eigen_kets, key =lambda x: x.eigen_val)
          return eigen_vector_space(self.quantum_state_bases,self.eigen_kets)

     def create_eigen_kets_vals_table(self, bases:hilber_space_bases)->pd.DataFrame:

          eigen_kets_dict = {}
          index_col_name = str(bases.qm_nums_names)
          eigen_kets_dict[index_col_name] = list( map( lambda x: str(x), bases._ket_states ) )
          eigen_vec_names = []
          eigen_vals = []
          for (i,eigen_ket) in zip( range(0, len(self.eigen_kets)) ,self.eigen_kets):

               eigen_vec_name = 'eigenstate_' + str(i)

               eigen_val = eigen_ket.eigen_val

               eigen_vec_names.append(eigen_vec_name)
               
               eigen_vals.append(eigen_val)

               eigen_kets_dict[eigen_vec_name] = eigen_ket.coeffs.tolist()


          eig_vecs_df = pd.DataFrame.from_dict(eigen_kets_dict )
          eig_vecs_df = eig_vecs_df.set_index(index_col_name)

          eig_val_dict = {}
          eig_val_dict['state_name'] = eigen_vec_names
          eig_val_dict['eigenenergy'] = eigen_vals

          eig_vals_df = pd.DataFrame.from_dict(eig_val_dict).set_index('state_name')

          return eig_vecs_df, eig_vals_df



     def get_eigen_vect(self, i):
        return np.array(self.eigen_vects[:,i])
     
     def get_eigen_val(self, i):
        return self.eigen_vals[i]

     def multiply(self, other):
        return MatrixOperator(self.matrix.multiply(other.matrix))

     def truncate_matrix(self, trunc_num):
          dim = len(self.matrix)
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



class reduction_factors:
     def __init__(self, deg_ket_vectors:list[ket_vector] ):
          self.deg_ket_vectors = deg_ket_vectors
     
     def p_red_factor(self, op:MatrixOperator):

          return abs( op.calc_expected_val(self.deg_ket_vectors[0])-op.calc_expected_val(self.deg_ket_vectors[1]))

  


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
     

     def pert_in_complex_basis(self, pert_ham:MatrixOperator):
          complex_pert_ham =  pert_ham.to_new_basis(self.complex_deg_ket_vectors)

          complex_pert_ham.calc_eigen_vals_vects()

          return (complex_pert_ham.eigen_kets[1].eigen_val - complex_pert_ham.eigen_kets[0].eigen_val)/2
          
     def add_perturbation(self, perturbation: MatrixOperator):
          pert_sys_mat =  super().add_perturbation(perturbation)
          pert_sys_mat.calc_eigen_vals_vects()


          self.p_red_fact= abs(pert_sys_mat.eigen_kets[0].eigen_val-pert_sys_mat.eigen_kets[1].eigen_val)/2


     def calc_p_factor(self, perturbation:MatrixOperator):
          p_1 = perturbation.calc_expected_val(self.deg_ket_vectors[0])
          p_2 = perturbation.calc_expected_val(self.deg_ket_vectors[1])
          return p_2 + p_1

class braket_to_matrix_formalism:
     def __init__(self, eig_states:hilber_space_bases, used_dimensions = None):
          self.eig_states = eig_states
          self.calculation_dimension = self.eig_states.dim
          self.used_dimension = used_dimensions

     def create_new_basis(self, gen_ops:list[MatrixOperator], generating_order:int)->list[ket_vector]:
          bases_vectors = []
          base_0 = ket_vector( maths.col_vector.zeros(self.used_dimension) )
          
          base_0.set_item(0,complex(1.0,0.0))

          

          bases_vectors.append(base_0)

    

          for i in range(0,generating_order):
               new_bases_vectors = []
               for base in bases_vectors:
                    
                    for j in range(0, len(gen_ops)):
                    



                         new_base = gen_ops[j]*base

                         if (new_base not in bases_vectors) and (new_base not in new_bases_vectors) :

                              new_bases_vectors.append(new_base)
               bases_vectors+=new_bases_vectors
          
          return [ base_vec.normalize() for base_vec in bases_vectors ]                

     def create_new_basis2(self, gen_ops:list[MatrixOperator], generating_order:int):
          bases_vectors_data = namedtuple('bases_vectors_data', 'vector create_num annil_num')

          basis_vector_datas = []

          base_0 = ket_vector( maths.col_vector.zeros(self.used_dimension) )
          
          base_0.set_item(0,complex(1.0,0.0))

          basis_vector_datas.append(bases_vectors_data(base_0,0,0))          

         

          for i in range(0,generating_order):
               new_bases_vector_datas = []
               for base_data in basis_vector_datas:
                    
                    for j in range(0, len(gen_ops)):
                    

                         if j==0:
                              new_create_num = base_data.create_num+1
                              new_annil_num = base_data.annil_num
                         else:
                              new_create_num = base_data.create_num
                              new_annil_num = base_data.annil_num+1


                         new_base = gen_ops[j]*base_data.vector

                         new_base_data = bases_vectors_data(new_base, new_create_num, new_annil_num)

                         if (new_base_data.vector not in [x.vector for x in basis_vector_datas]) and (new_base_data.vector not in [x.vector for x in new_bases_vector_datas]) :
                              

                              new_bases_vector_datas.append(bases_vectors_data(new_base,new_create_num, new_annil_num))

               basis_vector_datas+=new_bases_vector_datas
          
          new_hilbert_space = hilber_space_bases().from_qm_nums_list(qm_nums_list=[ [base_vectors_data.create_num, base_vectors_data.annil_num] for base_vectors_data in basis_vector_datas ], qm_nums_names=['+', '-'])
          return [  base_vectors_data.vector.normalize() for base_vectors_data in basis_vector_datas ]   ,new_hilbert_space             


     def create_basis_trf(self, gen_ops:list[MatrixOperator], generation_order:int)->MatrixOperator:
          basis =  self.create_new_basis(gen_ops, generation_order)
          
          return MatrixOperator.basis_trf_matrix(basis)



     def create_MatrixOperator(self, op: operator,name = '', subsys_name = ''):
          dim = len(self.eig_states)
          mx_op = np.zeros((dim, dim), dtype = maths.complex_number_typ)
          for i in range(0,len(self.eig_states._ket_states)):
               for j in range(0,len(self.eig_states._bra_states)):
                    bra = self.eig_states._bra_states[j]
                    ket = self.eig_states._ket_states[i]


                    mx_op[i][j] = bra*op*ket
          return MatrixOperator(maths.Matrix(mx_op), name = name,subsys_name=subsys_name)




class eigen_vectors:

     def bra_from_csv(states_fn:str, energies_fn:str):
          energies_df = pd.read_csv(energies_fn, sep=';',index_col='state_name')

          states_df = pd.read_csv(states_fn, sep=';')
          eigen_vectors = []
          for eigen_state_name in list(states_df)[1:]:
               coeff_strs = states_df[eigen_state_name].tolist()
               eigen_energy = energies_df['eigenenergy'][eigen_state_name]

               eigen_vectors.append(bra_vector.from_str_list(coeff_strs, eigen_energy))

          return eigen_vectors
     
     def ket_from_csv(states_fn:str, energies_fn:str):
          energies_df = pd.read_csv(energies_fn, sep=';',index_col='state_name')

          states_df = pd.read_csv(states_fn, sep=';')
          eigen_vectors = []
          for eigen_state_name in list(states_df)[1:]:
               coeff_strs = states_df[eigen_state_name].tolist()
               eigen_energy = energies_df['eigenenergy'][eigen_state_name]

               eigen_vectors.append(ket_vector.from_str_list(coeff_strs, eigen_energy))

          return eigen_vectors
     
class eigen_vector_space:
     def __init__(self,hilbert_space:hilber_space_bases, eigen_kets:list[ket_vector]):
          self.quantum_states_basis = hilbert_space
          self.eigen_kets = eigen_kets
     
     def __getitem__(self, item_num)->ket_vector:
          return self.eigen_kets[item_num]

     def transform_vector_space(self, new_hilbert_space:hilber_space_bases,trf_mx):
          return eigen_vector_space( new_hilbert_space, [  (trf_mx*eig_ket).round(4) for eig_ket in self.eigen_kets ]  )

     def save(self, eig_vec_fn:str, eig_val_fn:str):
          
          eig_vec_df, eig_val_df  = self.create_eigen_kets_vals_table(self.quantum_states_basis)

          eig_vec_df.to_csv( eig_vec_fn,sep = ';')
          eig_val_df.to_csv( eig_val_fn,sep = ';')


     def create_eigen_kets_vals_table(self, bases:hilber_space_bases)->pd.DataFrame:

          eigen_kets_dict = {}
          index_col_name = str(bases.qm_nums_names)
          eigen_kets_dict[index_col_name] = list( map( lambda x: str(x), bases._ket_states ) )
          eigen_vec_names = []
          eigen_vals = []
          for (i,eigen_ket) in zip( range(0, len(self.eigen_kets)) ,self.eigen_kets):

               eigen_vec_name = 'eigenstate_' + str(i+1)

               eigen_val = eigen_ket.eigen_val

               eigen_vec_names.append(eigen_vec_name)
               
               eigen_vals.append(eigen_val)

               eigen_kets_dict[eigen_vec_name] = eigen_ket.coeffs.tolist()


          eig_vecs_df = pd.DataFrame.from_dict(eigen_kets_dict )
          eig_vecs_df = eig_vecs_df.set_index(index_col_name)

          eig_val_dict = {}
          eig_val_dict['state_name'] = eigen_vec_names
          eig_val_dict['eigenenergy'] = eigen_vals

          eig_vals_df = pd.DataFrame.from_dict(eig_val_dict).set_index('state_name')

          return eig_vecs_df, eig_vals_df
