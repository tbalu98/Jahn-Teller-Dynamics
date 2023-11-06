import numpy as np
import collections
import copy
import math
import utilities.maths  as maths
import utilities.quantum_mechanics as qm
class ket_state:
     def __init__(self, qm_nums,qm_nums_names = None, amplitudo = complex(1.0, 0.0)):
          self.qm_nums = qm_nums
          self.qm_nums_names = qm_nums_names
          self.amplitudo = amplitudo

     def __eq__(self, other):
          return self.qm_nums == other.qm_nums and self.qm_nums_names == other.qm_nums_names
     
     
     
     def __rmul__(self, other):
          if type(other) == float:
               return ket_state(self.qm_nums, self.qm_nums_names, amplitudo=self.amplitudo*other)
          else:
               return ket_state(self.qm_nums, self.qm_nums_names, amplitudo=other*self.amplitudo)

     def __repr__(self):
          return str(self.amplitudo) + ' '  + str(self.qm_nums)
     
     def increase_qm_num(self, qm_num_index):

          new_qm_nums = copy.deepcopy(self.qm_nums)
          new_qm_nums_names = copy.deepcopy(self.qm_nums_names )

          new_qm_nums[qm_num_index] = new_qm_nums[qm_num_index] +1
          return bra_state(new_qm_nums, new_qm_nums_names,self.amplitudo)
     
     def decrease_qm_num(self, qm_num_index):
          #self.qm_nums[qm_num_index] -= 1
          #return self
          new_qm_nums = copy.deepcopy(self.qm_nums)
          new_qm_nums_names = copy.deepcopy(self.qm_nums_names )
          new_qm_nums[qm_num_index] = new_qm_nums[qm_num_index] -1
          return bra_state(new_qm_nums, new_qm_nums_names,self.amplitudo)
     

class bra_state:
     def __init__(self, qm_nums,qm_nums_names = None, amplitudo = complex(1.0, 0.0)):
          self.qm_nums = qm_nums
          self.qm_nums_names = qm_nums_names
          self.amplitudo = amplitudo

     def __eq__(self, other):
          return self.qm_nums == other.qm_nums and self.qm_nums_names == other.qm_nums_names
     
     def increase_qm_num(self, qm_num_index):
          self.qm_nums[qm_num_index] += 1
          #return self
          #new_qm_nums = copy.deepcopy(self.qm_nums)
          #new_qm_nums[qm_num_index] = new_qm_nums[qm_num_index] +1
          #return bra_state(new_qm_nums, self.qm_nums_names,self.amplitudo)
     
     def decrease_qm_num(self, qm_num_index):
          self.qm_nums[qm_num_index] -= 1
          #return self
          #new_qm_nums = copy.deepcopy(self.qm_nums)
          #new_qm_nums[qm_num_index] = new_qm_nums[qm_num_index] -1
          #return bra_state(new_qm_nums, self.qm_nums_names,self.amplitudo)
     
     def __repr__(self):
          return str(self.amplitudo) + ' '  + str(self.qm_nums)
     
     def __mul__(self, other: ket_state):
          
          
          if isinstance(other, ket_state):     
               if self == other:
                    return  np.conj(self.amplitudo)*other.amplitudo
               else:
                    return complex(0.0,0.0)
          if isinstance(other, operator):
               return other.operate(self)
     def __rmul__(self, other):
          if type(other) == float:
               return bra_state(self.qm_nums, self.qm_nums_names, amplitudo=self.amplitudo*other)
          else:
               return bra_state(self.qm_nums, self.qm_nums_names, amplitudo=other*self.amplitudo)

     def increase_qm_num(self, qm_num_index):

          new_qm_nums = copy.deepcopy(self.qm_nums)
          new_qm_nums_names = copy.deepcopy(self.qm_nums_names )

          new_qm_nums[qm_num_index] = new_qm_nums[qm_num_index] +1
          return bra_state(new_qm_nums, new_qm_nums_names,self.amplitudo)
     
     def decrease_qm_num(self, qm_num_index):
          #self.qm_nums[qm_num_index] -= 1
          #return self
          new_qm_nums = copy.deepcopy(self.qm_nums)
          new_qm_nums_names = copy.deepcopy(self.qm_nums_names )
          new_qm_nums[qm_num_index] = new_qm_nums[qm_num_index] -1
          return bra_state(new_qm_nums, new_qm_nums_names,self.amplitudo)

class harm_osc_ket_state(ket_state):
     def __init__(self, energy_quantum,qm_nums,qm_nums_names = None, amplitudo = complex(1.0, 0.0), ):
          super().__init__(qm_nums,qm_nums_names = qm_nums_names, amplitudo = amplitudo)
          self.energy_quantum = energy_quantum
     
     
     def calc_order(self):
          
          return sum(self.qm_nums)
    
     def __eq__(self, other):
          return super().__eq__(other) and self.energy_quantum == other.energy_quantum

     def calc_energy(self):
          return self.calc_order()*self.energy_quantum
     
     

     def increase_qm_num(self, qm_num_index):
          new_state  = copy.deepcopy(self)
          new_state.qm_nums[qm_num_index] +=1
          return new_state

     def decrease_qm_num(self, qm_num_index):
          new_state  = copy.deepcopy(self)
          new_state.qm_nums[qm_num_index] -=1
          return new_state






class harm_osc_bra_state(bra_state):
     def __init__(self, energy_quantum,qm_nums,qm_nums_names = None, amplitudo = complex(1.0, 0.0), ):
          super().__init__(qm_nums,qm_nums_names = qm_nums_names, amplitudo = amplitudo)
          self.energy_quantum = energy_quantum
     """
     def __mul__(self, other):
          if type(other) == creator_op:
               return other.operate(self)
          elif type(other) == annil_op:
               return other.operate(self)
     """
     def calc_order(self):
          
          return sum(self.qm_nums)

     def __eq__(self, other):
          return super().__eq__(other) and self.energy_quantum == other.energy_quantum

     def calc_energy(self):
          return self.calc_order()*self.energy_quantum
     
     def __repr__(self):
          return str(self.energy_quantum) + ' '  + str(super().__repr__())

     def increase_qm_num(self, qm_num_index):
          return super().increase_qm_num(qm_num_index)
     
     def increase_qm_num(self, qm_num_index):
          new_state  = copy.deepcopy(self)
          new_state.qm_nums[qm_num_index] +=1
          return new_state

     def decrease_qm_num(self, qm_num_index):
          new_state  = copy.deepcopy(self)
          new_state.qm_nums[qm_num_index] -=1
          return new_state


     def __mul__(self, other):
          if isinstance(other, harm_osc_ket_state):
               if self == other:
                    return self.amplitudo*other.amplitudo
               else:
                    return complex(0.0,0.0)
          elif isinstance(other, complex) or isinstance(other, float):
               new_state = copy.deepcopy(self)
               new_state.amplitudo *=other
               return new_state
          
          elif isinstance(other, operator):
               return other.operate(self)
          else:
               return complex(0.0,0.0)

     def __rmul__(self ,other):
          return self*other

"""    
     def __mul__(self, other):

          if self == other:
               return self.amplitudo*other.amplitudo
"""
class operator:
     def __init__():
          pass
     
class creator_op(operator):
     def __init__(self, spatial_index:int):
          self.spatial_index = spatial_index

     def operate(self, state):
          if type(state) == harm_osc_ket_state:
               state.increase_qm_num(self.spatial_index)
               return math.sqrt(state.qm_nums[self.spatial_index]+1)*state
          elif type(state) == harm_osc_bra_state:
               
               return math.sqrt(state.qm_nums[self.spatial_index]+1)*state.increase_qm_num(self.spatial_index)
     def __mul__(self, ket: harm_osc_ket_state):
          return math.sqrt(ket.qm_nums[self.spatial_index]+1)*ket.increase_qm_num(self.spatial_index)
     def __rmul__(self, bra):
          pass

class annil_op:
     def __init__(self, spatial_index:int):
          self.spatial_index = spatial_index

     def operate(self, state):
          if type(state) == harm_osc_ket_state:
               return math.sqrt(state.qm_nums[self.spatial_index])*state.decrease_qm_num(self.spatial_index)
          elif type(state) == harm_osc_bra_state:
               return math.sqrt(state.qm_nums[self.spatial_index])*state.decrease_qm_num(self.spatial_index)
               
     def __mul__(self, ket: harm_osc_ket_state):
          return math.sqrt(ket.qm_nums[self.spatial_index])*ket.increase_qm_num(self.spatial_index)
     def __rmul__(self, bra):
          pass



class harm_osc_eigen_states:
     def __init__(self, energy_quantum, dim,order):
          self._bra_states = []
          self._ket_states = []

          self.energy_quantum = energy_quantum
          self.create_qm_nums(dim,order, [])
          self._bra_states = sorted(self._bra_states, key=lambda x: x.calc_order() )
          self.h_space_dim = len(self)


     def create_qm_nums(self, dim, order, curr_osc_coeffs: list):
          if len(curr_osc_coeffs)<dim:

               for i in range(0,order+1):
                    temp_curr_osc_coeffs = copy.deepcopy(curr_osc_coeffs)
                    temp_curr_osc_coeffs.append(i)
                    if sum(temp_curr_osc_coeffs)>order:
                         return
                    else:
                         self.create_qm_nums(dim, order,temp_curr_osc_coeffs )
          elif len(curr_osc_coeffs) == dim and sum(curr_osc_coeffs)<=order:
               self._bra_states.append( harm_osc_bra_state(self.energy_quantum, curr_osc_coeffs))
               self._ket_states.append( harm_osc_ket_state(self.energy_quantum, curr_osc_coeffs))

               return
          else:
               return


     def __len__(self):
          return len(self._bra_states)
     
     def __getitem__(self, position):
          return self._bra_states[position]
     
     def order(self):
          return sum(self)

class operator_builder:
     def __init__(self, eig_states:harm_osc_eigen_states, op: operator):
          self.eig_states = eig_states
          self.op = op
          #self.mxtype = mxtype

     def create_MatrixOperator(self):
          dim = len(self.eig_states)
          mx_op = np.zeros((dim, dim), dtype = np.complex64)
          for i in range(0,len(self.eig_states._bra_states)):
               for j in range(0,len(self.eig_states._ket_states)):
                    bra = self.eig_states._bra_states[i]
                    ket = self.eig_states._ket_states[j]
                    
                    
                    mx_op[i][j] = bra*self.op*ket
                    print(mx_op[i][j])
          #if self.mxtype == 'ordinary':
          return qm.MatrixOperator(maths.Matrix(mx_op))
          #elif self.mxtype == 'sparse':
          #     return MatrixOperator( maths.SparseMatrix( operator ) )
