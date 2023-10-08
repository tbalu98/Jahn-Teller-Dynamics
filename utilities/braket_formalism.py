import math
import copy
import itertools
from collections import namedtuple
quantum_subsystem_signature = namedtuple('quantum_subsystem_signature','name dim qm_nums_names' )


class quantum_state:
    def __init__(self, qm_nums: list, amplitude = complex(1.0, 0.0)):
        self.qm_nums = qm_nums
        self.amplitude = amplitude

    def __eq__(self, other):
        if other == None:
            return False
        else:
            return self.qm_nums == other.qm_nums

    def __mul__(self, other):
        if isinstance(other, complex) or isinstance(other, float):
            return quantum_state(self.qm_nums, self.amplitude*other)
        
        elif isinstance(other, quantum_state):
            if self == other:
                return self.amplitude*other.amplitude
            else:
                return complex(0.0,0.0)
        
        else:
            return complex(0.0, 0.0)
    
    def __rmul__(self, other):
        return self*other
        
    def __pow__(self, other):
        new_qm_nums = self.qm_nums + other.qm_nums
        new_amplitude = self.amplitude*other.amplitude
        return quantum_state(new_qm_nums, new_amplitude)
    
    def increase_qm_num(self, index):
        new_qm_nums = copy.deepcopy(self.qm_nums)
        new_qm_nums[index] += 1
        return quantum_state(new_qm_nums, self.amplitude)
    
    def decrease_qm_num(self, index):
        new_qm_nums = copy.deepcopy(self.qm_nums)
        new_qm_nums[index] -= 1
        return quantum_state(new_qm_nums, self.amplitude)
    
    def __pow__(self, other):
        if isinstance(other,quantum_state):
            return quantum_state(qm_nums= self.qm_nums + other.qm_nums)
        else:
            return None
    
    def calc_order(self):
        return sum(self.qm_nums)

    def __repr__(self):
        return str(self.amplitude) + ' ' +  str(self.qm_nums)
    def __getitem__(self, key):
        return self.qm_nums[key]
    

    def get_sub_state(self, indexes):
        return [ self.qm_nums[index]  for index in indexes]
class bra_state:

    def calc_order(self):
        return self.qm_state.calc_order()

    def __init__(self, qm_state = None,qm_nums = None, amplitude = complex(1.0, 0.0)):
        if qm_state != None:
            self.qm_state = qm_state
        elif qm_nums != None:    
            qm_state = quantum_state(qm_nums, amplitude)
            self.qm_state = qm_state
        else:
            return None
        
    def __repr__(self):
        return str(self.qm_state)
    
    def __mul__(self, other):
        if isinstance(other, ket_state):
            if self.qm_state == other.qm_state:
                return self.qm_state.amplitude * other.qm_state.amplitude
            else:
                return complex(0.0,0.0)
        elif isinstance(other,complex):
            new_qm_state = self.qm_state*other
            return bra_state(new_qm_state)
        elif isinstance(other, operator):
            return other.operate(self)
    
    def __pow__(self, other):
        return bra_state(qm_state = self.qm_state**other.qm_state)

            
class ket_state:
    def __init__(self, qm_state = None,qm_nums = None, amplitude = complex(1.0, 0.0)):
        if qm_state != None:
            self.qm_state = qm_state
        elif qm_nums != None:    
            qm_state = quantum_state(qm_nums, amplitude)
            self.qm_state = qm_state
        else:
            return None
    def calc_order(self):
        return self.qm_state.calc_order()

    def __pow__(self, other):
        return ket_state(qm_state = self.qm_state**other.qm_state)

    def get_sub_state(self, indexes):
        return self.qm_state.get_sub_state(indexes)
    
    def __repr__(self):
        return str(self.qm_state)

class operator:
    def __init__(self, fun):
        self.fun = fun
    def operate(self, other):
        return self.fun(other)

class creator_operator(operator):
    def __init__(self, raise_index, name = ''):
        self.raise_index = raise_index
        self.name = name
    def operate(self, other: bra_state):
        qm_state = other.qm_state
        new_state =  math.sqrt( qm_state[self.raise_index] +1)* qm_state.increase_qm_num(self.raise_index)
        return bra_state(qm_state=new_state)

class annil_operator(operator):
    def __init__(self, raise_index, name = ''):
        self.raise_index = raise_index
        self.name = name
    def operate(self, other: bra_state):
        qm_state = other.qm_state
        new_state =  math.sqrt( qm_state[self.raise_index] )* qm_state.decrease_qm_num(self.raise_index)
        return bra_state(qm_state=new_state)


class hilber_space_bases:
    def kron_hilber_spaces(hilber_spaces:list):
          return list( itertools.accumulate(hilber_spaces , lambda x,y: x**y) )[-1]

        
       
#        self._names = { name:num for name,num in zip(names, range(0, len(names))) }
#
#    def get_qm_num(self,bra_state:bra_state, key):
#        return bra_state[ self._names[key] ]



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
               
               self._bra_states.append( bra_state(qm_nums = curr_osc_coeffs))
               self._ket_states.append( ket_state(qm_nums =  curr_osc_coeffs))
               return
          
          else:
               return


    def harm_osc_sys(self, dim,order):
        curr_osc_coeffs = []
        self._bra_states = []
        self._ket_states = []

        self.create_hosc_eigen_states(dim, order, curr_osc_coeffs)
        self._bra_states = sorted(self._bra_states,key=lambda x:x.calc_order())
        self._ket_states = sorted(self._ket_states,key=lambda x:x.calc_order())
        self.dim = len(self._bra_states)
        return self


    def from_qm_nums_list(self, qm_nums_list):
        self._bra_vectors = []
        self._ket_vectors = []
        for qm_nums in qm_nums_list:
            self._bra_vectors.append(bra_state( qm_nums=qm_nums ))
            self._ket_vectors.append(ket_state( qm_nums=qm_nums ))

        return self

    def __init__(self, bra_states, ket_states, names = None):
        self._bra_states = bra_states
        self._ket_states = ket_states

    
    def filter_sub_syss_by_order(self,order ):
        state_indexes = []
        for index in range(0, len(self._bra_states)):
            ket = self._bra_states[index]

    def __len__(self):
        return len(self._ket_states)
     
    def __getitem__(self, position):
        return self._ket_states[position]

    def __pow__(self, other):
        if isinstance(other, hilber_space_bases):
            new_bra_states = [ bra_a**bra_b for bra_a in self._bra_states 
                                            for bra_b in self._bra_states ]
            new_ket_states = [ ket_a**ket_b for ket_a in self._ket_states 
                                            for ket_b in self._ket_states ]
            
            return hilber_space_bases(new_bra_states, new_ket_states)
        
        else:
            return None

class quantum_system_signature:
    def __init__(self, qm_sub_sys_lst: list[quantum_subsystem_signature]):
        self.qm_sub_sys_lst = qm_sub_sys_lst

    def get_dim_before_and_after(self,sub_sys_name):
        
        dim_before = 0
        dim_after = 0
        

        
        qm_sub_sys_lst_it = iter(self.qm_sub_sys_lst)

        qm_sub_sys = next(qm_sub_sys_lst_it,None)
        
        while  qm_sub_sys is not None and qm_sub_sys.name != sub_sys_name:
            dim_before += qm_sub_sys.dim
            qm_sub_sys = next(qm_sub_sys_lst_it,None)

        if qm_sub_sys == None:
            return dim_before, 1

        qm_sub_sys = next(qm_sub_sys_lst_it,None)

        while qm_sub_sys != None:
            dim_after += qm_sub_sys.dim
            qm_sub_sys = next(qm_sub_sys_lst_it,None)

        if dim_before == 0:
            dim_before = 1
        if dim_after == 0:
            dim_after = 1

        return dim_before, dim_after



class quantum_sub_system:
    def __init__(self, bra_states, ket_states):
        self._bra_states = bra_states
        self._ket_states = ket_states

    def __len__(self):
        return len(self._bra_states)
     
    def __getitem__(self, position):
        return self._bra_states[position]

    def __pow__(self, other):
        if isinstance(other, quantum_sub_system):
            new_bra_states = []
            new_ket_states = []
            for (bra_state_a, bra_state_b) in zip(self._bra_states, other._bra_states):
                new_bra_states.append(bra_state_a**bra_state_b)

            for (ket_state_a, ket_state_b) in zip(self._ket_states, other._ket_states):
                new_ket_states.append(ket_state_a**ket_state_b)
            return quantum_sub_system(new_bra_states, new_ket_states)
        
        else:
            return None




class new_harm_osc:
     def __init__(self, energy_quantum, dim,order):
        bra_states = []
        ket_states = []

        self.energy_quantum = energy_quantum
        self.create_eigen_states(dim,order, [], bra_states, ket_states)
        self._bra_states = sorted(self._bra_states, key=lambda x: x.calc_order() )
        self.h_space_dim = len(self)


     def create_eigen_states(self, dim, order, curr_osc_coeffs: list, bra_states:list, ket_states:list):
          if len(curr_osc_coeffs)<dim:

               for i in range(0,order+1):
                    temp_curr_osc_coeffs = copy.deepcopy(curr_osc_coeffs)
                    temp_curr_osc_coeffs.append(i)
                    if sum(temp_curr_osc_coeffs)>order:
                         return
                    else:
                         self.create_eigen_states(dim, order,temp_curr_osc_coeffs )

          elif len(curr_osc_coeffs) == dim and sum(curr_osc_coeffs)<=order:
               
               bra_states.append( bra_state(qm_nums= curr_osc_coeffs))
               ket_states.append( ket_state(qm_nums =  curr_osc_coeffs))
               return
          
          else:
               return


     def __len__(self):
          return len(self._bra_states)
     
     def __getitem__(self, position):
          return self._bra_states[position]
     
     def order(self):
          return sum(self)


class harm_osc:
     def __init__(self, energy_quantum, dim,order):
        self._bra_states = []
        self._ket_states = []

        self.energy_quantum = energy_quantum
        self.create_eigen_states(dim,order, [])
        self._bra_states = sorted(self._bra_states, key=lambda x: x.calc_order() )
        self.h_space_dim = len(self)


     def create_eigen_states(self, dim, order, curr_osc_coeffs: list):
          if len(curr_osc_coeffs)<dim:

               for i in range(0,order+1):
                    temp_curr_osc_coeffs = copy.deepcopy(curr_osc_coeffs)
                    temp_curr_osc_coeffs.append(i)
                    if sum(temp_curr_osc_coeffs)>order:
                         return
                    else:
                         self.create_eigen_states(dim, order,temp_curr_osc_coeffs )
          elif len(curr_osc_coeffs) == dim and sum(curr_osc_coeffs)<=order:
               self._bra_states.append( bra_state(qm_nums= curr_osc_coeffs))
               self._ket_states.append( ket_state(qm_nums =  curr_osc_coeffs))

               return
          else:
               return


     def __len__(self):
          return len(self._bra_states)
     
     def __getitem__(self, position):
          return self._bra_states[position]
     
     def order(self):
          return sum(self)

"""

class operator_builder:
     def __init__(self, eig_states:harm_osc, op: operator):
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
          return mf.MatrixOperator(maths.Matrix(mx_op))
"""

class multi_mode_harm_osc:
    
    def __init__(self, one_mode_harm_oscs:list):
        self.one_mode_harm_oscs = one_mode_harm_oscs
    
    