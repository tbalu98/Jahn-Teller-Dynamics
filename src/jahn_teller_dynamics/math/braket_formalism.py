import math
import copy

from collections import namedtuple
import numpy as np


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

    def to_ket_state(self):
        new_qm_state = copy.deepcopy(self.qm_state)
        new_qm_state.amplitude = np.conj(self.qm_state.amplitude)
        return ket_state(qm_state = new_qm_state)



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
        txt = '<'

        qm_nums_len = len(self.qm_state.qm_nums)
        for i in range(0,qm_nums_len):
            txt+=str(self.qm_state[i])
            
            if i !=qm_nums_len-1:
                txt+= ','

        txt += '|'
        return txt
    
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
        txt = '|'

        qm_nums_len = len(self.qm_state.qm_nums)
        for i in range(0,qm_nums_len):
            txt+=str(self.qm_state[i])
            
            if i !=qm_nums_len-1:
                txt+= ','
        txt += '>'
        return txt
    
    def __eq__(self, other):
        return self.qm_state == other.qm_state
    
    def to_bra_state(self):
        new_qm_state = copy.deepcopy(self.qm_state)
        new_qm_state.amplitude = np.conj(self.qm_state.amplitude)

        return bra_state(self.qm_state)

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

class raise_index_operator(operator):
    def __init__(self, raise_index, name = ''):
        self.raise_index = raise_index
        self.name = name
    def operate(self, other: bra_state):
        qm_state = other.qm_state
        new_state = qm_state.increase_qm_num(self.raise_index)
        return bra_state(qm_state=new_state)


class annil_operator(operator):
    def __init__(self, raise_index, name = ''):
        self.raise_index = raise_index
        self.name = name
    def operate(self, other: bra_state):
        qm_state = other.qm_state
        new_state =  math.sqrt( qm_state[self.raise_index] )* qm_state.decrease_qm_num(self.raise_index)
        return bra_state(qm_state=new_state)

