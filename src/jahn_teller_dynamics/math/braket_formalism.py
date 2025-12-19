"""
Bra-ket formalism for quantum mechanics calculations.

This module provides classes for representing quantum states in bra-ket notation,
including quantum states, bra states, ket states, and various operators for
quantum mechanical calculations.
"""

import math
import cmath
import copy
from typing import Optional, List, Union, Any

import numpy as np


class quantum_state:
    """
    Represents a quantum state with quantum numbers and amplitude.
    
    This class encapsulates a quantum state with a list of quantum numbers
    and a complex amplitude. It supports operations like multiplication,
    tensor products, and quantum number manipulation.
    
    Attributes:
        qm_nums: List of quantum numbers (e.g., [n1, n2, n3] for occupation numbers)
        amplitude: Complex amplitude of the state
    """
    
    def __init__(self, qm_nums: List[int], amplitude: complex = complex(1.0, 0.0)) -> None:
        """
        Initialize a quantum state.
        
        Args:
            qm_nums: List of quantum numbers (can be negative)
            amplitude: Complex amplitude (default: 1.0 + 0.0j)
            
        Raises:
            ValueError: If qm_nums is empty
        """
        if not qm_nums:
            raise ValueError("qm_nums cannot be empty")
        self.qm_nums = qm_nums
        self.amplitude = amplitude

    def __eq__(self, other: Any) -> bool:
        """
        Check equality of quantum states.
        
        Two states are equal if they have the same quantum numbers
        (amplitude is not considered for equality).
        
        Args:
            other: Object to compare with
            
        Returns:
            True if quantum numbers match, False otherwise
        """
        if other is None:
            return False
        if not isinstance(other, quantum_state):
            return False
        return self.qm_nums == other.qm_nums

    def __mul__(self, other: Any) -> Union['quantum_state', complex]:
        """
        Multiply quantum state with scalar or compute inner product with another state.
        
        Args:
            other: Scalar (complex/float) or another quantum_state
            
        Returns:
            - If other is scalar: new quantum_state with multiplied amplitude
            - If other is quantum_state: inner product (complex) if states match, else 0.0
        """
        if isinstance(other, (complex, float, int)):
            return quantum_state(self.qm_nums, self.amplitude * other)
        
        elif isinstance(other, quantum_state):
            if self == other:
                return self.amplitude * other.amplitude
            else:
                return complex(0.0, 0.0)
        
        else:
            return complex(0.0, 0.0)
    
    def __rmul__(self, other: Any) -> Union['quantum_state', complex]:
        """Right multiplication (scalar * state)."""
        return self * other
        
    def __pow__(self, other: 'quantum_state') -> 'quantum_state':
        """
        Tensor product of two quantum states.
        
        Combines quantum numbers and multiplies amplitudes.
        
        Args:
            other: Another quantum_state to tensor with
            
        Returns:
            New quantum_state with combined quantum numbers and multiplied amplitudes
            
        Raises:
            TypeError: If other is not a quantum_state
        """
        if not isinstance(other, quantum_state):
            raise TypeError(f"Cannot tensor with {type(other)}")
        new_qm_nums = self.qm_nums + other.qm_nums
        new_amplitude = self.amplitude * other.amplitude
        return quantum_state(new_qm_nums, new_amplitude)
    
    def increase_qm_num(self, index: int) -> 'quantum_state':
        """
        Increase quantum number at given index by 1.
        
        Args:
            index: Index of quantum number to increase
            
        Returns:
            New quantum_state with increased quantum number
            
        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self.qm_nums):
            raise IndexError(f"Index {index} out of range for quantum numbers")
        new_qm_nums = copy.deepcopy(self.qm_nums)
        new_qm_nums[index] += 1
        return quantum_state(new_qm_nums, self.amplitude)
    
    def decrease_qm_num(self, index: int) -> 'quantum_state':
        """
        Decrease quantum number at given index by 1.
        
        Args:
            index: Index of quantum number to decrease
            
        Returns:
            New quantum_state with decreased quantum number
            
        Raises:
            IndexError: If index is out of range
        """
        if index < 0 or index >= len(self.qm_nums):
            raise IndexError(f"Index {index} out of range for quantum numbers")
        new_qm_nums = copy.deepcopy(self.qm_nums)
        new_qm_nums[index] -= 1
        return quantum_state(new_qm_nums, self.amplitude)
    
    def calc_order(self) -> int:
        """
        Calculate the order (sum) of quantum numbers.
        
        Returns:
            Sum of all quantum numbers
        """
        return sum(self.qm_nums)

    def __repr__(self) -> str:
        """String representation of the quantum state."""
        return f"{self.amplitude} {self.qm_nums}"
    
    def __getitem__(self, key: int) -> int:
        """
        Get quantum number at given index.
        
        Args:
            key: Index of quantum number
            
        Returns:
            Quantum number at index
        """
        return self.qm_nums[key]
    
    def get_sub_state(self, indexes: List[int]) -> List[int]:
        """
        Get subset of quantum numbers at specified indexes.
        
        Args:
            indexes: List of indices to extract
            
        Returns:
            List of quantum numbers at specified indices
        """
        return [self.qm_nums[index] for index in indexes]
    
    def __hash__(self) -> int:
        """
        Hash function for quantum state (based on quantum numbers only).
        
        Allows quantum_state to be used in sets and as dictionary keys.
        """
        return hash(tuple(self.qm_nums))


class bra_state:
    """
    Represents a bra state <ψ| in quantum mechanics.
    
    A bra state is the dual of a ket state, used for computing inner products
    and expectation values.
    """
    
    def __init__(
        self, 
        qm_state: Optional[quantum_state] = None,
        qm_nums: Optional[List[int]] = None, 
        amplitude: complex = complex(1.0, 0.0)
    ) -> None:
        """
        Initialize a bra state.
        
        Args:
            qm_state: Existing quantum_state object
            qm_nums: List of quantum numbers (alternative to qm_state)
            amplitude: Complex amplitude (used only if qm_nums is provided)
            
        Raises:
            ValueError: If neither qm_state nor qm_nums is provided
        """
        if qm_state is not None:
            self.qm_state = qm_state
        elif qm_nums is not None:
            self.qm_state = quantum_state(qm_nums, amplitude)
        else:
            raise ValueError("Either qm_state or qm_nums must be provided")

    def to_ket_state(self) -> 'ket_state':
        """
        Convert bra state to ket state (complex conjugate of amplitude).
        
        Returns:
            ket_state with conjugated amplitude
        """
        new_qm_state = copy.deepcopy(self.qm_state)
        new_qm_state.amplitude = np.conj(self.qm_state.amplitude)
        return ket_state(qm_state=new_qm_state)

    def calc_order(self) -> int:
        """
        Calculate the order of the quantum state.
        
        Returns:
            Sum of quantum numbers
        """
        return self.qm_state.calc_order()

    def __repr__(self) -> str:
        """String representation: <n1,n2,n3|"""
        txt = '<'
        qm_nums_len = len(self.qm_state.qm_nums)
        for i in range(qm_nums_len):
            txt += str(self.qm_state[i])
            if i != qm_nums_len - 1:
                txt += ','
        txt += '|'
        return txt
    
    def __mul__(self, other: Any) -> Union[complex, 'bra_state']:
        """
        Multiply bra state with ket state (inner product) or scalar.
        
        Args:
            other: ket_state, complex number, or operator
            
        Returns:
            - If other is ket_state: inner product (complex)
            - If other is complex: new bra_state with scaled amplitude
            - If other is operator: result of operator acting on bra
        """
        if isinstance(other, ket_state):
            if self.qm_state == other.qm_state:
                return self.qm_state.amplitude * other.qm_state.amplitude
            else:
                return complex(0.0, 0.0)
        elif isinstance(other, (complex, float, int)):
            new_qm_state = self.qm_state * other
            return bra_state(new_qm_state)
        elif isinstance(other, operator):
            return other.operate(self)
        else:
            raise TypeError(f"Cannot multiply bra_state with {type(other)}")
    
    def __pow__(self, other: 'ket_state') -> 'bra_state':
        """
        Tensor product of bra and ket states.
        
        Args:
            other: ket_state to tensor with
            
        Returns:
            New bra_state with combined quantum numbers
        """
        return bra_state(qm_state=self.qm_state ** other.qm_state)


class ket_state:
    """
    Represents a ket state |ψ> in quantum mechanics.
    
    A ket state represents a quantum state vector in Hilbert space.
    """
    
    def __init__(
        self, 
        qm_state: Optional[quantum_state] = None,
        qm_nums: Optional[List[int]] = None, 
        amplitude: complex = complex(1.0, 0.0)
    ) -> None:
        """
        Initialize a ket state.
        
        Args:
            qm_state: Existing quantum_state object
            qm_nums: List of quantum numbers (alternative to qm_state)
            amplitude: Complex amplitude (used only if qm_nums is provided)
            
        Raises:
            ValueError: If neither qm_state nor qm_nums is provided
        """
        if qm_state is not None:
            self.qm_state = qm_state
        elif qm_nums is not None:
            qm_state = quantum_state(qm_nums, amplitude)
            self.qm_state = qm_state
        else:
            raise ValueError("Either qm_state or qm_nums must be provided")
    
    def calc_order(self) -> int:
        """
        Calculate the order of the quantum state.
        
        Returns:
            Sum of quantum numbers
        """
        return self.qm_state.calc_order()

    def __pow__(self, other: 'ket_state') -> 'ket_state':
        """
        Tensor product of two ket states.
        
        Args:
            other: Another ket_state to tensor with
            
        Returns:
            New ket_state with combined quantum numbers
        """
        return ket_state(qm_state=self.qm_state ** other.qm_state)

    def get_sub_state(self, indexes: List[int]) -> List[int]:
        """
        Get subset of quantum numbers at specified indexes.
        
        Args:
            indexes: List of indices to extract
            
        Returns:
            List of quantum numbers at specified indices
        """
        return self.qm_state.get_sub_state(indexes)
    
    def __repr__(self) -> str:
        """String representation: |n1,n2,n3>"""
        txt = '|'
        qm_nums_len = len(self.qm_state.qm_nums)
        for i in range(qm_nums_len):
            txt += str(self.qm_state[i])
            if i != qm_nums_len - 1:
                txt += ','
        txt += '>'
        return txt
    
    def __eq__(self, other: Any) -> bool:
        """
        Check equality of ket states.
        
        Args:
            other: Object to compare with
            
        Returns:
            True if quantum states match, False otherwise
        """
        if not isinstance(other, ket_state):
            return False
        return self.qm_state == other.qm_state
    
    def to_bra_state(self) -> bra_state:
        """
        Convert ket state to bra state (complex conjugate of amplitude).
        
        Returns:
            bra_state with conjugated amplitude
        """
        new_qm_state = copy.deepcopy(self.qm_state)
        new_qm_state.amplitude = np.conj(self.qm_state.amplitude)
        return bra_state(new_qm_state)


class operator:
    """
    Base class for quantum operators.
    
    Operators act on quantum states to produce new states.
    """
    
    def __init__(self, fun: Any) -> None:
        """
        Initialize operator with a function.
        
        Args:
            fun: Function that defines the operator action
        """
        self.fun = fun
    
    def operate(self, other: bra_state) -> Any:
        """
        Apply operator to a bra state.
        
        Args:
            other: bra_state to operate on
            
        Returns:
            Result of operator action
        """
        return self.fun(other)


class creator_operator(operator):
    """
    Creation operator (a†) for raising quantum numbers.
    
    Implements the standard creation operator with proper normalization:
    a†|n> = √(n+1) |n+1>
    """
    
    def __init__(self, raise_index: int, name: str = '') -> None:
        """
        Initialize creation operator.
        
        Args:
            raise_index: Index of quantum number to raise
            name: Optional name for the operator
        """
        self.raise_index = raise_index
        self.name = name
    
    def operate(self, other: bra_state) -> bra_state:
        """
        Apply creation operator to bra state.
        
        Args:
            other: bra_state to operate on
            
        Returns:
            New bra_state with increased quantum number and proper normalization
            
        Note:
            Uses cmath.sqrt to handle cases where (n+1) might be negative.
        """
        qm_state = other.qm_state
        new_state = cmath.sqrt(qm_state[self.raise_index] + 1) * qm_state.increase_qm_num(self.raise_index)
        return bra_state(qm_state=new_state)


class raise_index_operator(operator):
    """
    Simple raising operator that increases quantum number without normalization.
    
    Unlike creator_operator, this does not include the √(n+1) factor.
    """
    
    def __init__(self, raise_index: int, name: str = '') -> None:
        """
        Initialize raising operator.
        
        Args:
            raise_index: Index of quantum number to raise
            name: Optional name for the operator
        """
        self.raise_index = raise_index
        self.name = name
    
    def operate(self, other: bra_state) -> bra_state:
        """
        Apply raising operator to bra state.
        
        Args:
            other: bra_state to operate on
            
        Returns:
            New bra_state with increased quantum number
        """
        qm_state = other.qm_state
        new_state = qm_state.increase_qm_num(self.raise_index)
        return bra_state(qm_state=new_state)


class annihilator_operator(operator):
    """
    Annihilation operator (a) for lowering quantum numbers.
    
    Implements the standard annihilation operator with proper normalization:
    a|n> = √n |n-1>
    """
    
    def __init__(self, raise_index: int, name: str = '') -> None:
        """
        Initialize annihilation operator.
        
        Args:
            raise_index: Index of quantum number to lower
            name: Optional name for the operator
        """
        self.raise_index = raise_index
        self.name = name
    
    def operate(self, other: bra_state) -> bra_state:
        """
        Apply annihilation operator to bra state.
        
        Args:
            other: bra_state to operate on
            
        Returns:
            New bra_state with decreased quantum number and proper normalization
            
        Note:
            Uses cmath.sqrt to handle negative quantum numbers (returns complex result).
            For standard quantum mechanics, quantum numbers should be non-negative,
            but this implementation supports negative values for flexibility.
        """
        qm_state = other.qm_state
        qm_num = qm_state[self.raise_index]
        # Use cmath.sqrt to handle negative quantum numbers (returns complex)
        sqrt_factor = cmath.sqrt(qm_num)
        new_state = sqrt_factor * qm_state.decrease_qm_num(self.raise_index)
        return bra_state(qm_state=new_state)
