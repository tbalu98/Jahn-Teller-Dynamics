"""
Matrix mechanics module for quantum mechanical calculations.

This module provides core classes and functions for quantum mechanical calculations,
including ket vectors, bra vectors, matrix operators, and eigenvalue/eigenvector
computations. It serves as the foundation for Jahn-Teller dynamics calculations.
"""

import itertools
import numpy as np
from jahn_teller_dynamics.math.braket_formalism import  operator
import jahn_teller_dynamics.math.maths as  maths
import copy
import pandas as pd
import jahn_teller_dynamics.math.braket_formalism as  bf
import math
from collections import namedtuple
from typing import Optional, List, Union, Tuple, Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from typing import TypeVar
    from jahn_teller_dynamics.math.eigen_solver import EigenSolver
    T = TypeVar('T')




# Constants
dtype = np.complex64
SQRT_2 = math.sqrt(2.0)  # Square root of 2, used in complex basis transformations
DEFAULT_ROUNDING_PRECISION = 10  # Default number of decimal places for rounding
EIGENVALUE_ROUNDING_PRECISION = 4  # Number of decimal places for eigenvalue rounding


class ket_vector:
     """
     Ket vector class representing a quantum state vector.
     
     A ket vector is a column vector in a Hilbert space, representing a quantum
     mechanical state. It contains coefficients for each basis state and optionally
     an associated eigenvalue (energy).
     """

     @staticmethod
     def from_str_list(str_list: list[str], eigen_energy: Optional[float] = None) -> 'ket_vector':
          """
          Create ket vector from list of string coefficients.
          
          Args:
              str_list: List of coefficient strings (e.g., ['1.0', '0.0'])
              eigen_energy: Optional eigenvalue/energy for this state
              
          Returns:
              ket_vector: New ket vector instance
          """
          coeffs = []
          for coeff_str in str_list:
               coeff = complex(coeff_str)
               coeffs.append(coeff)

          return ket_vector(coeffs, eigen_energy)



     def __init__(self, coeffs: Union[maths.col_vector, List[complex]], eigen_val: Optional[float] = None, name: str = '', subsys_name: str = ''):
          self.name = name
          self.subsys_name = subsys_name
          self.eigen_val = eigen_val
          if isinstance(coeffs, maths.col_vector):
               self.amplitudo = complex(1.0,0)
               self.coeffs = coeffs
               self.dim = len(self.coeffs.coeffs)
          elif isinstance(coeffs, list):
               self.amplitudo = complex(1.0,0)

               self.coeffs = maths.col_vector(np.matrix( [  [num] for num in coeffs ] ))
               self.dim = len(self.coeffs.coeffs)
          else:
               raise TypeError(f"coeffs must be maths.col_vector or list, got {type(coeffs)}")
     
     def to_dataframe(self, bases: 'hilber_space_bases') -> pd.DataFrame:
          ket_dict = {}
          index_col_name = str(bases.qm_nums_names)
          ket_dict[index_col_name] = list( map( lambda x: str(x), bases._ket_states ) )

          ket_dict['coefficients'] = self.coeffs.tolist()

          ket_dataframe = pd.DataFrame.from_dict(ket_dict)

          ket_dataframe = ket_dataframe.set_index(index_col_name)

          return ket_dataframe
     
     def map(self, fun: Callable) -> 'ket_vector':
          return ket_vector(self.coeffs.map(fun))

     def tolist(self) -> List[complex]:
          return self.coeffs.tolist()

     def abs_sq(self) -> float:
          return sum( [ abs(complex(coeff))**2 for coeff in self.coeffs ] )

     def normalize(self) -> 'ket_vector':
          norm_factor = self.abs_sq()**0.5
          if norm_factor == 0:
               raise ValueError("Cannot normalize zero vector")
          return self/norm_factor

     def set_item(self, index: int, item: Union[complex, float]) -> 'ket_vector':
          return ket_vector(self.coeffs.set_item(index, item))

     def __repr__(self) -> str:
          if self.eigen_val is None:
               return str(self.coeffs)
          else:
               return str('eigen value: ') +str(self.eigen_val) + '\n'  +str(self.coeffs)
     
     def __getitem__(self, key: int) -> Union[complex, float]:
          """
          Get coefficient at given index.
          
          Args:
              key: Index to access
              
          Returns:
              Coefficient value at index
              
          Raises:
              IndexError: If index is out of range
          """
          if not isinstance(key, int) or key < 0 or key >= self.dim:
               raise IndexError(f"Index {key} out of range for vector of dimension {self.dim}")
          return self.coeffs[key]
     
     def set_val(self, index: int, val: Union[complex, float]) -> None:
          """
          Set value at given index.
          
          Args:
              index: Index to set
              val: Value to set
              
          Raises:
              IndexError: If index is out of range
          """
          if not isinstance(index, int) or index < 0 or index >= self.dim:
               raise IndexError(f"Index {index} out of range for vector of dimension {self.dim}")
          self.coeffs.set_val(index,val)

     def to_bra_vector(self) -> 'bra_vector':
          return bra_vector( self.coeffs.to_row_vector(), eigen_val = self.eigen_val )
      
     def __add__(self, other: 'ket_vector') -> 'ket_vector':
          if isinstance(other, ket_vector):
               return ket_vector(self.coeffs + other.coeffs)
          return NotImplemented

     def __sub__(self, other: 'ket_vector') -> 'ket_vector':
          if isinstance(other, ket_vector):
               return ket_vector(self.coeffs - other.coeffs)
          return NotImplemented
     
     def __rsub__(self, other: 'ket_vector') -> 'ket_vector':
          if isinstance(other, ket_vector):
               return ket_vector(other.coeffs - self.coeffs)
          return NotImplemented

     def __mul__(self, other: Union[complex, float]) -> 'ket_vector':
          return ket_vector(self.coeffs* other)
     
     def __rmul__(self, other: Union[complex, float]) -> 'ket_vector':
          return ket_vector(self.coeffs.__rmul__(other))

     def __truediv__(self, other: Union[complex, float]) -> 'ket_vector':
          return ket_vector(self.coeffs/other)

     def __abs__(self) -> float:
          return abs(self.coeffs)

     def __eq__(self, other: Any) -> bool:
          if not isinstance(other, ket_vector):
               return NotImplemented
          return (self.name == other.name and 
                  self.subsys_name == other.subsys_name and 
                  self.coeffs == other.coeffs and 
                  self.eigen_val == other.eigen_val)

     def round(self, dig: int) -> 'ket_vector':
          return ket_vector(coeffs=self.coeffs.round(dig), name = self.name, subsys_name = self.subsys_name, eigen_val = self.eigen_val)

     def calc_abs_square(self) -> 'ket_vector':
          return  ket_vector(self.coeffs.calc_abs_square())

class bases_system:
     def __init__(self, bases_kets:list[ket_vector]):
          self.bases_kets = bases_kets



class bra_vector:
     """
     Bra vector class representing the dual of a ket vector.
     
     A bra vector is the Hermitian conjugate (adjoint) of a ket vector,
     represented as a row vector. Used for computing inner products.
     """

     @staticmethod
     def from_str_list(str_list: list[str], eigen_energy: Optional[float] = None) -> 'bra_vector':
          """
          Create bra vector from list of string coefficients.
          
          Args:
              str_list: List of coefficient strings
              eigen_energy: Optional eigenvalue/energy
              
          Returns:
              bra_vector: New bra vector instance
          """
          coeffs = []
          for coeff_str in str_list:
               coeff = complex(coeff_str)
               coeffs.append(coeff)

          return bra_vector(coeffs, eigen_energy)

     def __init__(self, coeffs: Union[maths.row_vector, List[complex]], eigen_val: Optional[float] = None) -> None:
          """
          Initialize bra vector.
          
          Args:
              coeffs: Row vector or list of coefficients
              eigen_val: Optional eigenvalue/energy
              
          Raises:
              TypeError: If coeffs is not a row_vector or list
          """
          self.eigen_val = eigen_val
          if isinstance(coeffs, maths.row_vector):
               self.coeffs = coeffs
               self.dim = len(self.coeffs.coeffs[0])
          elif isinstance(coeffs, list):
               self.coeffs = maths.row_vector(np.matrix( [  [num for num in coeffs ]  ] ) )
               self.dim = len(self.coeffs.coeffs[0])
          else:
               raise TypeError(f"coeffs must be maths.row_vector or list, got {type(coeffs)}")
          self.amplitudo = complex(1.0,0)
          
          

     def __mul__(self, other: Union[ket_vector, 'MatrixOperator']) -> Union[complex, 'bra_vector']:
          """
          Multiply bra vector with ket vector or matrix operator.
          
          Args:
              other: Ket vector or matrix operator
              
          Returns:
              Complex number (inner product) if other is ket_vector,
              bra_vector if other is MatrixOperator
          """
          if isinstance(other, ket_vector):
               return complex(self.coeffs*other.coeffs)
          elif isinstance(other, MatrixOperator):
               return bra_vector(self.coeffs*other.matrix)
          return NotImplemented

     def __rmul__(self, other: ket_vector) -> 'MatrixOperator':
          """
          Right multiply bra vector with ket vector.
          
          Args:
              other: Ket vector
              
          Returns:
              MatrixOperator: Outer product matrix
          """
          return MatrixOperator(other.coeffs*self.coeffs)
     
     def __repr__(self) -> str:
          """Return string representation of bra vector."""
          return str(self.coeffs)
     
     def __getitem__(self, key: int) -> Union[complex, float]:
          """
          Get coefficient at given index.
          
          Args:
              key: Index to access
              
          Returns:
              Coefficient value at index
          """
          return self.coeffs[key]
     
     def set_val(self, index: int, val: Union[complex, float]) -> None:
          """
          Set value at given index.
          
          Args:
              index: Index to set
              val: Value to set
          """
          self.coeffs.set_val(index,val)
          


class hilber_space_bases:
    """
    Hilbert space basis class for quantum mechanical calculations.
    
    This class manages bra and ket states that form a basis for a quantum
    mechanical Hilbert space. It provides methods for creating, manipulating,
    and transforming quantum states.
    """
    
    def __init__(self, bra_states: Optional[List[bf.bra_state]] = None, ket_states: Optional[List[bf.ket_state]] = None, names: Optional[List[str]] = None):
        """
        Initialize Hilbert space basis.
        
        Args:
            bra_states: List of bra states (optional)
            ket_states: List of ket states (optional)
            names: List of quantum number names (optional)
        """
        self._ket_states = []
        self._bra_states = []
        self.base_vectors = {}

        if bra_states is None:
            self._bra_states = []
        else:
            self._bra_states = bra_states

        if ket_states is None:
            self._ket_states = []
        else:
            self._ket_states = ket_states
        if names is None:
            self.qm_nums_names = []
        else:
            self.qm_nums_names = names
        self.dim = len(self._bra_states)

    def create_ket_vector(self, ket_states: List[bf.ket_state]) -> ket_vector:
          """
          Create a ket vector from a list of ket states.
          
          Args:
              ket_states: List of ket states to combine into a ket vector
              
          Returns:
              ket_vector: New ket vector with coefficients from the ket states
          """
          ket_vec = ket_vector(maths.col_vector(np.zeros((len(self._ket_states), 1), dtype=np.complex128  )))

          for ket_state in ket_states:
               ket_state_index = self.get_ket_state_index(ket_state)
               coeff = ket_state.qm_state.amplitude
               ket_vec.set_val(ket_state_index, coeff)

          return ket_vec
               



    def get_ket_index(self, find_ket_state: bf.ket_state) -> Optional[int]:
          """
          Get the index of a ket state in the basis.
          
          Args:
              find_ket_state: The ket state to find
              
          Returns:
              int: The index of the ket state, or None if not found
          """
          for (i, ket_state) in zip(range(0, len(self._ket_states)), self._ket_states):
               if find_ket_state == ket_state:
                    return i
          return None



    def create_trf_op(self, basis_name: str) -> 'MatrixOperator':
          """
          Create basis transformation operator.
          
          Args:
              basis_name: Name of the basis to transform to
              
          Returns:
              MatrixOperator: Basis transformation matrix
              
          Raises:
              KeyError: If basis_name is not found in base_vectors
          """
          return MatrixOperator.basis_trf_matrix(self.base_vectors[basis_name][0])


    @staticmethod
    def kron_hilber_spaces(hilber_spaces: List['hilber_space_bases']) -> 'hilber_space_bases':
          """
          Compute tensor product (Kronecker product) of multiple Hilbert spaces.
          
          Args:
              hilber_spaces: List of hilber_space_bases to combine
              
          Returns:
              hilber_space_bases: Combined Hilbert space
          """
          return list( itertools.accumulate(hilber_spaces , lambda x,y: x**y) )[-1]



    def create_hosc_eigen_states(self, dim: int, order: int, curr_osc_coeffs: List[int]) -> None:
          """
          Recursively create harmonic oscillator eigenstates.
          
          Args:
              dim: Spatial dimension
              order: Maximum order (total quantum number)
              curr_osc_coeffs: Current oscillator coefficients (used in recursion)
          """
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


    def harm_osc_sys(self, dim: int, order: int, qm_nums_names: List[str]) -> 'hilber_space_bases':
          """
          Create harmonic oscillator system basis.
          
          Args:
              dim: Spatial dimension
              order: Maximum order (total quantum number)
              qm_nums_names: List of quantum number names
              
          Returns:
              hilber_space_bases: Self (for method chaining)
          """
          curr_osc_coeffs = []
          self._bra_states = []
          self._ket_states = []

          self.create_hosc_eigen_states(dim, order, curr_osc_coeffs)
          self._bra_states = sorted(self._bra_states,key=lambda x:(x.calc_order(), *x.qm_state.qm_nums))
          self._ket_states = sorted(self._ket_states,key=lambda x:(x.calc_order() , *x.qm_state.qm_nums ))
          self.dim = len(self._bra_states)
          self.qm_nums_names = qm_nums_names
          return self


    def from_qm_nums_list(self, qm_nums_list: List[List[Union[int, float]]], qm_nums_names: Optional[List[str]] = None) -> 'hilber_space_bases':
          """
          Create basis from list of quantum number lists.
          
          Args:
              qm_nums_list: List of quantum number lists
              qm_nums_names: Optional list of quantum number names
              
          Returns:
              hilber_space_bases: Self (for method chaining)
          """
          self._bra_states = []
          self._ket_states = []
          for qm_nums in qm_nums_list:
              self._bra_states.append(bf.bra_state( qm_nums=qm_nums ))
              self._ket_states.append(bf.ket_state( qm_nums=qm_nums ))
          self.dim = len(self._bra_states)
          self.qm_nums_names = qm_nums_names
          return self

    def savetxt(self, filename: str) -> None:
          """
          Save basis to text file.
          
          Args:
              filename: Path to output file
          """
          txt = ''

          txt+=str(self.qm_nums_names) + "\n"

          for ket in self._ket_states:
              txt+= str(ket) + '\n'

          with open(filename, "w") as text_file:
              text_file.write(txt)

    def reduce_space(self, new_dim: int) -> 'hilber_space_bases':
          """
          Reduce Hilbert space to specified dimension.
          
          Args:
              new_dim: New dimension (number of basis states to keep)
              
          Returns:
              hilber_space_bases: New reduced Hilbert space
          """
          return hilber_space_bases(self._bra_states[0:new_dim], self._ket_states[0:new_dim],names = self.qm_nums_names)

    def get_ket_state_index(self, ks:bf.ket_state):
        """
        Get the index of a ket state in the basis.
        
        Args:
            ks: The ket state to find
            
        Returns:
            int: The index of the ket state
            
        Raises:
            ValueError: If the ket state is not found in the basis
        """
        exam_ket_states = np.array(self._ket_states)
        indices = np.where(exam_ket_states == ks)[0]
        
        if len(indices) == 0:
            raise ValueError(f"Ket state {ks} not found in basis")
        
        return int(indices[0])


    def __len__(self) -> int:
        return len(self._ket_states)
     
    def __getitem__(self, position: Union[int, slice]) -> Union[bf.ket_state, List[bf.ket_state]]:
        """
        Get ket state at given position.
        
        Args:
            position: Index or slice
            
        Returns:
            Ket state(s) at position
            
        Raises:
            IndexError: If position is out of range
        """
        if isinstance(position, int) and (position < 0 or position >= len(self._ket_states)):
             raise IndexError(f"Index {position} out of range for basis of dimension {len(self._ket_states)}")
        return self._ket_states[position]

    def __pow__(self, other: 'hilber_space_bases') -> Optional['hilber_space_bases']:
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
     """
     Matrix operator class for quantum mechanical operators.
     
     Represents a quantum mechanical operator as a matrix. Provides methods
     for matrix operations, eigenvalue/eigenvector calculations, and basis
     transformations.
     """

     def tolist(self) -> List[List[complex]]:
          """
          Convert matrix to list representation.
          
          Returns:
              list: Matrix as nested list
          """
          return self.matrix.tolist()

     @staticmethod
     def pauli_x_mx_op() -> 'MatrixOperator':
          """
          Create Pauli X (sigma_x) matrix operator.
          
          Returns:
              MatrixOperator: Pauli X matrix
          """
          mx = np.matrix( [ [ complex(0.0, 0.0), complex(1.0, 0.0)]  , [ complex( 1.0, 0.0 ), complex( 0.0, 0.0)] ] , dtype=dtype)
          return MatrixOperator(maths.Matrix(mx))

     @staticmethod
     def pauli_y_mx_op() -> 'MatrixOperator':
          """
          Create Pauli Y (sigma_y) matrix operator.
          
          Returns:
              MatrixOperator: Pauli Y matrix
          """
          mx = np.matrix( [ [ complex(0.0, 0.0), complex(0.0, -1.0)]  , [ complex( 0.0, 1.0 ), complex( 0.0, 0.0)] ], dtype=dtype )
          return MatrixOperator(maths.Matrix(mx))
     
     @staticmethod
     def pauli_z_mx_op() -> 'MatrixOperator':
          """
          Create Pauli Z (sigma_z) matrix operator.
          
          Returns:
              MatrixOperator: Pauli Z matrix
          """
          mx = np.matrix( [ [ complex(1.0, 0.0), complex(0.0, 0.0)]  , [ complex( 0.0, 0.0 ), complex( -1.0, 0.0)] ], dtype=dtype )
          return MatrixOperator(maths.Matrix(mx))

     def save_eigen_vals_vects_to_file(self, bases_states: 'hilber_space_bases', eig_vec_fn: str, eig_vals_fn: str) -> None:
          """
          Save eigenvalues and eigenvectors to CSV files.
          
          Args:
              bases_states: Hilbert space basis
              eig_vec_fn: Filename for eigenvectors CSV
              eig_vals_fn: Filename for eigenvalues CSV
          """
          eigen_vectors:eigen_vector_space = self.calc_eigen_vals_vects()
          eigen_vectors.save(eig_vec_fn, eig_vals_fn)
          eig_vec_df, eig_val_df  = self.create_eigen_kets_vals_table(bases_states)

          eig_vec_df.to_csv( eig_vec_fn,sep = ';')
          eig_val_df.to_csv( eig_vals_fn,sep = ';')


     @staticmethod
     def basis_trf_matrix( kets: List[ket_vector]) -> 'MatrixOperator':
          """
          Create basis transformation matrix from ket vectors.
          
          Args:
              kets: List of ket vectors forming the new basis
              
          Returns:
              MatrixOperator: Basis transformation matrix
          """
          return MatrixOperator(maths.Matrix.from_col_vectors([ ket.coeffs for ket in kets ]).transpose())


     def calc_expected_val(self, ket: ket_vector) -> float:
          """
          Calculate expected value (expectation value) of operator with given ket vector.
          
          Args:
              ket: Ket vector
              
          Returns:
              float: Expected value (real part)
          """
          return (ket.to_bra_vector()*self*ket).real

     @staticmethod
     def from_ket_vectors(kets: List[ket_vector]) -> 'MatrixOperator':
          """
          Create matrix operator from list of ket vectors (as columns).
          
          Args:
              kets: List of ket vectors
              
          Returns:
              MatrixOperator: Matrix with ket vectors as columns
          """
          return MatrixOperator(maths.Matrix.from_col_vectors([ ket.coeffs for ket in kets ]))

     @staticmethod
     def from_bra_vectors(bras: List[ket_vector]) -> 'MatrixOperator':
          """
          Create matrix operator from list of bra vectors (as rows).
          
          Args:
              bras: List of bra vectors (ket_vectors used as bra)
              
          Returns:
              MatrixOperator: Matrix with bra vectors as rows
          """
          return MatrixOperator(maths.Matrix.from_row_vectors([ bra.coeffs for bra in bras ]))

     def to_new_basis(self, basis_kets: List[ket_vector]) -> 'MatrixOperator':
          """
          Transform operator to new basis.
          
          Args:
              basis_kets: List of ket vectors forming new basis
              
          Returns:
              MatrixOperator: Operator in new basis
          """
          return MatrixOperator(self.matrix.to_new_bases( [ ket.coeffs for ket in basis_kets ]))

     def new_basis_system(self, bases: List[ket_vector]) -> 'MatrixOperator':
          """
          Transform operator to new basis system.
          
          Args:
              bases: List of ket vectors forming new basis
              
          Returns:
              MatrixOperator: Operator in new basis (transposed)
          """
          new_bases_matrix = self.matrix.to_new_bases(list(map(lambda x: x.coeffs  ,bases )))
          return MatrixOperator(new_bases_matrix.transpose(), name  = self.name, subsys_name=self.subsys_name)

     @staticmethod
     def accumulate_operators(mx_ops: List['MatrixOperator'], fun: Callable) -> 'MatrixOperator':
          """
          Accumulate matrix operators using a function.
          
          Args:
              mx_ops: List of matrix operators
              fun: Binary function to accumulate with
              
          Returns:
              MatrixOperator: Result of accumulation
          """
          return list( itertools.accumulate(mx_ops, fun) )[-1]

     def drop_base_states(self, indexes: Union[int, List[int]]) -> None:
          """
          Drop (remove) base states from the matrix operator.
          
          Args:
              indexes: Index or list of indices of base states to remove
          """
          # Note: This modifies the underlying matrix object if it supports in-place operations
          # If not, you may need to reassign: self.matrix = self.matrix.drop_rows_cols(indexes)
          # For now, we'll try to delete and reassign if the matrix supports it
          try:
               self.matrix = maths.Matrix(np.delete(self.matrix.matrix, indexes, axis=0))
               self.matrix = maths.Matrix(np.delete(self.matrix.matrix, indexes, axis=1))
          except AttributeError:
               # If matrix doesn't have .matrix attribute, try direct deletion
               new_matrix = np.delete(self.matrix, indexes, axis=0)
               new_matrix = np.delete(new_matrix, indexes, axis=1)
               self.matrix = maths.Matrix(new_matrix)



     def save(self, filename: str) -> None:
          """
          Save matrix to file.
          
          Args:
              filename: Path to output file
          """
          self.matrix.save(filename)
     
     def round(self, dig: int) -> 'MatrixOperator':
          """
          Round matrix elements to specified number of decimal places.
          
          Args:
              dig: Number of decimal places
              
          Returns:
              MatrixOperator: Rounded matrix operator
          """
          return MatrixOperator(self.matrix.round(dig), name = self.name, subsys_name=self.subsys_name)
     
     def change_type(self, dtype: type) -> 'MatrixOperator':
          """
          Change data type of matrix.
          
          Args:
              dtype: New data type
              
          Returns:
              MatrixOperator: Matrix operator with new data type
          """
          return MatrixOperator(self.matrix.change_type(dtype), name = self.name, subsys_name=self.subsys_name)
     def __add__(self, other: 'MatrixOperator') -> 'MatrixOperator':
          if isinstance(other, MatrixOperator):
               return MatrixOperator(self.matrix + other.matrix, name=self.name, subsys_name=self.subsys_name)
          return NotImplemented
     
     def __radd__(self, other: Union[int, float, complex]) -> 'MatrixOperator':
          if isinstance(other, (int, float, complex)):
               return self
          return NotImplemented

     def __sub__(self, other: 'MatrixOperator') -> 'MatrixOperator':
          if isinstance(other, MatrixOperator):
               return MatrixOperator(self.matrix - other.matrix, name=self.name, subsys_name=self.subsys_name)
          return NotImplemented
     
     def __mul__(self, other: Union['MatrixOperator', ket_vector]) -> Union['MatrixOperator', ket_vector]:
          if isinstance(other, MatrixOperator):
               return MatrixOperator(self.matrix.__mul__(other.matrix), name=self.name, subsys_name=self.subsys_name)
          elif isinstance(other, ket_vector):
               return ket_vector(self.matrix.__mul__(other.coeffs), eigen_val=other.eigen_val, name=self.name, subsys_name=self.subsys_name)
          return NotImplemented

     def __rmul__(self, other: Union[int, float, complex]) -> 'MatrixOperator':
          return MatrixOperator(self.matrix.__rmul__(other),name= self.name, subsys_name=self.subsys_name)

     def __truediv__(self, other: Union[int, float, complex]) -> 'MatrixOperator':
          return MatrixOperator(self.matrix.__truediv__(other),name= self.name, subsys_name=self.subsys_name)
     
     def __pow__(self, other: 'MatrixOperator') -> 'MatrixOperator':
          if isinstance(other, MatrixOperator):
               return MatrixOperator(self.matrix**other.matrix, name=self.name, subsys_name=self.subsys_name)
          return NotImplemented
     
     def __repr__(self) -> str:
          return self.matrix.__repr__()

     def __init__(self, matrix: maths.Matrix, name: str = "", subsys_name: str = "", solver: Optional['EigenSolver'] = None):
          """
          Initialize matrix operator.
          
          Args:
              matrix: Underlying matrix object
              name: Optional name for the operator
              subsys_name: Optional subsystem name
              solver: Optional custom eigenvalue solver (defaults to DenseEigenSolver)
          """
          self.name = name
          self.subsys_name = subsys_name
          self.matrix = matrix
          self.matrix_class = type(matrix)
          self.quantum_state_bases: Optional['hilber_space_bases'] = None
          
          # Set up eigenvalue solver
          if solver is None:
              # Lazy import to avoid circular dependency
              from jahn_teller_dynamics.math.eigen_solver import DenseEigenSolver
              self._eigen_solver = DenseEigenSolver()
          else:
              self._eigen_solver = solver
     
     def set_quantum_states(self, quantum_states: 'hilber_space_bases') -> None:
          self.quantum_state_bases = quantum_states
     
     def set_eigen_solver(self, solver: 'EigenSolver') -> None:
          """
          Set a custom eigenvalue solver for this matrix operator.
          
          This allows switching between different solver implementations
          (e.g., dense, sparse, iterative) without recreating the operator.
          
          Args:
              solver: EigenSolver instance to use
          """
          self._eigen_solver = solver
     
     @staticmethod
     def create_id_matrix_op(dim: int, matrix_type: type = maths.Matrix) -> 'MatrixOperator':
          return MatrixOperator(matrix_type.create_eye(dim))

     @staticmethod
     def create_null_matrix_op(dim: int, matrix_type: type = maths.Matrix) -> 'MatrixOperator':
          return MatrixOperator(matrix_type.create_zeros(dim))

     def __len__(self) -> int:
          return len(self.matrix)

     @staticmethod
     def create_Lz_op(matrix_type: type = maths.Matrix) -> 'MatrixOperator':
          return MatrixOperator(matrix_type.create_Lz_mx())


     def __getitem__(self, key: Union[int, Tuple[int, int]]) -> Union[complex, float]:
          return self.matrix[key]

     def calc_eigen_vals_vects_old(self, num_of_vals: Optional[int] = None, ordering_type: Optional[str] = None) -> None:
          """
          Calculate eigenvalues and eigenvectors (old implementation).
          
          .. deprecated:: Use calc_eigen_vals_vects() instead.
          This method is kept for backward compatibility but should not be used in new code.
          
          Args:
              num_of_vals: Number of eigenvalues to calculate (optional)
              ordering_type: Type of ordering to apply (optional)
          """
          eigen_vals, eigen_vects =  self.matrix.get_eig_vals(num_of_vals, ordering_type)

          self.eigen_kets = []

          for (eigen_val, eigen_vect) in zip( eigen_vals, eigen_vects  ):
               self.eigen_kets.append( ket_vector(maths.col_vector(np.transpose(np.matrix([eigen_vect]))),eigen_val) )
             
          self.eigen_kets = sorted(self.eigen_kets, key =lambda x: x.eigen_val)

     def calc_eigen_vals_vects(self, num_of_vals: Optional[int] = None, ordering_type: Optional[str] = None, quantum_states_bases: Optional['hilber_space_bases'] = None) -> 'eigen_vector_space':
          """
          Calculate eigenvalues and eigenvectors of the matrix operator.
          
          This method uses the configured eigenvalue solver (default: DenseEigenSolver).
          For backward compatibility, it also sets self.eigen_kets as a side effect.
          
          Args:
              num_of_vals: Number of eigenvalues to calculate (None = all)
              ordering_type: Type of ordering to apply (optional)
              quantum_states_bases: Hilbert space basis for quantum states (optional)
              
          Returns:
              eigen_vector_space: Object containing eigen kets and basis
          """
          # Update quantum state bases if provided
          if quantum_states_bases is not None:
               self.quantum_state_bases = quantum_states_bases
          
          # Use the eigenvalue solver
          eigen_vec_space = self._eigen_solver.solve(
              self,
              num_of_vals=num_of_vals,
              ordering_type=ordering_type,
              quantum_states_bases=self.quantum_state_bases
          )
          
          # For backward compatibility: set eigen_kets attribute
          # (some code accesses self.eigen_kets directly)
          self.eigen_kets = eigen_vec_space.eigen_kets
          
          return eigen_vec_space

     def create_eigen_kets_vals_table(self, bases: 'hilber_space_bases') -> Tuple[pd.DataFrame, pd.DataFrame]:
          """
          Create DataFrames containing eigenvector coefficients and eigenvalues.
          
          Args:
              bases: Hilbert space basis
              
          Returns:
              tuple: (eig_vecs_df, eig_vals_df) where:
                  - eig_vecs_df: DataFrame with eigenvector coefficients (indexed by quantum states)
                  - eig_vals_df: DataFrame with eigenvalues (indexed by eigenstate names)
          """
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



     def get_eigen_vect(self, i: int) -> np.ndarray:
          """
          Get eigenvector at given index.
          
          Args:
              i: Index of eigenvector
              
          Returns:
              numpy.ndarray: Eigenvector coefficients as flattened array
              
          Raises:
              AttributeError: If eigenvectors have not been calculated
              IndexError: If index is out of range
          """
          if not hasattr(self, 'eigen_kets') or self.eigen_kets is None:
               raise AttributeError("Eigenvectors not calculated. Call calc_eigen_vals_vects() first.")
          if not isinstance(i, int) or i < 0 or i >= len(self.eigen_kets):
               raise IndexError(f"Index {i} out of range for {len(self.eigen_kets)} eigenvectors")
          return np.array(self.eigen_kets[i].coeffs.coeffs.flatten())
     
     def get_eigen_val(self, i: int) -> float:
          """
          Get eigenvalue at given index.
          
          Args:
              i: Index of eigenvalue
              
          Returns:
              float: Eigenvalue at index
              
          Raises:
              AttributeError: If eigenvalues have not been calculated
              IndexError: If index is out of range
          """
          if not hasattr(self, 'eigen_kets') or self.eigen_kets is None:
               raise AttributeError("Eigenvalues not calculated. Call calc_eigen_vals_vects() first.")
          if not isinstance(i, int) or i < 0 or i >= len(self.eigen_kets):
               raise IndexError(f"Index {i} out of range for {len(self.eigen_kets)} eigenvalues")
          return self.eigen_kets[i].eigen_val

     def multiply(self, other: 'MatrixOperator') -> 'MatrixOperator':
          """
          Multiply this matrix operator with another.
          
          Args:
              other: Matrix operator to multiply with
              
          Returns:
              MatrixOperator: Result of multiplication
          """
          return MatrixOperator(self.matrix.multiply(other.matrix))

     def truncate_matrix(self, trunc_num: int) -> 'MatrixOperator':
          """
          Truncate matrix by removing last trunc_num rows and columns.
          
          Args:
              trunc_num: Number of rows/columns to remove
              
          Returns:
              MatrixOperator: Truncated matrix operator
          """
          dim = len(self.matrix)
          return MatrixOperator(maths.Matrix(self.matrix[0:dim-trunc_num, 0: dim-trunc_num]),self.name, self.subsys_name)
     
     def get_dim(self) -> int:
          """
          Get dimension of matrix operator.
          
          Returns:
              int: Matrix dimension
          """
          return len(self.matrix)



class FirstOrderPerturbation:
     
     def __init__(self,deg_eigen_vecs: list[ket_vector], ham_comma: MatrixOperator):
          self.deg_eigen_vecs = deg_eigen_vecs
          self.ham_comma = ham_comma
          self.create_pert_op()

     def create_pert_op_old(self) -> 'MatrixOperator':
          """
          Create perturbation operator matrix (old implementation).
          
          .. deprecated:: Use create_pert_op() instead.
          This method is kept for backward compatibility.
          
          Returns:
              MatrixOperator: The perturbation operator
          """
          left = np.matrix( [ x.coeffs.get_coeffs_list() for x in self.deg_eigen_vecs] )
          right = np.transpose(left)

          raw_pert_mat = np.matmul(left, np.matmul( self.ham_comma.matrix.matrix, right ))

          self.pert_op = MatrixOperator(maths.Matrix(raw_pert_mat))
          
          self.pert_op.calc_eigen_vals_vects()
          self.pert_eigen_vals = [ ket_vec.eigen_val for ket_vec in self.pert_op.eigen_kets] 

          return self.pert_op




     def create_pert_op(self) -> 'MatrixOperator':
          """
          Create perturbation operator matrix from degenerate eigen vectors.
          
          Returns:
              MatrixOperator: The perturbation operator
          """
          # Use get_coeffs_list() to get proper list format for matrix construction
          left = np.matrix( [ x.coeffs.get_coeffs_list() for x in self.deg_eigen_vecs] )
          right = np.transpose(left)

          raw_pert_mat = np.matmul(left, np.matmul( self.ham_comma.matrix.matrix, right ))

          self.pert_op = MatrixOperator(maths.Matrix(raw_pert_mat))

          self.pert_op.calc_eigen_vals_vects()
          self.pert_eigen_vals = [ket_vec.eigen_val for ket_vec in self.pert_op.eigen_kets]

          return self.pert_op



     def get_reduction_factor(self) -> float:
          """
          Get reduction factor from perturbation eigenvalues.
          
          Returns:
              float: Reduction factor
          """
          return abs( (self.pert_eigen_vals[1] - self.pert_eigen_vals[0]).real/2 )



class reduction_factors:
     """
     Class for calculating reduction factors from degenerate ket vectors.
     """
     
     def __init__(self, deg_ket_vectors: List[ket_vector]) -> None:
          """
          Initialize reduction factors calculator.
          
          Args:
              deg_ket_vectors: List of degenerate ket vectors
          """
          self.deg_ket_vectors = deg_ket_vectors
     
     def p_red_factor(self, op: MatrixOperator) -> float:
          """
          Calculate p reduction factor.
          
          Args:
              op: Matrix operator
              
          Returns:
              float: Reduction factor
          """
          return abs( op.calc_expected_val(self.deg_ket_vectors[0])-op.calc_expected_val(self.deg_ket_vectors[1]))

  


class degenerate_system:
     """
     Class representing a degenerate quantum system.
     """
     
     def __init__(self, deg_ket_vectors: List[ket_vector]) -> None:
          """
          Initialize degenerate system.
          
          Args:
              deg_ket_vectors: List of degenerate ket vectors
          """
          self.deg_ket_vectors = deg_ket_vectors
          self.deg_bra_vectors = [ ket.to_bra_vector() for ket in self.deg_ket_vectors ]
          self.eigen_val = deg_ket_vectors[0].eigen_val

     def __getitem__(self, key: int) -> ket_vector:
          """
          Get degenerate ket vector at index.
          
          Args:
              key: Index
              
          Returns:
              ket_vector: Degenerate ket vector
          """
          return self.deg_ket_vectors[key]
     
     def add_perturbation(self, perturbation: MatrixOperator) -> 'MatrixOperator':
          """
          Add perturbation to degenerate system.
          
          Args:
              perturbation: Perturbation operator
              
          Returns:
              MatrixOperator: Perturbed system matrix
          """
          left_op = MatrixOperator.from_bra_vectors(self.deg_bra_vectors)
          right_op = MatrixOperator.from_ket_vectors(self.deg_ket_vectors)

          return left_op*perturbation*right_op
     


class degenerate_system_2D(degenerate_system):
     """
     Class representing a 2D degenerate quantum system.
     """
     
     def __init__(self, deg_ket_vectors: List[ket_vector]) -> None:
          """
          Initialize 2D degenerate system.
          
          Args:
              deg_ket_vectors: List of exactly 2 degenerate ket vectors
              
          Raises:
              ValueError: If not exactly 2 ket vectors provided
          """
          if len(deg_ket_vectors) != 2:
               raise ValueError(f"degenerate_system_2D requires exactly 2 ket vectors, got {len(deg_ket_vectors)}")
          super().__init__(deg_ket_vectors)

     def to_complex_basis(self, basis_trf_matrix: MatrixOperator) -> None:
          """
          Transform degenerate ket vectors to complex basis.
          
          Args:
              basis_trf_matrix: Basis transformation matrix
          """
          phix = self.deg_ket_vectors[0]
          phiy = self.deg_ket_vectors[1]
          phiplus = basis_trf_matrix*(phix+complex(0.0,1.0)*phiy)/SQRT_2
          phiminus = basis_trf_matrix*(phix-complex(0.0,1.0)*phiy)/SQRT_2
          self.complex_deg_ket_vectors = [phiminus, phiplus]
     

     def pert_in_complex_basis(self, pert_ham: MatrixOperator) -> float:
          """
          Calculate perturbation in complex basis.
          
          Args:
              pert_ham: Perturbation Hamiltonian
              
          Returns:
              float: Energy splitting in complex basis
          """
          complex_pert_ham =  pert_ham.to_new_basis(self.complex_deg_ket_vectors)

          complex_pert_ham.calc_eigen_vals_vects()

          return (complex_pert_ham.eigen_kets[1].eigen_val - complex_pert_ham.eigen_kets[0].eigen_val)
          
     def add_perturbation(self, perturbation: MatrixOperator) -> None:
          """
          Add perturbation to 2D degenerate system.
          
          Args:
              perturbation: Perturbation operator
          """
          pert_sys_mat =  super().add_perturbation(perturbation)
          pert_sys_mat.calc_eigen_vals_vects()

          self.p_red_fact= pert_sys_mat.eigen_kets[1].eigen_val-pert_sys_mat.eigen_kets[0].eigen_val

     def calc_p_factor(self, perturbation: MatrixOperator) -> complex:
          """
          Calculate p factor.
          
          Args:
              perturbation: Perturbation operator
              
          Returns:
              complex: p factor value
          """
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

     @staticmethod
     def bra_from_csv(states_fn:str, energies_fn:str):
          energies_df = pd.read_csv(energies_fn, sep=';',index_col='state_name')

          states_df = pd.read_csv(states_fn, sep=';')
          eigen_vectors = []
          for eigen_state_name in list(states_df)[1:]:
               coeff_strs = states_df[eigen_state_name].tolist()
               eigen_energy = energies_df['eigenenergy'][eigen_state_name]

               eigen_vectors.append(bra_vector.from_str_list(coeff_strs, eigen_energy))

          return eigen_vectors
     
     @staticmethod
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
          return eigen_vector_space( new_hilbert_space, [  (trf_mx*eig_ket).round(EIGENVALUE_ROUNDING_PRECISION) for eig_ket in self.eigen_kets ]  )

     def save(self, eig_vec_fn:str, eig_val_fn:str):
          
          eig_vec_df, eig_val_df  = self.create_eigen_kets_vals_table(self.quantum_states_basis)

          eig_vec_df.to_csv( eig_vec_fn,sep = ';')
          eig_val_df.to_csv( eig_val_fn,sep = ';')


     def create_eigen_kets_vals_table(self, bases:hilber_space_bases):
          """
          Create DataFrames containing eigenvector coefficients and eigenvalues.
          
          Note: This method uses 1-based indexing for eigenstate names (eigenstate_1, eigenstate_2, ...)
          while MatrixOperator.create_eigen_kets_vals_table uses 0-based indexing (eigenstate_0, eigenstate_1, ...).
          
          Args:
              bases: Hilbert space basis
              
          Returns:
              tuple: (eig_vecs_df, eig_vals_df) where:
                  - eig_vecs_df: DataFrame with eigenvector coefficients (indexed by quantum states)
                  - eig_vals_df: DataFrame with eigenvalues (indexed by eigenstate names)
          """
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
