"""
Mathematical utilities for Jahn-Teller dynamics calculations.

This module provides vector and matrix classes for quantum mechanical calculations,
including column vectors, row vectors, and matrices with support for complex numbers.
"""

import numpy as np
from scipy.sparse import csr_matrix, block_diag
import scipy.sparse as sparse
from scipy.sparse.linalg import eigs as sparse_eigs, eigsh, spsolve
import scipy.sparse.linalg as sparse_linalg
from scipy.linalg import eig as eigs
import pandas as pd
from typing import Optional, List, Union, Any, Tuple
import warnings
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    warnings.warn("networkx not available. Block diagonalization features will be disabled.")
warnings.simplefilter("ignore", np.exceptions.ComplexWarning)


def meV_to_GHz(e: float) -> float:
    """
    Convert energy from meV to GHz.
    
    Args:
        e: Energy in meV
        
    Returns:
        Energy in GHz
    """
    return e * 241.798935


precision = 0.0000001
complex_number_typ = np.complex64


def equal_matrix(a: Union[np.matrix, np.ndarray], b: Union[np.matrix, np.ndarray]) -> bool:
    """
    Check if two matrices are equal within precision tolerance.
    
    Args:
        a: First matrix
        b: Second matrix
        
    Returns:
        True if matrices are equal within precision, False otherwise
    """
    if a.shape != b.shape:
        return False
    else:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if abs(a[i, j] - b[i, j]) > precision:
                    return False
        return True


class col_vector:
    """
    Column vector class for quantum mechanical calculations.
    
    Represents a column vector (n x 1 matrix) with support for complex numbers.
    Provides operations for vector arithmetic, normalization, basis transformations, etc.
    """
    
    @staticmethod
    def load_from_file(file_name: str) -> List['col_vector']:
        """
        Load column vectors from CSV file.
        
        Args:
            file_name: Path to CSV file
            
        Returns:
            List of col_vector objects (one per column, skipping first column)
        """
        df = pd.read_csv(file_name)
        columns = df.columns
        col_vector_list = []
        
        for column in columns[1:]:
            col_vector_list.append(col_vector(np.transpose(np.matrix(df[column].tolist()))))
        
        return col_vector_list

    def basis_trf(self, new_bases: List['col_vector']) -> 'Matrix':
        """
        Transform vector to new basis.
        
        Args:
            new_bases: List of new basis vectors
            
        Returns:
            Transformed vector as Matrix
        """
        new_bases_row_vectors = [new_base.to_row_vector() for new_base in new_bases]
        basis_trf_mx = Matrix.from_row_vectors(new_bases_row_vectors)
        return basis_trf_mx * self

    def length(self) -> float:
        """
        Calculate Euclidean length (norm) of the vector.
        
        Returns:
            Vector length
        """
        coeffs = self.tolist()
        res = 0.0
        for coeff in coeffs:
            res += abs(coeff)**2
        return res**0.5

    def normalize(self) -> 'col_vector':
        """
        Normalize the vector to unit length.
        
        Returns:
            Normalized col_vector
        """
        return (1 / self.length()) * self

    @staticmethod
    def from_list(coeff_list: List[Union[complex, float]]) -> 'col_vector':
        """
        Create col_vector from list of coefficients.
        
        Args:
            coeff_list: List of coefficients
            
        Returns:
            col_vector instance
        """
        coeffs_mx = np.matrix([coeff_list]).transpose()
        return col_vector(coeffs_mx)

    def in_new_basis(self, basis_vecs: List['col_vector']) -> 'col_vector':
        """
        Express vector in new basis.
        
        Args:
            basis_vecs: List of basis vectors
            
        Returns:
            col_vector expressed in new basis
        """
        coeffs = self.tolist()
        dim = self.coeffs.shape[0]
        res_vec = col_vector.from_list([0.0 for i in range(dim)])
        
        for coeff, basis_vec in zip(coeffs, basis_vecs):
            res_vec += coeff * basis_vec
        return res_vec

    @staticmethod
    def from_file(file_name: str) -> 'col_vector':
        """
        Load col_vector from text file.
        
        Args:
            file_name: Path to text file
            
        Returns:
            col_vector instance
        """
        coeffs = np.loadtxt(file_name)
        return col_vector(np.transpose(np.matrix(coeffs)))

    def calc_abs_square(self) -> 'col_vector':
        """
        Calculate absolute square of coefficients.
        
        Returns:
            col_vector with |coeff|^2 for each element
        """
        coeff_conj = np.conj(self.coeffs)
        return col_vector(np.multiply(coeff_conj, self.coeffs))

    def map(self, func: callable) -> 'col_vector':
        """
        Apply function to each coefficient.
        
        Args:
            func: Function to apply
            
        Returns:
            New col_vector with transformed coefficients
        """
        raw_mx = list(map(func, self.coeffs))
        return col_vector(np.matrix(np.reshape(raw_mx, (len(raw_mx), 1))))

    def round(self, dig: int) -> 'col_vector':
        """
        Round coefficients to specified number of decimal places.
        
        Args:
            dig: Number of decimal places
            
        Returns:
            Rounded col_vector
        """
        return col_vector(np.round(self.coeffs, dig))

    def tolist(self) -> List[complex]:
        """
        Convert to Python list.
        
        Returns:
            List of coefficients
        """
        flatten_coeffs = np.reshape(self.coeffs, (len(self.coeffs)))
        return flatten_coeffs.tolist()[0]

    @staticmethod
    def zeros(dim: int) -> 'col_vector':
        """
        Create zero vector of specified dimension.
        
        Args:
            dim: Dimension of vector
            
        Returns:
            Zero col_vector
        """
        matrix = np.matrix(np.zeros(shape=(dim, 1)), dtype=complex_number_typ)
        return col_vector(matrix)

    def set_item(self, index: int, item: Union[complex, float]) -> None:
        """
        Set value at given index.
        
        Args:
            index: Index to set
            item: Value to set
        """
        self.coeffs.itemset((index, 0), item)

    def __eq__(self, other: Any) -> bool:
        """
        Check equality of column vectors.
        
        Args:
            other: Object to compare with
            
        Returns:
            True if vectors are equal, False otherwise
        """
        if other is None:
            return False
        if not isinstance(other, col_vector):
            return False
        return equal_matrix(self.coeffs, other.coeffs)

    def __init__(self, coeffs: Union[np.matrix, np.ndarray]):
        """
        Initialize column vector from numpy matrix or array.
        
        Args:
            coeffs: numpy matrix or array with shape (n, 1)
            
        Raises:
            ValueError: If coeffs is not a column vector (shape[1] != 1)
        """
        # Convert to matrix if it's an array
        if isinstance(coeffs, np.ndarray) and not isinstance(coeffs, np.matrix):
            coeffs = np.asmatrix(coeffs)
        
        if coeffs.shape[1] == 1:
            self.coeffs = coeffs
        else:
            raise ValueError(f"col_vector requires shape (n, 1), got {coeffs.shape}")

    def get_coeffs_list(self) -> List[complex]:
        """
        Get list of coefficients as complex numbers.
        
        Returns:
            List of complex coefficients
        """
        coeffs_list = []
        for coeff in self.coeffs:
            coeffs_list.append(complex(coeff))
        return coeffs_list

    def __repr__(self) -> str:
        """String representation of the vector."""
        return str(self.coeffs)

    def __str__(self) -> str:
        """Human-readable string representation."""
        coeffs_list = self.tolist()
        res_str = ''
        for coeff in coeffs_list[0:-1]:
            res_str += str(coeff) + ', '
        res_str += str(coeffs_list[-1])
        return res_str

    def __getitem__(self, key: int) -> Union[complex, float]:
        """
        Get coefficient at given index.
        
        Args:
            key: Index
            
        Returns:
            Coefficient value
        """
        return self.coeffs[key][0]
    
    def set_val(self, index: int, val: Union[complex, float]) -> None:
        """
        Set value at given index.
        
        Args:
            index: Index to set
            val: Value to set
        """
        self.coeffs[index, 0] = val
    
    def to_row_vector(self) -> 'row_vector':
        """
        Convert to row vector (complex conjugate transpose).
        
        Returns:
            row_vector instance
        """
        return row_vector(np.conj(np.transpose(self.coeffs)))

    def __mul__(self, other: Any) -> Union['col_vector', 'Matrix']:
        """
        Multiply column vector with other objects.
        
        Args:
            other: row_vector, scalar, or other compatible type
            
        Returns:
            - col_vector * row_vector: Matrix
            - col_vector * scalar: col_vector
            
        Raises:
            TypeError: If multiplication is not supported
        """
        if isinstance(other, row_vector):
            return Matrix(np.matmul(self.coeffs, other.coeffs))
        elif isinstance(other, (complex, float, int)):
            return col_vector(self.coeffs * other)
        else:
            raise TypeError(f"Cannot multiply col_vector with {type(other)}")

    def __rmul__(self, other: Any) -> Union['col_vector', 'Matrix']:
        """Right multiplication (scalar * col_vector)."""
        return self * other

    def __truediv__(self, other: Union[complex, float]) -> 'col_vector':
        """
        Divide vector by scalar.
        
        Args:
            other: Scalar divisor
            
        Returns:
            col_vector divided by scalar
        """
        return col_vector(self.coeffs / other)

    def __add__(self, other: 'col_vector') -> 'col_vector':
        """
        Add two column vectors.
        
        Args:
            other: Another col_vector
            
        Returns:
            Sum of vectors
            
        Raises:
            TypeError: If other is not a col_vector
        """
        if not isinstance(other, col_vector):
            raise TypeError(f"Cannot add col_vector with {type(other)}")
        return col_vector(self.coeffs + other.coeffs)

    def __radd__(self, other: Any) -> 'col_vector':
        """Right addition."""
        return self + other

    def __sub__(self, other: 'col_vector') -> 'col_vector':
        """
        Subtract two column vectors.
        
        Args:
            other: Another col_vector
            
        Returns:
            Difference of vectors
            
        Raises:
            TypeError: If other is not a col_vector
        """
        if not isinstance(other, col_vector):
            raise TypeError(f"Cannot subtract {type(other)} from col_vector")
        return col_vector(self.coeffs - other.coeffs)

    def __rsub__(self, other: Any) -> 'col_vector':
        """
        Right subtraction (other - self).
        
        Args:
            other: Scalar or col_vector
            
        Returns:
            Result of subtraction
        """
        if isinstance(other, (complex, float, int)):
            return col_vector(other - self.coeffs)
        elif isinstance(other, col_vector):
            return col_vector(other.coeffs - self.coeffs)
        else:
            raise TypeError(f"Cannot subtract col_vector from {type(other)}")
    
    def __abs__(self) -> float:
        """
        Calculate magnitude (Euclidean norm) of vector.
        
        Returns:
            Magnitude as float
        """
        magnitude = 0.0
        for coeff in self.coeffs:
            magnitude += abs(coeff)**2
        return float(magnitude)


class row_vector:
    """
    Row vector class for quantum mechanical calculations.
    
    Represents a row vector (1 x n matrix) with support for complex numbers.
    Used for computing inner products and bra states.
    """
    
    def __init__(self, coeffs: Union[np.matrix, np.ndarray]):
        """
        Initialize row vector from numpy matrix or array.
        
        Args:
            coeffs: numpy matrix or array with shape (1, n)
            
        Raises:
            ValueError: If coeffs is not a row vector (shape[0] != 1)
        """
        # Convert to matrix if it's an array
        if isinstance(coeffs, np.ndarray) and not isinstance(coeffs, np.matrix):
            coeffs = np.asmatrix(coeffs)
        
        if coeffs.shape[0] == 1:
            self.coeffs = coeffs
        else:
            raise ValueError(f"row_vector requires shape (1, n), got {coeffs.shape}")
        
    def __mul__(self, other: Any) -> Union[complex, 'row_vector', 'Matrix']:
        """
        Multiply row vector with other objects.
        
        Args:
            other: col_vector, Matrix, scalar, or row_vector
            
        Returns:
            - row_vector * col_vector: complex (inner product)
            - row_vector * Matrix: row_vector
            - row_vector * scalar: row_vector
            - row_vector * row_vector: Matrix
            
        Raises:
            TypeError: If multiplication is not supported
        """
        if isinstance(other, col_vector):
            return complex(np.matmul(self.coeffs, other.coeffs))
        elif isinstance(other, Matrix):
            return row_vector(np.matmul(self.coeffs, other.matrix))
        elif isinstance(other, (complex, float, int)):
            return row_vector(self.coeffs * other)
        elif isinstance(other, row_vector):
            return Matrix(np.matmul(self.coeffs, other.coeffs))
        else:
            raise TypeError(f"Cannot multiply row_vector with {type(other)}")

    def __rmul__(self, other: Any) -> Union['row_vector', 'Matrix']:
        """
        Right multiplication.
        
        Args:
            other: col_vector, scalar, or other compatible type
            
        Returns:
            Result of right multiplication
        """
        if isinstance(other, col_vector):
            return Matrix(np.matmul(other.coeffs, self.coeffs))
        else:
            return self * other

    def __truediv__(self, other: Union[complex, float]) -> 'row_vector':
        """
        Divide row vector by scalar.
        
        Args:
            other: Scalar divisor
            
        Returns:
            row_vector divided by scalar
        """
        return row_vector(self.coeffs / other)

    def __add__(self, other: 'row_vector') -> 'row_vector':
        """
        Add two row vectors.
        
        Args:
            other: Another row_vector
            
        Returns:
            Sum of vectors
            
        Raises:
            TypeError: If other is not a row_vector
        """
        if not isinstance(other, row_vector):
            raise TypeError(f"Cannot add row_vector with {type(other)}")
        return row_vector(self.coeffs + other.coeffs)

    def __radd__(self, other: Any) -> 'row_vector':
        """Right addition."""
        return self + other

    def __sub__(self, other: 'row_vector') -> 'row_vector':
        """
        Subtract two row vectors.
        
        Args:
            other: Another row_vector
            
        Returns:
            Difference of vectors
            
        Raises:
            TypeError: If other is not a row_vector
        """
        if not isinstance(other, row_vector):
            raise TypeError(f"Cannot subtract {type(other)} from row_vector")
        return row_vector(self.coeffs - other.coeffs)

    def __rsub__(self, other: Any) -> 'row_vector':
        """
        Right subtraction (other - self).
        
        Args:
            other: Scalar or row_vector
            
        Returns:
            Result of subtraction
        """
        if isinstance(other, (complex, float, int)):
            return row_vector(other - self.coeffs)
        elif isinstance(other, row_vector):
            return row_vector(other.coeffs - self.coeffs)
        else:
            raise TypeError(f"Cannot subtract row_vector from {type(other)}")
    
    def __abs__(self) -> float:
        """
        Calculate magnitude of row vector.
        
        Returns:
            Magnitude as float
        """
        magnitude = 0.0
        for coeff in self.coeffs:
            magnitude += abs(coeff)**2
        return float(magnitude**0.5)
    
    def extend(self, l: List[Union[complex, float]]) -> 'row_vector':
        """
        Extend row vector with additional elements.
        
        Args:
            l: List of additional elements
            
        Returns:
            Extended row_vector
        """
        old_coeffs_list = self.tolist()
        new_coeffs_list = old_coeffs_list + l
        return row_vector(np.matrix([new_coeffs_list]))

    def set_val(self, index: int, val: Union[complex, float]) -> None:
        """
        Set value at given index.
        
        Args:
            index: Index to set
            val: Value to set
        """
        self.coeffs[0, index] = val

    def __repr__(self) -> str:
        """String representation of the vector."""
        return str(self.coeffs)

    def __str__(self) -> str:
        """Human-readable string representation."""
        return str(self.coeffs)
    
    def __getitem__(self, key: int) -> Union[complex, float]:
        """
        Get coefficient at given index.
        
        Args:
            key: Index
            
        Returns:
            Coefficient value
        """
        return self.coeffs[0][key]
    
    def tolist(self) -> List[complex]:
        """
        Convert to Python list.
        
        Returns:
            List of coefficients
        """
        return self.coeffs.tolist()[0]
    
    def __eq__(self, other: Any) -> bool:
        """
        Check equality of row vectors.
        
        Args:
            other: Object to compare with
            
        Returns:
            True if vectors are equal, False otherwise
        """
        if other is None:
            return False
        if not isinstance(other, row_vector):
            return False
        return self.coeffs.tolist() == other.coeffs.tolist()

    def norm_square(self, to_index: Optional[int] = None) -> float:
        """
        Calculate squared norm of row vector.
        
        Args:
            to_index: Optional index to calculate norm up to (default: all elements)
            
        Returns:
            Squared norm
        """
        coeffs_list = self.tolist()
        if to_index is None:
            to_index = len(coeffs_list)
        res = 0.0
        for coeff in coeffs_list[0:to_index]:
            res += abs(coeff)**2
        return res


class Matrix:
    """
    Matrix class for quantum mechanical calculations.
    
    Represents a 2D matrix with support for complex numbers.
    Provides operations for matrix arithmetic, transformations, eigenvalues, etc.
    """
    
    def tolist(self) -> List[List[complex]]:
        """
        Convert matrix to nested list.
        
        Returns:
            Nested list representation
        """
        return self.matrix.tolist()

    def calc_inverse(self) -> 'Matrix':
        """
        Calculate matrix inverse.
        
        Returns:
            Inverse matrix
            
        Raises:
            numpy.linalg.LinAlgError: If matrix is singular
        """
        return Matrix(np.linalg.inv(self.matrix))

    @staticmethod
    def from_col_vectors(bases: List[col_vector]) -> 'Matrix':
        """
        Create matrix from column vectors.
        
        Args:
            bases: List of col_vector objects
            
        Returns:
            Matrix with columns from basis vectors
        """
        raw_matrix = []
        for base in bases:
            base_trp = base.coeffs.transpose().tolist()[0]
            raw_matrix.append(base_trp)
        
        matrix = np.transpose(np.matrix(raw_matrix, dtype=complex_number_typ))
        return Matrix(matrix)
    
    @staticmethod
    def from_row_vectors(bases: List[row_vector]) -> 'Matrix':
        """
        Create matrix from row vectors.
        
        Args:
            bases: List of row_vector objects
            
        Returns:
            Matrix with rows from basis vectors
        """
        raw_matrix = []
        for base in bases:
            raw_matrix.append(base.coeffs.tolist()[0])
        matrix = np.matrix(raw_matrix, dtype=complex_number_typ)
        return Matrix(matrix)
    
    def to_new_bases(self, bases: List[col_vector]) -> 'Matrix':
        """
        Transform matrix to new basis.
        
        Args:
            bases: List of new basis vectors
            
        Returns:
            Transformed matrix
        """
        V: Matrix = Matrix.from_col_vectors(bases)
        V_inv: Matrix = V.calc_inverse()
        return V * self * V_inv

    def __str__(self) -> str:
        """String representation of the matrix."""
        return str(self.matrix)
    
    def __repr__(self) -> str:
        """String representation of the matrix."""
        return str(self.matrix)

    @staticmethod
    def create_Lz_mx() -> 'Matrix':
        """
        Create Lz matrix operator: [[0, -i], [i, 0]].
        
        Returns:
            Lz operator matrix
        """
        raw_mx = np.matrix([[0, complex(0, -1)], [complex(0, 1), 0]], dtype=complex_number_typ)
        return Matrix(raw_mx)

    @staticmethod
    def create_eye(dim: int) -> 'Matrix':
        """
        Create identity matrix.
        
        Args:
            dim: Dimension of matrix
            
        Returns:
            Identity matrix
        """
        return Matrix(np.eye(dim, dtype=complex_number_typ))

    @staticmethod
    def create_zeros(dim: Union[int, tuple]) -> 'Matrix':
        """
        Create zero matrix.
        
        Args:
            dim: Dimension(s) of matrix (int for square, tuple for rectangular)
            
        Returns:
            Zero matrix
        """
        return Matrix(np.zeros(dim, dtype=complex_number_typ))

    def save(self, filename: str) -> None:
        """
        Save matrix to text file.
        
        Args:
            filename: Path to output file
        """
        np.savetxt(filename, self.matrix)

    def __init__(self, matrix: Union[np.matrix, np.ndarray]):
        """
        Initialize Matrix from numpy matrix or array.
        
        Args:
            matrix: numpy matrix or array
            
        Raises:
            ValueError: If matrix is not 2D
        """
        # Convert to matrix if it's an array
        if isinstance(matrix, np.ndarray) and not isinstance(matrix, np.matrix):
            matrix = np.asmatrix(matrix)
        
        if len(matrix.shape) != 2:
            raise ValueError(f"Matrix requires 2D array, got shape {matrix.shape}")
        
        self.matrix = matrix
        self.dim = self.matrix.shape[0]

    def multiply(self, other: 'Matrix') -> 'Matrix':
        """
        Multiply two matrices.
        
        Args:
            other: Another Matrix
            
        Returns:
            Product of matrices
        """
        return Matrix(np.matmul(self.matrix, other.matrix))
    
    def __mul__(self, other: Any) -> Union['Matrix', 'col_vector']:
        """
        Multiply matrix with other objects.
        
        Args:
            other: Matrix, col_vector, or scalar
            
        Returns:
            - Matrix * Matrix: Matrix
            - Matrix * col_vector: col_vector
            
        Raises:
            TypeError: If multiplication is not supported
        """
        if isinstance(other, Matrix):
            return Matrix(np.matmul(self.matrix, other.matrix))
        elif isinstance(other, col_vector):
            return col_vector(np.matmul(self.matrix, other.coeffs))
        elif isinstance(other, (complex, float, int)):
            return Matrix(self.matrix * other)
        else:
            raise TypeError(f"Cannot multiply Matrix with {type(other)}")
    
    def __rmul__(self, other: Union[complex, float]) -> 'Matrix':
        """
        Right multiplication (scalar * Matrix).
        
        Args:
            other: Scalar
            
        Returns:
            Scaled matrix
        """
        return Matrix(self.matrix * other)

    def __truediv__(self, other: Union[complex, float]) -> 'Matrix':
        """
        Divide matrix by scalar.
        
        Args:
            other: Scalar divisor
            
        Returns:
            Matrix divided by scalar
        """
        return Matrix(self.matrix / other)

    def __rtruediv__(self, other: Union[complex, float]) -> np.matrix:
        """
        Right division (scalar / Matrix).
        
        Args:
            other: Scalar numerator
            
        Returns:
            Result of division
        """
        return self.matrix / other
    
    def __add__(self, other: 'Matrix') -> 'Matrix':
        """
        Add two matrices.
        
        Args:
            other: Another Matrix
            
        Returns:
            Sum of matrices
            
        Raises:
            TypeError: If other is not a Matrix
        """
        if not isinstance(other, Matrix):
            raise TypeError(f"Cannot add Matrix with {type(other)}")
        return Matrix(self.matrix + other.matrix)
    
    def __sub__(self, other: 'Matrix') -> 'Matrix':
        """
        Subtract two matrices.
        
        Args:
            other: Another Matrix
            
        Returns:
            Difference of matrices
            
        Raises:
            TypeError: If other is not a Matrix
        """
        if not isinstance(other, Matrix):
            raise TypeError(f"Cannot subtract {type(other)} from Matrix")
        return Matrix(self.matrix - other.matrix)
    
    def __rsub__(self, other: Any) -> 'Matrix':
        """
        Right subtraction (other - self).
        
        Args:
            other: Scalar or Matrix
            
        Returns:
            Result of subtraction
        """
        if isinstance(other, (complex, float, int)):
            return Matrix(other - self.matrix)
        elif isinstance(other, Matrix):
            return Matrix(other.matrix - self.matrix)
        else:
            raise TypeError(f"Cannot subtract Matrix from {type(other)}")

    def __pow__(self, other: 'Matrix') -> 'Matrix':
        """
        Kronecker product of two matrices.
        
        Args:
            other: Another Matrix
            
        Returns:
            Kronecker product as Matrix
        """
        return Matrix(np.kron(self.matrix, other.matrix))
    
    def __rpow__(self, other: Any) -> 'Matrix':
        """Right power (same as left power for Kronecker product)."""
        return self ** other
    
    def __radd__(self, other: Any) -> 'Matrix':
        """Right addition."""
        return self + other

    def __len__(self) -> int:
        """
        Get dimension of matrix.
        
        Returns:
            Number of rows
        """
        return len(self.matrix)

    def scale(self, scalar: Union[complex, float]) -> 'Matrix':
        """
        Scale matrix by scalar.
        
        Args:
            scalar: Scaling factor
            
        Returns:
            Scaled matrix
        """
        return Matrix(scalar * self.matrix)
    
    def count_occurrences(self, element: Union[complex, float]) -> int:
        """
        Count occurrences of element in matrix.
        
        Args:
            element: Element to count
            
        Returns:
            Number of occurrences
        """
        return np.count_nonzero(self.matrix == element)
    
    def kron(self, other: 'Matrix') -> 'Matrix':
        """
        Compute Kronecker product with another matrix.
        
        Args:
            other: Another Matrix
            
        Returns:
            Kronecker product as Matrix
        """
        return Matrix(np.kron(self.matrix, other.matrix))
    
    def to_sparse_matrix(self) -> 'SparseMatrix':
        """
        Convert to sparse matrix representation.
        
        Returns:
            SparseMatrix representation
        """
        return SparseMatrix(csr_matrix(self.matrix))

    def transpose(self) -> 'Matrix':
        """
        Compute transpose of matrix.
        
        Returns:
            Transposed matrix
        """
        return Matrix(np.transpose(self.matrix))
    
    def get_eig_vals(self, num_of_vals: Optional[int] = None, ordering_type: Optional[str] = None) -> tuple:
        """
        Calculate eigenvalues and eigenvectors.
        
        Args:
            num_of_vals: Number of eigenvalues to compute (default: all)
            ordering_type: Ordering type (default: 'SM')
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        if num_of_vals is None:
            num_of_vals = len(self.matrix)
        if ordering_type is None:
            ordering_type = 'SM'
        return eigs(self.matrix)
    
    def __getitem__(self, key: Union[int, tuple, slice]) -> Any:
        """
        Get matrix element(s) by index.
        
        Args:
            key: Index, tuple, or slice
            
        Returns:
            Matrix element(s)
        """
        return self.matrix[key]
    
    def len(self) -> int:
        """
        Get dimension of matrix.
        
        Returns:
            Number of rows
        """
        return len(self.matrix)

    def round(self, dec: int) -> 'Matrix':
        """
        Round matrix elements to specified decimal places.
        
        Args:
            dec: Number of decimal places
            
        Returns:
            Rounded matrix
        """
        return Matrix(np.round(self.matrix, dec))

    def change_type(self, dtype: type) -> 'Matrix':
        """
        Change data type of matrix.
        
        Args:
            dtype: New data type
            
        Returns:
            Matrix with new data type
        """
        return Matrix(self.matrix.astype(dtype))
    
    def save_text(self, filename: str) -> None:
        """
        Save matrix to text file.
        
        Args:
            filename: Path to output file
        """
        np.savetxt(filename, self.matrix)


class SparseMatrix:
    """
    Sparse matrix class for quantum mechanical calculations.
    
    Represents a 2D sparse matrix with support for complex numbers using scipy.sparse.
    Provides operations for matrix arithmetic, transformations, eigenvalues, etc.
    This is more memory-efficient for large, sparse matrices.
    """
    
    def __init__(self, matrix: Union[csr_matrix, np.matrix, np.ndarray]):
        """
        Initialize SparseMatrix from scipy sparse matrix, numpy matrix, or array.
        
        Args:
            matrix: scipy.sparse.csr_matrix, numpy matrix, or array
            
        Raises:
            ValueError: If matrix is not 2D
        """
        if isinstance(matrix, (np.matrix, np.ndarray)):
            matrix = csr_matrix(matrix)
        elif not isinstance(matrix, csr_matrix):
            matrix = csr_matrix(matrix)
        
        if len(matrix.shape) != 2:
            raise ValueError(f"SparseMatrix requires 2D array, got shape {matrix.shape}")
        
        self.matrix = matrix.tocsr()  # Ensure CSR format
        self.dim = self.matrix.shape[0]
    
    def tolist(self) -> List[List[complex]]:
        """
        Convert sparse matrix to nested list.
        
        Returns:
            Nested list representation
        """
        return self.matrix.todense().tolist()
    
    def calc_inverse(self) -> 'SparseMatrix':
        """
        Calculate sparse matrix inverse.
        
        Returns:
            Inverse sparse matrix
        """
        # Use sparse solver for inverse
        identity = sparse.identity(self.dim, format='csc', dtype=complex_number_typ)
        inv_matrix = spsolve(self.matrix.tocsc(), identity)
        return SparseMatrix(csr_matrix(inv_matrix))
    
    @staticmethod
    def from_col_vectors(bases: List[col_vector]) -> 'SparseMatrix':
        """
        Create sparse matrix from column vectors.
        
        Args:
            bases: List of col_vector objects
            
        Returns:
            SparseMatrix with columns from basis vectors
        """
        # Convert col_vectors to sparse column vectors
        sp_col_vecs = []
        for base in bases:
            # Convert col_vector to sparse format
            col_data = base.coeffs.transpose().tolist()[0]
            sp_col = csr_matrix((col_data, ([0]*len(col_data), range(len(col_data)))), 
                               shape=(len(col_data), 1), dtype=complex_number_typ)
            sp_col_vecs.append(sp_col)
        
        new_mx = sparse.hstack(sp_col_vecs).tocsr()
        return SparseMatrix(new_mx)
    
    @staticmethod
    def from_row_vectors(bases: List[row_vector]) -> 'SparseMatrix':
        """
        Create sparse matrix from row vectors.
        
        Args:
            bases: List of row_vector objects
            
        Returns:
            SparseMatrix with rows from basis vectors
        """
        raw_matrix = []
        for base in bases:
            raw_matrix.append(base.coeffs.tolist()[0])
        matrix = csr_matrix(raw_matrix, dtype=complex_number_typ)
        return SparseMatrix(matrix)
    
    def to_new_bases(self, bases: List[col_vector]) -> 'SparseMatrix':
        """
        Transform sparse matrix to new basis.
        
        Args:
            bases: List of new basis vectors
            
        Returns:
            Transformed sparse matrix
        """
        V: SparseMatrix = SparseMatrix.from_col_vectors(bases)
        V_inv: SparseMatrix = V.calc_inverse()
        return V_inv * self * V
    
    def __str__(self) -> str:
        """String representation of the sparse matrix."""
        return str(self.matrix)
    
    def __repr__(self) -> str:
        """String representation of the sparse matrix."""
        return f"SparseMatrix(shape={self.matrix.shape}, nnz={self.matrix.nnz})"
    
    @staticmethod
    def create_Lz_mx() -> 'SparseMatrix':
        """
        Create Lz matrix operator: [[0, -i], [i, 0]].
        
        Returns:
            Lz operator sparse matrix
        """
        raw_mx = csr_matrix([[0, complex(0, -1)], [complex(0, 1), 0]], 
                           dtype=complex_number_typ)
        return SparseMatrix(raw_mx)
    
    @staticmethod
    def create_eye(dim: int) -> 'SparseMatrix':
        """
        Create sparse identity matrix.
        
        Args:
            dim: Dimension of matrix
            
        Returns:
            Sparse identity matrix
        """
        return SparseMatrix(sparse.identity(dim, dtype=complex_number_typ))
    
    @staticmethod
    def create_zeros(dim: Union[int, tuple]) -> 'SparseMatrix':
        """
        Create sparse zero matrix.
        
        Args:
            dim: Dimension(s) of matrix (int for square, tuple for rectangular)
            
        Returns:
            Sparse zero matrix
        """
        if isinstance(dim, int):
            dim = (dim, dim)
        return SparseMatrix(sparse.csr_matrix(dim, dtype=complex_number_typ))
    
    @staticmethod
    def create_permutation_matrix(perm_arr: List[int]) -> 'SparseMatrix':
        """
        Create permutation matrix from permutation array.
        
        Matches the convention from previous implementation:
        P[perm_arr[i], i] = 1, which means row perm_arr[i] gets value from column i
        
        When applied: P * v, element v[i] goes to position perm_arr[i] in the result
        
        Args:
            perm_arr: List of integers defining the permutation
                     perm_arr[i] = j means element at position i goes to position j
            
        Returns:
            SparseMatrix: Permutation matrix
        """
        dim = len(perm_arr)
        data = [1.0] * dim
        col_ind = list(range(dim))
        
        # Create permutation matrix: P[perm_arr[i], i] = 1
        # This means: row perm_arr[i] gets value from column i
        # When applied: P * v, element v[i] goes to position perm_arr[i]
        mx = csr_matrix((data, (perm_arr, col_ind)), shape=(dim, dim), dtype=complex_number_typ)
        return SparseMatrix(mx)
    
    def save(self, filename: str) -> None:
        """
        Save sparse matrix to file.
        
        Args:
            filename: Path to output file
        """
        sparse.save_npz(filename, self.matrix)
    
    def multiply(self, other: 'SparseMatrix') -> 'SparseMatrix':
        """
        Multiply two sparse matrices.
        
        Args:
            other: Another SparseMatrix
            
        Returns:
            Product of sparse matrices
        """
        return SparseMatrix(self.matrix * other.matrix)
    
    def __mul__(self, other: Any) -> Union['SparseMatrix', 'col_vector']:
        """
        Multiply sparse matrix with other objects.
        
        Args:
            other: SparseMatrix, col_vector, or scalar
            
        Returns:
            - SparseMatrix * SparseMatrix: SparseMatrix
            - SparseMatrix * col_vector: col_vector
            
        Raises:
            TypeError: If multiplication is not supported
        """
        if isinstance(other, SparseMatrix):
            return SparseMatrix(self.matrix * other.matrix)
        elif isinstance(other, col_vector):
            # Convert col_vector to sparse, multiply, convert back
            sp_col = csr_matrix(other.coeffs)
            result = self.matrix * sp_col
            return col_vector(result.todense())
        elif isinstance(other, (complex, float, int)):
            return SparseMatrix(self.matrix * other)
        else:
            raise TypeError(f"Cannot multiply SparseMatrix with {type(other)}")
    
    def __rmul__(self, other: Union[complex, float]) -> 'SparseMatrix':
        """
        Right multiplication (scalar * SparseMatrix).
        
        Args:
            other: Scalar
            
        Returns:
            Scaled sparse matrix
        """
        return SparseMatrix(self.matrix * other)
    
    def __truediv__(self, other: Union[complex, float]) -> 'SparseMatrix':
        """
        Divide sparse matrix by scalar.
        
        Args:
            other: Scalar divisor
            
        Returns:
            Sparse matrix divided by scalar
        """
        return SparseMatrix(self.matrix / other)
    
    def __rtruediv__(self, other: Union[complex, float]) -> Any:
        """
        Right division (scalar / SparseMatrix).
        
        Args:
            other: Scalar numerator
            
        Returns:
            Result of division
        """
        return self.matrix / other
    
    def __add__(self, other: 'SparseMatrix') -> 'SparseMatrix':
        """
        Add two sparse matrices.
        
        Args:
            other: Another SparseMatrix
            
        Returns:
            Sum of sparse matrices
            
        Raises:
            TypeError: If other is not a SparseMatrix
        """
        if not isinstance(other, SparseMatrix):
            raise TypeError(f"Cannot add SparseMatrix with {type(other)}")
        return SparseMatrix(self.matrix + other.matrix)
    
    def __sub__(self, other: 'SparseMatrix') -> 'SparseMatrix':
        """
        Subtract two sparse matrices.
        
        Args:
            other: Another SparseMatrix
            
        Returns:
            Difference of sparse matrices
            
        Raises:
            TypeError: If other is not a SparseMatrix
        """
        if not isinstance(other, SparseMatrix):
            raise TypeError(f"Cannot subtract {type(other)} from SparseMatrix")
        return SparseMatrix(self.matrix - other.matrix)
    
    def __rsub__(self, other: Any) -> 'SparseMatrix':
        """
        Right subtraction (other - self).
        
        Args:
            other: Scalar or SparseMatrix
            
        Returns:
            Result of subtraction
        """
        if isinstance(other, (complex, float, int)):
            return SparseMatrix(other - self.matrix)
        elif isinstance(other, SparseMatrix):
            return SparseMatrix(other.matrix - self.matrix)
        else:
            raise TypeError(f"Cannot subtract SparseMatrix from {type(other)}")
    
    def __pow__(self, other: 'SparseMatrix') -> 'SparseMatrix':
        """
        Kronecker product of two sparse matrices.
        
        Args:
            other: Another SparseMatrix
            
        Returns:
            Kronecker product as SparseMatrix
        """
        return SparseMatrix(sparse.kron(self.matrix, other.matrix))
    
    def __rpow__(self, other: Any) -> 'SparseMatrix':
        """Right power (same as left power for Kronecker product)."""
        return self ** other
    
    def __radd__(self, other: Any) -> 'SparseMatrix':
        """Right addition."""
        return self + other
    
    def __len__(self) -> int:
        """
        Get dimension of sparse matrix.
        
        Returns:
            Number of rows
        """
        return self.dim
    
    def scale(self, scalar: Union[complex, float]) -> 'SparseMatrix':
        """
        Scale sparse matrix by scalar.
        
        Args:
            scalar: Scaling factor
            
        Returns:
            Scaled sparse matrix
        """
        return SparseMatrix(scalar * self.matrix)
    
    def count_occurrences(self, element: Union[complex, float]) -> int:
        """
        Count occurrences of element in sparse matrix.
        
        Args:
            element: Element to count
            
        Returns:
            Number of occurrences
        """
        data_list = self.matrix.data.tolist()
        return data_list.count(element)
    
    def kron(self, other: 'SparseMatrix') -> 'SparseMatrix':
        """
        Compute Kronecker product with another sparse matrix.
        
        Args:
            other: Another SparseMatrix
            
        Returns:
            Kronecker product as SparseMatrix
        """
        return SparseMatrix(sparse.kron(self.matrix, other.matrix))
    
    def to_sparse_matrix(self) -> 'SparseMatrix':
        """
        Convert to sparse matrix representation (already sparse).
        
        Returns:
            Self
        """
        return self
    
    def to_dense_matrix(self) -> 'Matrix':
        """
        Convert sparse matrix to dense Matrix.
        
        Returns:
            Dense Matrix representation
        """
        return Matrix(self.matrix.todense())
    
    def transpose(self) -> 'SparseMatrix':
        """
        Compute transpose of sparse matrix.
        
        Returns:
            Transposed sparse matrix
        """
        return SparseMatrix(self.matrix.transpose().tocsr())
    
    def get_eig_vals(self, num_of_vals: Optional[int] = None, ordering_type: Optional[str] = None) -> tuple:
        """
        Calculate eigenvalues and eigenvectors using sparse methods.
        
        Args:
            num_of_vals: Number of eigenvalues to compute (default: all, but sparse solvers
                        typically compute a subset)
            ordering_type: Ordering type ('SM' for smallest magnitude, 'SA' for smallest algebraic,
                         'LM' for largest magnitude, 'LA' for largest algebraic)
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        if num_of_vals is None:
            # For sparse matrices, compute all eigenvalues (may be slow for large matrices)
            # In practice, you'd want to specify num_of_vals for large matrices
            num_of_vals = min(self.dim, 10)  # Default to 10 for large matrices
        
        if ordering_type is None:
            ordering_type = 'SM'  # Smallest magnitude
        
        # For small matrices or when k >= N-1, use dense solver
        # Sparse iterative solvers require k < N-1
        if num_of_vals >= self.dim - 1:
            # Convert to dense and use dense solver
            dense_matrix = self.matrix.todense()
            eigen_vals, eigen_vects = eigs(dense_matrix)
            return eigen_vals, eigen_vects
        
        # Use sparse eigenvalue solver for large matrices
        try:
            eigen_vals, eigen_vects = sparse_eigs(self.matrix, k=num_of_vals, which=ordering_type)
            return eigen_vals, eigen_vects
        except (TypeError, ValueError) as e:
            # Fallback to dense solver if sparse solver fails
            dense_matrix = self.matrix.todense()
            eigen_vals, eigen_vects = eigs(dense_matrix)
            return eigen_vals, eigen_vects
    
    def __getitem__(self, key: Union[int, tuple, slice]) -> Any:
        """
        Get sparse matrix element(s) by index.
        
        Args:
            key: Index, tuple, or slice
            
        Returns:
            Matrix element(s)
        """
        return self.matrix[key]
    
    def len(self) -> int:
        """
        Get dimension of sparse matrix.
        
        Returns:
            Number of rows
        """
        return self.dim
    
    def round(self, dec: int) -> 'SparseMatrix':
        """
        Round sparse matrix elements to specified decimal places.
        
        Args:
            dec: Number of decimal places
            
        Returns:
            Rounded sparse matrix
        """
        rounded_matrix = self.matrix.copy()
        rounded_matrix.data = np.round(rounded_matrix.data, dec)
        return SparseMatrix(rounded_matrix)
    
    def change_type(self, dtype: type) -> 'SparseMatrix':
        """
        Change data type of sparse matrix.
        
        Args:
            dtype: New data type
            
        Returns:
            SparseMatrix with new data type
        """
        return SparseMatrix(self.matrix.astype(dtype))
    
    def save_text(self, filename: str) -> None:
        """
        Save sparse matrix to text file (converts to dense first).
        
        Args:
            filename: Path to output file
        """
        np.savetxt(filename, self.matrix.todense())
    
    def get_sparse_blocks(self) -> Tuple[List[csr_matrix], List[int]]:
        """
        Decompose sparse matrix into block-diagonal form using graph network analysis.
        
        Uses networkx to find connected components in the matrix graph representation.
        Each connected component corresponds to a block in the block-diagonal matrix.
        
        Returns:
            Tuple of (list of sparse matrix blocks, list of basis indices for reordering)
            
        Raises:
            ImportError: If networkx is not available
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for block diagonalization. Install with: pip install networkx")
        
        # Convert sparse matrix to graph
        G = nx.from_scipy_sparse_array(self.matrix)
        
        # Find connected components
        G_components = nx.connected_components(G)
        
        # Create subgraphs for each connected component
        connected_comp_sub_graphs = [G.subgraph(c).copy() for c in G_components]
        
        # Extract basis indices (for reordering if needed)
        bases_indexes = [e for c in connected_comp_sub_graphs for e in list(c.nodes)]
        
        # Convert subgraphs back to sparse matrices
        sp_mx_blocks = [nx.to_scipy_sparse_array(s) for s in connected_comp_sub_graphs]
        
        return sp_mx_blocks, bases_indexes
    
    def get_sparse_blocks_matrixes(self) -> Tuple[List['SparseMatrix'], List[int]]:
        """
        Decompose sparse matrix into block-diagonal form, returning SparseMatrix objects.
        
        Similar to get_sparse_blocks() but returns SparseMatrix objects instead of raw csr_matrix.
        
        Returns:
            Tuple of (list of SparseMatrix blocks, list of basis indices for reordering)
            
        Raises:
            ImportError: If networkx is not available
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for block diagonalization. Install with: pip install networkx")
        
        # Convert sparse matrix to graph
        G = nx.from_scipy_sparse_array(self.matrix)
        
        # Find connected components
        G_components = nx.connected_components(G)
        
        # Create subgraphs for each connected component
        connected_comp_sub_graphs = [G.subgraph(c).copy() for c in G_components]
        
        # Extract basis indices
        bases_indexes = [e for c in connected_comp_sub_graphs for e in list(c.nodes)]
        
        # Convert subgraphs to SparseMatrix objects
        sp_mx_blocks = [SparseMatrix(nx.to_scipy_sparse_array(s)) for s in connected_comp_sub_graphs]
        
        return sp_mx_blocks, bases_indexes
    
    def get_blocks(self) -> Tuple[List['Matrix'], List[int]]:
        """
        Decompose sparse matrix into block-diagonal form, returning dense Matrix objects.
        
        Similar to get_sparse_blocks() but converts each block to a dense Matrix.
        Useful for compatibility with existing dense matrix code.
        
        Returns:
            Tuple of (list of dense Matrix blocks, list of basis indices for reordering)
        """
        sp_mx_blocks, new_order_of_basis = self.get_sparse_blocks()
        dense_blocks = [Matrix(np.matrix(s.toarray(), dtype=complex_number_typ)) for s in sp_mx_blocks]
        return dense_blocks, new_order_of_basis
    
    def herm_op_eigsh(self, eig_state_per_block: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute eigenvalues/eigenvectors for Hermitian operator using sparse methods.
        
        Computes the lowest eigenvalues and eigenvectors for a Hermitian sparse matrix.
        
        Args:
            eig_state_per_block: Number of eigenstates to compute per block
            
        Returns:
            Tuple of (eigenvalues array, eigenvectors matrix)
        """
        dim = self.matrix.shape[0]
        
        # Limit k to be less than N-1 for sparse solver
        k = min(eig_state_per_block, dim - 1) if dim > 1 else min(eig_state_per_block, dim)
        
        # For small blocks or when k >= N-1, use dense solver
        if k >= dim - 1 or dim <= 2:
            # Convert to dense and use dense solver
            dense_matrix = self.matrix.todense()
            eigen_vals, eigen_vects = eigs(dense_matrix)
            # Sort by real part (for Hermitian matrices, eigenvalues are real)
            idx = np.argsort(eigen_vals.real)
            eigen_vals = eigen_vals[idx]
            eigen_vects = eigen_vects[:, idx]
            # Return only requested number
            return eigen_vals[:eig_state_per_block], eigen_vects[:, :eig_state_per_block]
        
        # Compute lowest eigenvalues (smallest algebraic) using sparse solver
        low_eigenvalues, low_eigenvectors = eigsh(
            self.matrix, 
            k=k, 
            which='SA'  # Smallest Algebraic
        )
        
        return low_eigenvalues, low_eigenvectors
    
    def calc_eigen_all_sparse_blocks(
        self, 
        eig_state_per_block: int
    ) -> Tuple[List[Any], List[int], Any]:
        """
        Compute eigenvalues/eigenvectors by block diagonalization.
        
        This method decomposes the sparse matrix into connected blocks using graph analysis,
        then computes eigenvalues/eigenvectors for each block separately. This is much more
        efficient than diagonalizing the full matrix when it has block-diagonal structure.
        
        Args:
            eig_state_per_block: Number of eigenstates to compute per block
            
        Returns:
            Tuple of:
            - list of eigen kets (ket_vector objects)
            - list of basis indices for reordering
            - MatrixOperator containing the eigenvector matrix
            
        Raises:
            ImportError: If networkx is not available
        """
        if not HAS_NETWORKX:
            raise ImportError("networkx is required for block diagonalization. Install with: pip install networkx")
        
        # Get sparse matrix blocks
        sp_mx_blocks: List[SparseMatrix]
        sp_mx_blocks, new_basis_order = self.get_sparse_blocks_matrixes()
        
        dim = self.matrix.shape[0]
        eig_vecs_blocks = []
        eig_vals_list = np.array([], dtype=complex_number_typ)
        
        # Diagonalize each block separately
        for i, sp_mx_block in enumerate(sp_mx_blocks):
            # Compute eigenvalues/eigenvectors for this block
            # herm_op_eigsh returns: (eigenvalues, eigenvectors) where eigenvectors shape is (n, k)
            # with n = block dimension, k = number of eigenvectors
            eig_vals, eig_vecs = sp_mx_block.herm_op_eigsh(eig_state_per_block)
            
            # Store eigenvectors as-is (shape: n × k, columns are eigenvectors)
            # These are in the block's local basis (part of the reordered basis)
            eig_vecs_blocks.append(eig_vecs)
            
            # Accumulate eigenvalues
            eig_vals_list = np.concatenate((eig_vals_list, eig_vals), axis=0)
        
        # Combine all eigenvectors into block-diagonal matrix
        # block_diag creates a matrix where rows correspond to the reordered basis
        # Shape will be (dim, k_total) where k_total is the total number of eigenvectors
        eigen_vects_sp_mx = SparseMatrix(block_diag(eig_vecs_blocks).tocsr())
        
        # CRITICAL: Reorder eigenvectors back to original basis order
        # new_basis_order[i] = original index at reordered position i
        # To transform back: we need inv_perm where inv_perm[original_idx] = reordered_pos
        # Then apply permutation to map reordered -> original
        dim = len(new_basis_order)
        
        # Create inverse permutation: inv_perm[original_index] = reordered_position
        inv_perm = [0] * dim
        for reordered_pos, original_index in enumerate(new_basis_order):
            inv_perm[original_index] = reordered_pos
        
        # Create permutation matrix to transform from reordered to original
        # We want: original position i gets value from reordered position inv_perm[i]
        # Our permutation matrix convention: P[perm[i], i] = 1 means row perm[i] gets value from column i
        # So we need: P[inv_perm[i], i] = 1, which means we use inv_perm as the permutation
        # But wait - that would give: result[inv_perm[i]] = input[i], which is not what we want
        # 
        # Actually, we want: result[i] = input[inv_perm[i]]
        # This means: row i gets value from column inv_perm[i]
        # So we need P[i, inv_perm[i]] = 1
        # But our convention is P[perm[i], i] = 1, so we need the inverse
        # Let's create the permutation that does: result[i] = input[inv_perm[i]]
        
        # Actually, simpler: create permutation matrix with inv_perm, then use its inverse
        # Or: create permutation where perm[i] = j such that inv_perm[j] = i
        # That is: find the inverse of inv_perm
        perm_to_original = [0] * dim
        for original_idx in range(dim):
            reordered_pos = inv_perm[original_idx]
            # We want: original position original_idx gets value from reordered position reordered_pos
            # So perm_to_original[reordered_pos] = original_idx
            perm_to_original[reordered_pos] = original_idx
        
        # Now create permutation matrix: P[perm_to_original[i], i] = 1
        # This means: row perm_to_original[i] gets value from column i
        # When applied: P * v, result[perm_to_original[i]] = v[i]
        # But we want: result[i] = v[inv_perm[i]], so we need the transpose
        perm_matrix = SparseMatrix.create_permutation_matrix(perm_to_original)
        
        # Apply permutation: P^T * eigen_vects reorders rows
        # P^T[i, j] = P[j, i], so P^T[i, perm_to_original[j]] = 1 if i = j
        # Actually, let's think: we want result[i] = input[inv_perm[i]]
        # P^T[i, j] = P[j, i], so P^T[i, perm_to_original[j]] = 1 if i = j
        # This gives: result[i] = input[perm_to_original[i]]? No...
        
        # Let me use a different approach: create the permutation matrix directly
        # We want a matrix Q where Q[i, inv_perm[i]] = 1
        # This means: row i gets value from column inv_perm[i]
        # So Q * v gives: result[i] = v[inv_perm[i]] ✓
        perm_data = [1.0] * dim
        perm_rows = list(range(dim))
        perm_cols = inv_perm
        perm_mx = csr_matrix((perm_data, (perm_rows, perm_cols)), shape=(dim, dim), dtype=complex_number_typ)
        perm_matrix = SparseMatrix(perm_mx)
        
        # Apply permutation to eigenvector matrix: perm_matrix * eigen_vects
        # This reorders the rows (basis elements) back to original order
        eigen_vects_sp_mx = SparseMatrix(perm_matrix.matrix * eigen_vects_sp_mx.matrix)
        
        # Convert to col_vectors and create ket_vectors
        # We need to extract each column from the eigenvector matrix
        # Use lazy import to avoid circular dependency
        from jahn_teller_dynamics.math.matrix_mechanics import ket_vector, MatrixOperator
        
        # The number of eigenvectors should match the number of eigenvalues
        num_eigen = len(eig_vals_list)
        num_cols = eigen_vects_sp_mx.matrix.shape[1]
        
        # Use the minimum to avoid index errors
        num_to_process = min(num_eigen, num_cols)
        
        eigen_kets = []
        for i in range(num_to_process):
            # Extract column i
            col_data = eigen_vects_sp_mx.matrix[:, i].todense()
            col_vec = col_vector(col_data)
            
            # Get corresponding eigenvalue
            eigen_val = eig_vals_list[i]
            
            # Create ket_vector
            eigen_ket = ket_vector(col_vec, eigen_val=float(eigen_val.real))
            eigen_kets.append(eigen_ket)
        
        # Sort by eigenvalue
        eigen_kets = sorted(eigen_kets, key=lambda x: x.eigen_val)
        
        # Create MatrixOperator for eigenvector matrix (now in original basis order)
        eigen_vects_op = MatrixOperator(eigen_vects_sp_mx)
        
        return eigen_kets, new_basis_order, eigen_vects_op


cartesian_basis = [
    col_vector.from_list([1, 0, 0]),
    col_vector.from_list([0, 1, 0]),
    col_vector.from_list([0, 0, 1])
]
