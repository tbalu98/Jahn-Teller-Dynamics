"""
Mathematical utilities for Jahn-Teller dynamics calculations.

This module provides vector and matrix classes for quantum mechanical calculations,
including column vectors, row vectors, and matrices with support for complex numbers.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import eig as eigs
import pandas as pd
from typing import Optional, List, Union, Any
import warnings
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
    
    def to_sparse_matrix(self) -> 'Matrix':
        """
        Convert to sparse matrix representation.
        
        Returns:
            Matrix with sparse representation
        """
        return Matrix(csr_matrix(self.matrix))

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


cartesian_basis = [
    col_vector.from_list([1, 0, 0]),
    col_vector.from_list([0, 1, 0]),
    col_vector.from_list([0, 0, 1])
]
