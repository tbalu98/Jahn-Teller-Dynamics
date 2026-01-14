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
complex_number_typ = np.complex128  # Use double precision for better numerical accuracy


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
    
    def to_sparse_vector(self) -> 'SparseColVector':
        """
        Convert to sparse column vector.
        
        Returns:
            SparseColVector instance
        """
        return SparseColVector(self.coeffs)

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
        elif isinstance(other, SparseRowVector):
            # Convert col_vector to sparse and multiply
            self_sparse = self.to_sparse_vector()
            return SparseMatrix(self_sparse.coeffs * other.coeffs).to_dense_matrix()
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

    def __add__(self, other: Union['col_vector', 'SparseColVector']) -> Union['col_vector', 'SparseColVector']:
        """
        Add two column vectors.
        
        Args:
            other: Another col_vector or SparseColVector
            
        Returns:
            Sum of vectors (SparseColVector if other is sparse, else col_vector)
            
        Raises:
            TypeError: If other is not a col_vector or SparseColVector
        """
        if isinstance(other, SparseColVector):
            # Convert self to sparse and add
            self_sparse = self.to_sparse_vector()
            return SparseColVector(self_sparse.coeffs + other.coeffs)
        elif isinstance(other, col_vector):
            return col_vector(self.coeffs + other.coeffs)
        else:
            raise TypeError(f"Cannot add col_vector with {type(other)}")

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
        elif isinstance(other, SparseColVector):
            # Convert row_vector to sparse and multiply
            # row_vector (1 x n) * col_vector (n x 1) = scalar
            self_sparse = self.to_sparse_vector()
            result = self_sparse.coeffs * other.coeffs
            return complex(result[0, 0])
        elif isinstance(other, Matrix):
            return row_vector(np.matmul(self.coeffs, other.matrix))
        elif isinstance(other, SparseMatrix):
            # Convert row_vector to sparse and multiply
            self_sparse = self.to_sparse_vector()
            # row_vector (1 x n) * matrix (n x m) = row_vector (1 x m)
            result = self_sparse.coeffs * other.matrix
            return SparseRowVector(result).to_dense_vector()
        elif isinstance(other, (complex, float, int)):
            return row_vector(self.coeffs * other)
        elif isinstance(other, row_vector):
            return Matrix(np.matmul(self.coeffs, other.coeffs))
        elif isinstance(other, SparseRowVector):
            # Convert row_vector to sparse and multiply
            self_sparse = self.to_sparse_vector()
            return SparseMatrix(self_sparse.coeffs * other.coeffs).to_dense_matrix()
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

    def to_sparse_vector(self) -> 'SparseRowVector':
        """
        Convert to sparse row vector.
        
        Returns:
            SparseRowVector instance
        """
        return SparseRowVector(self.coeffs)
    
    def __truediv__(self, other: Union[complex, float]) -> 'row_vector':
        """
        Divide row vector by scalar.
        
        Args:
            other: Scalar divisor
            
        Returns:
            row_vector divided by scalar
        """
        return row_vector(self.coeffs / other)

    def __add__(self, other: Union['row_vector', 'SparseRowVector']) -> Union['row_vector', 'SparseRowVector']:
        """
        Add two row vectors.
        
        Args:
            other: Another row_vector or SparseRowVector
            
        Returns:
            Sum of vectors (SparseRowVector if other is sparse, else row_vector)
            
        Raises:
            TypeError: If other is not a row_vector or SparseRowVector
        """
        if isinstance(other, SparseRowVector):
            # Convert self to sparse and add
            self_sparse = self.to_sparse_vector()
            return SparseRowVector(self_sparse.coeffs + other.coeffs)
        elif isinstance(other, row_vector):
            return row_vector(self.coeffs + other.coeffs)
        else:
            raise TypeError(f"Cannot add row_vector with {type(other)}")

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


class SparseColVector:
    """
    Sparse column vector class for quantum mechanical calculations.
    
    Represents a sparse column vector (n x 1 matrix) with support for complex numbers.
    Uses scipy.sparse.csr_matrix for efficient storage of sparse vectors.
    """
    
    def __init__(self, coeffs: Union[csr_matrix, np.matrix, np.ndarray, 'col_vector']):
        """
        Initialize sparse column vector from various input types.
        
        Args:
            coeffs: csr_matrix, numpy matrix/array, or col_vector
                   Must have shape (n, 1) for column vector
                   
        Raises:
            ValueError: If coeffs is not a column vector (shape[1] != 1)
        """
        # Convert from col_vector if needed
        if isinstance(coeffs, col_vector):
            coeffs = coeffs.coeffs
        
        # Convert to sparse matrix
        if isinstance(coeffs, csr_matrix):
            self.coeffs = coeffs.tocsr()
        elif isinstance(coeffs, (np.matrix, np.ndarray)):
            # Convert to array first, then to sparse
            if hasattr(coeffs, 'A'):
                arr = coeffs.A
            else:
                arr = np.asarray(coeffs)
            # Ensure it's a column vector
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            self.coeffs = csr_matrix(arr, dtype=complex_number_typ)
        else:
            raise TypeError(f"Cannot create SparseColVector from {type(coeffs)}")
        
        # Verify it's a column vector
        if self.coeffs.shape[1] != 1:
            raise ValueError(f"SparseColVector requires shape (n, 1), got {self.coeffs.shape}")
    
    def to_dense_vector(self) -> 'col_vector':
        """Convert to dense col_vector."""
        dense = self.coeffs.todense()
        return col_vector(dense)
    
    def to_sparse_row_vector(self) -> 'SparseRowVector':
        """Convert to sparse row vector (complex conjugate transpose)."""
        return SparseRowVector(self.coeffs.getH().tocsr())
    
    def length(self) -> float:
        """Calculate Euclidean length (norm) of the vector."""
        # Use sparse dot product for efficiency
        return float(np.sqrt((self.coeffs.getH() * self.coeffs)[0, 0].real))
    
    def normalize(self) -> 'SparseColVector':
        """Normalize the vector to unit length."""
        norm = self.length()
        if norm == 0.0:
            return SparseColVector(self.coeffs.copy())
        return SparseColVector(self.coeffs / norm)
    
    def tolist(self) -> List[complex]:
        """Convert to Python list."""
        return self.coeffs.todense().A1.tolist()
    
    def __len__(self) -> int:
        """Get dimension of the vector."""
        return self.coeffs.shape[0]
    
    def __getitem__(self, key: int) -> Union[complex, float]:
        """Get coefficient at given index."""
        return complex(self.coeffs[key, 0])
    
    def set_eigen_val(self, eigen_val: Union[complex, float]) -> 'SparseColVector':
        """
        Set eigenvalue on the vector (returns new vector with eigenvalue attribute).
        
        Args:
            eigen_val: Eigenvalue to set
            
        Returns:
            New SparseColVector with eigen_val attribute
        """
        new_vec = SparseColVector(self.coeffs.copy())
        new_vec.eigen_val = eigen_val
        return new_vec
    
    @staticmethod
    def from_sparse_matrix(sp_mx: 'SparseMatrix') -> List['SparseColVector']:
        """
        Extract column vectors from a sparse matrix.
        
        Args:
            sp_mx: SparseMatrix to extract columns from
            
        Returns:
            List of SparseColVector objects, one for each column
        """
        dim = sp_mx.matrix.shape[1]
        sparse_col_vectors = []
        for i in range(dim):
            sparse_col_vectors.append(SparseColVector(sp_mx.matrix[:, i]))
        return sparse_col_vectors
    
    def __mul__(self, other: Any) -> Union['SparseColVector', 'SparseMatrix', complex]:
        """Multiply sparse column vector with other objects."""
        if isinstance(other, (complex, float, int)):
            return SparseColVector(self.coeffs * other)
        elif isinstance(other, SparseRowVector):
            # Outer product: col * row = matrix
            return SparseMatrix(self.coeffs * other.coeffs)
        elif isinstance(other, row_vector):
            # Convert row_vector to sparse and multiply
            other_sparse = other.to_sparse_vector()
            return SparseMatrix(self.coeffs * other_sparse.coeffs)
        else:
            raise TypeError(f"Cannot multiply SparseColVector with {type(other)}")
    
    def __rmul__(self, other: Any) -> 'SparseColVector':
        """Right multiplication (scalar * SparseColVector)."""
        return self * other
    
    def __add__(self, other: Union['SparseColVector', 'col_vector']) -> 'SparseColVector':
        """Add two column vectors."""
        if isinstance(other, col_vector):
            other = other.to_sparse_vector()
        if isinstance(other, SparseColVector):
            return SparseColVector(self.coeffs + other.coeffs)
        raise TypeError(f"Cannot add SparseColVector with {type(other)}")
    
    def __sub__(self, other: Union['SparseColVector', 'col_vector']) -> 'SparseColVector':
        """Subtract two column vectors."""
        if isinstance(other, col_vector):
            other = other.to_sparse_vector()
        if isinstance(other, SparseColVector):
            return SparseColVector(self.coeffs - other.coeffs)
        raise TypeError(f"Cannot subtract {type(other)} from SparseColVector")
    
    @staticmethod
    def zeros(dim: int) -> 'SparseColVector':
        """Create zero sparse column vector."""
        return SparseColVector(csr_matrix((dim, 1), dtype=complex_number_typ))
    
    @staticmethod
    def from_list(coeff_list: List[Union[complex, float]]) -> 'SparseColVector':
        """Create sparse column vector from list."""
        arr = np.array([[c] for c in coeff_list], dtype=complex_number_typ)
        return SparseColVector(csr_matrix(arr))
    
    def __repr__(self) -> str:
        """String representation."""
        return f"SparseColVector(shape={self.coeffs.shape}, nnz={self.coeffs.nnz})"


class SparseRowVector:
    """
    Sparse row vector class for quantum mechanical calculations.
    
    Represents a sparse row vector (1 x n matrix) with support for complex numbers.
    Uses scipy.sparse.csr_matrix for efficient storage of sparse vectors.
    """
    
    def __init__(self, coeffs: Union[csr_matrix, np.matrix, np.ndarray, 'row_vector']):
        """
        Initialize sparse row vector from various input types.
        
        Args:
            coeffs: csr_matrix, numpy matrix/array, or row_vector
                   Must have shape (1, n) for row vector
                   
        Raises:
            ValueError: If coeffs is not a row vector (shape[0] != 1)
        """
        # Convert from row_vector if needed
        if isinstance(coeffs, row_vector):
            coeffs = coeffs.coeffs
        
        # Convert to sparse matrix
        if isinstance(coeffs, csr_matrix):
            self.coeffs = coeffs.tocsr()
        elif isinstance(coeffs, (np.matrix, np.ndarray)):
            # Convert to array first, then to sparse
            if hasattr(coeffs, 'A'):
                arr = coeffs.A
            else:
                arr = np.asarray(coeffs)
            # Ensure it's a row vector
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            self.coeffs = csr_matrix(arr, dtype=complex_number_typ)
        else:
            raise TypeError(f"Cannot create SparseRowVector from {type(coeffs)}")
        
        # Verify it's a row vector
        if self.coeffs.shape[0] != 1:
            raise ValueError(f"SparseRowVector requires shape (1, n), got {self.coeffs.shape}")
    
    def to_dense_vector(self) -> 'row_vector':
        """Convert to dense row_vector."""
        dense = self.coeffs.todense()
        return row_vector(dense)
    
    def to_sparse_col_vector(self) -> 'SparseColVector':
        """Convert to sparse column vector (complex conjugate transpose)."""
        return SparseColVector(self.coeffs.getH().tocsr())
    
    def length(self) -> float:
        """Calculate Euclidean length (norm) of the vector."""
        # Use sparse dot product for efficiency
        return float(np.sqrt((self.coeffs * self.coeffs.getH())[0, 0].real))
    
    def normalize(self) -> 'SparseRowVector':
        """Normalize the vector to unit length."""
        norm = self.length()
        if norm == 0.0:
            return SparseRowVector(self.coeffs.copy())
        return SparseRowVector(self.coeffs / norm)
    
    def tolist(self) -> List[complex]:
        """Convert to Python list."""
        return self.coeffs.todense().A1.tolist()
    
    def __len__(self) -> int:
        """Get dimension of the vector."""
        return self.coeffs.shape[1]
    
    def __getitem__(self, key: int) -> Union[complex, float]:
        """Get coefficient at given index."""
        return complex(self.coeffs[0, key])
    
    def __mul__(self, other: Any) -> Union[complex, 'SparseRowVector', 'SparseMatrix']:
        """Multiply sparse row vector with other objects."""
        if isinstance(other, SparseColVector):
            # Inner product: row * col = scalar
            result = self.coeffs * other.coeffs
            return complex(result[0, 0])
        elif isinstance(other, col_vector):
            # Convert col_vector to sparse and multiply
            other_sparse = other.to_sparse_vector()
            result = self.coeffs * other_sparse.coeffs
            return complex(result[0, 0])
        elif isinstance(other, SparseMatrix):
            # row * matrix = row
            return SparseRowVector(self.coeffs * other.matrix)
        elif isinstance(other, Matrix):
            # Convert Matrix to sparse and multiply
            other_sparse = other.to_sparse_matrix()
            return SparseRowVector(self.coeffs * other_sparse.matrix)
        elif isinstance(other, (complex, float, int)):
            return SparseRowVector(self.coeffs * other)
        elif isinstance(other, SparseRowVector):
            # Outer product: row * row = matrix (transpose needed)
            return SparseMatrix(self.coeffs.getH() * other.coeffs)
        elif isinstance(other, row_vector):
            # Convert row_vector to sparse and multiply
            other_sparse = other.to_sparse_vector()
            return SparseMatrix(self.coeffs.getH() * other_sparse.coeffs)
        else:
            raise TypeError(f"Cannot multiply SparseRowVector with {type(other)}")
    
    def __rmul__(self, other: Any) -> 'SparseRowVector':
        """Right multiplication (scalar * SparseRowVector)."""
        return self * other
    
    def __add__(self, other: Union['SparseRowVector', 'row_vector']) -> 'SparseRowVector':
        """Add two row vectors."""
        if isinstance(other, row_vector):
            other = other.to_sparse_vector()
        if isinstance(other, SparseRowVector):
            return SparseRowVector(self.coeffs + other.coeffs)
        raise TypeError(f"Cannot add SparseRowVector with {type(other)}")
    
    def __sub__(self, other: Union['SparseRowVector', 'row_vector']) -> 'SparseRowVector':
        """Subtract two row vectors."""
        if isinstance(other, row_vector):
            other = other.to_sparse_vector()
        if isinstance(other, SparseRowVector):
            return SparseRowVector(self.coeffs - other.coeffs)
        raise TypeError(f"Cannot subtract {type(other)} from SparseRowVector")
    
    @staticmethod
    def zeros(dim: int) -> 'SparseRowVector':
        """Create zero sparse row vector."""
        return SparseRowVector(csr_matrix((1, dim), dtype=complex_number_typ))
    
    @staticmethod
    def from_list(coeff_list: List[Union[complex, float]]) -> 'SparseRowVector':
        """Create sparse row vector from list."""
        arr = np.array([coeff_list], dtype=complex_number_typ)
        return SparseRowVector(csr_matrix(arr))
    
    def __repr__(self) -> str:
        """String representation."""
        return f"SparseRowVector(shape={self.coeffs.shape}, nnz={self.coeffs.nnz})"


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
    
    @staticmethod
    def create_permutation_matrix(perm_arr: List[int]) -> 'Matrix':
        """
        Create permutation matrix from permutation array.
        
        Matches the convention from SparseMatrix implementation:
        P[perm_arr[i], i] = 1, which means row perm_arr[i] gets value from column i
        
        When applied: P * v, element v[i] goes to position perm_arr[i] in the result
        
        Args:
            perm_arr: List of integers defining the permutation
                     perm_arr[i] = j means element at position i goes to position j
            
        Returns:
            Matrix: Permutation matrix
        """
        dim = len(perm_arr)
        # Create dense permutation matrix
        perm_matrix = np.zeros((dim, dim), dtype=complex_number_typ)
        for i, j in enumerate(perm_arr):
            perm_matrix[i, j] = 1.0
        return Matrix(np.matrix(perm_matrix, dtype=complex_number_typ))

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
    
    def __mul__(self, other: Any) -> Union['Matrix', 'col_vector', 'SparseColVector']:
        """
        Multiply matrix with other objects.
        
        Args:
            other: Matrix, col_vector, SparseColVector, or scalar
            
        Returns:
            - Matrix * Matrix: Matrix
            - Matrix * col_vector: col_vector
            - Matrix * SparseColVector: SparseColVector (converts matrix to sparse)
            
        Raises:
            TypeError: If multiplication is not supported
        """
        if isinstance(other, Matrix):
            return Matrix(np.matmul(self.matrix, other.matrix))
        elif isinstance(other, col_vector):
            return col_vector(np.matmul(self.matrix, other.coeffs))
        elif isinstance(other, SparseColVector):
            # Convert Matrix to sparse for multiplication
            self_sparse = self.to_sparse_matrix()
            return SparseColVector(self_sparse.matrix * other.coeffs)
        elif isinstance(other, (complex, float, int)):
            return Matrix(self.matrix * other)
        else:
            raise TypeError(f"Cannot multiply Matrix with {type(other)}")
    
    def __rmul__(self, other: Union[complex, float, 'row_vector', 'SparseRowVector']) -> Union['Matrix', 'row_vector', 'SparseRowVector']:
        """
        Right multiplication (scalar * Matrix or row_vector * Matrix).
        
        Args:
            other: Scalar, row_vector, or SparseRowVector
            
        Returns:
            Scaled matrix or row_vector
        """
        if isinstance(other, (complex, float, int)):
            return Matrix(self.matrix * other)
        elif isinstance(other, row_vector):
            # row_vector * matrix = row_vector
            return row_vector(np.matmul(other.coeffs, self.matrix))
        elif isinstance(other, SparseRowVector):
            # Convert SparseRowVector to dense, multiply, convert back
            other_dense = other.to_dense_vector()
            result = row_vector(np.matmul(other_dense.coeffs, self.matrix))
            return result.to_sparse_vector()
        else:
            return NotImplemented

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
        # If already a sparse matrix, return it wrapped
        if isinstance(self.matrix, (csr_matrix, SparseMatrix)):
            if isinstance(self.matrix, SparseMatrix):
                return self.matrix
            return SparseMatrix(self.matrix)
        
        # Convert numpy matrix to array first, then to sparse
        # Use .A attribute if available (numpy matrix), otherwise use np.array
        if hasattr(self.matrix, 'A'):
            matrix_array = self.matrix.A
        else:
            matrix_array = np.asarray(self.matrix)
        
        # Check if the array conversion gave us something unexpected
        # (e.g., if self.matrix contained sparse matrices)
        if isinstance(matrix_array, (csr_matrix, SparseMatrix)):
            if isinstance(matrix_array, SparseMatrix):
                return matrix_array
            return SparseMatrix(matrix_array)
        
        # Check if it's a numpy array with object dtype (might contain sparse matrices)
        if isinstance(matrix_array, np.ndarray) and matrix_array.dtype == object:
            # This shouldn't happen, but if it does, try to convert element by element
            # For now, raise an error with a helpful message
            raise TypeError(f"Cannot convert Matrix with object dtype to SparseMatrix. "
                          f"Matrix may contain mixed types.")
        
        # Ensure it's a numpy array
        if not isinstance(matrix_array, np.ndarray):
            # Try to convert to array
            try:
                matrix_array = np.asarray(matrix_array)
            except:
                raise TypeError(f"Cannot convert {type(matrix_array)} to numpy array for sparse conversion")
        
        # Ensure it's a 2D array
        if matrix_array.ndim == 1:
            matrix_array = matrix_array.reshape(-1, 1)
        elif matrix_array.ndim != 2:
            raise ValueError(f"Cannot convert {matrix_array.ndim}D array to sparse matrix")
        
        # Ensure correct dtype before creating sparse matrix
        # Only call astype if it's actually a numpy array with numeric dtype
        if matrix_array.dtype != complex_number_typ:
            try:
                matrix_array = matrix_array.astype(complex_number_typ)
            except (TypeError, ValueError) as e:
                raise TypeError(f"Cannot convert array with dtype {matrix_array.dtype} to {complex_number_typ}: {e}")
        
        # Create sparse matrix - let it infer dtype from the array
        sparse_mx = csr_matrix(matrix_array)
        return SparseMatrix(sparse_mx)

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
            # Row indices: 0 to len-1, Column indices: all 0 (single column)
            sp_col = csr_matrix((col_data, (range(len(col_data)), [0]*len(col_data))), 
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
        # CRITICAL: Use same transformation order as Matrix.to_new_bases for consistency
        # V * self * V_inv (not V_inv * self * V)
        V: SparseMatrix = SparseMatrix.from_col_vectors(bases)
        V_inv: SparseMatrix = V.calc_inverse()
        return V * self * V_inv
    
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
        
        Convention: P[i, perm_arr[i]] = 1
        This means: new position i gets original state perm_arr[i]
        
        When applied: P * v, we get v_new[i] = v_orig[perm_arr[i]]
        
        For operator transformation: O_perm = P * O * P^T
        
        Args:
            perm_arr: List of integers defining the permutation
                     perm_arr[i] = j means new position i gets original state j
            
        Returns:
            SparseMatrix: Permutation matrix
        """
        dim = len(perm_arr)
        data = [1.0] * dim
        row_ind = list(range(dim))
        
        # Create permutation matrix: P[i, perm_arr[i]] = 1
        # This means: row i gets value from column perm_arr[i]
        # When applied: P * v, we get v_new[i] = v_orig[perm_arr[i]]
        mx = csr_matrix((data, (row_ind, perm_arr)), shape=(dim, dim), dtype=complex_number_typ)
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
    
    def __mul__(self, other: Any) -> Union['SparseMatrix', 'col_vector', 'SparseColVector']:
        """
        Multiply sparse matrix with other objects.
        
        Args:
            other: SparseMatrix, col_vector, SparseColVector, or scalar
            
        Returns:
            - SparseMatrix * SparseMatrix: SparseMatrix
            - SparseMatrix * col_vector: SparseColVector (converts col_vector to sparse)
            - SparseMatrix * SparseColVector: SparseColVector
            
        Raises:
            TypeError: If multiplication is not supported
        """
        if isinstance(other, SparseMatrix):
            return SparseMatrix(self.matrix * other.matrix)
        elif isinstance(other, col_vector):
            # Convert col_vector to sparse for multiplication
            other_sparse = other.to_sparse_vector()
            return SparseColVector(self.matrix * other_sparse.coeffs)
        elif isinstance(other, SparseColVector):
            return SparseColVector(self.matrix * other.coeffs)
        elif isinstance(other, (complex, float, int)):
            return SparseMatrix(self.matrix * other)
        else:
            raise TypeError(f"Cannot multiply SparseMatrix with {type(other)}")
    
    def __rmul__(self, other: Union[complex, float, 'SparseRowVector', 'row_vector']) -> Union['SparseMatrix', 'SparseRowVector']:
        """
        Right multiplication (scalar * SparseMatrix or row_vector * SparseMatrix).
        
        Args:
            other: Scalar, SparseRowVector, or row_vector
            
        Returns:
            Scaled sparse matrix or SparseRowVector
        """
        if isinstance(other, (complex, float, int)):
            return SparseMatrix(self.matrix * other)
        elif isinstance(other, SparseRowVector):
            # row_vector * matrix = row_vector
            return SparseRowVector(other.coeffs * self.matrix)
        elif isinstance(other, row_vector):
            # Convert row_vector to sparse and multiply
            other_sparse = other.to_sparse_vector()
            return SparseRowVector(other_sparse.coeffs * self.matrix)
        else:
            return NotImplemented
    
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
        Calculate eigenvalues and eigenvectors using dense solver.
        
        For block diagonalization, always use dense solver (eigs) to ensure exact matching
        with the dense solver implementation.
        
        Args:
            num_of_vals: Number of eigenvalues to compute (default: all)
            ordering_type: Ordering type (not used, kept for compatibility)
            
        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        if num_of_vals is None:
            num_of_vals = self.dim
        
        if ordering_type is None:
            ordering_type = 'SM'  # Not used, but kept for compatibility
        
        # Always use dense solver for block diagonalization
        # Convert to dense and use dense solver
        dense_matrix = np.array(self.matrix.todense(), dtype=complex_number_typ)
        eigen_vals, eigen_vects = eigs(dense_matrix)
        # Ensure eigenvalues are complex128 for consistency
        eigen_vals = eigen_vals.astype(complex_number_typ)
        eigen_vects = eigen_vects.astype(complex_number_typ)
        # Sort by real part (for Hermitian matrices, eigenvalues are real)
        # This matches the dense solver behavior
        idx = np.argsort(eigen_vals.real)
        eigen_vals = eigen_vals[idx]
        eigen_vects = eigen_vects[:, idx]
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
        CRITICAL: Extracts blocks by selecting rows and columns to preserve Hermiticity.
        
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
        
        # Extract basis indices (sorted for consistency)
        bases_indexes = []
        sp_mx_blocks = []
        
        # Extract blocks by selecting rows and columns to preserve Hermiticity
        for component in G_components:
            # Sort component indices for consistency
            #comp_indices = sorted(list(component))
            comp_indices = sorted(list(component))
            
            
            bases_indexes.extend(comp_indices)
            
            # Extract block by selecting rows and columns at these indices
            # This preserves Hermiticity: if we select rows [i1, i2, ...], we select columns [i1, i2, ...]
            block_matrix = self.matrix[np.ix_(comp_indices, comp_indices)]
            sp_mx_blocks.append(SparseMatrix(block_matrix))
        
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
        Compute eigenvalues and eigenvectors for Hermitian sparse matrix.
        
        For small blocks or when requesting all eigenvalues, uses dense solver.
        For large blocks, uses sparse iterative solver.
        """
        """
        Compute eigenvalues/eigenvectors for Hermitian operator using sparse methods.
        
        Computes the lowest eigenvalues and eigenvectors for a Hermitian sparse matrix.
        
        Args:
            eig_state_per_block: Number of eigenstates to compute per block
            
        Returns:
            Tuple of (eigenvalues array, eigenvectors matrix)
        """
        dim = self.matrix.shape[0]
        
        # If requesting all or nearly all eigenvalues, use dense solver
        # This ensures we get all eigenvalues for proper matching with dense solver
        # Use >= dim to ensure we always use dense solver when requesting all eigenvalues
        if eig_state_per_block >= dim or dim <= 2:
            # Convert to dense and use dense solver to get ALL eigenvalues
            # This uses numpy.linalg.eig (via eigs) which is the same as dense solver
            dense_matrix = self.matrix.todense()
            dense_matrix = np.array(dense_matrix, dtype=complex_number_typ)
            eigen_vals, eigen_vects = eigs(dense_matrix)
            # Ensure complex128 precision
            eigen_vals = eigen_vals.astype(complex_number_typ)
            eigen_vects = eigen_vects.astype(complex_number_typ)
            # Sort by real part (for Hermitian matrices, eigenvalues are real)
            idx = np.argsort(eigen_vals.real)
            eigen_vals = eigen_vals[idx]
            eigen_vects = eigen_vects[:, idx]
            # Return only requested number (but we computed all for matching)
            return eigen_vals[:eig_state_per_block], eigen_vects[:, :eig_state_per_block]
        
        # Compute lowest eigenvalues (smallest algebraic) using sparse solver
        # Limit k to dim - 1 (ARPACK requirement: k < N-1)
        # But if k >= dim - 1, ARPACK will fail, so we should have used dense solver above
        k = min(eig_state_per_block, dim - 1)
        if k >= dim - 1:
            # This shouldn't happen if condition above worked, but fallback to dense
            dense_matrix = self.matrix.todense()
            dense_matrix = np.array(dense_matrix, dtype=complex_number_typ)
            eigen_vals, eigen_vects = eigs(dense_matrix)
            eigen_vals = eigen_vals.astype(complex_number_typ)
            eigen_vects = eigen_vects.astype(complex_number_typ)
            idx = np.argsort(eigen_vals.real)
            eigen_vals = eigen_vals[idx]
            eigen_vects = eigen_vects[:, idx]
            return eigen_vals[:eig_state_per_block], eigen_vects[:, :eig_state_per_block]
        
        low_eigenvalues, low_eigenvectors = eigsh(
            self.matrix, 
            k=k, 
            which='SA'  # Smallest Algebraic
        )
        
        # Ensure complex128 precision
        low_eigenvalues = low_eigenvalues.astype(complex_number_typ)
        low_eigenvectors = low_eigenvectors.astype(complex_number_typ)
        
        return low_eigenvalues, low_eigenvectors
    
    def calc_eigen_vals_vects(self) -> Tuple[List[Any], List[int], Any]:
        """
        Compute eigenvalues/eigenvectors by block diagonalization.
        
        Implementation matches Cavity_ionization project exactly.
        Uses get_sparse_blocks_matrixes() to get blocks, then get_eig_vals() on each block.
        
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
        
        # Get sparse matrix blocks (matching Cavity_ionization get_blocks())
        sp_mx_blocks: List[SparseMatrix]
        sp_mx_blocks, new_basis_order = self.get_sparse_blocks_matrixes()
        
        eig_vecs_blocks = []
        eig_vals_list = np.array([], dtype=complex_number_typ)
        
        # Diagonalize each block separately
        for i, sp_mx_block in enumerate(sp_mx_blocks):
            # Use get_eig_vals() which will use dense solver when appropriate
            # This matches Cavity_ionization: sp_mx_block.get_eig_vals()
            # Pass block dimension to compute all eigenvalues for this block
            block_dim = sp_mx_block.dim
            eig_vals, eig_vecs = sp_mx_block.get_eig_vals(num_of_vals=block_dim, ordering_type=None)
            
            # Transpose eigenvectors to match Cavity_ionization
            # eig_vecs from get_eig_vals has shape (n, k) where columns are eigenvectors
            # After transpose, shape is (k, n) where rows are eigenvectors
            eig_vecs_blocks.append(eig_vecs.transpose())
            
            # Accumulate eigenvalues
            eig_vals_list = np.concatenate((eig_vals_list, eig_vals), axis=0)
        
        # Combine all eigenvectors into block-diagonal matrix
        eigen_vects_sp_mx = SparseMatrix(block_diag(eig_vecs_blocks).tocsr())
        
        # Extract column vectors from eigenvector matrix (matching Cavity_ionization)
        sp_col_vects = SparseColVector.from_sparse_matrix(eigen_vects_sp_mx)
        
        # Set eigenvalues on vectors (matching Cavity_ionization)
        sp_col_vects = [vect.set_eigen_val(eig_val) for vect, eig_val in zip(sp_col_vects, eig_vals_list)]
        
        # Sort by eigenvalue (matching Cavity_ionization)
        sp_col_vects = sorted(sp_col_vects, key=lambda x: x.eigen_val)
        
        # Create ket_vectors with sorted eigenvalues (matching Cavity_ionization)
        from jahn_teller_dynamics.math.matrix_mechanics import ket_vector, MatrixOperator
        sorted_eig_vals = sorted(eig_vals_list)
        sp_kets = [ket_vector(k, eigen_val) for k, eigen_val in zip(sp_col_vects, sorted_eig_vals)]
        
        # Create MatrixOperator for eigenvector matrix
        eigen_vects_op = MatrixOperator(eigen_vects_sp_mx)
        
        return sp_kets, new_basis_order, eigen_vects_op
    
    def calc_eigen_all_sparse_blocks(
        self, 
        eig_state_per_block: int
    ) -> Tuple[List[Any], List[int], Any]:
        """
        Compute eigenvalues/eigenvectors by block diagonalization.
        
        This method decomposes the sparse matrix into connected blocks using graph analysis,
        then computes eigenvalues/eigenvectors for each block separately. This is much more
        efficient than diagonalizing the full matrix when it has block-diagonal structure.
        
        Implementation matches Cavity_ionization project exactly.
        
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
        
        eig_vecs_blocks = []
        eig_vals_list = np.array([], dtype=complex_number_typ)
        
        # Diagonalize each block separately
        for i, sp_mx_block in enumerate(sp_mx_blocks):
            # Compute eigenvalues/eigenvectors for this block
            # Use the block's dimension to ensure we compute all eigenvalues for this block
            block_dim = sp_mx_block.dim
            # Request all eigenvalues for this block (or the requested number, whichever is smaller)
            # This ensures we use dense solver when appropriate
            block_eig_state = min(eig_state_per_block, block_dim) if eig_state_per_block is not None else block_dim
            
            # herm_op_eigsh returns: (eigenvalues, eigenvectors) where eigenvectors shape is (n, k)
            # with n = block dimension, k = number of eigenvectors
            # If block_eig_state >= block_dim, it will use dense solver (eigs) for exact matching
            eig_vals, eig_vecs = sp_mx_block.herm_op_eigsh(block_eig_state)
            
            # CRITICAL: Transpose eigenvectors to match Cavity_ionization implementation
            # eig_vecs from eigsh/eigs has shape (n, k) where columns are eigenvectors
            # After transpose, shape is (k, n) where rows are eigenvectors
            eig_vecs_blocks.append(eig_vecs.transpose())
            
            # Accumulate eigenvalues
            eig_vals_list = np.concatenate((eig_vals_list, eig_vals), axis=0)
        
        # Combine all eigenvectors into block-diagonal matrix
        # block_diag creates a matrix where rows correspond to the reordered basis
        # Shape will be (dim, k_total) where k_total is the total number of eigenvectors
        eigen_vects_sp_mx = SparseMatrix(block_diag(eig_vecs_blocks).tocsr())
        
        # Convert to col_vectors and create ket_vectors
        # Use lazy import to avoid circular dependency
        from jahn_teller_dynamics.math.matrix_mechanics import ket_vector, MatrixOperator
        from jahn_teller_dynamics.math.maths import SparseColVector
        
        # Extract columns from eigenvector matrix (matching Cavity_ionization from_sparse_matrix)
        num_eigen = len(eig_vals_list)
        num_cols = eigen_vects_sp_mx.matrix.shape[1]
        num_to_process = min(num_eigen, num_cols)
        
        # Create sparse column vectors from eigenvector matrix
        sp_col_vects = []
        for i in range(num_to_process):
            col_sparse = eigen_vects_sp_mx.matrix[:, i]
            col_vec = SparseColVector(col_sparse)
            sp_col_vects.append(col_vec)
        
        # Set eigenvalues on vectors (matching Cavity_ionization line 243)
        # Note: SparseColVector doesn't have set_eigen_val, so we'll set it in ket_vector
        # Sort by eigenvalue (matching Cavity_ionization line 245)
        sorted_eig_vals = sorted(eig_vals_list)
        sp_col_vects = sorted(sp_col_vects, key=lambda x: x.eigen_val if hasattr(x, 'eigen_val') else 0)
        
        # Create ket_vectors with sorted eigenvalues (matching Cavity_ionization line 247)
        sp_kets = [ket_vector(k, eigen_val) for k, eigen_val in zip(sp_col_vects, sorted_eig_vals)]
        
        # Create MatrixOperator for eigenvector matrix
        eigen_vects_op = MatrixOperator(eigen_vects_sp_mx)
        
        return sp_kets, new_basis_order, eigen_vects_op


cartesian_basis = [
    col_vector.from_list([1, 0, 0]),
    col_vector.from_list([0, 1, 0]),
    col_vector.from_list([0, 0, 1])
]
