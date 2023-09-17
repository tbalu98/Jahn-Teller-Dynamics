import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigs
from scipy import sparse
import collections

class Matrix:
    
    def create_eye(dim):
        return Matrix(np.eye(dim))

    def __init__(self, matrix:np.matrix):
        self.matrix = matrix
        self.dim = self.matrix.shape[0]

    def multiply(self, other):
        return Matrix(np.matmul(self.matrix, other.matrix))
    
    def __mul__(self, other):
        #if ( hasattr(other, '__getitem__') ):
        #    return Matrix(np.matmul(self.matrix, other.matrix))
        #else:
        return Matrix(np.matmul(self.matrix,other.matrix))
    def __rmul__(self,  other):
        return self.matrix*other

    def __truediv__(self, other):
        return Matrix(self.matrix/other)

    def __rtruediv__(self, other):

        return self.matrix/other
    
    def __add__(self, other):
        return Matrix( self.matrix + other.matrix)
    
    def __sub__(self, other):
        return Matrix(  self.matrix - other.matrix  )
    
    def __rsub__(self, other):
        return self-other

    def __pow__(self, other):
        return Matrix(np.kron(self.matrix, other.matrix))
    
    def __radd__(self, other):
        return self+other

    def __sum__(self, other):
        return Matrix(self.matrix + other.matrix)
    
    def __len__(self):
        return len(self.matrix)
    """
    def add(self, other):
        return Matrix(self.matrix+ other.matrix)

    def sum(self, other):
        return Matrix(self.matrix+ other.matrix)
    """

    def scale(self, scalar):
        return Matrix(scalar*self.matrix)
    
    def count_occurrences(self, element):
        return np.count_nonzero(self.matrix==element)
    
    def kron(self, other):
        return np.kron(self.matrix, other.matrix)
    
    def to_sparse_matrix(self):
        return Matrix(csr_matrix(self.matrix))

    def transpose(self):
        return np.transpose(self.matrix)
    
    def get_eig_vals(self, num_of_vals=None, ordering_type=None):
        if num_of_vals == None:
            num_of_vals = len(self.matrix)
        if ordering_type == None:
            ordering_type = 'SM'
        return eigs(self.matrix, k = num_of_vals, which=ordering_type)
    
    def __getitem__(self,key):
        return self.matrix[key]
    
    #def __
    def len(self):
        return len(self.matrix)
    
    def round(self, dec):
        return Matrix(np.round( self.matrix, dec ))

    def change_type(self, type):
        matrix = np.matrix( self.matrix, dtype = type )
        return Matrix(matrix)
    

"""
class SparseMatrix:

    def create_eye(dim):
        return SparseMatrix( sparse.eye(dim))

    def __init__(self, matrix):
        self.matrix = csr_matrix(matrix)
        #self.type = matrix.dtype

    def __mul__(self, other):

        return SparseMatrix(other.matrix*self.matrix)

    def __rmul__(self,  other):
        return self*other

    def __truediv__(self, other):
        return self.matrix/other

    def __rtruediv__(self, other):
        return self.matrix/other
    def __add__(self, other):
        return SparseMatrix( self.matrix + other.matrix)

    def __sub__(self, other):
        return SparseMatrix(  self.matrix - other.matrix  )

    def __rsub__(self, other):
        return self-other

    def __pow__(self, other):
        return SparseMatrix(sparse.kron(self.matrix, other.matrix))
    
    def __radd__(self, other):
        return self+other

    def __sum__(self, other):
        return SparseMatrix(self.matrix + other.matrix)
    
    def __len__(self):
        return len(self.matrix)


    def scale(self, scalar):
        return SparseMatrix(scalar*self.matrix)
    
    #!!!
    #def count_occurrences(self, element):
    #    return np.count_nonzero(self.matrix==element)
    
    def count_occurrences( self, des_el, data_trf_fun):
        data_str = self.matrix.data

        arr = [ data_trf_fun(el)  for el in data_str]

        return collections.Counter(arr)[des_el]

    
    def to_Matrix(self):
        return Matrix(self.matrix.toarray())

    def truncate(self, drop_row_num, drop_col_num):
        row_dim = len(self.matrix)
        col_dim = len(self.matrix[0])
        return SparseMatrix(self.matrix[0:row_dim-drop_row_num, 0: col_dim-drop_col_num])

    def transpose(self):
        return SparseMatrix(self.matrix.transpose())
    
    def get_eig_vals(self, num_of_vals=None, ordering_type=None):
        if num_of_vals == None:
            num_of_vals = self.matrix.shape[0]
        if ordering_type == None:
            ordering_type = 'SM'
        return eigs(self.matrix, k = num_of_vals, which=ordering_type)
    
    def __getitem__(self,key):
        return self.matrix[key]
    
    
    def round(self, dec):
        return SparseMatrix( round( self.matrix, dec))

    #!!!
    #def change_type(self, type):
    #    matrix = np.matrix( self.matrix, dtype = type )
    #    return SparseMatrix(matrix)

"""