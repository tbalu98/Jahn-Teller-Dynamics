import numpy as np
from scipy.sparse import csr_matrix
#from scipy.sparse.linalg import eigs
from scipy.linalg import eig as eigs
from scipy import sparse
import collections



precision = 0.0000001
#precision = 0.0000001

complex_number_typ = np.complex64

def equal_matrix(a:np.matrix, b:np.matrix):
    if a.shape != b.shape:
        return False
    else:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                if (abs(a[i, j] - b[i, j]) > precision) == True:
                    return False
        
        return True



class col_vector:

    def from_file(file_name):
        coeffs = np.loadtxt(file_name)
        return col_vector(np.transpose(np.matrix(coeffs)))

    def calc_abs_square(self):
        coeff_conj = np.conj(self.coeffs)
        return col_vector(np.multiply(coeff_conj,self.coeffs))


    def map(self, func):
        raw_mx = list(map( func, self.coeffs))
        return col_vector(np.matrix(np.reshape(raw_mx, (len(raw_mx), 1))))

    def round(self, dig):
        return col_vector(np.round(self.coeffs,dig))

    def tolist(self):
        return self.coeffs.tolist()

    def zeros(dim):
        matrix = np.matrix(np.zeros(shape=( dim, 1 )), dtype=complex_number_typ)

        return col_vector(matrix)


    def set_item(self, index, item):
        self.coeffs.itemset((index, 0), item)

    def __eq__(self,other):
        #return np.array_equal(self.coeffs , other.coeffs)
        return equal_matrix(self.coeffs, other.coeffs)

    def __init__(self,coeffs:np.matrix):
        if coeffs.shape[1]==1:
            self.coeffs = coeffs
        else:
            return None

    def get_coeffs_list(self):
        coeffs_list = []
        for coeff in self.coeffs:
            coeffs_list.append(complex(coeff))
        return coeffs_list

    def __repr__(self):
        return str(self.coeffs)

    def __str__(self):
        return str(self.coeffs)

    def __getitem__(self, key):
        return self.coeffs[key][0]
    
    def set_val(self, index, val):
        self.coeffs[index, 0] = val
    def to_row_vector(self):
        return row_vector(np.transpose(self.coeffs))

    def __mul__(self, other):
        if type(other) is row_vector:
            return Matrix(np.matmul(self.coeffs,other.coeffs))
        elif (type(other) is complex) or (type(other) is float):
            return col_vector(self.coeffs* other)

        """
        elif isinstance(other,col_vector):
             
            return col_vector(np.multiply(self.coeffs,np.conj(other.coeffs)))
        """




    def __rmul__(self, other):

        return self*other

    def __truediv__(self, other):
        return col_vector(self.coeffs/other)

    def __add__(self, other):
        return col_vector(self.coeffs + other.coeffs)

    def __radd__(self, other):
        return self+ other

    def __sub__(self, other):
        return col_vector(self.coeffs-other.coeffs)

    def __rsub__(self, other):
        return self-other
    
    def __abs__(self):
        magnitude = 0.0
        for coeff in self.coeffs:
            magnitude+=abs(coeff)**2
        return float(magnitude)
    
    

class row_vector:
    def set_val(self, index, val):
        self.coeffs[0, index] = val
    def __init__(self,coeffs:np.matrix):
        if coeffs.shape[0]==1:
            self.coeffs = coeffs
        else:
            return None
        
    def __mul__(self, other:col_vector):
        if type(other) is col_vector:
            return complex(np.matmul(self.coeffs,other.coeffs))
        elif type(other) is Matrix:
            return row_vector(np.matmul( self.coeffs, other.matrix ))
        elif (type(other) is complex) or (type(other) is float):
            return row_vector(self.coeffs* other)


    def __rmul__(self, other: col_vector):
        return Matrix(np.matmul(other.coeffs,self.coeffs))



    def __repr__(self):
        return str(self.coeffs)

    def __str__(self):
        return str(self.coeffs)
    def __getitem__(self, key):
        return self.coeffs[0][key]

class Matrix:
    
    def calc_inverse(self):
        return Matrix(np.linalg.inv(self.matrix))

    def from_col_vectors( bases:list[col_vector]):
        raw_matrix = []
        for base in bases:
            base_trp = base.coeffs.transpose().tolist()[0]
            raw_matrix.append(base_trp)
        
        matrix = np.matrix(raw_matrix)

        return Matrix(matrix.transpose())
    
    def from_row_vectors(bases: list[row_vector]):
        raw_matrix = []
        for base in bases:
            raw_matrix.append(base.coeffs.tolist()[0])
        matrix = np.matrix(raw_matrix)
        return Matrix(matrix)
    
    def to_new_bases(self, bases: list[col_vector]):
        V = Matrix.from_col_vectors(bases)
        V_inv = V.calc_inverse()

        return V_inv*self*V


    def __str__(self):
        return str(self.matrix)
    
    def __repr__(self):
        return str(self.matrix)

    def create_Lz_mx():
        raw_mx =  np.matrix([[0, complex(0,1)], [complex(0,-1), 0]], dtype=complex_number_typ)
        return Matrix(raw_mx)

    def create_eye(dim):
        return Matrix(np.eye(dim,dtype= complex_number_typ))

    def save(self,filename):
        np.savetxt(filename,self.matrix)

    def __init__(self, matrix:np.matrix):
        self.matrix = matrix
        self.dim = self.matrix.shape[0]


    def multiply(self, other):
        return Matrix(np.matmul(self.matrix, other.matrix))
    
    def __mul__(self, other):
        if type(other) is Matrix:
            return Matrix(np.matmul(self.matrix,other.matrix))
        elif type(other) is col_vector:
            return col_vector(np.matmul(self.matrix,other.coeffs))
        
    def __rmul__(self,  other):
        return Matrix(self.matrix*other)

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
    def __rpow__(self, other):
        return self**other
    
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
        return Matrix(np.transpose(self.matrix))
    
    def get_eig_vals(self, num_of_vals=None, ordering_type=None):
        if num_of_vals == None:
            num_of_vals = len(self.matrix)
        if ordering_type == None:
            ordering_type = 'SM'
        #return eigs(self.matrix, k = num_of_vals, which=ordering_type)
        return eigs(self.matrix)
    
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
    def save_text(self, filename):
        np.savetxt(filename, self.matrix)

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