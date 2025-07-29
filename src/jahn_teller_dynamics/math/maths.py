import numpy as np
from scipy.sparse import csr_matrix
from scipy.linalg import eig as eigs
from scipy.ndimage.filters import minimum_filter
import matplotlib.pyplot as plt
import pandas as pd
import warnings
import itertools
warnings.simplefilter("ignore", np.exceptions.ComplexWarning)
from scipy.optimize import curve_fit
import numpy as np



def exp_func(x, a, b, c):
    return a * np.exp(-b * x) + c

def fit_function(x_data, y_data,p0,func):
    # Initial parameter guesses
    
    # Fit exponential to data
    popt, pcov = curve_fit(func, x_data, y_data, p0=p0)
    
    # Calculate R-squared value
    residuals = y_data - func(x_data, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_data - np.mean(y_data))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Calculate parameter uncertainties
    perr = np.sqrt(np.diag(pcov))
    
    return popt, r_squared, perr


def meV_to_GHz(e):
    return e*241.798935

def GHz_to_meV(G):
    return G/241.798935

class APES_second_order:

    F:float
    K:float
    G:float


    def from_pars(K,F,G):
        new_apes = APES_second_order()
        new_apes.K = K
        new_apes.F = F
        new_apes.G = G
        return new_apes
    
    def calc_apes_plus(self, rho, phi):
        return 0.5*self.K*rho**2 + rho*(self.F**2 + self.G**2*rho**2 + 2*self.F*self.G*rho*np.cos(3*phi))**0.5

    def calc_apes_negativ(self, rho, phi):
        return 0.5*self.K*rho**2 - rho*(self.F**2 + self.G**2*rho**2 + 2*self.F*self.G*rho*np.cos(3*phi))**0.5

    def calc_sec_order_potential(self, x,y):
        return self.F*(x -y) + self.G*(x**2 - y**2 +2*x*y)

    def calc_full_potential(self, xs,ys):
        
        data_mx = np.zeros((len(xs), len(ys)))
        for i in range(0, len(xs)):
            for j in range(0, len(ys)):
                x = xs[i]
                y = ys[j]

                data_mx[i][j] = self.calc_sec_order_potential(x,y)

        res_df =  pd.DataFrame(data_mx,index = ys, columns =xs)
        
        return res_df

    def calc_APES(self, xs,ys):
        data_mx = np.zeros((len(xs), len(ys)))
        for i in range(0, len(xs)):
            for j in range(0, len(ys)):
                x = xs[i]
                y = ys[j]

                rho = (x**2+ y**2)**0.5


                theta = np.arccos(x/rho) if rho!=0.0 else 0.0

                e_plus = self.calc_apes_plus(rho, theta)
                e_neg = self.calc_apes_negativ(rho, theta)

                data_mx[i][j] = e_plus if e_plus<e_neg else e_neg

        res_df =  pd.DataFrame(data_mx,index = ys, columns =xs)
        
        res_df.to_csv('APES_1_.csv')
        return res_df, data_mx

    def calc_full_apes_negativ_Descartes(self, xs, ys):

        
        data_mx = np.zeros((len(xs), len(ys)))
        for i in range(0, len(xs)):
            for j in range(0, len(ys)):
                x = xs[i]
                y = ys[j]

                rho = (x**2+ y**2)**0.5


                theta = np.arccos(x/rho) if rho!=0.0 else 0.0

                data_mx[i][j] = self.calc_apes_negativ(rho, theta)

        res_df =  pd.DataFrame(data_mx,index = ys, columns =xs)
        
        res_df.to_csv('APES_1_.csv')
        return res_df, data_mx


    def calc_full_apes_plus_Descartes(self, xs, ys):

        
        data_mx = np.zeros((len(xs), len(ys)))
        for i in range(0, len(xs)):
            for j in range(0, len(ys)):
                x = xs[i]
                y = ys[j]

                rho = (x**2+ y**2)**0.5


                theta = np.arccos(x/rho) if rho!=0.0 else 0.0

                data_mx[i][j] = self.calc_apes_plus(rho, theta)

        res_df =  pd.DataFrame(data_mx,index = ys, columns =xs)
        
        res_df.to_csv('APES_1_.csv')
        return res_df, data_mx


    def plot_polar_contour(values, zeniths ,azimuths):

        theta = np.radians(azimuths)
        zeniths = np.array(zeniths)
    
        values = np.array(values)
        values = values.reshape(len(azimuths), len(zeniths))
    
        r, theta = np.meshgrid(zeniths, np.radians(azimuths))
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'))
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        plt.autumn()
        cax = ax.contourf(theta, r, values, 30,cmap='jet')
        cax.changed()
        plt.autumn()
        cb = fig.colorbar(cax)
        cb.set_label("Pixel reflectance")
        return fig, ax, cax



def get_loc_min_indexes(data):
    minima = (data == minimum_filter(data, 3, mode='constant', cval=0.0))

    print('min: \n'  +str(minima))
    res = np.where(1 == minima)
    
    fmt_res = []

    for i, j in zip(res[0], res[1]):
        fmt_res.append((i,j))

    return fmt_res


def isFloat(s):
   try:
      float(s)
      return True
   except:
      return False

precision = 0.0000001

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



    def basis_trf(self,new_bases:list):
        new_bases_row_vectors = [ new_base.to_row_vector() for new_base in new_bases  ]
        basis_trf_mx =  Matrix.from_row_vectors(new_bases_row_vectors)
        return basis_trf_mx*self

    def length(self):
        coeffs = self.tolist()
        res = 0.0
        for coeff in coeffs:
            res+=abs(coeff)**2
        return res**0.5

    def normalize(self):
        return (1/self.length())*self

    def from_list(coeff_list:list):
        coeffs_mx = np.matrix([coeff_list]).transpose()
        return col_vector(coeffs_mx)


    def in_new_basis(self, basis_vecs:list):
        coeffs = self.tolist()

        res_vec = col_vector.from_list([ 0.0, 0.0, 0.0 ])

        for coeff,basis_vec in zip(coeffs, basis_vecs):
            res_vec+=coeff*basis_vec
        return res_vec

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

        flatten_coeffs = np.reshape(self.coeffs,(len(self.coeffs)))
        return flatten_coeffs.tolist()[0]

    def zeros(dim):
        matrix = np.matrix(np.zeros(shape=( dim, 1 )), dtype=complex_number_typ)

        return col_vector(matrix)


    def set_item(self, index, item):
        self.coeffs.itemset((index, 0), item)

    def __eq__(self,other):
        if other==None:
            return False
        else:
            return equal_matrix(self.coeffs, other.coeffs)

    def round(self, dig_num):
        return col_vector(coeffs=np.round(self.coeffs, dig_num))

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
        coeffs_list = self.tolist()
        res_str = ''

        for coeff in coeffs_list[0:-1]:
            res_str+=str(coeff)+', '

        res_str+= str(coeffs_list[-1])
        return res_str

    def __getitem__(self, key):
        return self.coeffs[key][0]
    
    def set_val(self, index, val):
        self.coeffs[index, 0] = val
    def to_row_vector(self):
        return row_vector(np.conj(np.transpose(self.coeffs)))

    def __mul__(self, other):
        if type(other) is row_vector:
            return Matrix(np.matmul(self.coeffs,other.coeffs))
        elif (type(other) is complex) or (type(other) is float):
            return col_vector(self.coeffs* other)




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




    def __mul__(self, other):
        if type(other) is row_vector:
            return Matrix(np.matmul(self.coeffs,other.coeffs))
        elif (type(other) is complex) or (type(other) is float):
            return col_vector(self.coeffs* other)




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
        return float(magnitude**0.5)
    
    def extend(self, l:list):
        old_coeffs_list = self.tolist()
        new_coeffs_list = old_coeffs_list + l
        return row_vector( np.matrix([ new_coeffs_list ]) )

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
    
    def tolist(self):
        return self.coeffs.tolist()[0]
    
    def __eq__(self,other):
        return self.coeffs.tolist()==other.coeffs.tolist()

    def norm_square(self, to_index = None):
        coeffs_list = self.tolist()
        if to_index is None:
            to_index = len(coeffs_list)
        res = 0.0
        for coeff in coeffs_list[0:to_index]:
            res+= abs(coeff)**2
        
        return res

class Matrix:
    
        

    def tolist(self):
        return self.matrix.tolist()


    def calc_inverse(self):
        return Matrix(np.linalg.inv(self.matrix))

    def from_col_vectors( bases:list[col_vector]):
        raw_matrix = []
        for base in bases:
            base_trp = base.coeffs.transpose().tolist()[0]
            raw_matrix.append(base_trp)
        
        #matrix = np.matrix(raw_matrix, dtype=complex_number_typ)
        matrix = np.transpose(np.matrix(raw_matrix, dtype=complex_number_typ))

        return Matrix(matrix)
    
    def from_row_vectors(bases: list[row_vector]):
        raw_matrix = []
        for base in bases:
            raw_matrix.append(base.coeffs.tolist()[0])
        matrix = np.matrix(raw_matrix,dtype=complex_number_typ)
        return Matrix(matrix)
    
    def to_new_bases(self, bases: list[col_vector]):
        V:Matrix = Matrix.from_col_vectors(bases)
        V_inv:Matrix = V.calc_inverse()

        #return V_inv*self*V
        return V*self*V_inv


    def __str__(self):
        return str(self.matrix)
    
    def __repr__(self):
        return str(self.matrix)

    def create_Lz_mx():
        raw_mx =  np.matrix([[0, complex(0,1)], [complex(0,-1), 0]], dtype=complex_number_typ)
        raw_mx =  np.matrix([[0, complex(0,-1)], [complex(0,1), 0]], dtype=complex_number_typ)
        
        return Matrix(raw_mx)

    def create_eye(dim):
        return Matrix(np.eye(dim,dtype= complex_number_typ))

    def create_zeros(dim):
        return Matrix(np.zeros(dim,dtype = complex_number_typ))


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
        return eigs(self.matrix)
    
    def __getitem__(self,key):
        return self.matrix[key]
    
    def len(self):
        return len(self.matrix)
    
    def round(self, dec):
        return Matrix(np.round( self.matrix, dec ))

    def change_type(self, type):
        return Matrix(self.matrix.astype(type))
    def save_text(self, filename):
        np.savetxt(filename, self.matrix)


class numeric_function:
    def __init__(self, input_data:np.matrix, val_data:np.ndarray, equidistant = True ):
        self.input_data = input_data
        self.val_data = val_data
        self.equidistant = equidistant
        self.dxs = [  abs(xs[1]-xs[0]) for xs in self.input_data   ] if equidistant==True else None

    
    def get_gradient(self):

        first_gradient = np.gradient(self.val_data, *self.dxs)
        return [numeric_function(self.input_data, first_gradient_i,equidistant=self.equidistant) for first_gradient_i in first_gradient] 

    def get_extreme_points(self, precision = 0.1):
        first_gradients = self.get_gradient()




        loc_min_xi_indexes = [ np.where(precision>abs( first_grad_i.val_data )) for first_grad_i in first_gradients ]
        
        extr_points_ij = []

    
        for loc_min_i, i in zip(loc_min_xi_indexes , range(len(loc_min_xi_indexes)) ):
            extr_points_ij.append([])
            index_coords = np.matrix( loc_min_i ).transpose()
            for index_coord,j in zip(index_coords, range(0,len(index_coords))):
                extr_points_ij[i].append( row_vector(index_coord))



        extrenum_points = []
        for ext_points_0j in extr_points_ij[0]:
            is_extr_point = True
            for grad_i in first_gradients:
                if abs(grad_i.get_value(tuple(ext_points_0j.tolist())))<precision:
                    is_extr_point=True
                else:
                    is_extr_point=False
                    break
            
            if is_extr_point == True:
                extrenum_points.append(ext_points_0j)



        return [ self.transform_row_vec_to_point(ext_point) for ext_point in extrenum_points]



    def get_value(self, indexes:tuple):
        return self.val_data[indexes]

    def transform_row_vec_to_point(self, row_vec:row_vector):
        new_coords=  []

        indexes = row_vec.tolist()

        

        for ind, axis_val in zip(indexes, self.input_data):
            new_coords.append( axis_val[ind]  )
        
        val_in_point = [self.get_value(tuple(indexes))]

        new_coords+=val_in_point

        return row_vector(np.matrix(new_coords))
    
cartesian_basis = [ col_vector.from_list([1,0,0]), col_vector.from_list([0,1,0]), col_vector.from_list([0,0,1]) ]
