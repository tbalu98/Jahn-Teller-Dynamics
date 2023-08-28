import numpy as np
import re
import sys
import collections
import math
import collections
#Coordinate = collections.namedtuple('Coordinate', 'x y z')

Control_data = collections.namedtuple('Control_data',[ 'case_name', 'symm_geom', 'less_symm_geom_1', 'less_symm_geom_2', 'symm_geom_energy', 'less_symm' ])



class Vector:
    def __init__(self, x, y, z ):
        self.x = x
        self.y= y
        self.z = z
    def add(self, other):
        return Vector(self.x+other.x, self.y+other.y, self.z+other.z)
    def norm(self  ):
        return self.x**2 + self.y**2 + self.z**2

    def length(self ):
        return math.sqrt(self.norm())
    def __repr__(self) -> str:
        return str(self.x)  +' '  + str(self.y) + ' '  +str(self.z)
    
    def subtract(self, vec2):
        return Vector(self.x - vec2.x, self.y - vec2.y, self.z - vec2.z)
    
    def scale(self, c):
        return Vector(self.x * c, self.y * c, self.z * c)

    def dist(self, vec2):
        return self.subtract(vec2).length()
    
    def get_dists(self, vecs: list):
        return [self.dist(x) for x  in vecs]

    def get_min_dist_vec(self, vecs: list):
        dxs = self.get_dists(vecs)
        min_dist =  np.min(dxs)
        index = np.where(dxs ==np.min(dxs) )[0][0]
        return vecs[index], min_dist
        
class Ions:
    def __init__(self, name,  vecs, m=None):
        self.name = name
        self._vecs = vecs
        self.m = m

    def __getitem__(self,i):
        return self._vecs[i]
    def __len__(self):
        return len(self._vecs)
    
    def calc_dist_sq(self, ions2: list):
        dist_sq = 0.0
        for vec in self._vecs:
            dist = vec.get_min_dist_vec(ions2)[1]
            dist_sq = dist_sq + dist**2
        return dist_sq
    
    def __lt__(self, other):
        return len(self)<= len(other)

class Lattice:
    def __init__(self, energy = None):
        self.ions_arr = []
        self.energy = energy
    def get_ions(self, name) -> Ions:
        found_ions = list(filter(  lambda x: x.name == name ,self.ions_arr ))
        
        if len(found_ions) == 1:
            return found_ions[0]
        else:
            print('Num of ions error')
            sys.exit(-1)

    def get_least_ions(self) -> Ions:
        return min(self.ions_arr)
    
    def get_dope_ions(self):
        return min(self.ions_arr)

    def get_maj_ions(self):
        return max(self.ions_arr)

    def same_type(self, other):
        return True

    def calc_dist(self, other_lattice):
        if self.same_type(other_lattice) == True:
            lattice_dist = 0.0
            for ions in self.ions_arr:
                other_ions = other_lattice.get_ions(ions.name)
                lattice_dist = lattice_dist + ions.calc_dist_sq(other_ions)*ions.m
            return lattice_dist**0.5



class POSCAR_data:


    def __init__(self, path):
        #Read from file:
        poscar_file = open(path, 'r')
        lines = poscar_file.readlines()

        self.build_data(lines)


    def build_data(self, lines):
        self.comment = lines[0]
        scaling_factors_line = lines[1]
        self.get_scaling_factors(scaling_factors_line)
        self.get_lattice_vectors(lines)
        self.get_ions(lines)

    def get_ions(self, lines):
        name_line = lines[5]
        numbers_line = lines[6]
        #selective_dynamics = lines[7]


        names = self.convert_line_elements(name_line, str)
        numbers = self.convert_line_elements(numbers_line, int)
        line_num = 8
        self.lattice = Lattice()
        
        for name, number_of_ion in zip( names, numbers):
            new_line_num = line_num + number_of_ion
            coordinates = [self.get_coordinate_from_line( line ) for line in lines[line_num: line_num+number_of_ion]]
            line_num = new_line_num
            self.lattice.ions_arr.append(Ions(name, coordinates))
            #self.ionss[name] = coordinates





    def get_scaling_factors(self, line):
        #scaling_factors = self.get_all_numbers_from_line(line)
        scaling_factors = self.convert_line_elements(line, float)
        if (len(scaling_factors)==1 or len(scaling_factors) ==3 ) is True:
            self.scaling_factors = scaling_factors
        else:
            print('Error in the scaling factors! ' + str(len(scaling_factors)) + ' number of arguments')
            sys.exit(-1)

    def get_lattice_vectors(self, lines):
        a1_line = lines[2]
        a2_line = lines[3]
        a3_line = lines[4]
        
        self.a1 = self.convert_line_elements(a1_line, float)
        self.a2 = self.convert_line_elements(a2_line, float)
        self.a3 = self.convert_line_elements(a3_line, float)

        a1_coords = self.convert_line_elements(a1_line,float)
        a2_coords = self.convert_line_elements(a2_line,float)
        a3_coords = self.convert_line_elements(a3_line,float)

        self.a1 = Vector(a1_coords[0], a1_coords[1], a1_coords[2])
        self.a2 = Vector(a2_coords[0], a2_coords[1], a2_coords[2])
        self.a3 = Vector(a3_coords[0], a3_coords[1], a3_coords[2])



    def convert_line_elements(self, line: str, conv_fun):
        str_elements =  line.split()
        try:
            return [ conv_fun(el) for el in  str_elements]
        except ValueError:
            print('Incorrect values in line:\n' + line)
            sys.exit(-1)

    def coordinate_conversion(self, x):
        if (x == 'T' or x == 'F'):
            return None
        else:
            return float(x)
        
    def get_coordinate_from_line(self, line):
        #numbers = self.get_all_numbers_from_line(line)
        numbers  =self.convert_line_elements(line, self.coordinate_conversion)
        position = Vector(numbers[0], numbers[1], numbers[2])
        #if (len(numbers) == 3) is True:
        return self.a1.scale(position.x).add(self.a2.scale(position.y)).add(self.a3.scale(position.z))


#        return Vector(numbers[0], numbers[1], numbers[2])
        """
        else:
            print('Error in coordinates' + str( len(numbers) )  +  ' of argument in one line' )
            sys.exit(-1)
    """
    def error_message(exception: Exception):
        print(Exception)