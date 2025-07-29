import numpy as np
import re
import sys
import collections
import math
import collections
import copy
import pandas as pd
from io import StringIO
from typing import List
import jahn_teller_dynamics.math.maths as maths
import jahn_teller_dynamics.io.JT_config_file_parsing as cfg_parsing

def list_of_nums_to_str(coeffs_list:list):
    
    res_str = ''

    for coeff in coeffs_list[0:-1]:
        res_str+=str(coeff)+', '

    res_str+= str(coeffs_list[-1])
    return res_str


class Vector:

    def tolist(self):
        return [ self.x, self.y, self.z ]

    def __init__(self, x, y, z ):
        self.x = x
        self.y= y
        self.z = z
    def from_str(txt:str):
        nums = [  float(t) for t in txt.split() ]
        return Vector(*nums[0:3])

    def add(self, other):
        return Vector(self.x+other.x, self.y+other.y, self.z+other.z)
    def norm(self  ):
        return self.x**2 + self.y**2 + self.z**2

    def length(self ):
        return math.sqrt(self.norm())
    def __repr__(self) -> str:
        return str(self.x)  +', '  + str(self.y) + ', '  +str(self.z)
    
    def subtract(self, vec2):
        return Vector(self.x - vec2.x, self.y - vec2.y, self.z - vec2.z)
    
    def scale(self, c):
        return Vector(self.x * c, self.y * c, self.z * c)

    def dist(self, vec2):
        return self.subtract(vec2).push_to_unit_cell().length()
    
    def get_dists(self, vecs: list):
        return [self.dist(x) for x  in vecs]

    def get_min_dist_vec(self, vecs: list):
        dxs = self.get_dists(vecs)
        min_dist =  np.min(dxs)
        index = np.where(dxs ==np.min(dxs) )[0][0]
        return self.subtract(vecs[index]).push_to_unit_cell(), min_dist
    
    def push_to_unit_cell(self):
        
        new_x = copy.deepcopy(self.x)
        new_y = copy.deepcopy(self.y)
        new_z = copy.deepcopy(self.z)

        if new_x<-0.5:
            new_x +=1.0
        elif new_x>0.5:
            new_x -= 1.0

        if new_y<-0.5:
            new_y +=1.0
        elif new_y>0.5:
            new_y -= 1.0

        if new_z<-0.5:
            new_z +=1.0
        elif new_z>0.5:
            new_z -= 1.0


        return Vector(new_x,new_y,new_z)

    def to_natural_bases(self, basis_vecs:list):
        
        x_vec:Vector = basis_vecs[0].scale(self.x)
        y_vec:Vector = basis_vecs[1].scale(self.y)
        z_vec:Vector = basis_vecs[2].scale(self.z)

        return x_vec.add(y_vec).add(z_vec)



    def push_to_cell(self,cell_x:float, cell_y:float, cell_z:float):

        new_x = copy.deepcopy(self.x)
        new_y = copy.deepcopy(self.y)
        new_z = copy.deepcopy(self.z)

        if new_x<0:
            new_x +=cell_x
        elif new_x>cell_x:
            new_x -= cell_x

        if new_y<0:
            new_y +=cell_y
        elif new_y>cell_y:
            new_y -= cell_y

        if new_z<0:
            new_z +=cell_z
        elif new_z>cell_z:
            new_z -= cell_z

        return Vector(new_x,new_y,new_z)            

class Ions:

    def __init__(self, name,  vecs, m=None, cell_x = None, cell_y = None, cell_z = None, basis_vecs = None):
        self.name = name
        self._vecs = vecs
        self.m = m

        self.basis_vecs = basis_vecs

    def __getitem__(self,i):
        return self._vecs[i]
    def __len__(self):
        return len(self._vecs)
    
    def to_dataframe(self):

        res_dict = {}
        res_dict['x coordinate'] = [ vec.to_natural_bases(self.basis_vecs).x for vec in self._vecs ]
        res_dict['y coordinate'] = [ vec.to_natural_bases(self.basis_vecs).y for vec in self._vecs ]
        res_dict['z coordinate'] = [ vec.to_natural_bases(self.basis_vecs).z for vec in self._vecs ]

        res_df =  pd.DataFrame(res_dict)
        res_df.index.name = 'index'

        return res_df



    def to_dataframe_string(self):

        df = self.to_dataframe()
        
        string_buffer = StringIO()
        
        df.to_csv(string_buffer, sep=';', index=True)
        csv_string = string_buffer.getvalue()

        return csv_string


    def calc_dist_sq(self, ions2: list):
        dist_sq = 0.0
        for i,vec in zip(range(0,len(self._vecs)),self._vecs):
            dist = vec.get_min_dist_vec(ions2)[1]

            unit_cell_dist_vec:Vector = vec.get_min_dist_vec(ions2)[0]

            dist_vec = unit_cell_dist_vec.to_natural_bases(self.basis_vecs)

            dist = dist_vec.length()

            dist_sq = dist_sq + (dist)**2
        return dist_sq
    
    def __lt__(self, other):
        return len(self)<= len(other)

class Lattice:


    def create_atom_structure_parameters(self):

        atom_pars_dict = self.create_atom_pars_dict()
        
        atom_pars_dict[cfg_parsing.basis_vector_1_opt] = str(self.basis_vecs[0])
        atom_pars_dict[cfg_parsing.basis_vector_1_opt] = str(self.basis_vecs[1])
        atom_pars_dict[cfg_parsing.basis_vector_3_opt] = str(self.basis_vecs[2])
        
        return atom_pars_dict

    def create_atom_pars_dict(self):
        res_dict = {}

        masses = [  ion.m for ion in self.ions_arr ]
        names = [ion.name for ion in self.ions_arr]
        numbers = [len(ion) for ion in self.ions_arr]



        res_dict[cfg_parsing.mass_of_atoms_op] = list_of_nums_to_str(masses)
        res_dict[cfg_parsing.names_of_atoms_op] = list_of_nums_to_str(names)
        res_dict[cfg_parsing.num_of_atoms_op] = list_of_nums_to_str(numbers)

        res_dict[cfg_parsing.basis_vector_1_opt] = str(self.basis_vecs[0])
        res_dict[cfg_parsing.basis_vector_2_opt] = str(self.basis_vecs[1])
        res_dict[cfg_parsing.basis_vector_3_opt] = str(self.basis_vecs[2])

        return res_dict


    def get_normalized_basis_vecs(self):
        res_vecs = [vec.normalize() for vec in self.basis_vecs]
        return res_vecs

    def minimize_lattice(self):
        vacancy_ion = self.ions_arr[0]
        vacancy_ion_vec = vacancy_ion[0]
        host_material = self.ions_arr[1]

        host_vecs = sorted(host_material, key= lambda x: x.dist(vacancy_ion_vec) )[0:7]

        new_latt = Lattice(energy=self.energy, basis_vecs=self.basis_vecs)

        new_latt.ions_arr.append(self.ions_arr[0])
        new_latt.ions_arr.append( Ions( host_material.name, host_vecs, host_material.m, basis_vecs=host_material.basis_vecs ))

        return new_latt
    def __init__(self, energy=None ,basis_vecs = None):
        self.basis_vecs = basis_vecs
        self.ions_arr: List[Ions] = []
        self.energy = energy
   
    def push_vec_back_to_cell(self, vec:Vector):
        return vec.push_to_cell(self.cell_x, self.cell_y, self.cell_z)

    def read_from_file(self, filename):
        f = open(filename, "r")
        print(f.read())

    def to_dataframe_string(self):
        
        df_str =""
        for ion_arr in self.ions_arr:
            df_str+=ion_arr.to_dataframe_string()
        return df_str

    def to_coordinates_data_frame(self):
        x_coordinates = []
        y_coordinates = []
        z_coordinates = []
        
        for ion_arr in self.ions_arr:
            x_coordinates+=[ vec.x for vec in ion_arr ]
            y_coordinates+=[ vec.y for vec in ion_arr ]
            z_coordinates+=[ vec.z for vec in ion_arr ]
        
        res_dict = {}
        res_dict['x_coordinates'] = x_coordinates
        res_dict['y_coordinates'] = y_coordinates
        res_dict['z_coordinates'] = z_coordinates

        res_df = pd.DataFrame(res_dict)
        res_df.index.name = 'index'
        return res_df

    def save_to_coordinates_dataframe(self):
        res_df = self.to_coordinates_data_frame()
        res_df.to_csv(self)

    def read_from_coordinates_dataframe(self,filename, atom_datas,basis_vecs,energy):

        df = pd.read_csv(filename)


        ions_arr = []
        to_index = 0
        from_index = 0
        for atom_data in atom_datas:
            atom_name = atom_data.name
            atom_number = int(atom_data.number)
            atom_mass = float(atom_data.mass)
            from_index = to_index
            to_index +=atom_number

            coord_vecs = []

            for ind in df.index[from_index:to_index]:
                
                x = float(df['x_coordinates'][ind])
                y = float(df['y_coordinates'][ind])
                z = float(df['z_coordinates'][ind])


                coord_vecs.append(Vector(x,y,z))
            
            ions_arr.append(Ions(name = atom_name, vecs= coord_vecs, m = atom_mass, basis_vecs=basis_vecs))


        lattice = Lattice(energy, [ maths.col_vector.from_list(b.tolist()) for b in basis_vecs ] )
        lattice.ions_arr =  ions_arr

        return lattice


    def push_to_cell(self):
        for ion in self.ions_arr:
            for vec in ion._vecs:
                
                if vec.x<0:
                    vec.x+=self.cell_x
                elif vec.x>self.cell_x:
                    vec.x-=self.cell_x
                
                if vec.y < 0:
                    vec.y+=self.cell_y
                elif vec.y>self.cell_y:
                    vec.y-=self.cell_y

                if vec.z < 0:
                    vec.z+=self.cell_z
                elif vec.z>self.cell_z:
                    vec.z-=self.cell_z


    def to_unit_cell(self):
        for ion in self.ions_arr:

            for vec in ion._vecs:

                vec.x/=self.cell_x
                vec.y/=self.cell_y
                vec.z/=self.cell_z
        

    def set_cell_size(self, cell_x, cell_y=None, cell_z = None):
        if cell_y==None and cell_z == None :
            self.cell_x = cell_x
            self.cell_y = cell_x
            self.cell_z = cell_x
            for ion in self.ions_arr:
                ion.cell_x = cell_x
                ion.cell_y = cell_x
                ion.cell_z = cell_x

        else:
            self.cell_x = cell_x
            self.cell_y = cell_y
            self.cell_z = cell_z

            for ion in self.ions_arr:
                ion.cell_x = cell_x
                ion.cell_y = cell_y
                ion.cell_z = cell_z



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
            for i in range(len(self.ions_arr)):
                self_ions = self.ions_arr[i]
                other_ions = other_lattice.ions_arr[i]
                lattice_dist = lattice_dist + self_ions.calc_dist_sq(other_ions)*(self_ions.m)
            return ((lattice_dist)**0.5)#*self.cell_x


"""

class POSCAR_data:


    def __init__(self, path):
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


        names = self.convert_line_elements(name_line, str)
        numbers = self.convert_line_elements(numbers_line, int)
        line_num = 9
        self.lattice = Lattice()
        
        for name, number_of_ion in zip( names, numbers):
            new_line_num = line_num + number_of_ion
            coordinates = [self.get_coordinate_from_line( line ) for line in lines[line_num: line_num+number_of_ion]]
            line_num = new_line_num
            self.lattice.ions_arr.append(Ions(name, coordinates))





    def get_scaling_factors(self, line):
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
        numbers  =self.convert_line_elements(line, self.coordinate_conversion)
        position = Vector(numbers[0], numbers[1], numbers[2])
        return self.a1.scale(position.x).add(self.a2.scale(position.y)).add(self.a3.scale(position.z))

    def error_message(exception: Exception):
        print(Exception)


"""