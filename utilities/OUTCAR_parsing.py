import numpy as np
import re
import sys
import collections
import math
import collections
import itertools

import utilities.VASP as VASP

class OUTCAR_data_parser:

    def find_last_positions(self)->str:
        pattern = re.compile(r".*POSITION.*\n \-*[\n|\s|\+|\-|.|\[0-9\]]*[\-]*", re.IGNORECASE)
        return pattern.findall(self.outcar_txt)[-1]

    def find_lattice_energy(self)->float:
        row_str = self.find_row_starts_with('free  energy   TOTEN  =')
        row_str = row_str.replace('free  energy   TOTEN  =','')
        row_str = row_str.replace('eV', '')
        return  [float(el) for el in row_str.split()][-1]

    def find_positions(self):
        pos_strs = self.find_last_positions()
        rows_strs = pos_strs.split('\n')
        rows_strs.pop(0)
        rows_strs.pop(0)
        rows_strs.pop(-1)
        rows_strs.pop(-1)
        return [ self.get_floats_from_str(row_str)[0:3] for row_str in rows_strs ]
        

    def get_floats_from_str(self, txt:str):
        return [ float(el) for el in txt.split() ]  

    def find_floating_point_numbers(self):
        pattern = re.compile(r"([+-]?(?=\.\d|\d)(?:\d+)?(?:\.?\d*))(?:[Ee]([+-]?\d+))?", re.IGNORECASE)
        return pattern.findall(self.outcar_txt)

    def find_row_starts_with(self, row_start)->str:
        pattern = re.compile(r""+row_start+".*", re.IGNORECASE)
        return pattern.findall(self.outcar_txt)[-1]

    def find_floats_in_row(self, row_start):
        row_str = self.find_row_starts_with(row_start)
        row_str = row_str.replace(row_start, '')
        return [float(el) for el in row_str.split()]

    def find_ints_in_row(self, row_start):
        row_str = self.find_row_starts_with(row_start)
        row_str = row_str.replace(row_start, '')
        return [int(el) for el in row_str.split()]


    def find_masses(self):
        return self.find_floats_in_row('POMASS =')

    def find_ion_nums(self):
        return self.find_ints_in_row('ions per type = ')



    def find_last_pomass_row(self):
        pattern = re.compile(r"POMASS.*", re.IGNORECASE)
        return pattern.findall(self.outcar_txt)[-1]

    def find_ions_per_type(self)->str:
        pattern = re.compile(r"ions per type.*", re.IGNORECASE)
        return pattern.findall(self.outcar_txt)[-1]


    def __init__(self, filename):
        file = open(filename, "r")
        self.outcar_txt = file.read()
        file.close()
        self.coordinates = self.find_positions()
        self.ion_nums = self.find_ion_nums()
        self.masses = self.find_masses()
        self.energy = self.find_lattice_energy()

        self.lattice = VASP.Lattice(self.energy)

        ion_type_boundies = [0] + list(itertools.accumulate(self.ion_nums,lambda x,y: x+y))


        self.ions = []

        for mass,  i in  zip(self.masses, range(0,len(ion_type_boundies)-1)):
            coordinate_vectors = [ VASP.Vector(*s) for s in self.coordinates[ ion_type_boundies[ i]:ion_type_boundies[i+1]]  ]

            self.lattice.ions_arr.append(VASP.Ions(name = None,vecs = coordinate_vectors, m=mass))
        
        print('fin')



    def get_masses(self):
        pass