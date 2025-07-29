import numpy as np
import collections
from numpy import linalg as LA
import math
import jahn_teller_dynamics.io.VASP as V
import collections
import copy
from scipy.sparse.linalg import eigs
import itertools
import jahn_teller_dynamics.math.maths as maths
import jahn_teller_dynamics.math.matrix_formalism as mf
import pandas as pd
import jahn_teller_dynamics.io.xml_parser
import jahn_teller_dynamics.io.JT_config_file_parsing as jt_parser
from collections import namedtuple



class Jahn_Teller_Theory:

     symm_lattice:V.Lattice
     JT_lattice:V.Lattice
     barrier_lattice:V.Lattice


     def from_JT_pars(self, E_JT, E_b, hw):
          
          jt_pars = Jahn_Teller_Theory()
          jt_pars.E_JT_meV = E_JT
          jt_pars.E_b = E_b
          jt_pars.hw_meV = hw
          
          return jt_pars

     def build_jt_theory_from_csv(csv_filenames:list):


          par_file_name =  csv_filenames[0] 
          symm_lattice_coords_filename =  csv_filenames[1] 
          JT_lattice_coords_filename = csv_filenames[2] 
          barrier_lattice_coords_filename =  csv_filenames[3] if len(csv_filenames)==4 else None

          at_parser = jt_parser.Atom_config_parser(par_file_name)

  
          basis_vec_1, basis_vec_2, basis_vec_3 =at_parser.get_basis_vectors()

          basis_vecs = [basis_vec_1, basis_vec_2,basis_vec_3]



          atom_data = namedtuple('atom_data', 'name mass number')

          atom_names = at_parser.get_names()
          atom_masses = at_parser.get_masses()
          atom_numbers = at_parser.get_numbers()

          
          atom_datas = []

          for atom_name,atom_mass,atom_number in zip(atom_names, atom_masses, atom_numbers):

               atom_datas.append( atom_data(atom_name, atom_mass, atom_number))

          sym_lattice_energy = float(at_parser.get_lattice_energy('symm_lattice_energy'))
          less_symm_lattice_1_energy = float(at_parser.get_lattice_energy('JT_lattice_energy'))
          if barrier_lattice_coords_filename!=None:
               less_symm_lattice_2_energy = float(at_parser.get_lattice_energy('barrier_lattice_energy'))
               less_symm_lattice_2 = V.Lattice().read_from_coordinates_dataframe(barrier_lattice_coords_filename, atom_datas, basis_vecs,less_symm_lattice_2_energy)
          else:
               less_symm_lattice_2 = None




          symm_lattice = V.Lattice().read_from_coordinates_dataframe(symm_lattice_coords_filename, atom_datas,basis_vecs,sym_lattice_energy)
    
          less_symm_lattice_1 = V.Lattice().read_from_coordinates_dataframe(JT_lattice_coords_filename, atom_datas, basis_vecs, less_symm_lattice_1_energy)
    

          return Jahn_Teller_Theory(symm_lattice,less_symm_lattice_1,less_symm_lattice_2), symm_lattice, less_symm_lattice_1, less_symm_lattice_2

     def build_jt_theory_from_csv_and_pars(csv_filenames:list, basis_vecs:list[V.Vector], atom_names:list[str], atom_masses:list[float], number_of_atoms:list[int]):
          

          par_file_name =  csv_filenames[0] 
          symm_lattice_coords_filename =  csv_filenames[1] 
          JT_lattice_coords_filename = csv_filenames[2] 
          barrier_lattice_coords_filename =  csv_filenames[3] if len(csv_filenames)==4 else None

          at_parser = jt_parser.Atom_config_parser(par_file_name)

  
          basis_vec_1, basis_vec_2, basis_vec_3 =at_parser.get_basis_vectors()

          basis_vecs = [basis_vec_1, basis_vec_2,basis_vec_3]


          atom_data = namedtuple('atom_data', 'name mass number')

          atom_names = at_parser.get_names()
          atom_masses = at_parser.get_masses()
          atom_numbers = at_parser.get_numbers()

          
          atom_datas = []

          for atom_name,atom_mass,atom_number in zip(atom_names, atom_masses, atom_numbers):

               atom_datas.append( atom_data(atom_name, atom_mass, atom_number))

          sym_lattice_energy = float(at_parser.get_lattice_energy('symm_lattice_energy'))
          less_symm_lattice_1_energy = float(at_parser.get_lattice_energy('JT_lattice_energy'))
          if barrier_lattice_coords_filename!=None:
               less_symm_lattice_2_energy = float(at_parser.get_lattice_energy('barrier_lattice_energy'))
               less_symm_lattice_2 = V.Lattice().read_from_coordinates_dataframe(barrier_lattice_coords_filename, atom_datas, basis_vecs,less_symm_lattice_2_energy)
          else:
               less_symm_lattice_2 = None




          symm_lattice = V.Lattice().read_from_coordinates_dataframe(symm_lattice_coords_filename, atom_datas,basis_vecs,sym_lattice_energy)
    
          less_symm_lattice_1 = V.Lattice().read_from_coordinates_dataframe(JT_lattice_coords_filename, atom_datas, basis_vecs, less_symm_lattice_1_energy)
    

          return Jahn_Teller_Theory(symm_lattice,less_symm_lattice_1,less_symm_lattice_2), symm_lattice, less_symm_lattice_1, less_symm_lattice_2

     def build_jt_theory_from_vasprunxmls(filenames):

          symm_lattice = utilities.xml_parser.xml_parser( filenames[0]).lattice
          less_symm_lattice_1 = utilities.xml_parser.xml_parser(filenames[1]).lattice
          less_symm_lattice_2 = utilities.xml_parser.xml_parser(filenames[2]).lattice if len(filenames)==3 else None
          JT_theory = Jahn_Teller_Theory(symm_lattice, less_symm_lattice_1, less_symm_lattice_2)
          
          return JT_theory, symm_lattice, less_symm_lattice_1, less_symm_lattice_2

     def from_df(self, theory_data:pd.DataFrame):
          case_index = 0

          self.E_JT_meV = theory_data['JT energy'][case_index]
          self.E_b = theory_data['barrier energy'][case_index]
          self.delta_meV = self.E_b
          
          dop_atom_name = theory_data['dopant atom'][case_index]
          maj_atom_name = theory_data['majority atom'][case_index]

          dop_atom_mass = theory_data[ dop_atom_name+' mass'][case_index]

          maj_atom_mass = theory_data[ maj_atom_name+' mass'][case_index]


          symm_JT_dist_maj = theory_data['symm-JT distance ' + maj_atom_name + ' atoms'][case_index]
          symm_JT_dist_dop = theory_data['symm-JT distance ' +dop_atom_name + ' atoms'][case_index]

          self.JT_dist = ((symm_JT_dist_dop**2)*dop_atom_mass + (symm_JT_dist_maj**2)*maj_atom_mass)**0.5

          symm_barrier_dist_maj = theory_data['symm-barrier distance ' + maj_atom_name + ' atoms'][case_index]
          symm_barrier_dist_dop = theory_data['symm-barrier distance ' +dop_atom_name + ' atoms'][case_index]

          self.barrier_dist = ((symm_barrier_dist_dop**2)*dop_atom_mass + (symm_barrier_dist_maj**2)*maj_atom_mass)**0.5
          
          self.order_flag = 2
          c = 64.654148236


          self.hw_mG = float(c*( 2*(-abs( self.E_b/1000 ) + abs(self.E_JT_meV/1000) ) / self.JT_dist**2  )**0.5)

          self.hw_pG = float(c*( 2*(abs( self.E_JT_meV/1000 ) ) / self.barrier_dist**2 )**0.5)
          self.hw_meV = (self.hw_mG + self.hw_pG)/2
          self.hw_meV = float(self.hw_meV)
          self.calc_Taylor_coeffs()

          return self

     def from_Taylor_coeffs(self, hw, F,G = None):
          self.F = F
          self.G = G
          self.hw_meV = hw
          self.order_flag = 3
          return self

     def from_parameters(self, E_JT:float, delta:float,energy_quantum:float):
          self.E_JT_meV = E_JT
          self.delta_meV = delta
          self.hw_meV = energy_quantum
          self.calc_Taylor_coeffs()
          return self

     def __init__(self, symm_lattice: V.Lattice=None, less_symm_lattice_1: V.Lattice=None, less_symm_lattice_2:V.Lattice=None):
          self.symm_lattice = symm_lattice
          self.JT_lattice = None
          self.barrier_lattice = None
          self.intrinsic_soc:float
          self.orbital_red_factor:float
          if less_symm_lattice_1!=None and less_symm_lattice_2!=None:
               self.JT_lattice = less_symm_lattice_1 if less_symm_lattice_1.energy< less_symm_lattice_2.energy else less_symm_lattice_2
               self.barrier_lattice = less_symm_lattice_1 if less_symm_lattice_1.energy> less_symm_lattice_2.energy else less_symm_lattice_2
               self.calc_paramters_until_second_order()
               self.order_flag = 2

          elif less_symm_lattice_1!=None and less_symm_lattice_2==None:
               self.JT_lattice = less_symm_lattice_1
               self.calc_paramters_until_first_order()
               self.order_flag = 1
          
               
     def calc_paramters_until_first_order(self):
          self.JT_dist = self.symm_lattice.calc_dist(self.JT_lattice)
          self.calc_E_JT()
          self.E_b = 0.0
          c = 64.654148236
          
          self.hw_meV = float(c*( 2*( abs(self.E_JT_meV/1000) ) / self.JT_dist**2  )**0.5)
          self.K = self.hw_meV
          
          self.F =  float((( 2*self.E_JT_meV*self.hw_meV )**0.5))
          self.G =  0.0

     def __repr__(self) -> str:
          if self.order_flag == 1:

               res_str = 'First order Jahn-Teller interaction parameters'
               res_str += '\n\tJahn-Teller energy: ' + str(round(self.E_JT_meV,4))+' meV' 
               
               res_str+= '\n\t' + 'vibration energy quantum = '+ str(round(self.hw_meV,4)) + ' meV'+'\n'

               res_str+='\n\t' +  'Taylor coefficient:'
               res_str+= '\n\t\t'+ 'F = ' + str(round(float(self.F),4)) +'meV'

               return res_str
          
          elif self.order_flag == 2:
               
               res_str = 'Second order dynamic Jahn-Teller interaction parameters: \n'

               res_str += '\n\tJahn-Teller energy: ' + str(round(self.E_JT_meV,4))+' meV' 
               res_str+='\n\t' + 'Barrier energy: '  + str(round(self.delta_meV,4))+ ' meV' 

               res_str+= '\n\t' + 'vibration energy quantum = '+ str(round(self.hw_meV,4)) + ' meV'+'\n'
               res_str+='\n\t' +  'Taylor coefficients:'
               res_str+= '\n\t\t'+ 'F = ' + str(round(float(self.F),4)) + ' meV'
               res_str+='\n\t\t' +'G = ' + str(round(float(self.G),4)) + ' meV'
               #res_str += '\n\t' + 'K = '+ str(round(float(self.K),4)) + ' meV'
               return res_str
          
          elif self.order_flag == 3:
               res_str = 'hw = '+ str(round(self.hw_meV,4)) + ' meV'+'\n' +  'Taylor coefficients:' + '\n'+ 'F = ' + str(round(self.F,4)) + ' meV'+'\n' +'G = ' + str(round(self.G,4)) + ' meV'
               return res_str
     def calc_dists(self):
          self.JT_dist = self.symm_lattice.calc_dist(self.JT_lattice)
          self.barrier_dist = self.symm_lattice.calc_dist(self.barrier_lattice)

     def calc_E_JT(self):
          self.E_JT_meV = abs(self.JT_lattice.energy - self.symm_lattice.energy)*1000
     def calc_delta(self):
          self.delta_meV = abs( self.JT_lattice.energy - self.barrier_lattice.energy)*1000


     def calc_hw(self):
          c = 64.654148236

          self.hw_mG = float(c*( 2*(-abs( self.delta_meV/1000 ) + abs(self.E_JT_meV/1000) ) / self.barrier_dist**2  )**0.5)

          self.hw_pG = float(c*( 2*(abs( self.E_JT_meV/1000 ) ) / self.JT_dist**2 )**0.5)
          self.hw_meV = (self.hw_mG + self.hw_pG)/2
          
          
          self.K = (self.hw_meV/(6.582120e-13))**2
          self.K = self.hw_meV


     def calc_paramters_until_second_order_from_JT_pars(self):

          self.K = self.hw_meV
          
          self.calc_Taylor_coeffs_K()

     def calc_paramters_until_second_order(self):
          self.calc_dists()
          self.calc_E_JT()
          self.calc_delta()


          


          self.calc_hw()


          self.calc_Taylor_coeffs_K()


     def set_quantum(self, hw):
          self.hw_meV = hw
          self.calc_Taylor_coeffs()
          self.repr_JT_pars()
     
     def calc_Taylor_coeffs_hw(self):
          self.F =  float((( 2*self.E_JT_meV*abs(self.hw_meV)*(1-self.delta_meV/(2*self.E_JT_meV-self.delta_meV)) )**0.5))#/(2**0.5)
          self.G = float(abs(self.hw_meV)*self.delta_meV/(4*self.E_JT_meV - 2*self.delta_meV))

     def calc_Taylor_coeffs_K(self):
          self.F =  float((( 2*self.E_JT_meV*abs(self.K)*(1-self.delta_meV/(2*self.E_JT_meV-self.delta_meV)) )**0.5))#/(2**0.5)
          self.G = float(abs(self.K)*self.delta_meV/(4*self.E_JT_meV - 2*self.delta_meV))

