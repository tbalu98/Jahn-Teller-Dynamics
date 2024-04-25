import numpy as np
import collections
from numpy import linalg as LA
import math
import utilities.VASP as VASP
import collections
import copy
from scipy.sparse.linalg import eigs
import itertools
import utilities.maths as maths
import utilities.matrix_formalism as mf
import pandas as pd
import utilities.xml_parser
import utilities.JT_config_file_parsing as jt_parser
Eigen_state_2D = collections.namedtuple('Eigen_state',  'x_fonon y_fonon' )

Jahn_Teller_Pars = collections.namedtuple('Jahn_Teller_Pars',  'E_JT E_b hwpG hwmG hw F G ' )


class Jahn_Teller_Theory:


     def build_jt_theory_from_csv(csv_filenames:list):


          par_file_name =  csv_filenames[0] 
          symm_lattice_coords_filename =  csv_filenames[1] 
          JT_lattice_coords_filename = csv_filenames[2] 
          barrier_lattice_coords_filename =  csv_filenames[3] if len(csv_filenames)==4 else None

          at_parser = jt_parser.Atom_config_parser(par_file_name)

  


          atom_1_name, atom_2_name = at_parser.get_names()

          basis_vec_1, basis_vec_2, basis_vec_3 =at_parser.get_basis_vectors()

          basis_vecs = [basis_vec_1, basis_vec_2,basis_vec_3]
    
          mass_1, mass_2 = at_parser.get_masses()

          mass_dict = {  atom_1_name: float( mass_1 ) , atom_2_name: float( mass_2)}

          sym_lattice_energy = float(at_parser.get_lattice_energy('symm_lattice_energy'))
          less_symm_lattice_1_energy = float(at_parser.get_lattice_energy('JT_lattice_energy'))
          if barrier_lattice_coords_filename!=None:
               less_symm_lattice_2_energy = float(at_parser.get_lattice_energy('barrier_lattice_energy'))
               less_symm_lattice_2 = VASP.Lattice().read_from_coordinates_dataframe(barrier_lattice_coords_filename, mass_dict, basis_vecs,less_symm_lattice_2_energy)
          else:
               less_symm_lattice_2 = None




          symm_lattice = VASP.Lattice().read_from_coordinates_dataframe(symm_lattice_coords_filename, mass_dict,basis_vecs,sym_lattice_energy)
    
          less_symm_lattice_1 = VASP.Lattice().read_from_coordinates_dataframe(JT_lattice_coords_filename, mass_dict, basis_vecs, less_symm_lattice_1_energy)
    

          return Jahn_Teller_Theory(symm_lattice,less_symm_lattice_1,less_symm_lattice_2), symm_lattice, less_symm_lattice_1, less_symm_lattice_2


     def build_jt_theory_from_vasprunxmls(filenames):
          symm_lattice = utilities.xml_parser.xml_parser(filenames[0]).lattice
          less_symm_lattice_1 = utilities.xml_parser.xml_parser(filenames[1]).lattice
          less_symm_lattice_2 = utilities.xml_parser.xml_parser(filenames[2]).lattice if len(filenames)==3 else None
          JT_theory = Jahn_Teller_Theory(symm_lattice, less_symm_lattice_1, less_symm_lattice_2)

          return JT_theory, symm_lattice, less_symm_lattice_1, less_symm_lattice_2

     def from_df(self, theory_data:pd.DataFrame):
          case_index = 0
          print('in_from_df')
          self.E_JT = theory_data['JT energy'][case_index]
          self.E_b = theory_data['barrier energy'][case_index]
          self.delta = self.E_b
          
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


          self.hw_mG = float(c*( 2*(-abs( self.E_b/1000 ) + abs(self.E_JT/1000) ) / self.JT_dist**2  )**0.5)

          self.hw_pG = float(c*( 2*(abs( self.E_JT/1000 ) ) / self.barrier_dist**2 )**0.5)
          self.hw = (self.hw_mG + self.hw_pG)/2
          self.hw = float(self.hw)
          self.calc_Taylor_coeffs()

          return self
          #dopant_atom_mass = theor

     def from_Taylor_coeffs(self, hw, F,G = None):
          self.F = F
          self.G = G
          self.hw = hw
          self.order_flag = 3
          return self

     def from_parameters(self, E_JT:float, delta:float,energy_quantum:float):
          self.E_JT = E_JT
          self.delta = delta
          self.hw = energy_quantum
          self.calc_Taylor_coeffs()
          return self

     def __init__(self, symm_lattice: VASP.Lattice=None, less_symm_lattice_1: VASP.Lattice=None, less_symm_lattice_2:VASP.Lattice=None):
          self.symm_lattice = symm_lattice
          
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
          
          self.hw = float(c*( 2*( abs(self.E_JT/1000) ) / self.JT_dist**2  )**0.5)
          
          self.F =  float((( 2*self.E_JT*self.hw )**0.5))
          self.G =  0.0

     def __repr__(self) -> str:
          if self.order_flag == 1:
               return 'Jahn-Teller energy: ' + str(self.E_JT) + '\n' +  'hw: '+ str(self.hw) 
          
          elif self.order_flag == 2:
               res_str = 'Jahn-Teller energy: ' + str(round(self.E_JT,4))+' meV' +'\n' + 'Barrier energy: '  + str(round(self.delta,4))+ ' meV' + '\n' + 'hw+G: ' + str(round(self.hw_pG,4)) +' meV' + '\n' + 'hw-G: ' + str(round(self.hw_mG,4)) + ' meV' + '\n' + 'hw = '+ str(round(self.hw,4)) + ' meV'+'\n' +  'Taylor coefficients:' + '\n'+ 'F = ' + str(round(self.F,4)) + ' meV'+'\n' +'G = ' + str(round(self.G,4)) + ' meV'
               return res_str
          
          elif self.order_flag == 3:
               res_str = 'hw = '+ str(round(self.hw,4)) + ' meV'+'\n' +  'Taylor coefficients:' + '\n'+ 'F = ' + str(round(self.F,4)) + ' meV'+'\n' +'G = ' + str(round(self.G,4)) + ' meV'
               return res_str
     def calc_dists(self):
          self.JT_dist = self.symm_lattice.calc_dist(self.JT_lattice)
          self.barrier_dist = self.symm_lattice.calc_dist(self.barrier_lattice)

     def calc_E_JT(self):
          self.E_JT = abs(self.JT_lattice.energy - self.symm_lattice.energy)*1000
     def calc_delta(self):
          self.delta = abs( self.JT_lattice.energy - self.barrier_lattice.energy)*1000


     def calc_paramters_until_second_order(self):
          self.calc_dists()
          c = 64.654148236
          self.calc_E_JT()
          self.calc_delta()
          #self.delta = self.E_JT - self.E_b
          #self.delta = self.E_b

          self.hw_mG = float(c*( 2*(-abs( self.delta/1000 ) + abs(self.E_JT/1000) ) / self.JT_dist**2  )**0.5)

          self.hw_pG = float(c*( 2*(abs( self.E_JT/1000 ) ) / self.barrier_dist**2 )**0.5)
          self.hw = (self.hw_mG + self.hw_pG)/2
          self.hw = float(self.hw)
          self.calc_Taylor_coeffs()
          self.repr_JT_pars()

     def repr_JT_pars(self):
          self.JT_pars= Jahn_Teller_Pars(self.E_JT, self.delta, self.hw_pG, self.hw_mG, self.hw, self.F, self.G)

     def set_quantum(self, hw):
          self.hw = hw
          self.calc_Taylor_coeffs()
          self.repr_JT_pars()
     
     def calc_Taylor_coeffs(self):
          self.F =  float((( 2*self.E_JT*self.hw*(1-self.delta/(2*self.E_JT-self.delta)) )**0.5))#/(2**0.5)
          self.G = float(self.hw*self.delta/(4*self.E_JT - 2*self.delta))
