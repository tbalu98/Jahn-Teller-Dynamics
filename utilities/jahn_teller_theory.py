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

def build_jt_theory_from_csv_2(csv_filenames:list):


     par_file_name =  csv_filenames[0] # sys.argv[3]
     symm_lattice_coords_filename =  csv_filenames[1] # sys.argv[4]
     JT_lattice_coords_filename = csv_filenames[2] #sys.argv[5]
     barrier_lattice_coords_filename =  csv_filenames[3] if len(csv_filenames)==4 else None #'barrier_lattice.csv'    sys.argv[6]

     at_parser = jt_parser.Atom_config_parser(par_file_name)

    #Par dataframe


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


def build_jt_theory_from_csv(csv_filenames:list):


    par_file_name =  csv_filenames[0] # sys.argv[3]
    symm_lattice_coords_filename =  csv_filenames[1] # sys.argv[4]
    JT_lattice_coords_filename = csv_filenames[2] #sys.argv[5]
    barrier_lattice_coords_filename =  csv_filenames[3] if len(csv_filenames)==4 else None #'barrier_lattice.csv'    sys.argv[6]

    #Par dataframe

    par_df = pd.read_csv(par_file_name,sep=';')

    atom_1_name = str(par_df.iloc[0]['atom_1_name'])
    atom_2_name = str(par_df.iloc[0]['atom_2_name'])

    basis_vec_1 = VASP.Vector(float(par_df['basis_vec_1_x']), float(par_df['basis_vec_1_y']), float(par_df['basis_vec_1_z']))
    basis_vec_2 = VASP.Vector(float(par_df['basis_vec_2_x']), float(par_df['basis_vec_2_y']), float(par_df['basis_vec_2_z']))
    basis_vec_3 = VASP.Vector(float(par_df['basis_vec_3_x']), float(par_df['basis_vec_3_y']), float(par_df['basis_vec_3_z']))

    basis_vecs = [basis_vec_1, basis_vec_2,basis_vec_3]
    
    mass_dict = {  atom_1_name: float( par_df.iloc[0]['atom_1_mass'] ) , atom_2_name: float( par_df.iloc[0]['atom_2_mass'])}

    sym_lattice_energy = float(par_df.iloc[0]['symm_lattice_energy'])
    less_symm_lattice_1_energy = float(par_df.iloc[0]['JT_lattice_energy'])
    if barrier_lattice_coords_filename!=None:
          less_symm_lattice_2_energy = float(par_df.iloc[0]['barrier_lattice_energy'])
          less_symm_lattice_2 = VASP.Lattice().read_from_coordinates_dataframe(barrier_lattice_coords_filename, mass_dict, basis_vecs,less_symm_lattice_2_energy)
    else:
          less_symm_lattice_2 = None




    symm_lattice = VASP.Lattice().read_from_coordinates_dataframe(symm_lattice_coords_filename, mass_dict,basis_vecs,sym_lattice_energy)
    
    less_symm_lattice_1 = VASP.Lattice().read_from_coordinates_dataframe(JT_lattice_coords_filename, mass_dict, basis_vecs, less_symm_lattice_1_energy)
    

    return Jahn_Teller_Theory(symm_lattice,less_symm_lattice_1,less_symm_lattice_2), symm_lattice, less_symm_lattice_1, less_symm_lattice_2



class Jahn_Teller_Theory:


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
               res_str = 'Jahn-Teller energy: ' + str(round(self.E_JT,4))+' meV' +'\n' + 'Barrier energy: '  + str(round(self.E_b,4))+ ' meV' + '\n' + 'hw+G: ' + str(round(self.hw_pG,4)) +' meV' + '\n' + 'hw-G: ' + str(round(self.hw_mG,4)) + ' meV' + '\n' + 'hw = '+ str(round(self.hw,4)) + ' meV'+'\n' +  'Taylor coefficients:' + '\n'+ 'F = ' + str(round(self.F,4)) + ' meV'+'\n' +'G = ' + str(round(self.G,4)) + ' meV'
               return res_str
          
          elif self.order_flag == 3:
               res_str = 'hw = '+ str(round(self.hw,4)) + ' meV'+'\n' +  'Taylor coefficients:' + '\n'+ 'F = ' + str(round(self.F,4)) + ' meV'+'\n' +'G = ' + str(round(self.G,4)) + ' meV'
               return res_str
     def calc_dists(self):
          self.JT_dist = self.symm_lattice.calc_dist(self.JT_lattice)
          self.barrier_dist = self.symm_lattice.calc_dist(self.barrier_lattice)

     def calc_E_JT(self):
          self.E_JT = abs(self.JT_lattice.energy - self.symm_lattice.energy)*1000
     def calc_E_b(self):
          self.E_b = abs( self.JT_lattice.energy - self.barrier_lattice.energy)*1000


     def calc_paramters_until_second_order(self):
          self.calc_dists()
          c = 64.654148236
          self.calc_E_JT()
          self.calc_E_b()
          self.delta = self.E_JT - self.E_b
          self.delta = self.E_b

          self.hw_mG = float(c*( 2*(-abs( self.E_b/1000 ) + abs(self.E_JT/1000) ) / self.JT_dist**2  )**0.5)

          self.hw_pG = float(c*( 2*(abs( self.E_JT/1000 ) ) / self.barrier_dist**2 )**0.5)
          self.hw = (self.hw_mG + self.hw_pG)/2
          self.hw = float(self.hw)
          self.calc_Taylor_coeffs()
          self.repr_JT_pars()

     def repr_JT_pars(self):
          self.JT_pars= Jahn_Teller_Pars(self.E_JT, self.E_b, self.hw_pG, self.hw_mG, self.hw, self.F, self.G)

     def set_quantum(self, hw):
          self.hw = hw
          self.calc_Taylor_coeffs()
          self.repr_JT_pars()
     
     def calc_Taylor_coeffs(self):
          self.F =  float((( 2*self.E_JT*self.hw*(1-self.delta/(2*self.E_JT-self.delta)) )**0.5))#/(2**0.5)
          self.G = float(self.hw*self.delta/(4*self.E_JT - 2*self.delta))

"""
class Exe_JT_int:
     
     H_int:mf.MatrixOperator
     def __init__(self,Jahn_Teller_pars: Jahn_Teller_Pars, el_states: mf.symmetric_electron_system, fonon_system: mf.n_dim_harm_osc):
          self.JT_pars = Jahn_Teller_pars
          self.el_states = el_states
          self.fonon_system = fonon_system
          self.create_hamiltonian_op()
     
     def create_hamiltonian_op(self):
          X = self.fonon_system.get_pos_i_op(0)
          Y = self.fonon_system.get_pos_i_op(1)
          XX = self.fonon_system.get_pos_i_i_op(0)
          XY = self.fonon_system.get_pos_i_j_op(0,1)
          YX = self.fonon_system.get_pos_i_j_op(1,0)
          YY = self.fonon_system.get_pos_i_i_op(1)
          K = self.fonon_system.get_ham_op()       
          s0 = self.el_states.symm_ops['s0']
          sz = self.el_states.symm_ops['sz']
          sx = self.el_states.symm_ops['sx']

          self.H_int = self.JT_pars.hw * (K**s0) +self.JT_pars.F*( X**sz + Y**sx ) + self.JT_pars.G * ( (XX-YY)**sz - (XY + YX) ** sx) 
"""
"""
class multi_mode_Exe_jt_int2:
     H_int:mf.MatrixOperator
     def __init__(self,JT_theory: Jahn_Teller_Theory, el_states: mf.symmetric_electron_system, fonon_systems: mf.fast_multimode_fonon_sys):
          self.JT_theory = JT_theory

          self.el_states = el_states
          self.fonon_systems = fonon_systems
          
          JT_pars = JT_theory.JT_pars

          #Hs = [self.create_ham_op_one_mode(mode) for mode in self.fonon_systems.fonon_syss.keys()]
          Hs = [ Exe_JT_int( JT_pars,el_states,fonon_systems.fonon_syss[mode]).H_int for mode in self.fonon_systems.fonon_syss.keys()  ]
          #for mode in fonon_systems.fonon_syss.keys():
          #     Hs.append( self.create_ham_op_one_mode(mode))
          self.H_int = sum(Hs)
          
          
          #self.H_int = MatrixOperator(all_mode_ham)

"""
"""
class multi_mode_Exe_jt_int:
     H_int:mf.MatrixOperator
     def __init__(self,JT_theory: Jahn_Teller_Theory, el_states: mf.symmetric_electron_system, fonon_systems: mf.fast_multimode_fonon_sys):
          self.JT_theory = JT_theory

          self.el_states = el_states
          self.fonon_systems = fonon_systems
          
          Hs = [self.create_ham_op_one_mode(mode) for mode in self.fonon_systems.fonon_syss.keys()]

          #for mode in fonon_systems.fonon_syss.keys():
          #     Hs.append( self.create_ham_op_one_mode(mode))
          self.H_int = sum(Hs)
          
          #self.H_int = MatrixOperator(all_mode_ham)

     
     def create_ham_op_one_mode(self,mode):
          
          fonon_system = self.fonon_systems

     

          X = fonon_system.calc_multi_mode_op(mode, lambda x: x.get_pos_i_op(0) )
          Y = fonon_system.calc_multi_mode_op(mode, lambda x: x.get_pos_i_op(1) )

          #Y = fonon_system.calc_pos_operator(mode, 1)

          XX = fonon_system.calc_multi_mode_op(mode, lambda x: x.get_pos_i_j_op(0,0) )
          YY = fonon_system.calc_multi_mode_op(mode, lambda x: x.get_pos_i_j_op(1,1) )

          XY = fonon_system.calc_multi_mode_op(mode, lambda x: x.get_pos_i_j_op(0,1) )
          YX = fonon_system.calc_multi_mode_op(mode, lambda x: x.get_pos_i_j_op(1,0) )

          


          K = fonon_system.calc_multi_mode_op(mode, lambda x: x.get_ham_op())

          self.JT_theory.set_quantum(mode)
          

          s0 = self.el_states.symm_ops['s0']
          sz = self.el_states.symm_ops['sz']
          sx = self.el_states.symm_ops['sx']

          H_int_mat = self.JT_theory.hw * K** s0 + self.JT_theory.F*(X**sz + Y**sx) + 1.0*self.JT_theory.G* ( (XX-YY) **sz - (XY + YX)**sx)

          #np.savetxt('H_int.csv', H_int_mat)


          #self.H_int = MatrixOperator(H_int_mat)
          return H_int_mat

"""
"""
          
class Tet_JT_int:
     H_int:mf.MatrixOperator
     def __init__(self,F_T, F_E,omega, el_states: mf.symmetric_electron_system, fonon_system: mf.n_dim_harm_osc):
          self.F_T = F_T
          self.F_E = F_E
          self.omega = omega
          self.el_states = el_states
          self.fonon_system = fonon_system
          self.create_hamiltonian_op()

     def create_hamiltonian_op(self):
          K = self.fonon_system.get_ham_op()


          X =  self.fonon_system.get_pos_i_op(0)
          Y =  self.fonon_system.get_pos_i_op(1)
          Z =  self.fonon_system.get_pos_i_op(2)
          V =  self.fonon_system.get_pos_i_op(3)
          W =  self.fonon_system.get_pos_i_op(4)

          O0 = self.el_states.symm_ops['O0']
          Ox = self.el_states.symm_ops['Ox']
          Oy = self.el_states.symm_ops['Oy']
          Oz = self.el_states.symm_ops['Oz']

          Ov = self.el_states.symm_ops['Ov']
          Ow = self.el_states.symm_ops['Ow']
          omega = self.omega
          
          self.H_int = omega* K**  O0- self.F_T* ( X**Ox  + Y **Oy  + Z** Oz) + self.F_E* (  V **Ov  + W **Ow )  


"""