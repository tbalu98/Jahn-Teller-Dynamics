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


Eigen_state_2D = collections.namedtuple('Eigen_state',  'x_fonon y_fonon' )

Jahn_Teller_Pars = collections.namedtuple('Jahn_Teller_Pars',  'E_JT E_b hwpG hwmG hw F G ' )


class Jahn_Teller_Theory:


     def __init__(self, symm_lattice: VASP.Lattice, less_symm_lattice_1: VASP.Lattice, less_symm_lattice_2:VASP.Lattice):
          self.symm_lattice = symm_lattice
          self.JT_lattice = less_symm_lattice_1 if less_symm_lattice_1.energy< less_symm_lattice_2.energy else less_symm_lattice_2
          self.barrier_lattice = less_symm_lattice_1 if less_symm_lattice_1.energy> less_symm_lattice_2.energy else less_symm_lattice_2
          self.calc_paramters()



     def __repr__(self) -> str:
          return 'Jahn-Teller energy: ' + str(self.E_JT) + '\n' + 'Barrier energy: '  + str(self.E_b) + '\n' + 'hw+G: ' + str(self.hw_pG) + '\n' + 'hw-G: ' + str(self.hw_mG) + '\n' + 'hw: '+ str(self.hw) 

     def calc_dists(self):
          self.JT_dist = self.symm_lattice.calc_dist(self.JT_lattice)
          self.barrier_dist = self.symm_lattice.calc_dist(self.barrier_lattice)

     def calc_E_JT(self):
          self.E_JT = abs(self.JT_lattice.energy - self.symm_lattice.energy)*1000
     def calc_E_b(self):
          self.E_b = abs( self.JT_lattice.energy - self.barrier_lattice.energy)*1000


     def calc_paramters(self):
          self.calc_dists()
          c = 64.654148236
          self.calc_E_JT()
          self.calc_E_b()
          self.delta = self.E_JT - self.E_b

          self.hw_mG = c*( 2*(-abs( self.E_b/1000 ) + abs(self.E_JT/1000) ) / self.JT_dist**2  )**0.5

          self.hw_pG = c*( 2*(abs( self.E_JT/1000 ) ) / self.barrier_dist**2 )**0.5
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