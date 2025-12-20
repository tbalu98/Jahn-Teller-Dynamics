import numpy as np
import collections
from numpy import linalg as LA
import math
import jahn_teller_dynamics.io.file_io.vasp as V
import copy
from scipy.sparse.linalg import eigs
import itertools
import jahn_teller_dynamics.math.maths as maths
import jahn_teller_dynamics.math.matrix_mechanics as mm
import pandas as pd
import jahn_teller_dynamics.io.file_io.xml_parser
# Lazy import to avoid circular dependency - only import when needed
from collections import namedtuple


def dimensionless_to_generalized_coordinate(q: float, energy_quantum: float) -> float:
    """
    Convert dimensionless coordinate to generalized coordinate in units of Å √amu.
    
    Formula: V * q where V = 2.044543799 * 1/sqrt(e)
    
    Args:
        q: Dimensionless coordinate
        energy_quantum: Energy quantum (e) in meV
        
    Returns:
        Generalized coordinate in units of Å √amu
    """
    if energy_quantum <= 0:
        raise ValueError(f"Energy quantum must be positive, got {energy_quantum}")
    
    V = 2.044543799 / math.sqrt(energy_quantum)
    return V * q



class Jahn_Teller_Theory:

     symm_lattice:'V.Lattice'
     JT_lattice:'V.Lattice'
     barrier_lattice:'V.Lattice'


     @staticmethod
     def from_JT_pars(E_JT, E_b, hw):
          """
          Create a Jahn_Teller_Theory instance from JT parameters.
          
          Args:
              E_JT: Jahn-Teller energy
              E_b: Barrier energy
              hw: Vibration energy quantum
              
          Returns:
              Jahn_Teller_Theory: New instance with specified parameters
          """
          jt_pars = Jahn_Teller_Theory()
          jt_pars.E_JT_meV = E_JT
          jt_pars.E_b = E_b
          jt_pars.hw_meV = hw
          
          return jt_pars



     def from_Taylor_coeffs(self, hw, F, G=None):
          self.F = F
          self.G = G
          self.hw_meV = hw
          self.order_flag = 3
          return self

     def from_parameters(self, E_JT:float, delta:float, energy_quantum:float):
          self.E_JT_meV = E_JT
          self.delta_meV = delta
          self.hw_meV = energy_quantum
          # Use calc_Taylor_coeffs_K if K is set, otherwise use calc_Taylor_coeffs_hw
          if hasattr(self, 'K') and self.K is not None:
               self.calc_Taylor_coeffs_K()
          else:
               self.calc_Taylor_coeffs_hw()
          return self

     def from_model_parameters(self, lambda_DFT, KJT, gL, delta_f, Yx, Yy, f_factor):
          #self.p_factor = p_factor
          self.lambda_DFT = lambda_DFT
          self.KJT = KJT
          self.gL = gL
          self.delta_f = delta_f
          self.Yx = Yx
          self.Yy = Yy
          self.f_factor = f_factor
          return self

     def from_minimal_model_parameters(self, lambda_DFT, gL, delta_f, Yx, Yy, f_factor):
          #self.p_factor = p_factor
          self.lambda_DFT = lambda_DFT
          
          self.gL = gL
          self.delta_f = delta_f
          self.Yx = Yx
          self.Yy = Yy
          self.f_factor = f_factor
          self.order_flag = 0
          return self
     
     def from_energy_split_and_factors(self, energy_split: float, orientation_basis: list[maths.col_vector], gL: float, delta_f: float, f_factor: float, Yx: float = 0.0, Yy: float = 0.0):
          """
          Build Jahn-Teller theory from energy split, orientation basis, and reduction factors.
          
          Args:
              energy_split: Energy splitting (typically spin-orbit coupling splitting) in meV
              orientation_basis: List of column vectors defining the system orientation
              gL: Orbital reduction factor (gL)
              delta_f: Delta f factor
              f_factor: f factor
              Yx: Yx parameter (default: 0.0)
              Yy: Yy parameter (default: 0.0)
              
          Returns:
              self: The configured Jahn_Teller_Theory instance
          """
          self.lambda_DFT = energy_split
          self.orientation_basis = orientation_basis
          self.gL = gL
          self.delta_f = delta_f
          self.f_factor = f_factor
          self.Yx = Yx
          self.Yy = Yy
          self.order_flag = 0  # Model Hamiltonian parameters
          return self
     

     def __init__(self, symm_lattice: V.Lattice=None, less_symm_lattice_1: V.Lattice=None, less_symm_lattice_2:V.Lattice=None):
          self.symm_lattice = symm_lattice
          self.JT_lattice = None
          self.barrier_lattice = None
          self.intrinsic_soc: float = None
          self.orbital_red_factor: float = None
          
          if less_symm_lattice_1 is not None and less_symm_lattice_2 is not None:
               if less_symm_lattice_1.energy < less_symm_lattice_2.energy:
                    self.JT_lattice = less_symm_lattice_1
                    self.barrier_lattice = less_symm_lattice_2
               else:
                    self.JT_lattice = less_symm_lattice_2
                    self.barrier_lattice = less_symm_lattice_1
               self.calc_parameters_until_second_order()
               self.order_flag = 2

          elif less_symm_lattice_1 is not None and less_symm_lattice_2 is None:
               self.JT_lattice = less_symm_lattice_1
               self.calc_parameters_until_first_order()
               self.order_flag = 1
          
               
     def calc_parameters_until_first_order(self):
          if self.symm_lattice is None or self.JT_lattice is None:
               raise ValueError("symm_lattice and JT_lattice must be set before calculating parameters")
          self.JT_dist = self.symm_lattice.calc_dist(self.JT_lattice)
          self.calc_E_JT()
          self.E_b = 0.0
          c = 64.654148236
          
          self.hw_meV = float(c*( 2*( abs(self.E_JT_meV/1000) ) / self.JT_dist**2  )**0.5)
          self.K = self.hw_meV
          
          self.F = float((( 2*self.E_JT_meV*self.hw_meV )**0.5))
          self.G = 0.0

     def __repr__(self) -> str:
          if self.order_flag == 0:
               res_str = 'Model Hamiltonian parameters'
               #res_str += '\n\tHam reduction factor = ' + str(round(self.p_factor,4)) if self.p_factor != None else ''
               res_str += '\n\tEnergy splitting = ' + str(abs(round(self.lambda_DFT,4))) + ' meV'
               res_str += ('\n\tOrbital reduction factor = ' + str(round(self.gL,4))) if self.gL != 0.0 and self.gL is not None else ''
               res_str += '\n\tDelta f factor = ' + str(round(self.delta_f,4)) + ' meV'
               res_str += '\n\tYx = ' + str(round(self.Yx,4)) if self.Yx is not None else ''
               res_str += '\n\tYy = ' + str(round(self.Yy,4)) if self.Yy is not None else ''
               res_str += '\n\tf factor = ' + str(round(self.f_factor,4))
               res_str += '\n\t'
               return res_str
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
          if self.symm_lattice is None or self.JT_lattice is None:
               raise ValueError("symm_lattice and JT_lattice must be set")
          self.JT_dist = self.symm_lattice.calc_dist(self.JT_lattice)
          if self.barrier_lattice is None:
               raise ValueError("barrier_lattice must be set for second order calculations")
          self.barrier_dist = self.symm_lattice.calc_dist(self.barrier_lattice)

     def calc_E_JT(self):
          if self.JT_lattice is None or self.symm_lattice is None:
               raise ValueError("JT_lattice and symm_lattice must be set")
          self.E_JT_meV = abs(self.JT_lattice.energy - self.symm_lattice.energy)*1000
     
     def calc_delta(self):
          if self.JT_lattice is None or self.barrier_lattice is None:
               raise ValueError("JT_lattice and barrier_lattice must be set")
          self.delta_meV = abs( self.JT_lattice.energy - self.barrier_lattice.energy)*1000


     def calc_hw(self):
          if not hasattr(self, 'barrier_dist') or not hasattr(self, 'JT_dist'):
               raise ValueError("barrier_dist and JT_dist must be calculated first (call calc_dists())")
          c = 64.654148236

          self.hw_mG = float(c*( 2*(-abs( self.delta_meV/1000 ) + abs(self.E_JT_meV/1000) ) / self.barrier_dist**2  )**0.5)

          self.hw_pG = float(c*( 2*(abs( self.E_JT_meV/1000 ) ) / self.JT_dist**2 )**0.5)
          self.hw_meV = (self.hw_mG + self.hw_pG)/2
          
          
          #self.K = (self.hw_meV/(6.582120e-13))**2
          self.K = self.hw_meV


     def calc_parameters_until_second_order_from_JT_pars(self):
          self.K = self.hw_meV
          self.calc_Taylor_coeffs_K()

     def calc_parameters_until_second_order(self):
          self.calc_dists()
          self.calc_E_JT()
          self.calc_delta()


          


          self.calc_hw()


          self.calc_Taylor_coeffs_K()


     def set_quantum(self, hw):
          self.hw_meV = hw
          # Use calc_Taylor_coeffs_K if K is set, otherwise use calc_Taylor_coeffs_hw
          if hasattr(self, 'K') and self.K is not None:
               self.calc_Taylor_coeffs_K()
          else:
               self.calc_Taylor_coeffs_hw()
     
     def calc_Taylor_coeffs_hw(self):
          if not hasattr(self, 'E_JT_meV') or not hasattr(self, 'hw_meV'):
               raise ValueError("E_JT_meV and hw_meV must be set before calculating Taylor coefficients")
          if not hasattr(self, 'delta_meV'):
               self.delta_meV = 0.0
          sqrt_2 = math.sqrt(2.0)
          self.F = float((( 2*self.E_JT_meV*abs(self.hw_meV)*(1-self.delta_meV/(2*self.E_JT_meV-self.delta_meV)) )**0.5))
          self.G = float(abs(self.hw_meV)*self.delta_meV/(4*self.E_JT_meV - 2*self.delta_meV))

     def calc_Taylor_coeffs_K(self):
          if not hasattr(self, 'E_JT_meV') or not hasattr(self, 'K'):
               raise ValueError("E_JT_meV and K must be set before calculating Taylor coefficients")
          if not hasattr(self, 'delta_meV'):
               self.delta_meV = 0.0
          sqrt_2 = math.sqrt(2.0)
          self.F = float((( 2*self.E_JT_meV*abs(self.K)*(1-self.delta_meV/(2*self.E_JT_meV-self.delta_meV)) )**0.5))
          self.G = float(abs(self.K)*self.delta_meV/(4*self.E_JT_meV - 2*self.delta_meV))

     def calc_parameters_from_Taylor_coeffs(self):
          """
          Calculate E_JT, delta, JT_dist, and barrier_dist from Taylor coefficients F, G, and hw.
          
          This is the inverse of calc_Taylor_coeffs_hw().
          Uses direct formulas:
          - E_JT = F^2/(2*K - 2*|G|) where K = hw
          - barrier (delta) = 4*E_JT*G/(K + 2*|G|) where K = hw
          - JT_dist = F/(K - 2*G) where K = hw (dimensionless), then converted to Å √amu
          - barrier_dist = F/(K + 2*G) where K = hw (dimensionless), then converted to Å √amu
            (only when G is present, None otherwise)
          
          The dimensionless distances are converted to generalized coordinates using:
          V * q where V = 2.044543799 * 1/sqrt(hw)
          
          Returns:
              tuple: (E_JT_meV, delta_meV, JT_dist, barrier_dist) calculated from F, G, hw
              where distances are in units of Å √amu
          """
          if not hasattr(self, 'F') or not hasattr(self, 'hw_meV'):
               raise ValueError("F and hw_meV must be set before calculating parameters from Taylor coefficients")
          
          F = self.F
          hw = abs(self.hw_meV)
          K = hw  # K = hw in this context
          
          if F == 0 or hw == 0:
               raise ValueError("F and hw_meV must be non-zero")
          
          # Get G, default to 0 if not set
          if not hasattr(self, 'G') or self.G is None:
               G = 0.0
          else:
               G = self.G
          
          abs_G = abs(G)
          
          # Calculate E_JT using direct formula: E_JT = F^2/(2*K - 2*|G|)
          denominator = 2 * K - 2 * abs_G
          if denominator <= 0:
               raise ValueError(f"Cannot calculate E_JT: denominator (2*K - 2*|G|) = {denominator} must be positive")
          
          E_JT = (F**2) / denominator
          
          # Calculate barrier (delta) using direct formula: barrier = 4*E_JT*G/(K + 2*|G|)
          if abs_G == 0:
               delta = 0.0
          else:
               delta = (4 * E_JT * G) / (K + 2 * abs_G)
          
          # Calculate JT_dist using direct formula: JT_dist = F/(K - 2*G)
          # This gives dimensionless coordinate, convert to generalized coordinate
          dist_denominator = K - 2 * G
          if dist_denominator == 0:
               raise ValueError(f"Cannot calculate JT_dist: denominator (K - 2*G) = {dist_denominator} must be non-zero")
          
          JT_dist_dimensionless = F / dist_denominator
          # Convert to generalized coordinate (Å √amu)
          JT_dist = dimensionless_to_generalized_coordinate(JT_dist_dimensionless, hw)
          
          # Calculate barrier_dist using direct formula: barrier_dist = F/(K + 2*G)
          # Only when G is present (non-zero), otherwise barrier_dist = None
          if abs_G > 0:
               barrier_dist_denominator = K + 2 * G
               if barrier_dist_denominator == 0:
                    raise ValueError(f"Cannot calculate barrier_dist: denominator (K + 2*G) = {barrier_dist_denominator} must be non-zero")
               barrier_dist_dimensionless = F / barrier_dist_denominator
               # Convert to generalized coordinate (Å √amu)
               barrier_dist = dimensionless_to_generalized_coordinate(barrier_dist_dimensionless, hw)
          else:
               # When only F is present (G = 0), barrier_dist is None
               barrier_dist = None
          
          self.E_JT_meV = float(E_JT)
          self.delta_meV = float(delta)
          self.JT_dist = float(JT_dist)
          if barrier_dist is not None:
               self.barrier_dist = float(barrier_dist)
          else:
               self.barrier_dist = None
          
          return (self.E_JT_meV, self.delta_meV, self.JT_dist, self.barrier_dist)

